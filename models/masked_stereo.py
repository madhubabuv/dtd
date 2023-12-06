import torch
import torch.nn as nn
import torch.nn.functional as F
from unimatch.unimatch.transformer import FeatureTransformer
from unimatch.unimatch.matching import global_correlation_softmax_stereo, local_correlation_softmax_stereo
from unimatch.unimatch.attention import SelfAttnPropagation
from unimatch.unimatch.geometry import flow_warp
from unimatch.unimatch.reg_refine import BasicUpdateBlock
from unimatch.unimatch.utils import feature_add_position, upsample_flow_with_mask
from matplotlib import pyplot as plt

def histogram_test(features, post_fix = 'full'):


    feat = features.clone().squeeze().detach().cpu()
    feat = feat.view(feat.shape[0], -1)
    feat = feat.permute(1, 0)
    norm_feat = torch.nn.functional.normalize(feat, dim=1)

    dot = torch.mm(norm_feat, norm_feat.t())

    #fill diagonal with -1
    n = dot.shape[0]
    dot.view(-1)[:: (n + 1)].fill_(-1)
    
    top_100_vals, top_100_inds = torch.topk(dot, 100, largest=True)
    
    r1_dists = torch.nn.functional.pairwise_distance(norm_feat, norm_feat[top_100_inds[:, 0]])
    r5_dists = torch.nn.functional.pairwise_distance(norm_feat, norm_feat[top_100_inds[:, 5]])

    r1_dists = r1_dists.numpy()
    r5_dists = r5_dists.numpy()

    plt.clf()
    plt.hist(r1_dists, bins = 50, histtype='step')
    plt.hist(r5_dists, bins = 50, histtype='step')
    plt.legend(["NN - 1", "NN - 5"])
    plt.xlabel("Pairwise Distance")
    plt.ylabel("Count")
    
    plt.savefig("distance_hist_{}.png".format(post_fix))



class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def mu(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the average across
        it's spatial dimensions as (h,w) tensor [See eq. 5 of paper]"""
        return torch.sum(x,(2,3))/(x.shape[2]*x.shape[3])

    def sigma(self, x):
        """ Takes a (n,c,h,w) tensor as input and returns the standard deviation
        across it's spatial dimensions as (h,w) tensor [See eq. 6 of paper] Note
        the permutations are required for broadcasting"""
        return torch.sqrt((torch.sum((x.permute([2,3,0,1])-self.mu(x)).permute([2,3,0,1])**2,(2,3))+0.000000023)/(x.shape[2]*x.shape[3]))

    def forward(self, feat):
        """ Takes a content embeding x and a style embeding y and changes
        transforms the mean and standard deviation of the content embedding to
        that of the style. [See eq. 8 of paper] Note the permutations are
        required for broadcasting"""

        batch_size = feat.shape[0]//2
        y = feat[:batch_size] # night
        x = feat[batch_size:] # day 

        #we first remove the day style and add night style 
        # so that idea is that we want to remove the day style and add the night style and estimate disparity
        # and calculate photometric losses on day-time images with night style

        trans_x =  (self.sigma(y)*((x.permute([2,3,0,1])-self.mu(x))/self.sigma(x)) + self.mu(y)).permute([2,3,0,1])

        return torch.cat([y, trans_x], dim = 0)

class UniMatch(nn.Module):
    def __init__(self,
                 num_scales=1,
                 feature_channels=128,
                 upsample_factor=8,
                 num_head=1,
                 ffn_dim_expansion=4,
                 num_transformer_layers=6,
                 reg_refine=False,  # optional local regression refinement
                 task='flow',
                 input_dim = 384,
                 output_dim = 128
                 ):
        super(UniMatch, self).__init__()

        self.feature_channels = feature_channels
        self.num_scales = num_scales
        self.upsample_factor = upsample_factor
        self.reg_refine = reg_refine

        # CNN
        self.backbone = None
        self.projector_1 = nn.Sequential(nn.Conv2d(input_dim, 256, 1),
                               nn.ReLU(inplace=True),
                               nn.Conv2d(256, output_dim, 1))

        self.projector_2 = nn.Sequential(nn.Conv2d(input_dim, 256, 1),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(256, output_dim, 1))

        self.projector = [self.projector_1, self.projector_2]

        # self.projector = nn.Sequential(nn.Conv2d(input_dim, input_dim, 1, bias=False),
        #                                 nn.BatchNorm2d(input_dim),
        #                                 nn.ReLU(inplace=True), # first layer
        #                                 nn.Conv2d(input_dim, input_dim,1,bias=False),
        #                                 nn.BatchNorm2d(input_dim),
        #                                 nn.ReLU(inplace=True), # second layer
        #                                 nn.Conv2d(input_dim, output_dim,1, bias=False),
        #                                 nn.BatchNorm2d(output_dim, affine=False)) # output layer


        # Transformer
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )
        self.d_n_transformer = FeatureTransformer(num_layers=num_transformer_layers,
                                              d_model=feature_channels,
                                              nhead=num_head,
                                              ffn_dim_expansion=ffn_dim_expansion,
                                              )

        self.predictor = nn.Sequential(nn.Conv2d(output_dim, output_dim, 1),
                        nn.BatchNorm2d(output_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(output_dim, output_dim, 1))


        # propagation with self-attn
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)


        if not self.reg_refine:
            # convex upsampling simiar to RAFT
            # concat feature0 and low res flow as input
            self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))
            # thus far, all the learnable parameters are task-agnostic

        if reg_refine:
            # optional task-specific local regression refinement
            self.refine_proj = nn.Conv2d(feature_channels, 256, 1)
            self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2,
                                           downsample_factor=upsample_factor,
                                           flow_dim=2 if task == 'flow' else 1,
                                           bilinear_up=task == 'depth',
                                           )

        self.threshold = torch.nn.Parameter(torch.tensor(0.2), requires_grad=False)

        if self.train:
            self.ada_in = AdaIN()

    
    def extract_feature(self, img0, img1):
        
        # this should be from DINO

        return None, None

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8,
                      is_depth=False):
        if bilinear:
            multiplier = 1 if is_depth else upsample_factor
            up_flow = F.interpolate(flow, scale_factor=upsample_factor,
                                    mode='bilinear', align_corners=True) * multiplier
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            up_flow = upsample_flow_with_mask(flow, mask, upsample_factor=self.upsample_factor,
                                              is_depth=is_depth)

        return up_flow

    def get_mask(self, features:torch.tensor, thr:float = 0.2)->torch.tensor:

        '''
        This function is compute the mask the features that are not unique enought to have 
        the distance grather than a threshold

        Args:
            feature: the feature map of the image
            thr: the threshold of the distance

        Returns:
            mask: the mask of the feature map
    
        '''

        normlize_features = torch.nn.functional.normalize(features, dim=1)
        re_feat = normlize_features.permute(0, 2, 3, 1).contiguous()
        re_feat = re_feat.view(re_feat.shape[0], -1, re_feat.shape[3])
        similarity = torch.bmm(re_feat, re_feat.permute(0, 2, 1).contiguous())
        nearest_nn_inds = torch.topk(similarity, k=2, dim=-1, largest=True)[1][:, :, 1]
        nearest_nn = torch.gather(re_feat, dim=1, index=nearest_nn_inds.unsqueeze(-1).expand(-1, -1, re_feat.shape[-1]))
        distance = torch.norm(re_feat - nearest_nn, dim=-1)
        mask = torch.where(distance > thr, torch.ones_like(distance), torch.zeros_like(distance))
        mask = mask.view(features.shape[0],1, features.shape[2], features.shape[3])

        return mask, distance


    def forward(self, img0, img1,
                attn_type=None,
                attn_splits_list=None,
                corr_radius_list=None,
                prop_radius_list=None,
                task='stereo',
                ):


        results_dict = {}
        flow_preds = []
        residual_flows = []


        with torch.no_grad():
            # list of features, resolution low to high
            feature0_list, feature1_list = self.extract_feature(img0, img1)  # list of features
            #print(self.train)
            # if self.train:


        flow = None

        assert len(attn_splits_list) == len(corr_radius_list) == len(prop_radius_list) == self.num_scales

        thr = [0.2,0.2] # change this to [0.2,0.2] in case of dino_v1 and the resolution is 192x320
        distances = []
        masks = []
        all_features = []
        before_features = []
        after_features = []
        for scale_idx in range(self.num_scales):
            feature0, feature1 = feature0_list[scale_idx].detach(), feature1_list[scale_idx].detach()
            #if scale_idx == 0:
            #    histogram_test(feature0,post_fix = 'before')
            feature0 = self.projector[scale_idx](feature0)
            feature1 = self.projector[scale_idx](feature1)
            #all_features.append(feature0)

            #if scale_idx == 0:
            #    histogram_test(feature0.detach(),post_fix = 'after')        
            #bad_pixel_mask, nearest_neighbour_distances = self.get_mask(feature0,thr[scale_idx])
            #distances.append(nearest_neighbour_distances)
            #masks.append(bad_pixel_mask)
            
            # if not scale_idx:
            #     flat_features = feature0.flatten(2).contiguous()  # [B, C, H*W]
                
            #     #flat_mask = bad_pixel_mask.flatten(2).contiguous()
            #     # masked_features = flat_features * flat_mask
            #     # feature_sum = masked_features.sum(2,True) 
            #     # mask_sum = flat_mask.sum(2,True)
            #     # feature_mean = feature_sum / mask_sum
            #     # flat_features = feature_mean.unsqueeze(2)

            #     flat_features = flat_features.mean(dim=2, keepdim=True).unsqueeze(2)  # [B, C, 1, 1]
            #     before_features.append(flat_features.squeeze())
            #     pred_feature0 = self.predictor(flat_features)
            #     after_features.append(pred_feature0.squeeze())

            #bad_pixel_mask = bad_pixel_mask.detach()
            upsample_factor = self.upsample_factor * (2 ** (self.num_scales - 1 - scale_idx))

            if scale_idx > 0:
                assert task != 'depth'  # not supported for multi-scale depth model
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2

            if flow is not None:
                assert task != 'depth'
                flow = flow.detach()

                if task == 'stereo':
                    # construct flow vector for disparity
                    # flow here is actually disparity
                    zeros = torch.zeros_like(flow)  # [B, 1, H, W]
                    # NOTE: reverse disp, disparity is positive
                    displace = torch.cat((-flow, zeros), dim=1)  # [B, 2, H, W]
                    feature1 = flow_warp(feature1, displace)  # [B, C, H, W]
                elif task == 'flow':
                    feature1 = flow_warp(feature1, flow)  # [B, C, H, W]
                else:
                    raise NotImplementedError

            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            prop_radius = prop_radius_list[scale_idx]

            # add position to features
            feature0, feature1 = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
        

            # lets mix day-night features

        




            # lets mix left right featurs
            feature0, feature1 = self.transformer(feature0, feature1,
                                                  attn_type=attn_type,
                                                  attn_num_splits=attn_splits,
                                                  )



                      
            if corr_radius == -1:  # global matching
                flow_pred = global_correlation_softmax_stereo(feature0, feature1)[0]
            elif task == 'stereo':
                flow_pred = local_correlation_softmax_stereo(feature0, feature1, corr_radius)[0]
            else:
                raise NotImplementedError

            #flow_pred = flow_pred * bad_pixel_mask


            # flow or residual flow
            flow = flow + flow_pred if flow is not None else flow_pred


            flow = flow.clamp(min=0)  # positive disparity

            # upsample to the original resolution for supervison at training time only
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor,
                                                   is_depth=task == 'depth')
                
                flow_preds.append(flow_bilinear)

            flow = self.feature_flow_attn(feature0, flow.detach(),
                                          local_window_attn=prop_radius > 0,
                                          local_window_radius=prop_radius,
                                          )
            # bilinear exclude the last one and for the last one use RAFT - style upsampling
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True,
                                             upsample_factor=upsample_factor,
                                             is_depth=task == 'depth')
                flow_preds.append(flow_up)

            if scale_idx == self.num_scales - 1:
                flow_pad = torch.cat((-flow, torch.zeros_like(flow)), dim=1)  # [B, 2, H, W]
                flow_up_pad = self.upsample_flow(flow_pad, feature0)
                flow_up = -flow_up_pad[:, :1]  # [B, 1, H, W]


                flow_preds.append(flow_up)
                
        for i in range(len(flow_preds)):
            flow_preds[i] = flow_preds[i].squeeze(1)  # [B, H, W]

        for j in range(len(residual_flows)):
            residual_flows[j] = residual_flows[j].squeeze(1)
        results_dict.update({'flow_preds': flow_preds})
        results_dict.update({'nn_distance': distances})
        results_dict.update({'bad_pixel_mask': masks})
        results_dict.update({'features': all_features})
        results_dict.update({'before_features': before_features})
        results_dict.update({'after_features': after_features})

        return results_dict
