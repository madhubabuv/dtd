import torch


#These log_sinkhorn_iterations and log_optimal_transport functions are from superglue
def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z

class PartialOT(torch.nn.Module):

    def __init__(self,iters=10, temperature=0.1):

        super(PartialOT, self).__init__()
        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        self.temperature = temperature
        self.iters = iters

    def forward(self, feat0, feat1):

        shape = feat0.shape
        batch_size, dim, h, w = shape


        feat0 = feat0.permute(0,2,3,1)
        feat1 = feat1.permute(0,2,3,1)

        feat0 = feat0.reshape(batch_size, h*w, dim)
        feat1 = feat1.reshape(batch_size, h*w, dim)

        #feat0 = torch.nn.functional.normalize(feat0, dim=2)
        #feat1 = torch.nn.functional.normalize(feat1, dim=2)

        # similarity = torch.einsum('bik,bjk->bij', feat0, feat1)
        # similarity = similarity / self.temperature
        distance = torch.cdist(feat0, feat1, p=2)
        # optimial weights for similarity using partial optimal transport
        gamma = log_optimal_transport(distance, self.bin_score, self.iters)
        gamma = gamma.exp() # converting back to normal space from log space

        outputs = {}
        # matching loss
        matching_loss = torch.sum(gamma[:,:-1,:-1] * distance, dim = 2)
        matching_loss = matching_loss.mean()
        outputs['matching_loss'] = matching_loss

        #dustbin_loss
        dustbin_loss=gamma[:,:, -1].mean() + gamma[:,:-1,:].mean()
        outputs['dustbin_loss'] = dustbin_loss
        outputs['gamma'] = gamma

        return outputs





        









