import torch
import torch.nn as nn

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

class PhotometricLoss:

    def __init__(self, weights = [0.15,0.85]):

        self.weights = weights
        self.ssim = SSIM()

    def simple_photometric_loss(self,original_image, reconstructed_image, weights = [0.15,0.85]):

        l1_loss = torch.abs(original_image - reconstructed_image).mean(1,True)
        ssim_loss = self.ssim(original_image, reconstructed_image).mean(1,True)

        losses = [l1_loss, ssim_loss]

        weighted_loss = 0
        for i in range(len(weights)):
            weighted_loss += weights[i] * losses[i]

        return weighted_loss

    def identiy_photometric_loss(self, source_image, targate_image, weights = [0.15,0.85]):

        return self.simple_photometric_loss(source_image, targate_image, weights)

    def minimum_photometric_loss(self,original_image, reconstructed_images):
        

        losses = []
        for idx, recon_image in enumerate(reconstructed_images):
            photometric_loss = self.simple_photometric_loss(original_image, recon_image, self.weights)
            losses.append(photometric_loss)
        losses = torch.stack(losses, dim=1)

        return torch.min(losses, dim=1)[0]


class MultiScalePhotometricLoss(nn.Module):

    def __init__(self, full_scale = False):

        super(MultiScalePhotometricLoss, self).__init__()

        self.ssim = SSIM()

        self.full_scale = full_scale

    def simple_loss(self,original_image, reconstructed_image, weights = [0.15,0.85]):

        assert original_image.shape == reconstructed_image.shape

        l1_loss = torch.abs(original_image - reconstructed_image).mean(1,True)
        ssim_loss = self.ssim(original_image, reconstructed_image).mean(1,True)

        losses = [l1_loss, ssim_loss]

        weighted_loss = 0
        for i in range(len(weights)):
            weighted_loss += weights[i] * losses[i]

        return weighted_loss

    def forward(self, reconstructed_images,original_images, reduce_mean = True):

        assert len(reconstructed_images) == len(original_images)

        total_loss = 0.0
        for original, recon in zip(original_images, reconstructed_images):
            
            if self.full_scale:
                loss = self.simple_loss(original_images[0], recon)
            else:
                loss = self.simple_loss(original, recon)

            total_loss += loss.mean()

        total_loss = total_loss / len(original_images)
        
        return total_loss


if __name__ == "__main__":
    import cv2

    image = cv2.imread("/mnt/nas/kaichen/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/0000000000.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
    image = image / 255.0

    



   
