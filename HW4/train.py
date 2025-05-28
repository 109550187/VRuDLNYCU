import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from net.model import PromptIR
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.functional as F
import os
import numpy as np
import random
from PIL import Image
from torchvision.transforms import ToTensor
from utils.image_utils import crop_img

# ========== DATASET CLASSES ==========
class PromptTrainDataset(Dataset):
    def __init__(self, args):
        super(PromptTrainDataset, self).__init__()
        self.args = args
        self.rs_ids = []
        self.snow_ids = []
        self.de_type = self.args.de_type
        print(f"Dataset types: {self.de_type}")

        self.de_dict = {'derain': 3, 'desnow': 4}
        self._init_ids()
        self._merge_ids()
        self.toTensor = ToTensor()

    def _init_ids(self):
        if 'derain' in self.de_type:
            self._init_rs_ids()
        if 'desnow' in self.de_type:
            self._init_snow_ids()

    def _init_rs_ids(self):
        """Initialize rain image pairs"""
        degraded_rain_files = [f for f in os.listdir(self.args.train_degraded_dir) 
                              if f.startswith('rain-') and f.endswith('.png')]
        
        print(f"Found {len(degraded_rain_files)} rain degraded images")
        
        for degraded_file in degraded_rain_files:
            degraded_path = os.path.join(self.args.train_degraded_dir, degraded_file)
            clean_file = degraded_file.replace('rain-', 'rain_clean-')
            clean_path = os.path.join(self.args.train_clean_dir, clean_file)
            
            if os.path.exists(clean_path):
                self.rs_ids.append({
                    "degraded_id": degraded_path, 
                    "clean_id": clean_path, 
                    "de_type": 3
                })

        print(f"Total Rain image pairs: {len(self.rs_ids)}")

    def _init_snow_ids(self):
        """Initialize snow image pairs"""
        degraded_snow_files = [f for f in os.listdir(self.args.train_degraded_dir) 
                              if f.startswith('snow-') and f.endswith('.png')]
        
        print(f"Found {len(degraded_snow_files)} snow degraded images")
        
        for degraded_file in degraded_snow_files:
            degraded_path = os.path.join(self.args.train_degraded_dir, degraded_file)
            clean_file = degraded_file.replace('snow-', 'snow_clean-')
            clean_path = os.path.join(self.args.train_clean_dir, clean_file)
            
            if os.path.exists(clean_path):
                self.snow_ids.append({
                    "degraded_id": degraded_path, 
                    "clean_id": clean_path, 
                    "de_type": 4
                })

        print(f"Total Snow image pairs: {len(self.snow_ids)}")

    def _crop_patch(self, img_1, img_2):
        """Crop matching patches from both images"""
        H, W = img_1.shape[:2]
        ind_H = random.randint(0, H - self.args.patch_size)
        ind_W = random.randint(0, W - self.args.patch_size)

        patch_1 = img_1[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.args.patch_size, ind_W:ind_W + self.args.patch_size]

        return patch_1, patch_2

    def _merge_ids(self):
        """Combine rain and snow datasets"""
        self.sample_ids = []
        if "derain" in self.de_type:
            self.sample_ids += self.rs_ids
        if "desnow" in self.de_type:
            self.sample_ids += self.snow_ids
        
        print(f"Total training samples: {len(self.sample_ids)}")
        random.shuffle(self.sample_ids)

    def __getitem__(self, idx):
        sample = self.sample_ids[idx]
        de_id = sample["de_type"]

        try:
            degrad_img = crop_img(np.array(Image.open(sample["degraded_id"]).convert('RGB')), base=16)
            clean_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
            clean_name = os.path.basename(sample["clean_id"]).split('.')[0]
            
        except Exception as e:
            print(f"Error loading images: {e}")
            return self.__getitem__((idx + 1) % len(self.sample_ids))

        # Crop patches
        degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)

        # Convert to tensors
        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)

        return [clean_name, de_id], degrad_patch, clean_patch

    def __len__(self):
        return len(self.sample_ids)

# ========== LOSS FUNCTIONS ==========
class SSIMLoss(nn.Module):
    """SSIM loss for better perceptual quality"""
    def __init__(self, window_size=11):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, img1, img2):
        # Simplified SSIM implementation
        mu1 = F.avg_pool2d(img1, self.window_size, 1, self.window_size//2)
        mu2 = F.avg_pool2d(img2, self.window_size, 1, self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, 1, self.window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, 1, self.window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, 1, self.window_size//2) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

class EdgeLoss(nn.Module):
    """Edge-preserving loss"""
    def __init__(self):
        super(EdgeLoss, self).__init__()
        
    def forward(self, x, y):
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        
        if x.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()
            
        sobel_x = sobel_x.repeat(x.shape[1], 1, 1, 1)
        sobel_y = sobel_y.repeat(x.shape[1], 1, 1, 1)
        
        edge_x = F.conv2d(x, sobel_x, padding=1, groups=x.shape[1])
        edge_y = F.conv2d(y, sobel_y, padding=1, groups=y.shape[1])
        
        edge_pred = torch.sqrt(edge_x**2 + edge_y**2)
        
        edge_x_target = F.conv2d(y, sobel_x, padding=1, groups=y.shape[1])
        edge_y_target = F.conv2d(y, sobel_y, padding=1, groups=y.shape[1])
        edge_target = torch.sqrt(edge_x_target**2 + edge_y_target**2)
        
        return F.l1_loss(edge_pred, edge_target)

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps)
        return torch.mean(loss)

# ========== LIGHTNING MODEL ==========
class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        
        # Multiple loss functions for better PSNR
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.edge_loss = EdgeLoss()
        
        self.lambda_l1 = 1.0
        self.lambda_ssim = 0.3
        self.lambda_edge = 0.1
        
        self.psnr_values = []

    def forward(self, x, de_id=None):
        return self.net(x)

    def calculate_psnr(self, pred, target, data_range=1.0):
        mse = F.mse_loss(pred, target, reduction='mean')
        if mse == 0:
            return float('inf')
        max_value = torch.tensor(data_range, device=pred.device) 
        psnr = 20 * torch.log10(max_value) - 10 * torch.log10(mse)
        return psnr

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored, task = self.net(degrad_patch)

        # Multi-loss approach for better PSNR
        l1_loss = self.l1_loss(restored, clean_patch)
        ssim_loss = self.ssim_loss(restored, clean_patch)
        edge_loss = self.edge_loss(restored, clean_patch)
        
        # total_loss = (self.lambda_l1 * l1_loss + 
        #              self.lambda_ssim * ssim_loss + 
        #              self.lambda_edge * edge_loss)
        #total_loss = l1_loss

        try:
            ssim_loss = self.ssim_loss(restored, clean_patch)
            if torch.isnan(ssim_loss):
                ssim_loss = torch.tensor(0.0, device=restored.device)
        except:
            ssim_loss = torch.tensor(0.0, device=restored.device)
        
        # Weighted combination (conservative weights)
        total_loss = l1_loss + 0.1 * ssim_loss

        psnr = self.calculate_psnr(restored, clean_patch, data_range=1.0)
        self.psnr_values.append(psnr.item())

        # Log metrics
        self.log("train_loss", total_loss, on_step=True, on_epoch=True)
        self.log("train_l1_loss", l1_loss, on_step=True, on_epoch=True)
        self.log("train_ssim_loss", ssim_loss, on_step=True, on_epoch=True)
        self.log("train_edge_loss", edge_loss, on_step=True, on_epoch=True)
        self.log("train_psnr", psnr.item(), on_step=True, on_epoch=True)

        return total_loss

    def on_train_epoch_end(self):
        if self.psnr_values:
            avg_psnr = sum(self.psnr_values) / len(self.psnr_values)
            self.log("epoch_avg_psnr", avg_psnr, on_epoch=True, prog_bar=True)
            print(f"Epoch {self.current_epoch} - Average PSNR: {avg_psnr:.2f} dB")
            self.psnr_values = []

    def configure_optimizers(self):
        # Enhanced optimizer settings
        optimizer = optim.AdamW(self.parameters(), lr=opt.lr, weight_decay=1e-4)
        
        # Cosine annealing with warm restarts for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        }

# ========== MAIN TRAINING FUNCTION ==========
def main():
    torch.set_float32_matmul_precision('medium')
    
    print("Enhanced PromptIR Training")
    print(f"Batch size: {opt.batch_size}")
    print(f"Epochs: {opt.epochs}")
    print(f"Learning rate: {opt.lr}")
    print(f"Dataset paths:")
    print(f"  Train degraded: {opt.train_degraded_dir}")
    print(f"  Train clean: {opt.train_clean_dir}")
    print(f"  Test: {opt.test_dir}")
    
    # Test dataset loading
    print("\nTesting dataset loading...")
    try:
        trainset = PromptTrainDataset(opt)
        print(f"Dataset loaded successfully with {len(trainset)} samples")
        
        sample = trainset[0]
        print(f"Sample shapes: degraded={sample[1].shape}, clean={sample[2].shape}")
        print("Dataset test passed!")
        
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        return
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir, 
        filename='enhanced-promptir-{epoch:02d}-{epoch_avg_psnr:.2f}',
        monitor='epoch_avg_psnr',
        mode='max',
        save_top_k=3,
        every_n_epochs=1
    )
    
    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers,
        persistent_workers=True
    )
    
    model = PromptIRModel()
    
    ckpt_path = None
    if hasattr(opt, 'ckpt_name') and opt.ckpt_name: 
        ckpt_path = os.path.join(opt.ckpt_dir, opt.ckpt_name)
        print(f"Resuming training from checkpoint: {ckpt_path}")
    
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu",
        devices=[0, 1],
        precision=16,
        strategy='ddp_find_unused_parameters_true',
        logger=TensorBoardLogger(save_dir="logs/", name="enhanced_promptir"),
        callbacks=[checkpoint_callback],
        gradient_clip_val=0.5,
        accumulate_grad_batches=1,
    )
    
    print("\nStarting enhanced training...")
    trainer.fit(model=model, train_dataloaders=trainloader, ckpt_path=ckpt_path)
    print("Training completed!")

if __name__ == '__main__':
    main()