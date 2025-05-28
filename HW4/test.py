import os
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from net.model import PromptIR
from options import options as opt
import lightning.pytorch as pl
import glob
import torch.nn.functional as F
from utils.image_utils import crop_img

# ========== TEST DATASET ==========
class TestSpecificDataset(Dataset):
    def __init__(self, args):
        super(TestSpecificDataset, self).__init__()
        self.args = args
        self.degraded_ids = []
        self._init_test_ids(args.test_dir)
        self.toTensor = ToTensor()

    def _init_test_ids(self, root):
        """Initialize test dataset - expects files named 0.png to 99.png"""
        extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
        
        if os.path.isdir(root):
            name_list = []
            for image_file in os.listdir(root):
                if any([image_file.endswith(ext) for ext in extensions]):
                    name_list.append(image_file)
            
            if len(name_list) == 0:
                raise Exception('The input directory does not contain any image files')
            
            name_list.sort(key=lambda x: int(x.split('.')[0]))
            self.degraded_ids = [os.path.join(root, id_) for id_ in name_list]
        else:
            raise Exception('Test directory not found')
        
        print(f"Total test images: {len(self.degraded_ids)}")
        self.num_img = len(self.degraded_ids)

    def __getitem__(self, idx):
        degraded_img = crop_img(np.array(Image.open(self.degraded_ids[idx]).convert('RGB')), base=16)
        name = os.path.basename(self.degraded_ids[idx]).split('.')[0]
        degraded_img = self.toTensor(degraded_img)
        
        return [name], degraded_img

    def __len__(self):
        return self.num_img

# ========== LIGHTNING MODEL (same as training) ==========
class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)

    def forward(self, x, de_id=None):
        return self.net(x)

    def calculate_psnr(self, pred, target, data_range=1.0):
        mse = F.mse_loss(pred, target, reduction='mean')
        if mse == 0:
            return float('inf')
        max_value = torch.tensor(data_range, device=pred.device)
        psnr = 20 * torch.log10(max_value) - 10 * torch.log10(mse)
        return psnr.item()

# ========== TESTING FUNCTION ==========
def test_and_generate_predictions(net, dataset):
    """Test the model and generate pred.npz file for competition submission"""
    predictions = {}
    output_path_derain = os.path.join(opt.output_path, "derain")
    output_path_desnow = os.path.join(opt.output_path, "desnow")
    os.makedirs(output_path_derain, exist_ok=True)
    os.makedirs(output_path_desnow, exist_ok=True)

    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    net.eval()
    with torch.no_grad():
        for ([degraded_name], degrad_patch) in tqdm(testloader, desc="Processing test images"):
            degrad_patch = degrad_patch.cuda()

            restored, task = net(degrad_patch) 
            print(f"Image {degraded_name[0]}: detected as {task[0]}")
            restored_np = restored.cpu().numpy().squeeze()  # Remove batch dimension
            if restored_np.ndim == 3 and restored_np.shape[0] == 3:
                restored_np = np.clip(restored_np * 255, 0, 255).astype(np.uint8)
            else:
                print(f"Warning: Unexpected shape {restored_np.shape} for image {degraded_name[0]}")
                continue

            predictions[degraded_name[0] + '.png'] = restored_np
            restored_img = restored.cpu().squeeze().permute(1, 2, 0).numpy()
            restored_img = np.clip(restored_img * 255, 0, 255).astype(np.uint8)
            output_path = output_path_derain if task[0] == "derain" else output_path_desnow
            Image.fromarray(restored_img).save(os.path.join(output_path, degraded_name[0] + '.png'))

    # Save the final submission file
    np.savez(os.path.join(opt.output_path, 'pred.npz'), **predictions)
    print(f"Saved pred.npz with {len(predictions)} images to {os.path.join(opt.output_path, 'pred.npz')}")
    verify_submission_file(os.path.join(opt.output_path, 'pred.npz'))
    
    return predictions

def verify_submission_file(npz_path):
    """Verify the submission file format"""
    try:
        data = np.load(npz_path)
        print(f"\nSubmission file verification:")
        print(f"Number of images: {len(data.files)}")
        print(f"First few filenames: {list(data.files)[:5]}")
        
        sample_key = list(data.files)[0]
        sample_img = data[sample_key]
        print(f"Sample image '{sample_key}' shape: {sample_img.shape}")
        print(f"Sample image data type: {sample_img.dtype}")
        print(f"Sample image value range: [{sample_img.min()}, {sample_img.max()}]")
        
        all_correct = True
        for filename in data.files:
            img = data[filename]
            if img.shape[0] != 3 or img.ndim != 3:
                print(f"Error: Image {filename} has incorrect shape {img.shape}")
                all_correct = False
        
        if all_correct:
            print("✓ All images have correct format (3, H, W)")
        else:
            print("✗ Some images have incorrect format")
            
    except Exception as e:
        print(f"Error verifying submission file: {e}")

def main():
    print("Enhanced PromptIR Testing")
    torch.cuda.set_device(opt.cuda)
    
    # Find best checkpoint
    if opt.ckpt_name:
        ckpt_path = os.path.join(opt.ckpt_dir, opt.ckpt_name)
    else:
        ckpt_files = glob.glob(os.path.join(opt.ckpt_dir, "*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files found in {opt.ckpt_dir}/")
        # Get latest checkpoint
        ckpt_path = max(ckpt_files, key=os.path.getctime)
    
    print(f"Loading checkpoint: {ckpt_path}")
    
    # Load trained model
    net = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()
    
    # Load test dataset
    print(f"Loading test dataset from: {opt.test_dir}")
    test_dataset = TestSpecificDataset(opt)
    
    # Create output directory
    os.makedirs(opt.output_path, exist_ok=True)
    
    # Run testing and generate predictions
    print("Starting inference...")
    predictions = test_and_generate_predictions(net, test_dataset)
    
    print(f"\nTesting completed!")
    print(f"Results saved to: {opt.output_path}")
    print(f"Submission file: {os.path.join(opt.output_path, 'pred.npz')}")
    print("Ready for competition submission!")

if __name__ == '__main__':
    main()