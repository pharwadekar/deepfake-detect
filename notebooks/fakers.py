import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.models as models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

def extract_fft(image):
    """
    Computes the Fast Fourier Transform of an RGB image.
    Returns a 3-channel magnitude spectrum image.
    """
    # Convert to grayscale for frequency analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform 2D FFT
    f = np.fft.rfft2(gray)
    fshift = np.fft.fftshift(f)
    
    # Calculate magnitude spectrum (log scale for visualization/training stability)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
    
    # Normalize to 0-255
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Convert back to 3 channels so it can be passed through standard CNNs like EfficientNet
    fft_3channel = cv2.cvtColor(magnitude_spectrum, cv2.COLOR_GRAY2BGR)
    return fft_3channel

class HybridDeepfakeDetector(nn.Module):
    def __init__(self, mode='hybrid'):
        super(HybridDeepfakeDetector, self).__init__()
        self.mode = mode
        assert self.mode in ['rgb', 'fft', 'hybrid'], "Mode must be 'rgb', 'fft', or 'hybrid'"
        
        # RGB Branch (EfficientNet-B0 as feature extractor)
        if self.mode in ['rgb', 'hybrid']:
            self.rgb_branch = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.num_ftrs = self.rgb_branch.classifier[1].in_features
            self.rgb_branch.classifier = nn.Identity() # Remove final classification layer
            
        # FFT Branch (Another EfficientNet-B0) Let's assume it also takes 3 channel input
        if self.mode in ['fft', 'hybrid']:
            self.fft_branch = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.num_ftrs = self.fft_branch.classifier[1].in_features
            self.fft_branch.classifier = nn.Identity()
            
        # Fully Connected classification head
        feature_dim = self.num_ftrs * 2 if self.mode == 'hybrid' else self.num_ftrs
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),   # 1 output node
            nn.Sigmoid()         # Output probabilities between 0 and 1
        )

    def forward(self, rgb_x=None, fft_x=None):
        features = []
        
        if self.mode in ['rgb', 'hybrid'] and rgb_x is not None:
            rgb_features = self.rgb_branch(rgb_x)
            features.append(rgb_features)
            
        if self.mode in ['fft', 'hybrid'] and fft_x is not None:
            fft_features = self.fft_branch(fft_x)
            features.append(fft_features)
            
        # Join branches together if hybrid
        combined = torch.cat(features, dim=1) if len(features) > 1 else features[0]
        
        out = self.fc(combined)
        return out.squeeze()

class DeepfakeImageDataset(Dataset):
    def __init__(self, image_paths, labels, mode='hybrid', transform=None):
        """
        image_paths: List of paths to .jpg files (pre-extracted from videos via preprocessing/extract_frames.py)
        labels: 0 for Real (Original), 1 for Fake (FaceSwap, Deepfakes, Face2Face)
        mode: 'rgb', 'fft', or 'hybrid'
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode
        assert self.mode in ['rgb', 'fft', 'hybrid']
        self.base_transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # FAST LOADING: Read from extracted .jpg files rather than opening a massive .mp4!
        try:
            frame = cv2.imread(self.image_paths[idx])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception: 
            # Fallback dummy array if path parsing fails
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            
        label = self.labels[idx]
        
        rgb_tensor = None
        fft_tensor = None
        
        # 1. Process RGB if required
        if self.mode in ['rgb', 'hybrid']:
            if self.transform:
                rgb_tensor = self.transform(image=frame)['image']
                
        # 2. Process FFT Frequency if required
        if self.mode in ['fft', 'hybrid']:
            fft_img = extract_fft(frame)
            # FFT frequency map rarely benefits from Gaussian noises/blur (they destroy it)
            # Thus, we typically just use the base_transform on the FFT channel
            fft_tensor = self.base_transform(image=fft_img)['image']
            
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        # Determine strict Tuple return formats depending on the architecture mode
        if self.mode == 'hybrid':
            return rgb_tensor, fft_tensor, label_tensor
        elif self.mode == 'rgb':
            return rgb_tensor, None, label_tensor
        elif self.mode == 'fft':
            return None, fft_tensor, label_tensor

def evaluate_model(model, dataloader, mode='hybrid', device="cpu"):

    model.eval()
    loss_fn = nn.BCELoss()
    
    y_true = []
    y_pred_probs = []
    running_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            rgb_imgs, fft_imgs, labels = batch
            labels = labels.to(device).float()
            
            if mode in ['rgb', 'hybrid']:
                rgb_imgs = rgb_imgs.to(device)
            if mode in ['fft', 'hybrid']:
                fft_imgs = fft_imgs.to(device)
                
            outputs = model(rgb_x=rgb_imgs, fft_x=fft_imgs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            num_batches += 1
            
            y_true.extend(labels.cpu().numpy())
            y_pred_probs.extend(outputs.cpu().numpy())

    avg_loss = running_loss / max(num_batches, 1)
    y_pred_classes = (np.array(y_pred_probs) > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred_classes)
    f1 = f1_score(y_true, y_pred_classes)
    auc = roc_auc_score(y_true, y_pred_probs)
    
    return avg_loss, accuracy, f1, auc

def evaluate_generalization_gap(model, in_domain_loader, cross_domain_loader, mode='hybrid', device="cpu"):
    """
    Specifically executes the evaluation framework to measure domain shift
    This is the core metric to track the true generalization improvement.
    """
    _, acc_in, f1_in, auc_in = evaluate_model(model, in_domain_loader, mode, device=device)
    _, acc_cross, f1_cross, auc_cross = evaluate_model(model, cross_domain_loader, mode, device=device)
    
    generalization_gap = acc_in - acc_cross
    
    print(f"--- Model Setup [{mode.upper()}] ---")
    print(f"In-domain (FF++)      : Acc={acc_in*100:.2f}%, F1={f1_in:.3f}, AUC={auc_in:.3f}")
    print(f"Cross-domain (CelebDF): Acc={acc_cross*100:.2f}%, F1={f1_cross:.3f}, AUC={auc_cross:.3f}")
    print(f"=====================================")
    print(f"GENERALIZATION GAP: {generalization_gap*100:.2f}%\n")
    
    return generalization_gap, (acc_in, acc_cross), (f1_in, f1_cross), (auc_in, auc_cross)