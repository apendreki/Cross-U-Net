import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    accuracy_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import numpy as np
import re

# ------------------------------
# 손실 함수 정의
# ------------------------------

def dice_loss(pred, target, smooth=1.):
    """
    Dice Loss 계산 함수
    """
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

def combined_loss(pred, target):
    """
    BCEWithLogitsLoss와 Dice Loss를 결합한 손실 함수
    """
    bce = nn.BCEWithLogitsLoss()(pred, target)
    d_loss = dice_loss(pred, target)
    return bce + d_loss

# ------------------------------
# CrossUNet 모델 정의
# ------------------------------
# Attention Mechanism
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# UNet Block with Residual Connection
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # Residual Connection
        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        residual = self.residual(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x + residual  # Residual Connection

# Decoder Block with bilinear upsampling
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(DecoderBlock, self).__init__()
        # 수정된 부분: in_channels + skip_channels로 변경
        self.conv = EncoderBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        # Bilinear upsampling
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            if x.size()[2:] != skip.size()[2:]:
                skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# CrossUNet Definition
class CrossUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super(CrossUNet, self).__init__()

        # Shared Encoder
        self.shared_encoder1 = EncoderBlock(input_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.shared_encoder2 = EncoderBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.shared_encoder3 = EncoderBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # A Path (original structure)


        # B Path (channel reduction)


        # Attention Gate


        # Final Decoder
        self.final_decoder1 = DecoderBlock(in_channels=128, out_channels=128, skip_channels=256)  # 수정됨
        self.final_decoder2 = DecoderBlock(in_channels=128, out_channels=64, skip_channels=128)    # 수정됨
        self.final_decoder3 = DecoderBlock(in_channels=64, out_channels=64, skip_channels=64)      # 수정됨

        # Final Output Layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Shared Encoder Path
        enc1 = self.shared_encoder1(x) # [B, 64, H, W]
        enc1_pool = self.pool1(enc1) # [B, 64, H/2, W/2]
        enc2 = self.shared_encoder2(enc1_pool) # [B, 128, H/2, W/2]
        enc2_pool = self.pool2(enc2) # [B, 128, H/4, W/4]
        enc3 = self.shared_encoder3(enc2_pool) # [B, 256, H/4, W/4]
        enc3_pool = self.pool3(enc3) # [B, 256, H/8, W/8]

        # B Path


        # A Path


        # B Path Decoder

        # Attention Gate

 
        # Final Decoder
        final_dec1 = self.final_decoder1(merged, skip=enc3) # [B, 128, H/2, W/2]
        final_dec2 = self.final_decoder2(final_dec1, skip=enc2) # [B, 64, H, W]

        # 마지막 디코더에서 fovs 크기와 맞추기 위해 크기를 [576, 576]으로 유지
        final_dec3 = F.interpolate(self.final_decoder3(final_dec2, skip=enc1), size=(576, 576), mode='bilinear', align_corners=False)# [B, 64, H, W]

        output = self.final_conv(final_dec3) # [B, num_classes, H, W]

        return output


# ------------------------------
# 데이터셋 클래스 정의
# ------------------------------

class DriveDataset(Dataset):
    def __init__(self, image_dir, mask_dir, fov_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.fov_dir = fov_dir
        self.transform = transform

        self.image_list = sorted(os.listdir(image_dir))
        self.mask_list = sorted(os.listdir(mask_dir))
        self.fov_list = sorted(os.listdir(fov_dir))

        # 이미지, 마스크, FOV 파일의 매칭을 위한 리스트 생성
        self.image_mask_fov_pairs = []
        pattern = re.compile(r'^(\d+)')
        for image_file in self.image_list:
            # 이미지 번호 추출
            image_num_match = pattern.match(image_file)
            if image_num_match:
                image_num = image_num_match.group(1)
                # 마스크 파일 찾기 (1st_manual)
                mask_file = next(
                    (f for f in self.mask_list if pattern.match(f)
                     and pattern.match(f).group(1) == image_num), None)
                # FOV 파일 찾기 (mask)
                fov_file = next(
                    (f for f in self.fov_list if pattern.match(f)
                     and pattern.match(f).group(1) == image_num), None)
                if mask_file and fov_file:
                    self.image_mask_fov_pairs.append(
                        (image_file, mask_file, fov_file)
                    )
                else:
                    print(
                        f"No corresponding mask or FOV found for image {image_file}"
                    )
            else:
                print(
                    f"Could not extract image number from filename {image_file}"
                )

    def __len__(self):
        return len(self.image_mask_fov_pairs)

    def __getitem__(self, idx):
        image_file, mask_file, fov_file = self.image_mask_fov_pairs[idx]
        image_path = os.path.join(self.image_dir, image_file)
        mask_path = os.path.join(self.mask_dir, mask_file)
        fov_path = os.path.join(self.fov_dir, fov_file)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        fov = Image.open(fov_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            fov = self.transform(fov)
        else:
            transform_ops = transforms.Compose([
                transforms.Resize((576, 576)),  # 입력 이미지 크기 조정
                transforms.ToTensor()
            ])
            image = transform_ops(image)
            mask = transform_ops(mask)
            fov = transform_ops(fov)

        # 마스크와 FOV를 이진화 (임계값 0.5 기준)
        mask = (mask > 0.5).float()
        fov = (fov > 0.5).float()

        return image, mask, fov

# ------------------------------
# 클래스 불균형 처리 (선택 사항)
# ------------------------------

def calculate_pos_weight(dataset, device='cuda'):
    """
    전체 데이터셋을 반복하여 pos_weight를 계산합니다.
    """
    total_positive = 0
    total_negative = 0
    for _, mask, fov in dataset:
        mask = mask.to(device)
        fov = fov.to(device)
        mask = mask * fov  # FOV 마스크 적용
        total_positive += mask.sum().item()
        total_negative += (1 - mask).sum().item()
    pos_weight = torch.tensor([total_negative / (total_positive + 1e-6)]).to(device)
    return pos_weight

# ------------------------------
# 학습 및 테스트 코드
# ------------------------------

def train_model(
    model, train_loader, optimizer, num_epochs=25, device='cuda'
):
    """
    모델을 학습시키는 함수입니다.
    검증 데이터셋을 사용하지 않습니다.
    """
    model.train()
    train_losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels, fovs in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            fovs = fovs.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            # FOV 마스크 적용
            outputs = outputs * fovs
            labels = labels * fovs

            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')

        # 에포크마다 일부 예측 시각화
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                sample_inputs, sample_labels, sample_fovs = next(iter(train_loader))
                sample_inputs = sample_inputs.to(device)
                sample_labels = sample_labels.to(device)
                sample_fovs = sample_fovs.to(device)

                sample_outputs = model(sample_inputs)
                sample_outputs = torch.sigmoid(sample_outputs)
                sample_preds = sample_outputs > 0.5

                # FOV 마스크 적용
                sample_preds = sample_preds & sample_fovs.bool()
                sample_labels = sample_labels.bool() & sample_fovs.bool()

                # 첫 번째 샘플 시각화
                input_img = sample_inputs[0].cpu().permute(1, 2, 0).numpy()
                input_img = (input_img * 255).astype(np.uint8)
                label_img = sample_labels[0].cpu().squeeze().numpy().astype(np.uint8) * 255
                pred_img = sample_preds[0].cpu().squeeze().numpy().astype(np.uint8) * 255
                fov_img = sample_fovs[0].cpu().squeeze().numpy().astype(np.uint8) * 255

                # 시각화 코드 유지
                plt.figure(figsize=(16, 4))
                plt.subplot(1, 4, 1)
                plt.imshow(input_img)
                plt.title('Input Image')
                plt.axis('off')

                plt.subplot(1, 4, 2)
                plt.imshow(label_img, cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')

                plt.subplot(1, 4, 3)
                plt.imshow(pred_img, cmap='gray')
                plt.title('Prediction')
                plt.axis('off')

                plt.subplot(1, 4, 4)
                plt.imshow(fov_img, cmap='gray')
                plt.title('FOV Mask')
                plt.axis('off')

                plt.tight_layout()
                os.makedirs('test_results/visualizations', exist_ok=True)
                plt.savefig(f'test_results/visualizations/epoch_{epoch+1}_prediction.png')
                plt.close()
            model.train()

    # 학습 손실 곡선 저장
    plt.figure()
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig('training_loss_curve.png')
    plt.close()

    return model

def compute_metrics(preds, labels):
    """
    예측값과 실제값을 기반으로 다양한 성능 지표를 계산합니다.
    """
    preds = preds.flatten()
    labels = labels.flatten()

    # 데이터 타입 변환
    preds = preds.astype(np.uint8)
    labels = labels.astype(np.uint8)

    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    accuracy = accuracy_score(labels, preds)

    # ROC AUC 계산 (예외 처리 추가)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = 0.0  # 모든 샘플이 한 클래스일 경우

    # 혼동 행렬 계산 (예외 처리 추가)
    cm = confusion_matrix(labels, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    # 민감도(Sensitivity) 계산
    if (tp + fn) > 0:
        sensitivity = tp / (tp + fn)
    else:
        sensitivity = 0.0

    # 특이도(Specificity) 계산
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    return precision, recall, f1, accuracy, auc, specificity, sensitivity

def test_model(
    model, dataloader, device='cuda', save_results=False, result_dir='results'
):
    """
    모델을 테스트하고 성능 지표를 계산하는 함수입니다.
    """
    model.eval()
    all_preds = []
    all_labels = []

    if save_results:
        os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (inputs, labels, fovs) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            fovs = fovs.to(device)

            outputs = model(inputs)
            outputs = outputs.squeeze(1)  # [B, H, W]
            outputs = torch.sigmoid(outputs)
            preds = outputs > 0.5  # Threshold

            # 데이터 타입 변환 및 FOV 마스크 적용
            preds = preds.bool()
            labels = labels.squeeze(1).bool()
            fovs = fovs.squeeze(1).bool()

            preds = preds & fovs
            labels = labels & fovs

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            if save_results:
                # 예측 결과를 이미지로 저장
                pred_img = preds.cpu().numpy()[0].astype(np.uint8) * 255  # [H, W]
                Image.fromarray(pred_img).save(
                    os.path.join(result_dir, f'pred_{idx+1}.png')
                )

                # 원본 이미지, 실제 마스크, 예측 마스크 시각화하여 저장
                input_img = inputs.cpu().numpy()[0].transpose(1, 2, 0)  # [H, W, C]
                input_img = (input_img * 255).astype(np.uint8)
                label_img = labels.cpu().numpy()[0].astype(np.uint8) * 255
                fov_img = fovs.cpu().numpy()[0].astype(np.uint8) * 255

                plt.figure(figsize=(16, 4))
                plt.subplot(1, 4, 1)
                plt.imshow(input_img)
                plt.title('Input Image')
                plt.axis('off')

                plt.subplot(1, 4, 2)
                plt.imshow(label_img, cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')

                plt.subplot(1, 4, 3)
                plt.imshow(pred_img, cmap='gray')
                plt.title('Prediction')
                plt.axis('off')

                plt.subplot(1, 4, 4)
                plt.imshow(fov_img, cmap='gray')
                plt.title('FOV Mask')
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, f'result_{idx+1}.png'))
                plt.close()

    all_preds = np.concatenate(all_preds).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    # 데이터 타입 변환
    all_preds = all_preds.astype(np.uint8)
    all_labels = all_labels.astype(np.uint8)

    # 평가지표 계산
    precision, recall, f1, accuracy, auc, specificity, sensitivity = compute_metrics(
        all_preds, all_labels
    )
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'Sensitivity: {sensitivity:.4f}')

    # ROC Curve 저장 (예외 처리 추가)
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_preds)
        plt.figure()
        plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.4f})'.format(auc))
        plt.plot([0, 1], [0, 1], 'k--')  # 랜덤 예측선
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend()
        plt.savefig(os.path.join(result_dir, 'roc_curve.png'))
        plt.close()
    except ValueError:
        print("ROC Curve를 그리기 위해 충분한 데이터가 없습니다.")

# ------------------------------
# 메인 실행 코드
# ------------------------------

if __name__ == "__main__":
    # 학습 또는 테스트 모드 선택
    mode = input("Select mode ('train' or 'test'): ").strip().lower()
    while mode not in ['train', 'test']:
        print("Invalid mode selected. Please choose 'train' or 'test'.")
        mode = input("Select mode ('train' or 'test'): ").strip().lower()

    # 하이퍼파라미터 설정
    batch_size = 2
    num_epochs = 25
    learning_rate = 0.0001
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 데이터셋 경로 설정
    train_image_dir = './data/DRIVE/training/images'
    train_mask_dir = './data/DRIVE/training/1st_manual'
    train_fov_dir = './data/DRIVE/training/mask'
    test_image_dir = './data/DRIVE/test/images'
    test_mask_dir = './data/DRIVE/test/1st_manual'
    test_fov_dir = './data/DRIVE/test/mask'

    # 모델 정의
    model = CrossUNet(input_channels=3, num_classes=1).to(device)

    if mode == 'train':
        # 데이터셋 및 데이터로더 생성
        transform = transforms.Compose([
            transforms.Resize((576, 576)),
            transforms.ToTensor()
        ])
        train_dataset = DriveDataset(train_image_dir, train_mask_dir, train_fov_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # 클래스 불균형 처리 (선택 사항)
        # pos_weight = calculate_pos_weight(train_dataset, device)
        # combined_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight) + dice_loss
        # 현재 combined_loss는 별도로 정의된 함수를 사용하므로 pos_weight를 적용하려면 combined_loss 함수를 수정해야 합니다.
        # 여기서는 pos_weight를 사용하지 않고, 단순히 BCEWithLogitsLoss + Dice Loss를 사용합니다.

        # 옵티마이저 정의
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 모델 학습
        print("Starting training...")
        model = train_model(
            model, train_loader, optimizer,
            num_epochs=num_epochs, device=device
        )

        # 모델 저장
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), 'models/crossunet_drive.pth')
        print("Model saved successfully!")

    elif mode == 'test':
        # 데이터셋 및 데이터로더 생성
        transform = transforms.Compose([
            transforms.Resize((576, 576)),
            transforms.ToTensor()
        ])
        test_dataset = DriveDataset(test_image_dir, test_mask_dir, test_fov_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # 모델 로드
        model.load_state_dict(torch.load('models/crossunet_drive.pth', map_location=device))
        model.to(device)
        print("Model loaded successfully!")

        # 테스트 수행
        print("Starting testing...")
        test_model(
            model, test_loader, device=device, save_results=True,
            result_dir='test_results'
        )
