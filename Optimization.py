import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import time
import copy
import numpy as np
from collections import OrderedDict

# ----------------------------
# 1. 모델 정의 (BasicUNet, CrossUNet)
# ----------------------------

# BasicUNet 정의
class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(BasicUNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.enc1 = self.conv_block(in_channels, 64, name='enc1')
        self.enc2 = self.conv_block(64, 128, name='enc2')
        self.enc3 = self.conv_block(128, 256, name='enc3')
        self.enc4 = self.conv_block(256, 512, name='enc4')
        self.enc5 = self.conv_block(512, 1024, name='enc5')

        self.dec1 = self.upconv_block(1024, 512, name='dec1')
        self.dec2 = self.upconv_block(512, 256, name='dec2')
        self.dec3 = self.upconv_block(256, 128, name='dec3')
        self.dec4 = self.upconv_block(128, 64, name='dec4')
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels, name):
        return nn.Sequential(OrderedDict([
            (f'{name}_conv1', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
            (f'{name}_bn1', nn.BatchNorm2d(out_channels)),
            (f'{name}_relu1', nn.ReLU(inplace=False)),
            (f'{name}_conv2', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            (f'{name}_bn2', nn.BatchNorm2d(out_channels)),
            (f'{name}_relu2', nn.ReLU(inplace=False)),
        ]))

    def upconv_block(self, in_channels, out_channels, name):
        return nn.Sequential(OrderedDict([
            (f'{name}_upconv', nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)),
            (f'{name}_conv', nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)),
            (f'{name}_bn', nn.BatchNorm2d(out_channels)),
            (f'{name}_relu', nn.ReLU(inplace=False)),
        ]))

    def forward(self, x):
        x = self.quant(x)
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        e5 = self.enc5(F.max_pool2d(e4, 2))

        d1 = self.dec1(F.interpolate(e5, scale_factor=2, mode='bilinear', align_corners=False))
        d2 = self.dec2(F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False))
        d3 = self.dec3(F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=False))
        d4 = self.dec4(F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=False))
        out = self.final(F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=False))
        out = self.dequant(out)
        return out

# Attention Mechanism 정의
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, name):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(OrderedDict([
            (f'{name}_W_g_conv', nn.Conv2d(F_g, F_int, kernel_size=1)),
            (f'{name}_W_g_bn', nn.BatchNorm2d(F_int))
        ]))
        self.W_x = nn.Sequential(OrderedDict([
            (f'{name}_W_x_conv', nn.Conv2d(F_l, F_int, kernel_size=1)),
            (f'{name}_W_x_bn', nn.BatchNorm2d(F_int))
        ]))
        self.psi = nn.Sequential(OrderedDict([
            (f'{name}_psi_conv', nn.Conv2d(F_int, 1, kernel_size=1)),
            (f'{name}_psi_bn', nn.BatchNorm2d(1)),
            (f'{name}_psi_hardsigmoid', nn.Hardsigmoid(inplace=False))  # Sigmoid 대신 Hardsigmoid 사용
        ]))
        self.relu = nn.ReLU(inplace=False)  # inplace=False로 변경

    def forward(self, g, x):
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# UNet Block 정의
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=False)
        self.name = name

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

# Decoder Block 정의
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, name):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = UNetBlock(out_channels + skip_channels, out_channels, name=f'{name}_conv_block')
        self.relu = nn.ReLU(inplace=False)
        self.name = name

    def forward(self, x, skip=None):
        x = self.upconv(x)
        x = self.relu(x)
        if skip is not None:
            if x.size() != skip.size():
                skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# CrossUNet 정의
class CrossUNet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1):
        super(CrossUNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        # Shared Encoder
        self.shared_encoder1 = UNetBlock(input_channels, 64, name='shared_enc1')
        self.pool1 = nn.MaxPool2d(2)
        self.shared_encoder2 = UNetBlock(64, 128, name='shared_enc2')
        self.pool2 = nn.MaxPool2d(2)
        self.shared_encoder3 = UNetBlock(128, 256, name='shared_enc3')
        self.pool3 = nn.MaxPool2d(2)

        # A Path
        self.a_bottleneck1 = UNetBlock(256, 512, name='a_bottleneck1')
        self.a_decoder1 = DecoderBlock(512, 256, skip_channels=512, name='a_dec1')
        self.a_decoder2 = DecoderBlock(256, 128, skip_channels=1024, name='a_dec2')
        self.a_bottleneck2 = UNetBlock(128, 64, name='a_bottleneck2')
        self.a_encoder1 = UNetBlock(64, 128, name='a_enc1')
        self.pool_a1 = nn.MaxPool2d(2)
        self.a_encoder2 = UNetBlock(128, 256, name='a_enc2')
        self.pool_a2 = nn.MaxPool2d(2)
        self.a_bottleneck3 = UNetBlock(256, 512, name='a_bottleneck3')

        # B Path
        self.b_encoder1 = UNetBlock(256, 512, name='b_enc1')
        self.pool_b1 = nn.MaxPool2d(2)
        self.b_encoder2 = UNetBlock(512, 1024, name='b_enc2')
        self.pool_b2 = nn.MaxPool2d(2)
        self.b_bottleneck = UNetBlock(1024, 1024, name='b_bottleneck')
        self.b_decoder1 = DecoderBlock(1024, 512, skip_channels=128, name='b_dec1')
        self.b_decoder2 = DecoderBlock(512, 256, skip_channels=256, name='b_dec2')

        # Attention Gate
        self.attention_gate = AttentionGate(F_g=512, F_l=256, F_int=256, name='attention_gate')

        # Final Decoder
        self.final_decoder1 = DecoderBlock(256, 128, skip_channels=256, name='final_dec1')
        self.final_decoder2 = DecoderBlock(128, 64, skip_channels=128, name='final_dec2')
        self.final_decoder3 = DecoderBlock(64, 64, skip_channels=64, name='final_dec3')

        # Final Output Layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.quant(x)
        # Shared Encoder Path
        enc1 = self.shared_encoder1(x)
        enc1_pool = self.pool1(enc1)
        enc2 = self.shared_encoder2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        enc3 = self.shared_encoder3(enc2_pool)
        enc3_pool = self.pool3(enc3)

        # B Path
        b_enc1 = self.b_encoder1(enc3_pool)
        b_enc1_pool = self.pool_b1(b_enc1)
        b_enc2 = self.b_encoder2(b_enc1_pool)
        b_enc2_pool = self.pool_b2(b_enc2)
        b_bottleneck = self.b_bottleneck(b_enc2_pool)

        # A Path
        a_bottleneck1 = self.a_bottleneck1(enc3_pool)
        a_dec1 = self.a_decoder1(a_bottleneck1, skip=b_enc1)
        a_dec2 = self.a_decoder2(a_dec1, skip=b_enc2)
        a_bottleneck2 = self.a_bottleneck2(a_dec2)
        a_enc1 = self.a_encoder1(a_bottleneck2)
        a_enc1_pool = self.pool_a1(a_enc1)
        a_enc2 = self.a_encoder2(a_enc1_pool)
        a_enc2_pool = self.pool_a2(a_enc2)
        a_bottleneck3 = self.a_bottleneck3(a_enc2_pool)

        # B Path Decoder
        b_dec1 = self.b_decoder1(b_bottleneck, skip=a_enc1)
        b_dec2 = self.b_decoder2(b_dec1, skip=a_enc2)

        # Attention Gate for merging A and B paths
        merged = self.attention_gate(a_bottleneck3, b_dec2)

        # Final Decoder
        final_dec1 = self.final_decoder1(merged, skip=enc3)
        final_dec2 = self.final_decoder2(final_dec1, skip=enc2)
        final_dec3 = self.final_decoder3(final_dec2, skip=enc1)

        output = self.final_conv(final_dec3)
        output = self.dequant(output)
        return output

# ----------------------------
# 2. 맞춤형 qconfig 설정
# ----------------------------

# ConvTranspose2d를 위한 per-tensor 양자화 qconfig 설정
custom_qconfig = torch.quantization.QConfig(
    activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
    weight=torch.quantization.MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
)

# ----------------------------
# 3. 정적 양자화 함수 수정
# ----------------------------

def fuse_model(model):
    for module_name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            fusion_list = []
            module_children = list(module.named_children())
            idx = 0
            while idx < len(module_children):
                # 시퀀스 내 Conv -> BatchNorm -> ReLU 패턴 찾기
                if idx + 2 < len(module_children):
                    layer1_name, layer1 = module_children[idx]
                    layer2_name, layer2 = module_children[idx + 1]
                    layer3_name, layer3 = module_children[idx + 2]
                    if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.BatchNorm2d) and isinstance(layer3, nn.ReLU):
                        fusion_list.append([layer1_name, layer2_name, layer3_name])
                        idx += 3
                        continue
                # 시퀀스 내 Conv -> BatchNorm 패턴 찾기
                if idx + 1 < len(module_children):
                    layer1_name, layer1 = module_children[idx]
                    layer2_name, layer2 = module_children[idx + 1]
                    if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.BatchNorm2d):
                        fusion_list.append([layer1_name, layer2_name])
                        idx += 2
                        continue
                # 시퀀스 내 Conv2d -> ReLU 패턴 찾기 (ConvTranspose2d + ReLU는 퓨전하지 않음)
                if idx + 1 < len(module_children):
                    layer1_name, layer1 = module_children[idx]
                    layer2_name, layer2 = module_children[idx + 1]
                    if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.ReLU):
                        fusion_list.append([layer1_name, layer2_name])
                        idx += 2
                        continue
                # 시퀀스 내 ConvTranspose2d -> ReLU는 퓨전하지 않음
                if idx + 1 < len(module_children):
                    layer1_name, layer1 = module_children[idx]
                    layer2_name, layer2 = module_children[idx + 1]
                    if isinstance(layer1, nn.ConvTranspose2d) and isinstance(layer2, nn.ReLU):
                        # 퓨전하지 않고 건너뜀
                        idx += 2
                        continue
                # 다음으로 이동
                idx += 1
            if fusion_list:
                try:
                    torch.quantization.fuse_modules(module, fusion_list, inplace=True)
                except RuntimeError as e:
                    print(f"Fusion failed for {module}: {e}")
        else:
            fuse_model(module)

def apply_static_quantization(model, calibration_data):
    quantized_model = copy.deepcopy(model)
    quantized_model.eval()

    # 모듈 퓨전
    fuse_model(quantized_model)

    # 양자화 설정 (맞춤형 qconfig 적용)
    for name, module in quantized_model.named_modules():
        if isinstance(module, nn.ConvTranspose2d):
            module.qconfig = custom_qconfig  # ConvTranspose2d에는 맞춤형 qconfig 사용
        else:
            module.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # 양자화 준비
    torch.quantization.prepare(quantized_model, inplace=True)

    # Calibration: 실제 데이터로 스케일 및 제로 포인트 설정
    with torch.no_grad():
        for data in calibration_data:
            quantized_model(data)

    # 양자화 변환
    torch.quantization.convert(quantized_model, inplace=True)

    return quantized_model

# ----------------------------
# 4. 정밀도 검증 (MSE 및 Dice Score)
# ----------------------------

def verify_accuracy(original_model, quantized_model, test_data, threshold=0.5):
    original_model.eval()
    quantized_model.eval()
    original_outputs = []
    quantized_outputs = []

    with torch.no_grad():
        for data in test_data:
            original_outputs.append(original_model(data).cpu().numpy())
            quantized_outputs.append(quantized_model(data).cpu().numpy())

    original_outputs = np.array(original_outputs)
    quantized_outputs = np.array(quantized_outputs)

    # Mean Squared Error (MSE)
    mse = np.mean((original_outputs - quantized_outputs) ** 2)

    # Dice Score
    dice_scores = []
    for orig, quant in zip(original_outputs, quantized_outputs):
        orig_bin = (orig > threshold).astype(np.float32)
        quant_bin = (quant > threshold).astype(np.float32)
        intersection = np.sum(orig_bin * quant_bin)
        dice = (2. * intersection) / (np.sum(orig_bin) + np.sum(quant_bin) + 1e-6)
        dice_scores.append(dice)
    mean_dice = np.mean(dice_scores)

    return mse, mean_dice

# ----------------------------
# 5. 추론 시간 측정 함수
# ----------------------------

def measure_inference_time(model, input_tensor, device, num_runs=100):
    model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(input_tensor)

        # Timing
        start_time = time.time()
        for _ in range(num_runs):
            _ = model(input_tensor)
        end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return avg_time

# ----------------------------
# 6. 메인 실행
# ----------------------------

def main():
    # 장치 설정
    cpu_device = torch.device("cpu")
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using GPU: {gpu_device.type == 'cuda'}")

    # 데이터 생성
    # BasicUNet: 1채널 입력
    input_tensor_unet = torch.rand(1, 1, 128, 128)
    calibration_data_unet = [torch.rand(1, 1, 128, 128) for _ in range(10)]  # Calibration 데이터
    test_data_unet = [torch.rand(1, 1, 128, 128) for _ in range(100)]  # 테스트 데이터

    # CrossUNet: 3채널 입력
    input_tensor_crossunet = torch.rand(1, 3, 128, 128)
    calibration_data_crossunet = [torch.rand(1, 3, 128, 128) for _ in range(10)]  # Calibration 데이터
    test_data_crossunet = [torch.rand(1, 3, 128, 128) for _ in range(100)]  # 테스트 데이터

    # 모델 생성 및 평가 모드 전환
    unet_model = BasicUNet(in_channels=1, out_channels=1).to(cpu_device)
    unet_model.eval()

    crossunet_model = CrossUNet(input_channels=3, num_classes=1).to(cpu_device)
    crossunet_model.eval()

    # ----------------------------
    # 6.1. BasicUNet 평가
    # ----------------------------
    print("=== BasicUNet Evaluation ===")
    # 원본 모델 추론 시간 측정 (CPU)
    unet_model_cpu = copy.deepcopy(unet_model).to(cpu_device)
    unet_model_cpu.eval()
    unet_base_time_cpu = measure_inference_time(unet_model_cpu, input_tensor_unet.to(cpu_device), cpu_device)
    print(f"U-Net Base Model (CPU) Inference Time: {unet_base_time_cpu:.6f} seconds")

    # 정적 양자화 적용 (전체 양자화, CPU)
    unet_full_quant = apply_static_quantization(unet_model, calibration_data_unet)
    unet_full_quant.eval()
    unet_full_quant_time = measure_inference_time(unet_full_quant, input_tensor_unet, cpu_device)
    print(f"U-Net Fully Quantized Model (CPU) Inference Time: {unet_full_quant_time:.6f} seconds")

    # 정밀도 검증
    mse_unet, dice_unet = verify_accuracy(unet_model, unet_full_quant, test_data_unet)
    print(f"U-Net Fully Quantized Mean Squared Error (MSE): {mse_unet:.6f}")
    print(f"U-Net Fully Quantized Dice Score: {dice_unet:.6f}\n")

    # ----------------------------
    # 6.2. CrossUNet 평가
    # ----------------------------
    print("=== CrossUNet Evaluation ===")
    # 원본 모델 추론 시간 측정 (CPU)
    crossunet_model_cpu = copy.deepcopy(crossunet_model).to(cpu_device)
    crossunet_model_cpu.eval()
    crossunet_base_time_cpu = measure_inference_time(crossunet_model_cpu, input_tensor_crossunet.to(cpu_device), cpu_device)
    print(f"CrossUNet Base Model (CPU) Inference Time: {crossunet_base_time_cpu:.6f} seconds")

    # 정적 양자화 적용 (전체 양자화, CPU)
    crossunet_full_quant = apply_static_quantization(crossunet_model, calibration_data_crossunet)
    crossunet_full_quant.eval()
    crossunet_full_quant_time = measure_inference_time(crossunet_full_quant, input_tensor_crossunet, cpu_device)
    print(f"CrossUNet Fully Quantized Model (CPU) Inference Time: {crossunet_full_quant_time:.6f} seconds")

    # 정밀도 검증
    mse_crossunet, dice_crossunet = verify_accuracy(crossunet_model, crossunet_full_quant, test_data_crossunet)
    print(f"CrossUNet Fully Quantized Mean Squared Error (MSE): {mse_crossunet:.6f}")
    print(f"CrossUNet Fully Quantized Dice Score: {dice_crossunet:.6f}\n")

    # ----------------------------
    # 6.3. 성능 요약
    # ----------------------------
    print("=== Performance Summary ===")
    print("U-Net:")
    print(f"  Base Model (CPU) Inference Time: {unet_base_time_cpu:.6f} seconds")
    print(f"  Fully Quantized Model (CPU) Inference Time: {unet_full_quant_time:.6f} seconds")
    print(f"  Fully Quantized Mean Squared Error (MSE): {mse_unet:.6f}")
    print(f"  Fully Quantized Dice Score: {dice_unet:.6f}\n")

    print("CrossUNet:")
    print(f"  Base Model (CPU) Inference Time: {crossunet_base_time_cpu:.6f} seconds")
    print(f"  Fully Quantized Model (CPU) Inference Time: {crossunet_full_quant_time:.6f} seconds")
    print(f"  Fully Quantized Mean Squared Error (MSE): {mse_crossunet:.6f}")
    print(f"  Fully Quantized Dice Score: {dice_crossunet:.6f}")

if __name__ == "__main__":
    main()
