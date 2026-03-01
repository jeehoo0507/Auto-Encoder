import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 컬러 ConvAE 모델
class ColorConvAE(nn.Module):
    def __init__(self):
        super(ColorConvAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# 2. 이미지 불러오기
img_path = '건터.jpg'
if not os.path.exists(img_path):
    img = np.random.rand(512, 512, 3).astype(np.float32) # 10001번 대비 512로 조정
else:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512)) # 소음/속도 타협점
    img = img.astype(np.float32) / 255.0

input_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

# 3. 모델 및 설정
model = ColorConvAE().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 결과를 담을 리스트
results = []
target_epochs = [101, 1001, 10001]

# 4. 학습 루프
print(f"학습 시작... (최대 {target_epochs[-1]} 에포크)")
start_time = time.time()

for epoch in range(1, target_epochs[-1] + 1):
    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, input_tensor)
    loss.backward()
    optimizer.step()
    
    # 고주파 소리 조절 (계산 부하 분산)
    if epoch < 1000:
        time.sleep(0.002)
    else:
        time.sleep(0.0005) # 만 번 돌릴 때는 조금 더 빠르게

    # 목표 에포크 도달 시 이미지 저장
    if epoch in target_epochs:
        print(f"에포크 {epoch} 완료! Loss: {loss.item():.6f}")
        with torch.no_grad():
            res = output.squeeze().cpu().permute(1, 2, 0).numpy()
            results.append((epoch, res))

# 5. 최종 결과 한꺼번에 출력
plt.figure(figsize=(20, 5))

# 원본 이미지
plt.subplot(1, 4, 1)
plt.title("Original")
plt.imshow(img)
plt.axis('off')

# 에포크별 결과
for i, (ep, res_img) in enumerate(results):
    plt.subplot(1, 4, i + 2)
    plt.title(f"Epoch {ep}")
    plt.imshow(res_img)
    plt.axis('off')

plt.tight_layout()
plt.show()

print(f"전체 학습 완료! 소요 시간: {(time.time() - start_time)/60:.2f}분")