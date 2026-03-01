import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. 데이터 준비
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 2. 모델 정의 (크기에 따라 유연하게 생성)
class MNIST_AE(nn.Module):
    def __init__(self, capacity):
        super(MNIST_AE, self).__init__()
        if capacity == 'small':
            # 아주 작은 신경망: 784 -> 32 (끝)
            self.encoder = nn.Sequential(nn.Linear(784, 32))
            self.decoder = nn.Sequential(nn.Linear(32, 784), nn.Sigmoid())
        else:
            # 큰 신경망: 784 -> 512 -> 256 -> 128 (깊고 넓음)
            self.encoder = nn.Sequential(
                nn.Linear(784, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128)
            )
            self.decoder = nn.Sequential(
                nn.Linear(128, 256), nn.ReLU(),
                nn.Linear(256, 512), nn.ReLU(),
                nn.Linear(512, 784), nn.Sigmoid()
            )

    def forward(self, x):
        x = x.view(-1, 784)
        return self.decoder(self.encoder(x))

# 3. 모델 두 개 생성 및 설정
model_small = MNIST_AE('small').to(device)
model_large = MNIST_AE('large').to(device)

opt_small = optim.Adam(model_small.parameters(), lr=0.001)
opt_large = optim.Adam(model_large.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 4. 동시 학습
epochs = 10
print("두 모델 동시 학습 시작...")

for epoch in range(1, epochs + 1):
    for data, _ in train_loader:
        img = data.to(device)
        target = img.view(-1, 784)

        # Small 모델 학습
        out_s = model_small(img)
        loss_s = criterion(out_s, target)
        opt_small.zero_grad(); loss_s.backward(); opt_small.step()

        # Large 모델 학습
        out_l = model_large(img)
        loss_l = criterion(out_l, target)
        opt_large.zero_grad(); loss_l.backward(); opt_large.step()

    print(f"Epoch {epoch} 완료")

# 5. 결과 비교 시각화
model_small.eval()
model_large.eval()

with torch.no_grad():
    test_img, _ = next(iter(train_loader))
    test_img = test_img[:5].to(device)
    recon_small = model_small(test_img).view(-1, 28, 28).cpu()
    recon_large = model_large(test_img).view(-1, 28, 28).cpu()
    test_img = test_img.cpu()

    plt.figure(figsize=(12, 8))
    for i in range(5):
        # 1행: 원본
        plt.subplot(3, 5, i + 1); plt.imshow(test_img[i].squeeze(), cmap='gray')
        if i == 0: plt.ylabel("Original", fontsize=15)
        plt.axis('off')
        
        # 2행: Small Recon (적은 신경망)
        plt.subplot(3, 5, i + 6); plt.imshow(recon_small[i], cmap='gray')
        if i == 0: plt.ylabel("Small (32 dim)", fontsize=15)
        plt.axis('off')
        
        # 3행: Large Recon (많은 신경망)
        plt.subplot(3, 5, i + 11); plt.imshow(recon_large[i], cmap='gray')
        if i == 0: plt.ylabel("Large (128 dim)", fontsize=15)
        plt.axis('off')

    plt.tight_layout()
    plt.show()