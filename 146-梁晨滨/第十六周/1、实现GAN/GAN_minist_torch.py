import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

# 用gpu训练
device = torch.device("cuda")


# 生成器网络
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model_g = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, x):
        img_generate = self.model_g(x)

        return img_generate


# 判别器网络
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model_d = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        result = self.model_d(x)

        return result


def train(epochs, batch_size, sample_interval):
    # 数据加载，用torchvision里的mnist,没有自动从网上下
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)

    # 构建生成器和判别器的模型函数 + 优化器 + 损失函数
    generate = Generator().to(device)
    generate_optimizer = optim.Adam(generate.parameters(), lr=0.0002, betas=(0.5, 0.999))
    discriminate = Discriminator().to(device)
    discriminate_optimizer = optim.Adam(discriminate.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_loader):
            # 数据预处理
            real_images = images.view(-1, 784).to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            # 训练判别网络
            real_score = discriminate(real_images)
            d_loss_real = criterion(real_score, real_labels)

            noise = torch.randn(batch_size, 100).to(device)
            fake_images = generate(noise)
            fake_score = discriminate(fake_images)
            d_loss_fake = criterion(fake_score, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            discriminate_optimizer.zero_grad()
            d_loss.backward()
            discriminate_optimizer.step()

            # 训练生成网络
            noise = torch.randn(batch_size, 100).to(device)
            fake_images = generate(noise)
            outputs = discriminate(fake_images)

            # 损失反向传播
            g_loss = criterion(outputs, real_labels)
            generate_optimizer.zero_grad()
            g_loss.backward()
            generate_optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                      .format(epoch, epochs, i+1, len(train_loader), d_loss.item(), g_loss.item(),
                              real_score.mean().item(), fake_score.mean().item()))

        if (epoch+1) % sample_interval == 0:
            torchvision.utils.save_image(fake_images.data, 'torch_images/mnist_%d.png' % (epoch+1), nrow=5, normalize=True)


if __name__ == '__main__':
    train(epochs=2000, batch_size=32, sample_interval=10)
