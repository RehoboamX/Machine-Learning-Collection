import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard

# 可以尝试提高实验效果的方法：
# 1.使用更大更深的网络
# 2.使用BatchNorm做归一化
# 3.使用别的学习率
# 4.使用CNN代替全连接网络


class Discriminator(nn.Module):
    def __init__(self, img_dim):    # img_dim = 28*28 = 784
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)

class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),    # 1*28*28 = 784
            nn.Tanh()   # 将生成的数据转换为-1~1的数，因为Mnist经过标准化后范围也为-1~1
        )

    def forward(self, x):
        return self.gen(x)

# 超参数的选择
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64  # 也可以尝试128，256
image_dim = 28 * 28 * 1  # 784
batch_size = 64
num_epochs = 100

disc = Discriminator(image_dim).to(device)
gen = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)   # 高斯（正态）分布
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]   # Mnist的均值和标准差
)

dataset = datasets.MNIST(root="mnist_data/", transform=transforms, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.4, 0.8), weight_decay=0.0001)
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.4, 0.8), weight_decay=0.0001)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(dataloader):
        real = real.view(real.shape[0], -1).to(device)  # 将除了batch_size的其他维度展平
        batch_size = real.shape[0]

        # 训练判别器：最大化 log(D(real)) + log(1 - D(G(z))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)  # 将判别器对训练样本的输出结果（batchsize*1）展平
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).view(-1)    # 不加detach()的话判别器参数更新后计算图会被释放，下面生成器无法再使用D(G(z))，况且这里也不更新G的参数
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_real + lossD_fake) / 2
        opt_disc.zero_grad()
        lossD.backward()    # 如果括号里加上 retain_graph=True 则上面无需使用detach()
        opt_disc.step()

        # 训练生成器：最小化 log(1 - D(G(z))  <--->  最大化 log(D(G(z)))，这样的话不容易梯度饱和
        output = disc(fake).view(-1)    # 重新构建计算图
        lossG = criterion(output, torch.ones_like(output))
        opt_gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] \ "
                f"Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )

                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step += 1