import torch
import torch.nn as nn


def gradient_penalty(critic, real, fake, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    epsilon = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)  # 每个样本的 CxHxW 对应的 epsilon 为同一值
    interpolated_images = epsilon * real + (1 - epsilon) * fake

    # 计算判别器分数
    mixed_scores = critic(interpolated_images)  # BATCH_SIZE x 1 x 1 x 1

    # 计算判别器分数对插值图像的梯度
    gradient = torch.autograd.grad(  # 这里返回的gradient的shape为 BATCH_SIZE x C x H x W，与inputs形状相同
        inputs=interpolated_images,  # 要被计算导数的叶子节点
        outputs=mixed_scores,  # 待被求导的tensor
        grad_outputs=torch.ones_like(mixed_scores),  # vector-Jacobian 乘积中的 “vector”，outputs不是常数时必须设置此项
        create_graph=True,  # 对反向传播过程中再次构建计算图，可求高阶导数
        retain_graph=True,  # 求导后不释放计算图
    )[0]  # 这里的[0]为梯度的张量本身，[1]则为grad_fn，追踪了上一步的操作函数

    gradient = gradient.view(gradient.shape[0], -1)  # BATCH_SIZE x C*H*W
    gradient_norm = gradient.norm(2, dim=1)  # 对第1维度求L2范数，实际上对每C*H*W个元素求L2范数,形状为BATCH_SIZE
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)  # 形状为BATCH_SIZE
    return gradient_penalty




