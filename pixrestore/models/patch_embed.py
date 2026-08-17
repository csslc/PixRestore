"""Bottleneck image-to-patch embedding."""
from torch import nn

class BottleneckPatchEmbed(nn.Module):
    """
    Image to Patch Embedding with bottleneck structure
    
    Args:
        img_size: 输入图像尺寸，可以是int或tuple (height, width)
        patch_size: 每个patch的尺寸，可以是int或tuple
        in_chans: 输入图像的通道数
        pca_dim: 第一个投影层的输出维度（bottleneck维度）
        embed_dim: 最终嵌入的维度
        bias: 第二个投影层是否使用偏置
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size[1] // patch_size[1] * (img_size[0] // patch_size[0])
        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 (B, C, H, W)
        Returns:
            x: 输出张量，形状为 (B, num_patches, embed_dim)
        """
        (B, C, H, W) = x.shape
        (ph, pw) = (self.patch_size[0], self.patch_size[1])
        if H % ph != 0 or W % pw != 0:
            raise ValueError(f'BottleneckPatchEmbed: spatial size ({H}, {W}) must be divisible by patch_size ({ph}, {pw}).')
        x = self.proj1(x)
        x = self.proj2(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
