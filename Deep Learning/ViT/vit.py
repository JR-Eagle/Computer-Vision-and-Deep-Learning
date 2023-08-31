import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Embeds image patches using a convolution operation."""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) * (image_size // patch_size)

        # Using a convolutional layer to implement patch embedding
        self.patch_embedding = nn.Conv2d(in_channels=in_channels,
                                         out_channels=embed_dim,
                                         kernel_size=patch_size,
                                         stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.image_size and W == self.image_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.image_size}*{self.image_size})."

        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)  # Flatten and rearrange dimensions for token sequences
        return x


class PositionEmbedding(nn.Module):
    """Implements positional embeddings."""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dim=768, dropout=0.):
        super().__init__()

        # Patch embedding
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.num_patches + 1, embed_dim))

        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Dropout layer
        self.pos_dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand((x.shape[0], -1, -1))
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.pos_dropout(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        return self.norm(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_head_size, qkv_bias=True, dropout=0, attn_dropout=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if attn_head_size is not None:
            self.attn_head_size = attn_head_size
        else:
            assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
            self.attn_head_size = embed_dim // num_heads
        self.all_head_size = self.attn_head_size * num_heads

        self.qkv = nn.Linear(embed_dim, self.all_head_size * 3)
        self.scale = self.attn_head_size ** -0.5

        self.out = nn.Linear(self.all_head_size, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_multihead(self, x):
        """Transpose for multi-head attention."""
        new_shape = x.shape[:-1] + (self.num_heads, self.attn_head_size)
        x = x.reshape(new_shape).transpose(1, 2)
        return x

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(self.transpose_multihead, qkv)

        q = q * self.scale
        attn = torch.matmul(q, k.transpose(2, 3))
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = torch.matmul(attn, v)
        z = z.transpose(1, 2).reshape(x.shape[0], -1, self.all_head_size)
        z = self.out(z)
        z = self.proj_dropout(z)
        return z


class Transformer(nn.Module):
    """A single Transformer block."""

    def __init__(self, embed_dim, num_heads, attn_head_size, qkv_bias=True, mlp_ratio=4.0, dropout=0., attn_dropout=0., droppath=None):
        super().__init__()
        self.attn_norm = PreNorm(embed_dim)
        self.attn = Attention(embed_dim, num_heads, attn_head_size, qkv_bias, dropout, attn_dropout)
        self.mlp_norm = PreNorm(embed_dim)
        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = h + x

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = h + x

        return x


class Mlp(nn.Module):
    """A small feed-forward network, used in the Transformer."""

    def __init__(self, embed_dim, mlp_ratio, dropout=0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Encoder(nn.Module):
    """The main encoder (stack of Transformers)."""

    def __init__(self, embed_dim, num_heads, depth, attn_head_size, qkv_bias, mlp_ratio, dropout, attn_dropout, droppath=None):
        super().__init__()
        self.layers = nn.ModuleList([Transformer(embed_dim, num_heads, attn_head_size, qkv_bias, mlp_ratio, dropout, attn_dropout, droppath)
                                     for _ in range(depth)])
        self.encoder_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.encoder_norm(x)
        return x


class VisionTransformer(nn.Module):
    """Main Vision Transformer model."""

    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 dropout=0., attn_dropout=0., droppath=0., attn_head_size=None, representation_size=None):
        super().__init__()
        self.pos_embedding = PositionEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.pos_dropout = nn.Dropout(dropout)
        self.encoder = Encoder(embed_dim, num_heads, depth, attn_head_size, qkv_bias, mlp_ratio, dropout, attn_dropout, droppath)

        if representation_size is not None:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(nn.Linear(embed_dim, representation_size), nn.ReLU())
        else:
            self.pre_logits = nn.Identity()

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.pos_embedding(x)
        x = self.encoder(x)
        x = self.pre_logits(x[:, 0])
        logits = self.classifier(x)
        return logits


def main():
    """Debugging main function."""
    x = torch.rand(100, 3, 224, 224)
    model = VisionTransformer(in_channels=3, patch_size=16, embed_dim=192)
    output = model(x)
    print(output.shape)


if __name__ == "__main__":
    main()
