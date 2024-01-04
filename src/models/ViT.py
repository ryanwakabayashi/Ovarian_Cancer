import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, img_size):
        super().__init__()
        self.patch_size = patch_size

        self.project = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.project(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.positions
        return x


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.att_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(emb_size)
        self.multihead_attention = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.att_dropout(x)
        attn_output, attn_output_weights = self.multihead_attention(x, x, x)
        return attn_output


class MachineLearningPerceptronBlock(nn.Module):
    def __init__(self, embedding_dims, mlp_size, mlp_dropout):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.mlp_size = mlp_size
        self.dropout = mlp_dropout

        self.layernorm = nn.LayerNorm(embedding_dims)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dims, mlp_size),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_size, embedding_dims),
            nn.Dropout(mlp_dropout)
        )

    def forward(self, x):
        return self.mlp(self.layernorm(x))


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dims=768, mlp_dropout=0.1, attn_dropout=0.0, mlp_size=1024, num_heads=3):
        super().__init__()
        self.msa_block = MultiHeadSelfAttentionBlock(embedding_dims, num_heads, attn_dropout)
        self.mlp_block = MachineLearningPerceptronBlock(embedding_dims, mlp_size, mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, img_size=512, in_channels=3, patch_size=32, num_classes=6, emb_size=2048, num_heads=16, num_layers=3, attn_dropout=0.0, mlp_size=4096, mlp_dropout=0.2):
        super(ViT, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.transformer_encoder = nn.Sequential(
            *[TransformerBlock(emb_size, mlp_dropout, attn_dropout, mlp_size, num_heads) for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.transformer_encoder(self.patch_embedding(x))[:, 0])
