import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from omegaconf import DictConfig


# setting value in class and function just avoid
# note: text 768 -> 512
# note: mfcc 13 -> 1024 # change latter
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int = 1024,  # emb_dim
            n_head: int = 8,  # embed_dim // num_heads
            attn_mask: torch.Tensor = None,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model*4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model*4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# basic transformer, not include class and position encoding
# use on text and maybe try on waveform
class Transformer(nn.Module):
    def __init__(
            self,
            width: int = 768,
            layers: int = 6,
            heads: int = 8,
            attn_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

# create text transformer


# use on mfcc
class AudioTransformer(nn.Module):
    def __init__(
        self,
        width: int = 13,  # transformer dim (要再改)
        layers: int = 6,  # # of layers
        heads: int = 8,  # # of head
        output_dim: int = 1024,
        max_length: int = 2048,  # to use positional embedding
    ):
        super().__init__()
        scale = width ** -0.5
        self.max_length = max_length
        # same as ViT
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(max_length, width))

        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width=width, layers=layers, heads=heads)

        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = torch.cat([self.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device), x], dim=1)
        if x.shape[1] > self.max_length:
            x = x[:, :self.max_length, :]

        x = x + self.positional_embedding[:x.shape[1], :x.shape[2]]
        x = self.ln_pre(x)

        x = self.transformer(x)
        x = self.ln_post(x[:, 0, :])  # take the class_embedding to calculate the projection

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLAP(nn.Module):
    def __init__(
            self,
            text_cfg: DictConfig,
            audio_cfg: DictConfig,
            embed_dim: int = 512,
    ):
        super().__init__()
        assert audio_cfg.name in ["transformer"], "Not Implemented Model"
        assert text_cfg.name in ["transformer"], "Not Implemented Model"

        audio_width = audio_cfg.audio_width  # def 13
        audio_layers = audio_cfg.audio_layers  # def 6
        audio_heads = audio_cfg.audio_heads  # def 8
        audio_max_length = audio_cfg.audio_max_length  # def 2048

        vocab_size = text_cfg.vocab_size  # 30000
        text_context_length = text_cfg.text_max_length  # 256
        text_width = text_cfg.text_width
        text_layers = text_cfg.text_layers
        text_heads = text_cfg.text_heads

        if audio_cfg.name == "transformer":
            self.audio = AudioTransformer(
                width=audio_width,
                layers=audio_layers,
                heads=audio_heads,
                max_length=audio_max_length,
                output_dim=embed_dim,
            )
        else:
            raise NotImplemented

        if text_cfg.name == "transformer":
            self.text = Transformer(
                width=text_width,
                layers=text_layers,
                heads=text_heads,
            )

        self.token_embedding = nn.Embedding(vocab_size, text_width)
        self.text_positional_embedding = nn.Parameter(torch.empty(text_context_length, text_width))
        self.ln_final = nn.LayerNorm(text_width)

        self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

        self.initialize_parameters()

    # ref form openai/clip
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.text_positional_embedding, std=0.01)

        proj_std = (self.text.width ** -0.5) * ((2 * self.text.layers) ** -0.5)
        attn_std = self.text.width ** -0.5
        fc_std = (2 * self.text.width) ** -0.5
        for block in self.text.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.text.width ** -0.5)

    def encode_audio(self, audio):
        return self.audio(audio)

    def encode_text(self, text):
        x = self.token_embedding(text)

        x = x + self.text_positional_embedding[:x.shape[1], :]
        x = self.text(x)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, audio, text):
        audio_features = self.encode_audio(audio)
        text_features = self.encode_text(text)

        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_features @ text_features.t()
        logits_per_text = logits_per_audio.t()

        return logits_per_audio, logits_per_text







