from hydra.utils import get_original_cwd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict, namedtuple
from omegaconf import DictConfig
from typing import List, Tuple
import hydra
from transformers import AutoModelForPreTraining, AutoModel
from transformers import DistilBertModel, DistilBertConfig


# setting value in class and function just avoid
# note: text 768 -> 512
# note: mfcc 13 -> 1024 # change latter
# not using mfcc anymore
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


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        projection_dim: int,
        dropout: int = 0.5,
    ):
        super().__init__()
        self.projection = nn.Linear(embed_dim, projection_dim)
        # self.gelu = nn.GELU()
        # self.fc = nn.Linear(projection_dim, projection_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        # x = self.gelu(projected)
        # x = self.fc(x)
        # x = self.dropout(x)
        # x = x + projected
        # x = self.layer_norm(x)
        return projected


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


# from fairseq/wav2vec2
class ConvFeatureExtractionModel(nn.Module):
    def __init__(
            self,
            conv_layers: List[Tuple[int, int, int]],
            dropout: float= 0.0,
            conv_bias: bool=False,
    ):
        super().__init__()

        def block(
                n_in,
                n_out,
                k,
                stride,
                conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        in_d = 1
        self.conv_layers = nn.ModuleList()

        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):
        x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x.transpose(1, 2)


class WaveformTransformer(nn.Module):
    def __init__(
        self,
        conv_feature_layers: str,
        width: int = 512,
        layers: int = 6,
        heads: int = 4,
        output_dim: int = 1024,
        max_length: int = 256,
    ):
        super().__init__()
        scale = width ** -0.5
        self.max_length = max_length

        feature_enc_layers = eval(conv_feature_layers)

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.01,
        )

        # use same as ViT
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(max_length, width))

        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width=width, layers=layers, heads=heads)

        self.ln_post = nn.LayerNorm(width)
        # self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.proj = ProjectionHead(embed_dim=width, projection_dim=output_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.cat([self.class_embedding.to(x.dtype) +
                       torch.zeros(x.shape[0], 1, x.shape[-1], device=x.device), x], dim=1)

        x = x + self.positional_embedding[:x.shape[1], :x.shape[2]]
        x = self.ln_pre(x)

        x = self.transformer(x)
        x = self.ln_post(x[:, 0, :])  # take the class_embedding to calculate the projection

        if self.proj is not None:
            x = self.proj(x)

        return x


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



class AudioPretrainedModel(nn.Module):
    def __init__(
        self, 
        pretrained_model_name: str,
        width: int=768,
        output_dim: int = 256,
        trainable: bool=True,
    ):
        super().__init__()
        scale = width ** -0.5
        self.model = AutoModel.from_pretrained(pretrained_model_name)
        for p in self.model.parameters():
            p.requires_grad = trainable

        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x):
        encode = self.model(x)
        last_hidden_state = encode.last_hidden_state
        output = last_hidden_state[:,0,:] @ self.proj
        return output


class DistilBertPretrain(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        trainable: bool=True,
    ):
        super().__init__()
        self.model = DistilBertModel.from_pretrained(pretrained_model_name)
        for p in self.model.parameters():
            p.requires_grad = trainable
    
    def forward(self, x):
        hidden_state = self.model(**x)['last_hidden_state'][:,0,:]
        return hidden_state


class CLAP(nn.Module):
    def __init__(
            self,
            text_cfg: DictConfig,
            audio_cfg: DictConfig,
            embed_dim: int = 512,
    ):
        super().__init__()
        assert audio_cfg.name in ["transformer", "wave_transformer", "hubert"], "Not Implemented Model"
        assert text_cfg.name in ["transformer", "distilbert"], "Not Implemented Model"

        self.audio_cfg = audio_cfg
        self.text_cfg = text_cfg

        vocab_size = text_cfg.vocab_size  # 30000
        text_context_length = text_cfg.text_max_length  # 256
        text_width = text_cfg.text_width
        text_layers = text_cfg.text_layers
        text_heads = text_cfg.text_heads

        if audio_cfg.name == "transformer":
            self.audio = AudioTransformer(
                width=audio_cfg.audio_width,
                layers=audio_cfg.audio_layers,
                heads=audio_cfg.audio_heads,
                max_length=audio_cfg.audio_max_length,
                output_dim=embed_dim,
            )
        elif audio_cfg.name == "wave_transformer":
            self.audio = WaveformTransformer(
                width=audio_cfg.audio_width,
                layers=audio_cfg.audio_layers,
                heads=audio_cfg.audio_heads,
                conv_feature_layers=audio_cfg.conv_feature_layers,
                output_dim=embed_dim,
                max_length=audio_cfg.audio_max_length,
            )
        elif audio_cfg.name == "hubert":
            self.audio = AudioPretrainedModel(
                pretrained_model_name=audio_cfg.pretrained_model_name,
                width=audio_cfg.audio_width,
                output_dim=embed_dim,
                trainable=audio_cfg.trainable,
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

        elif text_cfg.name == "distilbert":
            self.text = DistilBertPretrain(
                pretrained_model_name=text_cfg.pretrained_model_name,
                trainable=text_cfg.trainable,
            )
        else:
            raise NotImplemented

        self.text_projection = nn.Parameter(torch.randn(text_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

        # self.initialize_parameters()

    # ref form openai/clip
    # def initialize_parameters(self):
    #     nn.init.normal_(self.token_embedding.weight, std=0.02)
    #     nn.init.normal_(self.text_positional_embedding, std=0.01)

    #     proj_std = (self.text.width ** -0.5) * ((2 * self.text.layers) ** -0.5)
    #     attn_std = self.text.width ** -0.5
    #     fc_std = (2 * self.text.width) ** -0.5
    #     for block in self.text.resblocks:
    #         nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
    #         nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
    #         nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
    #         nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def encode_audio(self, audio):
        return self.audio(audio)

    def encode_text(self, text):
        if self.text_cfg.name == "transformer":
            x = self.token_embedding(text)

            x = x + self.text_positional_embedding[:x.shape[1], :]
            x = self.text(x)
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), 0]
        elif self.text_cfg.name == "distilbert":
            x = self.text(text)
        else:
            raise NotImplemented
            
        x = x @ self.text_projection

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


# for downstream prediction
class E2ESLU(nn.Module):
    def __init__(
            self,
            text_config: DictConfig,
            audio_cfg: DictConfig,
            check_point_path: str,
            embed_dim: int = 128,
            hidden_dim: int = 64,
            output_dim: int = 31,
            trainable: bool = True,
            from_pretrain: bool = True,
    ):
        super().__init__()

        self.audio_encoder = AudioTransformer(
            width=audio_cfg.audio_width,
            layers=audio_cfg.audio_layers,
            heads=audio_cfg.audio_heads,
            output_dim=embed_dim,
            max_length=audio_cfg.audio_max_length,

        )
           
        self.final_classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
        model = CLAP(
            text_cfg=text_config,
            audio_cfg=audio_cfg,
            embed_dim=embed_dim,
        )

        model.load_state_dict(torch.load(hydra.utils.get_original_cwd()+"/"+check_point_path))

        if from_pretrain:
            self.audio_encoder = model.audio

            for p in self.audio_encoder.parameters():
                p.requires_grad = trainable

    def forward(self, x):
        output = self.audio_encoder(x)
        output = self.final_classifier(output)
        return output
    




