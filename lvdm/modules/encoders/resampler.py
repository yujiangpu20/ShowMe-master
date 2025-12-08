# modified from https://github.com/mlfoundations/open_flamingo/blob/main/open_flamingo/src/helpers.py
# and https://github.com/lucidrains/imagen-pytorch/blob/main/imagen_pytorch/imagen_pytorch.py
# and https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/resampler.py
import math
import torch
import torch.nn as nn
from lvdm.modules.attention import LoRALayer
from collections import defaultdict


class ImageProjModel(nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        # embeds = image_embeds
        embeds = image_embeds.type(list(self.proj.parameters())[0].dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens,
                                                              self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8, tuning_type=None, lora_configs=None, **kwargs):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        self.block_type = 'spatial'
        self.tuning_type = tuning_type
        self.lora_configs = lora_configs if lora_configs is not None else {}
        self.lora_tuning = (tuning_type == 'lora')

        # Initialize LoRA layers based on lora_configs
        self.lora_layers = defaultdict(dict)
        if self.lora_tuning and len(self.lora_configs) > 0:
            self._init_lora_layers()

    def _init_lora_layers(self):
        for stage_name, config in self.lora_configs.items():
            apply_to = config['apply_to']
            rank = config['rank']
            dropout = config['dropout']
            scaling = config.get('scaling', 1.0)

            attn_key = self.block_type
            if attn_key not in apply_to:
                continue

            # Q-Former = cross-attn by default
            if 'cross_attn' in apply_to[attn_key]:
                layers_to_add = apply_to[attn_key]['cross_attn']
                prefix = f"{stage_name}_lora_cross_attn"

                if 'q' in layers_to_add:
                    layer = LoRALayer(self.to_q.in_features, self.to_q.out_features, rank, scaling, dropout)
                    setattr(self, f"{prefix}_q", layer)
                    self.lora_layers[stage_name]['cross_attn_q'] = layer

                if 'k' in layers_to_add:
                    layer = LoRALayer(self.to_kv.in_features, (self.to_kv.out_features // 2), rank, scaling, dropout)
                    setattr(self, f"{prefix}_k", layer)
                    self.lora_layers[stage_name]['cross_attn_k'] = layer

                if 'v' in layers_to_add:
                    layer = LoRALayer(self.to_kv.in_features, (self.to_kv.out_features // 2), rank, scaling, dropout)
                    setattr(self, f"{prefix}_v", layer)
                    self.lora_layers[stage_name]['cross_attn_v'] = layer

    def apply_lora(self, x, lora_key, orig_input):
        if not self.lora_tuning:
            return x
        for stage_name, stage_layers in self.lora_layers.items():
            if lora_key in stage_layers:
                x = x + stage_layers[lora_key](orig_input)
        return x

    def merge_lora(self):
        if not self.lora_tuning:
            return

        for stage_name, stage_layers in self.lora_layers.items():
            # 1. Merge Query
            if 'cross_attn_q' in stage_layers:
                lora_module = stage_layers['cross_attn_q']
                delta_w = lora_module.scaling * (lora_module.lora_B.t() @ lora_module.lora_A.t())
                self.to_q.weight.data.add_(delta_w.to(self.to_q.weight.device))

            # 2. Merge Key (Fused into first half of to_kv)
            if 'cross_attn_k' in stage_layers:
                lora_module = stage_layers['cross_attn_k']
                delta_w = lora_module.scaling * (lora_module.lora_B.t() @ lora_module.lora_A.t())

                # to_kv outputs [dim_inner * 2], Key is the first half
                dim_out = delta_w.shape[0]  # should be inner_dim
                self.to_kv.weight.data[:dim_out, :].add_(delta_w.to(self.to_kv.weight.device))

            # 3. Merge Value (Fused into second half of to_kv)
            if 'cross_attn_v' in stage_layers:
                lora_module = stage_layers['cross_attn_v']
                delta_w = lora_module.scaling * (lora_module.lora_B.t() @ lora_module.lora_A.t())

                # Value is the second half
                dim_out = delta_w.shape[0]
                self.to_kv.weight.data[dim_out:, :].add_(delta_w.to(self.to_kv.weight.device))

        self.lora_tuning = False
        self.lora_layers.clear()

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # Apply LoRA layers
        q = self.apply_lora(q, 'cross_attn_q', latents)
        k = self.apply_lora(k, 'cross_attn_k', kv_input)
        v = self.apply_lora(v, 'cross_attn_v', kv_input)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
            self,
            dim=1024,
            depth=8,
            dim_head=64,
            heads=16,
            num_queries=8,
            embedding_dim=768,
            output_dim=1024,
            ff_mult=4,
            video_length=None,  # using frame-wise version or not
            tuning_type=None,
            lora_configs=None,
            **kwargs,
    ):
        super().__init__()
        ## queries for a single frame / image
        self.num_queries = num_queries
        self.video_length = video_length

        ## <num_queries> queries for each frame
        if video_length is not None:
            num_queries = num_queries * video_length

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads,
                                           tuning_type=tuning_type,
                                           lora_configs=lora_configs,
                                           **kwargs),  # newly added
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def merge_lora(self):
        """Recursively call merge_lora on all Perceiver blocks"""
        for layer_list in self.layers:
            attn = layer_list[0]
            ff = layer_list[1]
            if hasattr(attn, 'merge_lora'):
                attn.merge_lora()
            if hasattr(ff, 'merge_lora'):
                ff.merge_lora()

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)  ## B (T L) C
        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)  # B L C or B (T L) C

        return latents