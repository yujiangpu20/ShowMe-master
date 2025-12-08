import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from collections import defaultdict

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False
from lvdm.common import (
    checkpoint,
    exists,
    default,
)
from lvdm.basics import zero_module


class RelativePosition(nn.Module):
    """ https://github.com/evelinehong/Transformer_Relative_Position_PyTorch/blob/master/relative_position.py """

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        device = self.embeddings_table.device
        range_vec_q = torch.arange(length_q, device=device)
        range_vec_k = torch.arange(length_k, device=device)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.long()
        embeddings = self.embeddings_table[final_mat]
        return embeddings


class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank, scaling=1.0, dropout=0.0):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank) * std_dev)
        self.lora_B = nn.Parameter(torch.zeros(rank, output_dim))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = scaling

    def forward(self, x):
        x = self.lora_dropout(x)
        x = self.scaling * (x @ self.lora_A @ self.lora_B)
        return x


class CrossAttention(nn.Module):

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.,
                 relative_position=False, temporal_length=None, video_length=None, image_cross_attention=False,
                 image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False, text_context_len=77,
                 block_type='spatial',  # spatial or temporal
                 **kwargs):
        super().__init__()
        # Determine this attention layer's type based on context_dim
        # - If context_dim is None, this block is used as self-attention (attn1).
        # - If context_dim is not None, this block is used as cross-attention (attn2).
        self.is_self_attn = (context_dim is None)
        self.attn_type = 'self_attn' if self.is_self_attn else 'cross_attn'

        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        self.block_type = block_type
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.relative_position = relative_position
        if self.relative_position:
            assert (temporal_length is not None)
            self.relative_position_k = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
            self.relative_position_v = RelativePosition(num_units=dim_head, max_relative_position=temporal_length)
        else:
            ## only used for spatial attention, while NOT for temporal attention
            if XFORMERS_IS_AVAILBLE and temporal_length is None:
                self.forward = self.efficient_forward

        self.video_length = video_length
        self.image_cross_attention = image_cross_attention
        self.image_cross_attention_scale = image_cross_attention_scale
        self.text_context_len = text_context_len
        self.image_cross_attention_scale_learnable = image_cross_attention_scale_learnable
        if self.image_cross_attention:
            self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)

            if image_cross_attention_scale_learnable:
                self.register_parameter('alpha', nn.Parameter(torch.tensor(0.)))

        # Handle LoRA initialization
        self.lora_tuning = (kwargs.get('tuning_type', None) == 'lora')
        self.lora_configs = kwargs.get('lora_configs', {})
        self.lora_layers = defaultdict(dict)  # stage_name -> { logical_key -> LoRALayer }

        # For storing LoRA layers per stage and per Q/K/V (and image cross-attention if applicable)
        if self.lora_tuning and len(self.lora_configs) > 0:
            self._init_lora_layers()

    def _init_lora_layers(self):
        for stage_name, config in self.lora_configs.items():
            apply_to = config['apply_to']
            rank = config['rank']
            dropout = config['dropout']
            scaling = config.get('scaling', 1.0)

            # Determine if this block_type is specified in the LoRA config
            attn_key = self.block_type
            if attn_key not in apply_to:
                continue

            # Only add LoRA layers for this block's attn_type (self_attn or cross_attn)
            if self.attn_type not in apply_to[attn_key]:
                continue

            layers_to_add = apply_to[attn_key][self.attn_type]

            # For self-attn: we apply q, k, v LoRA only if self_attn is required.
            # For cross-attn: we apply q, k, v and if image_cross_attention, also k_ip, v_ip.
            prefix = f"{stage_name}_lora_{self.attn_type}"

            if 'q' in layers_to_add:
                layer = LoRALayer(self.to_q.in_features, self.to_q.out_features, rank, scaling, dropout)
                setattr(self, f"{prefix}_q", layer)
                self.lora_layers[stage_name][f"{self.attn_type}_q"] = layer

            if 'k' in layers_to_add:
                layer = LoRALayer(self.to_k.in_features, self.to_k.out_features, rank, scaling, dropout)
                setattr(self, f"{prefix}_k", layer)
                self.lora_layers[stage_name][f"{self.attn_type}_k"] = layer

            if 'v' in layers_to_add:
                layer = LoRALayer(self.to_v.in_features, self.to_v.out_features, rank, scaling, dropout)
                setattr(self, f"{prefix}_v", layer)
                self.lora_layers[stage_name][f"{self.attn_type}_v"] = layer

            # If this is a cross-attn block and image_cross_attention is enabled,
            # we may also have k_ip, v_ip layers:
            if self.attn_type == 'cross_attn' and self.image_cross_attention:
                prefix_ip = f"{stage_name}_lora_{self.attn_type}_ip"
                if 'k_ip' in layers_to_add and hasattr(self, 'to_k_ip'):
                    layer = LoRALayer(self.to_k_ip.in_features, self.to_k_ip.out_features, rank, scaling, dropout)
                    setattr(self, f"{prefix_ip}_k", layer)
                    self.lora_layers[stage_name][f"{self.attn_type}_ip_k"] = layer

                if 'v_ip' in layers_to_add and hasattr(self, 'to_v_ip'):
                    layer = LoRALayer(self.to_v_ip.in_features, self.to_v_ip.out_features, rank, scaling, dropout)
                    setattr(self, f"{prefix_ip}_v", layer)
                    self.lora_layers[stage_name][f"{self.attn_type}_ip_v"] = layer

    def apply_lora(self, x, lora_key, orig_input):
        if not self.lora_tuning:
            return x
        for stage_name, stage_layers in self.lora_layers.items():
            if lora_key in stage_layers:
                x = x + stage_layers[lora_key](orig_input)
        return x

    def merge_lora(self):
        """
        Merges LoRA weights into the base linear layers to speed up inference.
        Sets self.lora_tuning = False so apply_lora becomes a no-op.
        """
        if not self.lora_tuning:
            return

        for stage_name, stage_layers in self.lora_layers.items():
            for lora_key, lora_module in stage_layers.items():
                scale = lora_module.scaling
                # Calculate delta weight: scale * (A @ B).T -> scale * (B.T @ A.T)
                # lora_A: [in, rank], lora_B: [rank, out]
                # B.T: [out, rank], A.T: [rank, in] -> result: [out, in]
                delta_w = scale * (lora_module.lora_B.t() @ lora_module.lora_A.t())

                # Identify target base layer based on key suffix
                target_module = None
                if lora_key.endswith('_q'):
                    target_module = self.to_q
                elif lora_key.endswith('_k'):
                    target_module = self.to_k
                elif lora_key.endswith('_v'):
                    target_module = self.to_v
                elif lora_key.endswith('_ip_k'):
                    target_module = self.to_k_ip
                elif lora_key.endswith('_ip_v'):
                    target_module = self.to_v_ip

                if target_module is not None:
                    # Merge weights
                    target_module.weight.data.add_(delta_w.to(target_module.weight.device))
                    # Note: LoRA usually doesn't have bias, so we only update weight.

        # Disable LoRA path to avoid double-counting and loop overhead
        self.lora_tuning = False
        # Clear the dictionary to free memory (optional)
        self.lora_layers.clear()

    def forward(self, x, context=None, mask=None):
        # Determine attn_type at runtime (for temporal/spatial):
        spatial_self_attn = (context is None)
        attn_type = 'self_attn' if spatial_self_attn else 'cross_attn'

        h = self.heads
        q = self.to_q(x)
        context = default(context, x)

        # Apply LoRA to q
        q = self.apply_lora(q, f'{attn_type}_q', x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:, :self.text_context_len, :], context[:, self.text_context_len:, :]
            k = self.to_k(context)
            v = self.to_v(context)
            k = self.apply_lora(k, f'{attn_type}_k', context)
            v = self.apply_lora(v, f'{attn_type}_v', context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
            k_ip = self.apply_lora(k_ip, f'{attn_type}_ip_k', context_image)
            v_ip = self.apply_lora(v_ip, f'{attn_type}_ip_v', context_image)
        else:
            if not spatial_self_attn:
                context = context[:, :self.text_context_len, :]
            k = self.to_k(context)
            v = self.to_v(context)
            k = self.apply_lora(k, f'{attn_type}_k', context)
            v = self.apply_lora(v, f'{attn_type}_v', context)
            k_ip, v_ip = None, None

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        if self.relative_position:
            len_q, len_k, len_v = q.shape[1], k.shape[1], v.shape[1]
            k2 = self.relative_position_k(len_q, len_k)
            sim2 = einsum('b t d, t s d -> b t s', q, k2) * self.scale  # TODO check
            sim += sim2
        del k

        if exists(mask):
            ## feasible for causal attention mask only
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b i j -> (b h) i j', h=h)
            sim.masked_fill_(~(mask > 0.5), max_neg_value)

        # attention, what we cannot get enough of
        sim = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', sim, v)
        if self.relative_position:
            v2 = self.relative_position_v(len_q, len_v)
            out2 = einsum('b t s, t s d -> b t d', sim, v2)  # TODO check
            out += out2
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k_ip, v_ip))
            sim_ip = torch.einsum('b i d, b j d -> b i j', q, k_ip) * self.scale
            del k_ip
            sim_ip = sim_ip.softmax(dim=-1)
            out_ip = torch.einsum('b i j, b j d -> b i d', sim_ip, v_ip)
            out_ip = rearrange(out_ip, '(b h) n d -> b n (h d)', h=h)
        else:
            out_ip = None

        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
            else:
                out = out + self.image_cross_attention_scale * out_ip

        return self.to_out(out)

    def efficient_forward(self, x, context=None, mask=None):
        spatial_self_attn = (context is None)
        attn_type = 'self_attn' if spatial_self_attn else 'cross_attn'
        k_ip, v_ip, out_ip = None, None, None

        q = self.to_q(x)
        context = default(context, x)
        q = self.apply_lora(q, f'{attn_type}_q', x)

        if self.image_cross_attention and not spatial_self_attn:
            context, context_image = context[:, :self.text_context_len, :], context[:, self.text_context_len:, :]
            k = self.to_k(context)
            v = self.to_v(context)
            k = self.apply_lora(k, f'{attn_type}_k', context)
            v = self.apply_lora(v, f'{attn_type}_v', context)
            k_ip = self.to_k_ip(context_image)
            v_ip = self.to_v_ip(context_image)
            k_ip = self.apply_lora(k_ip, f'{attn_type}_ip_k', context_image)
            v_ip = self.apply_lora(v_ip, f'{attn_type}_ip_v', context_image)
        else:
            if not spatial_self_attn:
                context = context[:, :self.text_context_len, :]
            k = self.to_k(context)
            v = self.to_v(context)
            k = self.apply_lora(k, f'{attn_type}_k', context)
            v = self.apply_lora(v, f'{attn_type}_v', context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )
        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        ## for image cross-attention
        if k_ip is not None:
            k_ip, v_ip = map(
                lambda t: t.unsqueeze(3)
                .reshape(b, t.shape[1], self.heads, self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b * self.heads, t.shape[1], self.dim_head)
                .contiguous(),
                (k_ip, v_ip),
            )
            out_ip = xformers.ops.memory_efficient_attention(q, k_ip, v_ip, attn_bias=None, op=None)
            out_ip = (
                out_ip.unsqueeze(0)
                .reshape(b, self.heads, out.shape[1], self.dim_head)
                .permute(0, 2, 1, 3)
                .reshape(b, out.shape[1], self.heads * self.dim_head)
            )

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        if out_ip is not None:
            if self.image_cross_attention_scale_learnable:
                out = out + self.image_cross_attention_scale * out_ip * (torch.tanh(self.alpha) + 1)
            else:
                out = out + self.image_cross_attention_scale * out_ip

        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, attention_cls=None, video_length=None, image_cross_attention=False,
                 image_cross_attention_scale=1.0, image_cross_attention_scale_learnable=False,
                 text_context_len=77, block_type='spatial', **kwargs):

        super().__init__()
        attn_cls = CrossAttention if attention_cls is None else attention_cls
        # For spatial transformers:
        # attn1: self-attention (context_dim=None)
        # attn2: cross-attention (context_dim=context_dim)
        # For temporal transformers:
        # attn1: self-attention (context_dim=None)
        # attn2: self-attention (context_dim=None)
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None,
                              block_type=block_type,
                              **kwargs)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, **kwargs)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              video_length=video_length, image_cross_attention=image_cross_attention,
                              image_cross_attention_scale=image_cross_attention_scale,
                              image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                              text_context_len=text_context_len,
                              block_type=block_type,
                              **kwargs)
        self.image_cross_attention = image_cross_attention

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, **kwargs):
        ## implementation tricks: because checkpointing doesn't support non-tensor (e.g. None or scalar) arguments
        input_tuple = (x,)      ## should not be (x), otherwise *input_tuple will decouple x into multiple arguments
        if context is not None:
            input_tuple = (x, context)
        if mask is not None:
            forward_mask = partial(self._forward, mask=mask)
            return checkpoint(forward_mask, (x,), self.parameters(), self.checkpoint)
        return checkpoint(self._forward, input_tuple, self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x  # self-attention
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x  # dual cross-attention
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data in spatial axis.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 conv
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, disable_self_attn=False, use_linear=False, video_length=None,
                 image_cross_attention=False, image_cross_attention_scale_learnable=False,
                 tuning_type=None, lora_configs=None,
                 **kwargs):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                disable_self_attn=disable_self_attn,
                checkpoint=use_checkpoint,
                video_length=video_length,
                image_cross_attention=image_cross_attention,
                image_cross_attention_scale_learnable=image_cross_attention_scale_learnable,
                block_type='spatial',
                tuning_type=tuning_type,
                lora_configs=lora_configs,
                **kwargs,
                ) for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None, **kwargs):
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context, **kwargs)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)

        return x + x_in


class TemporalTransformer(nn.Module):
    """
    Transformer block for image-like data in temporal axis.
    First, reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None,
                 use_checkpoint=True, use_linear=False, only_self_att=True, causal_attention=False, causal_block_size=1,
                 relative_position=False, temporal_length=None, tuning_type=None, lora_configs=None, **kwargs):
        super().__init__()
        self.only_self_att = only_self_att
        self.relative_position = relative_position
        self.causal_attention = causal_attention
        self.causal_block_size = causal_block_size

        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        if not use_linear:
            self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        if relative_position:
            assert(temporal_length is not None)
            attention_cls = partial(CrossAttention, relative_position=True, temporal_length=temporal_length)
        else:
            attention_cls = partial(CrossAttention, temporal_length=temporal_length)
        if self.causal_attention:
            assert(temporal_length is not None)
            self.mask = torch.tril(torch.ones([1, temporal_length, temporal_length]))

        if self.only_self_att:
            context_dim = None

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim,
                n_heads,
                d_head,
                dropout=dropout,
                context_dim=context_dim,
                attention_cls=attention_cls,
                checkpoint=use_checkpoint,
                block_type='temporal',
                tuning_type=tuning_type,
                lora_configs=lora_configs,
                **kwargs,
            )
            for d in range(depth)
        ])
        if not use_linear:
            self.proj_out = zero_module(nn.Conv1d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(inner_dim, in_channels))
        self.use_linear = use_linear

    def forward(self, x, context=None):
        b, c, t, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = rearrange(x, 'b c t h w -> (b h w) c t').contiguous()
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'bhw c t -> bhw t c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        temp_mask = None
        if self.causal_attention:
            # slice from mask map
            temp_mask = self.mask[:, :t, :t].to(x.device)

        if temp_mask is not None:
            mask = temp_mask.to(x.device)
            mask = repeat(mask, 'l i j -> (l bhw) i j', bhw=b*h*w)
        else:
            mask = None

        if self.only_self_att:
            ## note: if no context is given, cross-attention defaults to self-attention
            for i, block in enumerate(self.transformer_blocks):
                x = block(x, mask=mask)
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
        else:
            x = rearrange(x, '(b hw) t c -> b hw t c', b=b).contiguous()
            context = rearrange(context, '(b t) l con -> b t l con', t=t).contiguous()
            for i, block in enumerate(self.transformer_blocks):
                # calculate each batch one by one (since number in shape could not greater than 65,535 for some package)
                for j in range(b):
                    context_j = repeat(
                        context[j],
                        't l con -> (t r) l con', r=(h * w) // t, t=t).contiguous()
                    ## note: causal mask will not applied in cross-attention case
                    x[j] = block(x[j], context=context_j)

        if self.use_linear:
            x = self.proj_out(x)
            x = rearrange(x, 'b (h w) t c -> b c t h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = rearrange(x, 'b hw t c -> (b hw) c t').contiguous()
            x = self.proj_out(x)
            x = rearrange(x, '(b h w) c t -> b c t h w', b=b, h=h, w=w).contiguous()

        return x + x_in

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., tuning_type=None, lora_configs=None, **kwargs):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

        self.lora_tuning = (tuning_type == 'lora')
        self.lora_configs = lora_configs if lora_configs is not None else {}
        self.lora_layers = defaultdict(dict)
        if self.lora_tuning and len(self.lora_configs) > 0:
            self._init_lora_layers(dim, inner_dim, dim_out)

    def _init_lora_layers(self, dim, inner_dim, dim_out):
        for stage_name, config in self.lora_configs.items():
            apply_to = config['apply_to']
            if 'ff' in apply_to and apply_to['ff'] == True:
                rank = config['ff_rank']
                dropout = config['dropout']
                scaling = config.get('scaling', 1.0)

                if isinstance(self.net[0], GEGLU):
                    in_layer_out_dim = 2 * inner_dim
                else:
                    in_layer_out_dim = inner_dim

                in_lora = LoRALayer(dim, in_layer_out_dim, rank, scaling, dropout)
                out_lora = LoRALayer(inner_dim, dim_out, rank, scaling, dropout)

                setattr(self, f"{stage_name}_lora_ff_in", in_lora)
                setattr(self, f"{stage_name}_lora_ff_out", out_lora)

                self.lora_layers.setdefault(stage_name, {})
                self.lora_layers[stage_name]['in'] = in_lora
                self.lora_layers[stage_name]['out'] = out_lora

    def merge_lora(self):
        """
        Merges LoRA weights into FeedForward layers.
        """
        if not self.lora_tuning:
            return

        for stage_name, stage_layers in self.lora_layers.items():
            # 1. Merge 'in' projection
            if 'in' in stage_layers:
                lora_module = stage_layers['in']
                scale = lora_module.scaling
                delta_w = scale * (lora_module.lora_B.t() @ lora_module.lora_A.t())

                # Handle GEGLU structure
                if isinstance(self.net[0], GEGLU):
                    target_module = self.net[0].proj
                else:
                    target_module = self.net[0][0]

                target_module.weight.data.add_(delta_w.to(target_module.weight.device))

            # 2. Merge 'out' projection
            if 'out' in stage_layers:
                lora_module = stage_layers['out']
                scale = lora_module.scaling
                delta_w = scale * (lora_module.lora_B.t() @ lora_module.lora_A.t())

                target_module = self.net[2]
                target_module.weight.data.add_(delta_w.to(target_module.weight.device))

        self.lora_tuning = False
        self.lora_layers.clear()

    def forward(self, x):
        if self.lora_tuning:
            if isinstance(self.net[0], GEGLU):
                proj_linear = self.net[0].proj
                raw_out = proj_linear(x)

                # gather sum of all 'in' LoRA from different stages
                lora_sum_in = 0
                for stage_name, stage_layers in self.lora_layers.items():
                    if 'in' in stage_layers:
                        lora_sum_in += stage_layers['in'](x)

                raw_out = raw_out + lora_sum_in
                x_out, gate_out = raw_out.chunk(2, dim=-1)  # chunk for gating
                x = x_out * F.gelu(gate_out)

            else:
                in_linear = self.net[0][0]
                activation = self.net[0][1]
                raw_out = in_linear(x)

                # gather LoRA
                lora_sum_in = 0
                for stage_name, stage_layers in self.lora_layers.items():
                    if 'in' in stage_layers:
                        lora_sum_in += stage_layers['in'](x)

                raw_out = raw_out + lora_sum_in
                x = activation(raw_out)

            x = self.net[1](x)
            out_linear = self.net[2]
            raw_out = out_linear(x)

            # gather LoRA
            lora_sum_out = 0
            for stage_name, stage_layers in self.lora_layers.items():
                if 'out' in stage_layers:
                    lora_sum_out += stage_layers['out'](x)

            x = raw_out + lora_sum_out
            return x
        else:
            return self.net(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
