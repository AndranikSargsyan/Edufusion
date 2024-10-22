import math
from abc import abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.IntTensor, dim, max_period: int = 10000) -> torch.FloatTensor:
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x: torch.FloatTensor, emb: torch.FloatTensor):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x: int, emb: torch.FloatTensor, context: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class GroupNorm32(nn.GroupNorm):
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return super().forward(x.float()).type(x.dtype)
    
    
class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: Optional[int] = None, heads: int = 8, dim_head: int = 64, dropout: float=0.0):
        super().__init__()

        inner_dim = dim_head * heads
        context_dim = context_dim or query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x: torch.FloatTensor, context: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        h = self.heads
        q = self.to_q(x)
        context = x if context is None else context
        k = self.to_k(context)
        v = self.to_v(context)
        
        q, k, v = [
            t.view(t.shape[0], t.shape[1], h, t.shape[2] // h)
                .permute(0, 2, 1, 3)
                .reshape(t.shape[0] * h, t.shape[1], t.shape[2] // h) 
            for t in (q, k, v)
        ]

        with torch.autocast(enabled=False, device_type="cuda"):
            q, k = q.float(), k.float()
            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
        del q, k
        sim = sim.softmax(dim=-1)
        out = torch.einsum("b i j, b j d -> b i d", sim, v)
        # (b h) n d -> b n (h d)
        out = out.view(out.shape[0] // h, h, out.shape[1], out.shape[2])\
            .permute(0, 2, 1, 3)\
            .reshape(out.shape[0] // h, out.shape[1], -1) 
        
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, dim_out: int = None, mult: int = 4, glu: bool = False, dropout: float = 0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out or dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.net(x)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self, dim: int, n_heads: int, d_head: int, dropout: float = 0.0,
        context_dim: Optional[int] = None, gated_ff: bool = True, checkpoint: bool = True
    ):
        super().__init__()

        self.attn1 = CrossAttention(query_dim=dim,heads=n_heads,dim_head=d_head,dropout=dropout,context_dim=None)
        # is self-attn if context is none
        self.attn2 = CrossAttention(query_dim=dim,context_dim=context_dim,heads=n_heads,dim_head=d_head,dropout=dropout)
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x: torch.FloatTensor, context: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        if self.checkpoint:
            return checkpoint(self._forward, x, context)
        else:
            return self._forward(x, context)

    def _forward(self, x: torch.FloatTensor, context: Optional[torch.FloatTensor]=None) -> torch.FloatTensor:
        x = x + self.attn1(self.norm1(x), context=None)
        x = x + self.attn2(self.norm2(x), context=context)
        x = x + self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        n_heads: int,
        d_head: int,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: Optional[int] = None,
        use_linear: bool = False,
        use_checkpoint: bool = True
    ):
        super().__init__()
        
        if context_dim is not None and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
       
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim[d],
                    checkpoint=use_checkpoint
                )
                for d in range(depth)]
            )
        
        if not use_linear:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1,padding=0)
        else:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        
        self.use_linear = use_linear

    def forward(self, x: int, context: torch.FloatTensor = None) -> torch.FloatTensor:
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, context=context[i])
        if self.use_linear:
            x = self.proj_out(x)
        x = x.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in


class Upsample(nn.Module):
    """
    An upsampling layer with convolution.
    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels: int, out_channels: Optional[int] = None, padding: int = 1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.Conv2d(self.channels, self.out_channels, 3, padding=padding)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        assert x.shape[1] == self.channels
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with convolution.
    :param channels: channels in the inputs and outputs.
    """

    def __init__(self, channels: int, out_channels: Optional[int] = None, padding: int = 1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.op = nn.Conv2d(self.channels, self.out_channels, 3, stride=2, padding=padding)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        out_channels: Optional[int] = None,
        use_checkpoint: bool = False,
        up: bool = False,
        down: bool = False
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, self.out_channels),
        )
        self.out_layers = nn.Sequential(
            GroupNorm32(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x: torch.FloatTensor, emb: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.use_checkpoint:
            x = checkpoint(self._forward, x, emb)
        else:
            x = self._forward(x, emb)
        return x

    def _forward(self, x: torch.FloatTensor, emb: torch.FloatTensor) -> torch.FloatTensor:
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int] = (4, 2, 1),
        dropout: float = 0.0,
        channel_mult: Tuple[int] = (1, 2, 4, 4),
        use_checkpoint=False,
        num_heads: int = 8,
        num_head_channels: int = -1,
        context_dim: int = 768,
        num_attention_blocks: Optional[int] = None,
        use_linear_in_transformer: bool = False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int): self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else: self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
            
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1)
            )
        ])
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [ResBlock(ch,time_embed_dim,dropout,out_channels=mult * model_channels,use_checkpoint=use_checkpoint)]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if num_attention_blocks is None or nr < num_attention_blocks[level]:
                        layers.append(
                            SpatialTransformer(ch,num_heads,dim_head,context_dim=context_dim,
                                use_linear=use_linear_in_transformer,use_checkpoint=use_checkpoint)
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(Downsample(ch, out_channels=ch)))
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch,time_embed_dim, dropout, use_checkpoint=use_checkpoint),
            SpatialTransformer(
                ch,num_heads,dim_head,context_dim=context_dim,
                use_linear=use_linear_in_transformer,use_checkpoint=use_checkpoint
            ),
            ResBlock(ch, time_embed_dim, dropout, use_checkpoint=use_checkpoint)
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [ResBlock(ch + ich,time_embed_dim,dropout,out_channels=model_channels * mult,use_checkpoint=use_checkpoint)]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if (num_attention_blocks is None or i < num_attention_blocks[level]):
                        layers.append(
                            SpatialTransformer(ch,num_heads,dim_head,context_dim=context_dim,
                                use_linear=use_linear_in_transformer,use_checkpoint=use_checkpoint)
                        )
                if level and i == self.num_res_blocks[level]:
                    layers.append(Upsample(ch, out_channels=ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            GroupNorm32(32, ch),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.FloatTensor, timesteps: torch.IntTensor, context: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of integer timesteps.
        :param context: conditioning plugged in via crossattn
        :return: an [N x C x ...] Tensor of outputs.
        """
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_embed(t_emb)
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        return self.out(h)
