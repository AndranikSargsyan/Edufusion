from typing import Optional, Tuple

import torch
import torch.nn as nn

from .tokenizer import SimpleTokenizer


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class CLIPTextEmbeddings(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        embed_dim: int = 768,
        max_position_embeddings: int = 77,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids", torch.arange(max_position_embeddings).expand((1, -1)), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        q = self.q_proj(hidden_states) * self.scale
        k = self._shape(self.k_proj(hidden_states), -1, bsz)
        v = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        q = self._shape(q, tgt_len, bsz).view(*proj_shape)
        k = k.view(*proj_shape)
        v = v.view(*proj_shape)

        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        causal_attention_mask = torch.ones(
            size=(bsz, 1, tgt_len, tgt_len),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        ).triu(1) * (-3.0**39)
            
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
        attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, v)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


class CLIPMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 768,
        intermediate_size: int = 3072
    ):
        super().__init__()

        self.activation_fn = QuickGELUActivation()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 77,
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = CLIPAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout
        )
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)
        self.mlp = CLIPMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `num_hidden_layers` self attention layers. Each layer is a [`CLIPEncoderLayer`].
    """

    def __init__(self, num_hidden_layers: int, **config):
        super().__init__()
        self.layers = nn.ModuleList([CLIPEncoderLayer(**config) for _ in range(num_hidden_layers)])

    def forward(self, inputs_embeds: torch.FloatTensor) -> torch.Tensor:
        """
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
        """
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(hidden_states)
        return hidden_states


class CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 49408,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        max_position_embeddings: int = 77,
        layer_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0
    ):
        super().__init__()

        embed_dim = hidden_size
        self.embeddings = CLIPTextEmbeddings(
            vocab_size=vocab_size,
            embed_dim=hidden_size,
            max_position_embeddings=max_position_embeddings,
        )
        self.encoder = CLIPEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            attention_dropout=attention_dropout
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        last_hidden_state = self.encoder(inputs_embeds=hidden_states)
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        return last_hidden_state


class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    def __init__(self, device: str = "cuda", max_length: int = 77, freeze: bool = True, bpe_path: str = None):
        super().__init__()
        
        self.tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        self.text_model = CLIPTextTransformer(
            vocab_size=self.tokenizer.vocab_size,
            max_position_embeddings=max_length
        )
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.text_model = self.text_model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.tokenize(text, truncate=True, context_length=self.max_length)
        tokens = tokens.to(self.device)
        output = self.text_model(input_ids=tokens)
        return output

    def encode(self, text: str) -> torch.Tensor:
        return self(text)
