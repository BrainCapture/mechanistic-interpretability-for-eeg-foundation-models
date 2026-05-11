import torch
from torch import nn
import torch.nn.functional as F

import math
from einops import rearrange
from typing import List, Optional

# Import the channel positions utility
#from channel_positions import get_channel_positions

class ChannelPositionalEncoding(nn.Module):
    """
    Fourier positional encoding for 3D channel positions (x, y, z).
    Adapted from REVE's FourierEmb4D for spatial-only encoding.
    """
    def __init__(self, embed_dim: int, freqs: int = 8, margin: float = 0.4):
        super().__init__()
        self.embed_dim = embed_dim
        self.freqs = freqs
        self.margin = margin

    def forward(self, positions):
        """
        Args:
            positions: (B, C, 3) or (C, 3) tensor of (x, y, z) coordinates
        Returns:
            pos_embed: (B, C, embed_dim) positional encodings
        """
        # Handle (C, 3) input
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)  # (1, C, 3)
        
        B, C, _ = positions.shape
        positions = positions.clone()
        
        # Create frequency grids for x, y, z dimensions
        freqs_z = torch.arange(self.freqs, device=positions.device)
        freqs_y = freqs_z[:, None]
        freqs_x = freqs_y[:, None]
        
        width = 1 + 2 * self.margin
        positions = positions + self.margin
        
        # Compute phase shifts
        p_x = 2 * math.pi * freqs_x / width
        p_y = 2 * math.pi * freqs_y / width
        p_z = 2 * math.pi * freqs_z / width
        
        # Expand positions for broadcasting
        positions = positions[..., None, None, None, :]
        
        # Compute location encodings (REVE's approach for x, y, z)
        loc = (positions[..., 0] * p_x + 
               positions[..., 1] * p_y + 
               positions[..., 2] * p_z).view(B, C, -1)
        
        # Adjust dimensionality if needed
        target_size = self.embed_dim // 2
        if loc.shape[-1] > target_size:
            loc = loc[..., :target_size]
        elif loc.shape[-1] < target_size:
            padding = target_size - loc.shape[-1]
            loc = F.pad(loc, (0, padding))
        
        # Create sine and cosine embeddings (REVE's approach)
        emb = torch.cat([torch.cos(loc), torch.sin(loc)], dim=-1)
        return emb


class SetTransformerWithChannelPE(nn.Module):
    """
    SetTransformer with channel positional encodings.
    Uses REVE's Fourier-based spatial encoding for channel positions.
    """
    def __init__(
        self, 
        in_channels, 
        patch_size, 
        embed_dim, 
        num_heads, 
        num_layers, 
        pooling_head=4, 
        dropout=0.1, 
        max_seq_length=128,
        use_channel_pe=True,
        freqs=8
    ):
        super().__init__()
        
        # Patch embedding (tokenizer)
        self.patch_embedding = Tokenizer(input_size=patch_size, output_size=embed_dim)
        
        # Channel positional encoding
        self.use_channel_pe = use_channel_pe
        if use_channel_pe:
            self.channel_pe = ChannelPositionalEncoding(embed_dim, freqs=freqs)
            # MLP refinement like REVE's mlp4d
            self.channel_pe_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim, bias=False),
                nn.GELU(),
                nn.LayerNorm(embed_dim)
            )
        
        # Spatial pooling
        self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)
        
        # Temporal positional encoding
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=True, 
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Temporal pooling
        self.temporal_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

    def forward(self, x, mask=None, channel_names: Optional[List[str]] = None):
        """
        Args:
            x: (B, C, T) input signal
            mask: (B, T, C) or (B, C, T) mask
            channel_names: List of channel names (e.g., ['Fp1', 'Fp2', ...])
        """
        # Patch embedding
        x = self.patch_embedding(x)  # (B, C, S, E)
        B, C, S, E = x.shape
        
        # Add channel positional encodings if channel names provided
        if self.use_channel_pe and channel_names is not None:
            # Convert channel names to positions
            channel_pos = get_channel_positions(channel_names, device=x.device)  # (C, 3)
            
            # Compute channel positional encodings (REVE's fourier4d approach)
            ch_pe = self.channel_pe(channel_pos)  # (1, C, E) or (B, C, E)
            
            # Refine with MLP (REVE's mlp4d approach)
            ch_pe = self.channel_pe_mlp(ch_pe)  # (B, C, E)
            
            # Broadcast to all time steps and add (adapted from REVE's x + pos_embed)
            ch_pe = ch_pe.unsqueeze(2).expand(-1, -1, S, -1)  # (B, C, S, E)
            x = x + ch_pe
        
        # Original SetTransformer processing
        x = rearrange(x, 'b c s e -> (b s) c e')
        
        # Prepare mask for spatial pooling
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, S, -1)
            mask = rearrange(mask, 'b t c -> (b t) c')
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)
        
        # Spatial pooling
        x = self.spatial_pooling(x, mask)
        x = x.view(B, S, E)
        
        # Temporal positional encoding
        x = self.positional_encoding(x)
        x = self.layer_norm(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        embedding = x.clone()
        
        # Temporal pooling
        x = self.temporal_pooling(x)
        
        return x, embedding  # (B, E), (B, S, E)

##################################################


class PatchEmbedding(nn.Module):
	def __init__(self, in_channels, patch_size, embed_dim):
		super(PatchEmbedding, self).__init__()
		self.patch_size = patch_size
		self.embed_dim = embed_dim

		self.conv = nn.Sequential(
			nn.Conv1d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # 2560 -> 1280
			nn.ReLU(),
			nn.BatchNorm1d(16),
			nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # 1280 -> 640
			nn.ReLU(),
			nn.BatchNorm1d(32),
			nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),  # 640 -> 320
			nn.ReLU(),
			nn.BatchNorm1d(64),
			nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),  # 320 -> 160
			nn.ReLU(),
			nn.BatchNorm1d(128),
			nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),  # 160 -> 80
			nn.ReLU(),
			nn.BatchNorm1d(256),
			nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),  # 80 -> 40
			nn.Flatten(),  # Flatten the output
			nn.Linear(40 * 512, 512)  # Compress further to 512
		)

	def forward(self, x):
		B, C, T = x.shape
		x = x.view(B, C, -1, self.patch_size)
		x = x.permute(0, 1, 2, 3).contiguous().view(-1, 1, self.patch_size)
		x = self.conv(x)
		x = x.view(B, C, -1, self.embed_dim)
		return x


class TokenizerV2(nn.Module):
	"""
	CNN tokenizer used by SleepFM v2.x models.

	Architecture: six Conv1d → LayerNorm → ELU blocks (no BatchNorm),
	followed by AdaptiveAvgPool1d(1) + Linear. Trained with input_size=640
	(5-second patches at 128 Hz).

	Unlike the v1 Tokenizer, LayerNorm shapes are fixed to input_size at
	construction time, so this class must be instantiated with the same
	input_size that was used during training.
	"""

	def __init__(self, input_size=640, output_size=128):
		super().__init__()
		self.input_size  = input_size
		self.output_size = output_size
		s = input_size
		self.tokenizer = nn.Sequential(
			nn.Conv1d(1,   4,   kernel_size=5, stride=2, padding=2), nn.LayerNorm([4,   s // 2]),  nn.ELU(),
			nn.Conv1d(4,   8,   kernel_size=5, stride=2, padding=2), nn.LayerNorm([8,   s // 4]),  nn.ELU(),
			nn.Conv1d(8,   16,  kernel_size=5, stride=2, padding=2), nn.LayerNorm([16,  s // 8]),  nn.ELU(),
			nn.Conv1d(16,  32,  kernel_size=5, stride=2, padding=2), nn.LayerNorm([32,  s // 16]), nn.ELU(),
			nn.Conv1d(32,  64,  kernel_size=5, stride=2, padding=2), nn.LayerNorm([64,  s // 32]), nn.ELU(),
			nn.Conv1d(64,  128, kernel_size=5, stride=2, padding=2), nn.LayerNorm([128, s // 64]), nn.ELU(),
			nn.AdaptiveAvgPool1d(1),
			nn.Flatten(),
			nn.Linear(128, output_size),
		)

	def forward(self, x):
		B, C, T = x.shape
		x = x.view(B, C, -1, self.input_size).contiguous().view(-1, 1, self.input_size)
		x = self.tokenizer(x)
		return x.view(B, C, -1, self.output_size)


class Tokenizer(nn.Module):
	#input shappe: (batch, time X embedding, channel)
	#output shape: (batch, time, embedding, channel)
	def __init__(self, input_size=640, output_size=128):
		super(Tokenizer, self).__init__()
		self.input_size = input_size
		self.output_size = output_size

		# TODO: only one norm, having both shouldn't help
		# TODO: does batch norm share information between time points? This was one of the motivations for LN
		# TODO: residual connections
		# TODO: are there recent papers using modernized 1D CNN variants?
		self.tokenizer = nn.Sequential(
			nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm1d(4),
			nn.ELU(),
			nn.LayerNorm([4, self.input_size//2]),  
			
			nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm1d(8),
			nn.ELU(),
			nn.LayerNorm([8, self.input_size//4]),
			
			nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm1d(16),
			nn.ELU(),
			nn.LayerNorm([16, self.input_size//8]),
			
			nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm1d(32),
			nn.ELU(),
			nn.LayerNorm([32, self.input_size//16]),

			nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm1d(64),
			nn.ELU(),
			nn.LayerNorm([64, self.input_size//32]),

			nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
			nn.BatchNorm1d(128),
			nn.ELU(),
			nn.LayerNorm([128, self.input_size//64]),

			# adaptive avg pool 
			nn.AdaptiveAvgPool1d(1),

			nn.Flatten(),  # Flatten the output
			#nn.Linear(self.input_size//64 * 128, output_size)  # Compress further
			nn.Linear(128, self.output_size)
		)

	def forward(self, x):
		# input: (batch, time x embedding, channels)
		# output: (batch, time, embedding, channels)
		# future improvements: only process items in the batch that are not padded
		
		B, C, T = x.shape
		x = x.view(B, C, -1, self.input_size)
		x = x.permute(0, 1, 2, 3).contiguous().view(-1, 1, self.input_size)
		x = self.tokenizer(x)
		x = x.view(B, C, -1, self.output_size)
		
		return x


class AttentionPooling(nn.Module):
	def __init__(self, input_dim, num_heads=1, dropout=0.1):
		super(AttentionPooling, self).__init__()
		self.transformer_layer = nn.TransformerEncoderLayer(
			d_model=input_dim, 
			nhead=num_heads, 
			dropout=dropout, 
			batch_first=True
		)

	def forward(self, x, key_padding_mask=None):
		batch_size, seq_len, input_dim = x.size()
		
		if key_padding_mask is not None:
			if key_padding_mask.size(1) == 1:
				return x.mean(dim=1)
			if key_padding_mask.dtype != torch.bool:
				key_padding_mask = key_padding_mask.to(dtype=torch.bool)
				
		transformer_output = self.transformer_layer(x, src_key_padding_mask=key_padding_mask)

		# handle the masking here directly as well
		# pooled_output = (transformer_output * key_padding_mask).sum(dim=1) / key_padding_mask.sum(dim=1).clamp(min=1)
		pooled_output = transformer_output.mean(dim=1)  # Average pooling over the sequence length
		
		return pooled_output


class PositionalEncoding(nn.Module):
	def __init__(self, max_seq_len, d_model):
		super().__init__()
		position = torch.arange(max_seq_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(max_seq_len, d_model)
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
		self.register_buffer('pe', pe)

	def forward(self, x):
		# x shape: (batch_size, seq_len, d_model)
		x = x + self.pe[:, :x.size(1), :]
		return x


class SetTransformer(nn.Module):
	def __init__(self, in_channels, patch_size, embed_dim, num_heads, num_layers, pooling_head=4, dropout=0.1, max_seq_length=128):
		super(SetTransformer, self).__init__()
		# self.patch_embedding = PatchEmbeddingLinear(in_channels, patch_size, embed_dim)
		self.patch_embedding = Tokenizer(input_size=patch_size, output_size=embed_dim)

		self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

		self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
		self.layer_norm = nn.LayerNorm(embed_dim)

		encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

		self.temporal_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

	def forward(self, x, mask=None):

		x = self.patch_embedding(x)
		B, C, S, E = x.shape
		x = rearrange(x, 'b c s e -> (b s) c e')

		if mask is not None:
			mask = mask.unsqueeze(1).expand(-1, S, -1)
			mask = rearrange(mask, 'b t c -> (b t) c')

			if mask.dtype != torch.bool:
				mask = mask.to(dtype=torch.bool)

		x = self.spatial_pooling(x, mask)
		x = x.view((B, S, E))

		x = self.positional_encoding(x)
		x = self.layer_norm(x)

		x = self.transformer_encoder(x)
		embedding = x.clone()
		x = self.temporal_pooling(x)
		return x, embedding # (128, 1), (128, seconds)