import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize weights.
def init_w(module, gain=1.0):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

# Down-sampling block: a stride-2 conv.
class Down(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        init_w(self.conv)
    def forward(self, inp, time_emb=None):
        return self.conv(inp)

# Up-sampling block: nearest neighbor upsample then conv.
class Up(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        init_w(self.conv)
    def forward(self, inp, time_emb=None):
        inp = F.interpolate(inp, scale_factor=2, mode='nearest')
        return self.conv(inp)

# Self-attention block.
class Attn(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        for conv in (self.q, self.k, self.v, self.proj):
            init_w(conv)
    def forward(self, inp):
        batch, channel, height, width = inp.shape
        normed = self.norm(inp)
        query = self.q(normed).permute(0, 2, 3, 1).reshape(batch, height * width, channel)
        key = self.k(normed).reshape(batch, channel, height * width)
        attn_scores = torch.bmm(query, key) * (channel ** -0.5)
        attn_scores = F.softmax(attn_scores, dim=-1)
        value = self.v(normed).permute(0, 2, 3, 1).reshape(batch, height * width, channel)
        out = torch.bmm(attn_scores, value).reshape(batch, height, width, channel).permute(0, 3, 1, 2)
        return inp + self.proj(out)

""" # Residual block with optional attention.
class Res(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout, use_attn=False):
        super().__init__()
        self.attn = Attn(out_channels) if use_attn else nn.Identity()
        self.apply(lambda module: init_w(module) if isinstance(module, (nn.Conv2d, nn.Linear)) else None)
    def forward(self, inp, time_emb):
        h = inp
        # TODO: Implement the forward pass for the residual block and store the output in `h`.
        return self.attn(h)
 """

class Res(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim, dropout, use_attn=False):
        super().__init__()

        # Block 1
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        # Time embedding projection
        self.time_proj = nn.Linear(time_dim, out_channels)

        # Block 2
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

        # Optional attention
        self.attn = Attn(out_channels) if use_attn else nn.Identity()

        # Initialize weights
        self.apply(lambda module: init_w(module) if isinstance(module, (nn.Conv2d, nn.Linear)) else None)

    def forward(self, inp, time_emb):

        # ----- Block 1 -----
        h = self.norm1(inp)
        h = self.act1(h)
        h = self.conv1(h)

        # Inject time embedding
        time_out = self.time_proj(time_emb)
        h = h + time_out[:, :, None, None]

        # ----- Block 2 -----
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # ----- Residual connection -----
        #h = h + self.shortcut(inp)
        # ----- Residual connection -----
        h = (h + self.shortcut(inp)) / math.sqrt(2)

        # ----- Attention -----
        h = self.attn(h)

        return h

# Time embedding using sinusoidal features.
class TimeEmbed(nn.Module):
    def __init__(self, hidden_dim, freq=256, max_period=10000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freq = freq
        self.max_period = max_period
        # MLP: Linear → SiLU → Linear
        self.mlp = nn.Sequential(
            nn.Linear(freq, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Inicialización Xavier
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def sin_emb(time_input, dim, max_period=10000):
        """
        Genera embeddings sinusoidales para el timestep.
        Args:
            time_input: [B] tensor de timesteps
            dim: dimensión del embedding (número de frecuencias * 2)
        Returns:
            embedding [B, dim]
        """
        device = time_input.device
        half_dim = dim // 2
        # Exponentes para las frecuencias
        exponents = torch.arange(half_dim, device=device).float() * (-math.log(max_period) / half_dim)
        freqs = torch.exp(exponents)  # [half_dim]
        # Reshape time_input a [B,1] para multiplicar
        args = time_input[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return emb

    def forward(self, time_input):
        """
        Args:
            time_input: [B] tensor de timesteps
        Returns:
            embedding: [B, hidden_dim]
        """
        # Crear embedding sinusoidal
        sin_emb = self.sin_emb(time_input, self.freq, self.max_period)
        # Pasarlo por MLP
        return self.mlp(sin_emb)

# U-Net model with classifier-free guidance.
class UNet(nn.Module):
    def __init__(
            self, 
            num_timesteps=1000, 
            image_size=64, 
            base_channels=128, 
            channel_mults=[1, 2, 2, 2],
            attn_levels=[1], 
            num_res_blocks=4, 
            dropout=0.1, 
            use_cfg=False, 
            cfg_dropout=0.1, 
            num_classes=None,
            device=None
            ):
        super().__init__()
        time_dim = base_channels * 4
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.time_embed = TimeEmbed(time_dim).to(self.device)
        self.image_size = image_size
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        if use_cfg:
            assert num_classes is not None, "num_cls must be provided for CFG"
            self.cls_emb = nn.Embedding(num_classes + 1, time_dim)
        self.head = nn.Conv2d(3, base_channels, 3, padding=1)
        self.down_blocks = nn.ModuleList()
        skip_channels = [base_channels]
        current_channels = base_channels
        for i, multiplier in enumerate(channel_mults):
            out_channels = base_channels * multiplier
            for _ in range(num_res_blocks):
                self.down_blocks.append(Res(current_channels, out_channels, time_dim, dropout, use_attn=(i in attn_levels)).to(self.device))
                current_channels = out_channels
                skip_channels.append(current_channels)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Down(current_channels).to(self.device))
                skip_channels.append(current_channels)
        self.mid_blocks = nn.ModuleList([Res(current_channels, current_channels, time_dim, dropout, True),
                                         Res(current_channels, current_channels, time_dim, dropout)])
        self.up_blocks = nn.ModuleList()
        for i, multiplier in reversed(list(enumerate(channel_mults))):
            out_channels = base_channels * multiplier
            for _ in range(num_res_blocks + 1):
                self.up_blocks.append(Res(skip_channels.pop() + current_channels, out_channels, time_dim, dropout, use_attn=(i in attn_levels)).to(self.device))
                current_channels = out_channels
            if i:
                self.up_blocks.append(Up(current_channels).to(self.device))
        self.tail = nn.Sequential(
            nn.GroupNorm(32, current_channels), nn.SiLU(), 
            nn.Conv2d(current_channels, 3, 3, padding=1).to(self.device)
            )
        init_w(self.head)
        init_w(self.tail[-1], gain=1e-5)
    def _forward(self, inp, time_emb):
        h = self.head(inp)
        skip_connections = [h]
        for block in self.down_blocks:
            h = block(h, time_emb)
            skip_connections.append(h)
        for block in self.mid_blocks:
            h = block(h, time_emb)
        for block in self.up_blocks:
            if isinstance(block, Res):
                h = torch.cat([h, skip_connections.pop()], dim=1)
            h = block(h, time_emb)
        return self.tail(h)
    def forward(self, inp, time_step, cls=None, scale=1.0):
        inp, time_step = inp.to(self.device), time_step.to(self.device)
        base_time_emb = self.time_embed(time_step)
        if self.use_cfg:
            null_class = self.cls_emb.num_embeddings - 1
            if self.training and cls is not None:
                drop_mask = torch.rand(inp.size(0), device=inp.device) < self.cfg_dropout
                cls = torch.where(drop_mask, torch.full_like(cls, null_class), cls)
            if cls is None:
                cond_emb = self.cls_emb(torch.full((inp.size(0),), null_class, device=inp.device))
                time_emb = base_time_emb + cond_emb
            else:
                if scale != 1.0:
                    uncond_emb = self.cls_emb(torch.full((inp.size(0),), null_class, device=inp.device))
                    cond_emb = self.cls_emb(cls)
                    time_uncond = base_time_emb + uncond_emb
                    time_cond = base_time_emb + cond_emb
                    return self._forward(inp, time_uncond) + scale * (self._forward(inp, time_cond) - self._forward(inp, time_uncond))
                else:
                    cond_emb = self.cls_emb(cls)
                    time_emb = base_time_emb + cond_emb
        else:
            time_emb = base_time_emb
        return self._forward(inp, time_emb)
