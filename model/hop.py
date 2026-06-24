import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

##########################################################################
## Utility Functions
##########################################################################

def pad_to_multiple(x, multiple=16, mode='reflect', value=0):
    b, c, h, w = x.shape
    h_pad = (multiple - h % multiple) % multiple
    w_pad = (multiple - w % multiple) % multiple
    if h_pad == 0 and w_pad == 0:
        return x, (0, 0)
    padding = [0, w_pad, 0, h_pad]
    x_padded = F.pad(x, padding, mode=mode, value=value) if mode != 'constant' else F.pad(x, padding, mode=mode, value=value)
    return x_padded, (h_pad, w_pad)

def crop_to_original(x, pad_info, original_size):
    h_pad, w_pad = pad_info
    h_ori, w_ori = original_size
    if h_pad == 0 and w_pad == 0:
        return x
    return x[:, :, :h_ori, :w_ori]

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super(LayerNorm, self).__init__()
        self.body = nn.LayerNorm(dim, eps=1e-5) if LayerNorm_type == 'WithBias' else \
                    nn.LayerNorm(dim, eps=1e-5, elementwise_affine=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        return rearrange(self.body(rearrange(x, 'b c h w -> b (h w) c')), 'b (h w) c -> b c h w', h=h, w=w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # 1. Project to Tangent Space (Expansion)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        # 2. Tangent Space Operations
        # We split the space into 'Direction' (v) and 'Metric' (g)
        # Standard GDFN uses a simple depth-wise conv here.
        # We add a fixed mathematical prior: Laplacian Smoothing.
        
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        # Mathematical Prior: Discrete Laplacian Kernel for smoothing
        # [ 0, -1,  0]
        # [-1,  4, -1]
        # [ 0, -1,  0]
        # This acts as a high-pass filter. Adding X - Laplacian(X) acts as a sharpener/feature enhancer
        # purely based on neighboring geometry.
        self.laplacian_branch = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, padding=1, groups=hidden_features * 2, bias=False)
        
        # Initialize as fixed Laplacian (non-learnable to save parameters, or learnable for flexibility)
        # Here we make it learnable but initialized to Laplacian math.
        self._init_laplacian()

        # Learnable mixing weight between learned dwconv and math prior
        self.mix_alpha = nn.Parameter(torch.zeros(1, hidden_features * 2, 1, 1))

        # 3. Retraction (Projection back to manifold)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def _init_laplacian(self):
        # Initialize weights to a negative Laplacian kernel (edge detector)
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
        kernel = kernel.reshape(1, 1, 3, 3).repeat(self.laplacian_branch.out_channels, 1, 1, 1)
        self.laplacian_branch.weight.data = kernel
        # Optional: Lock gradient if you want purely fixed math prior (saving computation)
        # self.laplacian_branch.weight.requires_grad = False 

    def forward(self, x):
        # 1. Expand to high-dim tangent space
        x = self.project_in(x)
        
        # 2. Apply Geometry-aware processing
        # Learned local features
        x_learned = self.dwconv(x)
        
        # Geometric curvature (Edges/Noise) via Laplacian
        x_geom = self.laplacian_branch(x)
        
        # Combine: We modulate the learned features with geometric edge info
        # The equation: h = f(x) + alpha * Laplacian(x)
        # This resembles a diffusion process step or a PDE regularization.
        x_processed = x_learned + self.mix_alpha * x_geom
        
        # 3. Gating (Non-linearity)
        x1, x2 = x_processed.chunk(2, dim=1)
        
        # GeLU is the approximation of the area under Gaussian curve
        x = F.gelu(x1) * x2
        
        # 4. Project back
        x = self.project_out(x)
        return x

## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

##########################################################################
## Basic Self-Attention Block
##########################################################################
class HouseholderRejector(nn.Module):

    def __init__(self, dim, bias):
        super(HouseholderRejector, self).__init__()
        
        # [Lightweight 1] Use the difference map instead of concat (2*dim -> dim input channels)
        # [Lightweight 2] Depthwise separable conv (DW-Conv) to capture local ghost patterns
        self.v_gen = nn.Sequential(
            # DW Conv: spatial context with minimal parameters (9 * dim)
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=bias),
            nn.GELU(),
            # PW Conv: channel mixing (dim * dim params)
            # ~9x fewer params than a standard 3x3 conv (9 * dim * dim)
            nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        )
        
        # Learnable scaling factor, initialized to zero
        self.alpha = nn.Parameter(torch.zeros(1, dim, 1, 1)) 

    def forward(self, ref, aux):
        # 1. Difference prior: ghosting manifests as misalignment between ref and aux
        diff = ref - aux 
        
        # 2. Learn projection direction v (ghost direction and strength)
        v = self.v_gen(diff) 
        
        # 3. Normalize to a unit direction vector (orthogonalization)
        v_norm = F.normalize(v, dim=1, eps=1e-6)
        # 4. Project aux onto the ghost direction
        # dot_prod shape: [B, 1, H, W] (sum over channels)
        dot_prod = torch.sum(v_norm * aux, dim=1, keepdim=True)
        ghost_component = v_norm * dot_prod
        
        # 5. Gating and rejection: scale by diff magnitude to avoid amplifying noise
        # When diff is small, magnitude ~ 0 and rejection is suppressed
        magnitude = torch.tanh(torch.mean(torch.abs(diff), dim=1, keepdim=True))
        
        out = aux - (self.alpha * magnitude * ghost_component)
        
        return out




class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(SelfAttentionBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # Using the same CrossAttn structure for Self-Attn (y=x) for simplicity & power
        self.attn = Attention(dim, num_heads, bias) # We can reuse HCA with x=y
        self.rejector = HouseholderRejector(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # Self-Attention: Ref queries itself
        x = x + self.rejector(self.norm1(x), self.attn(self.norm1(x))) 
        x = x + self.ffn(self.norm2(x))
        return x

class IlluminationAlignment(nn.Module):
    def __init__(self, dim=3): # Modified default dim to 3 for RGB inputs
        super(IlluminationAlignment, self).__init__()
        # Downsample to capture global/low-frequency illumination
        # Use Average Pooling to extract brightness statistics
        self.illum_pool = nn.AvgPool2d(kernel_size=16, stride=16)
        
        # Simple transform to learn the illumination mapping
        self.map_transform = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid() # Illumination is strictly positive
        )
        
        # Light-weight refinement after alignment
        self.refine = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, ref, aux):
        """
        ref: Reference features (Target brightness)
        aux: Short/Long exposure features (Source brightness)
        """
        b, c, h, w = ref.shape
        
        # 1. Extract Low-Frequency Illumination Maps
        low_ref = self.illum_pool(ref)
        low_aux = self.illum_pool(aux)
        
        map_ref = self.map_transform(low_ref)
        map_aux = self.map_transform(low_aux)
        
        # Resize back to (H, W) efficiently
        map_ref = F.interpolate(map_ref, size=(h, w), mode='bilinear', align_corners=False)
        map_aux = F.interpolate(map_aux, size=(h, w), mode='bilinear', align_corners=False)

        # 2. Compute Illumination Ratio
        # We want Aux to have similar brightness to Ref
        # ratio ~ Ref_brightness / Aux_brightness
        # Add epsilon to prevent division by zero
        ratio = (map_ref + 1e-4) / (map_aux + 1e-4)
        
        # 3. Apply Alignment (Broadcast multiplication)
        aux_aligned = aux * ratio
        
        # 4. Refine slightly to smooth out artifacts
        return self.refine(aux_aligned)

##########################################################################
## Main Model: Orthogonal Householder HDR (Single Stream Variant)
##########################################################################

class HouseholderHDR(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim=48,
        encoder_depth=2,
        num_blocks=[2, 4, 4], # [Stage1, Stage2, Stage3, Latent]
        heads=[1, 2, 4],
        ffn_expansion_factor=2.0,
        bias=False
    ):
        super(HouseholderHDR, self).__init__()
        self.pad_multiple = 2 ** encoder_depth
        self.iam_s = IlluminationAlignment(dim=3)
        self.iam_l = IlluminationAlignment(dim=3)
        
        # Embedding: 3 frames * 3 channels = 9 channels -> dim
        self.embed = nn.Conv2d(9, dim, 3, 1, 1, bias=bias)

        self.enc_stages = nn.ModuleList()
        self.downs = nn.ModuleList()

        # --- Encoder Levels ---
        for i in range(encoder_depth):
            c_dim = int(dim * (2 ** i))
            c_head = heads[i]
            
            # Feature Processing (Self-Attn)
            # Single stream encoder
            self.enc_stages.append(nn.Sequential(*[
                SelfAttentionBlock(c_dim, c_head, ffn_expansion_factor, bias, 'WithBias')
                for _ in range(num_blocks[i])
            ]))
            
            # Downsampling
            self.downs.append(nn.Sequential(nn.Conv2d(c_dim, c_dim//2, 3, 1, 1, bias=False), nn.PixelUnshuffle(2)))

        # --- Latent ---
        l_dim = int(dim * (2 ** encoder_depth))
        self.latent = nn.Sequential(*[
            SelfAttentionBlock(l_dim, heads[-1], ffn_expansion_factor, bias, 'WithBias')
            for _ in range(num_blocks[-1])
        ])

        # --- Decoder ---
        self.dec_stages = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.reduces = nn.ModuleList()
        
        for i in range(encoder_depth - 1, -1, -1):
            c_dim = int(dim * (2 ** i))
            c_head = heads[i]
            deep_dim = int(dim * (2 ** (i + 1)))
            
            self.ups.append(nn.Sequential(nn.Conv2d(deep_dim, deep_dim*2, 3, 1, 1, bias=False), nn.PixelShuffle(2)))
            self.reduces.append(nn.Conv2d(deep_dim, c_dim, 1, bias=bias))
            
            self.dec_stages.append(nn.Sequential(*[
                SelfAttentionBlock(c_dim, c_head, ffn_expansion_factor, bias, 'WithBias')
                for _ in range(num_blocks[i])
            ]))

        self.refinement = nn.Sequential(
             SelfAttentionBlock(dim, heads[0], ffn_expansion_factor, bias, 'WithBias'),
             SelfAttentionBlock(dim, heads[0], ffn_expansion_factor, bias, 'WithBias')
        )
        
        self.tail = nn.Conv2d(dim, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp_img):
        # Input: [B, 3 (frames), C, H, W] or [B, 9, H, W]
        if inp_img.dim() == 5:
            # Assuming [B, T, C, H, W] where T=0:Short, T=1:Mid(Ref), T=2:Long
            x_s = inp_img[:, 0]
            x_m = inp_img[:, 1]
            x_l = inp_img[:, 2]
        else:
            x_s = inp_img[:, 0:3]
            x_m = inp_img[:, 3:6]
            x_l = inp_img[:, 6:9]

        h_ori, w_ori = x_m.shape[-2:]
        x_m, pad_info = pad_to_multiple(x_m, self.pad_multiple)
        x_s, _ = pad_to_multiple(x_s, self.pad_multiple)
        x_l, _ = pad_to_multiple(x_l, self.pad_multiple)
        
        x_ref_res = x_m

        x_s_aligned = self.iam_s(ref=x_m, aux=x_s)
        x_l_aligned = self.iam_l(ref=x_m, aux=x_l)
        
        
        # 2. Concat & Embed (Single Stream Start)
        # Concatenate aligned frames: [B, 9, H, W]
        x_concat = torch.cat((x_s_aligned, x_m, x_l_aligned), dim=1)
        
        # Initial Features [B, dim, H, W]
        x = self.embed(x_concat)
        
        skips = []
        
        # 3. Encoder Loop (Single Stream)
        for i in range(len(self.enc_stages)):
            x = self.enc_stages[i](x)
            skips.append(x)
            x = self.downs[i](x)
            
        # 4. Latent
        x = self.latent(x)
        
        # 5. Decoder
        for i in range(len(self.dec_stages)):
            skip_idx = len(skips) - 1 - i
            
            x = self.ups[i](x)
            x = torch.cat([x, skips[skip_idx]], dim=1)
            x = self.reduces[i](x)
            x = self.dec_stages[i](x)
            
        x = self.refinement(x)
        out = self.tail(x) + x_ref_res # Residual Learning
        
        out = crop_to_original(out, pad_info, (h_ori, w_ori))
        return torch.sigmoid(out) # Output is usually 0-1 range

if __name__ == '__main__':
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HouseholderHDR(
    ).to(device)
    
    # Dummy Input: Batch=2, 3 Frames (S, M, L), 3 Channels, 256x256
    dummy = torch.randn(2, 3, 3, 256, 256).to(device) 
    
    print("Testing HouseholderHDR_SingleStream Forward...")
    with torch.no_grad():
        out = model(dummy)
    print(f"Output shape: {out.shape}")
    
    try:
        from thop import profile
        # Profile using a smaller input to save time
        dummy_small = torch.randn(1, 3, 3, 128, 128).to(device)
        flops, params = profile(model, inputs=(dummy_small,), verbose=False)
        print(f"Params: {params/1e6:.4f}M")
        print(f"GFLOPs (128x128): {flops/1e9:.4f}")
    except ImportError:
        print("Install 'thop' to see FLOPs/Params stats.")
