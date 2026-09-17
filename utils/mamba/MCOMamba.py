import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numbers
from torch import Tensor

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

from .selective_scan_interface_mco import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, \
        mamba_inner_fn_no_out_proj

from mamba_ssm.ops.triton.selective_state_update import selective_state_update


try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=4,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="v2",
            if_devide_out=False,
            init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.in_proj_rgb = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.in_proj_event = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.norm_rgb = nn.LayerNorm(self.d_inner)
        self.norm_event = nn.LayerNorm(self.d_inner)
        self.conv1d_rgb = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_event = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act_rgb = nn.SiLU()
        self.act_event = nn.SiLU()
        self.x_proj_rgb = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.x_proj_event = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.x_proj_peep = nn.Linear(
            (self.dt_rank + self.d_state * 2) * 2, 16, bias=False, **factory_kwargs
        )
        self.x_proj_b_peep = nn.Linear(
            (self.dt_rank + self.d_state * 2) * 2, 16, bias=False, **factory_kwargs
        )
        self.x_proj_Cpeep = nn.Linear(
            (self.dt_rank + self.d_state * 2) * 2, 16, bias=False, **factory_kwargs
        )
        self.x_proj_b_Cpeep = nn.Linear(
            (self.dt_rank + self.d_state * 2) * 2, 16, bias=False, **factory_kwargs
        )
        self.dt_proj_rgb = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_event = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj_rgb.weight, dt_init_std)
            nn.init.constant_(self.dt_proj_event.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj_rgb.weight, -dt_init_std, dt_init_std)
            nn.init.uniform_(self.dt_proj_event.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj_rgb.bias.copy_(inv_dt)
            self.dt_proj_event.bias.copy_(inv_dt)
        self.dt_proj_rgb.bias._no_reinit = True
        self.dt_proj_event.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log_rgb = nn.Parameter(A_log)
        self.A_log_event = nn.Parameter(A_log)
        self.A_log_rgb._no_weight_decay = True
        self.A_log_event._no_weight_decay = True
        # D "skip" parameter
        self.D_rgb = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_event = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b_rgb = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b_event = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_rgb._no_weight_decay = True
        self.D_event._no_weight_decay = True
        self.D_b_rgb._no_weight_decay = True
        self.D_b_event._no_weight_decay = True

        # bidirectional
        if bimamba_type == "v1":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True
        elif bimamba_type == "v2" or bimamba_type == 'v3':
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log_rgb = nn.Parameter(A_b_log)
            self.A_b_log_rgb._no_weight_decay = True
            self.A_b_log_event = nn.Parameter(A_b_log)
            self.A_b_log_event._no_weight_decay = True

            self.conv1d_b_rgb = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )
            self.conv1d_b_event = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b_rgb = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.x_proj_b_event = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b_rgb = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
            self.dt_proj_b_event = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b_rgb = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b_rgb._no_weight_decay = True
            self.D_b_event = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b_event._no_weight_decay = True
            self.in_proj_extra = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.out_proj_rgb = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj_event = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.fusion_gate = nn.Sequential(
            nn.Linear(2 * d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, hidden_states_rgb, hidden_states_event, inference_params=None, extra_emb=None):
        """
        hidden_states_rgb: (B, L, D)
        Returns: same shape as hidden_states_rgb
        """
        batch_rgb, seqlen_rgb, dim_rgb = hidden_states_rgb.shape
        batch_event, seqlen_event, dim_event = hidden_states_rgb.shape

        xz_rgb = rearrange(
            self.in_proj_rgb.weight @ rearrange(hidden_states_rgb, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen_rgb,
        )

        xz_event = rearrange(
            self.in_proj_event.weight @ rearrange(hidden_states_event, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen_event,
        )
        A_rgb = -torch.exp(self.A_log_rgb.float())
        A_event = -torch.exp(self.A_log_event.float())
        A_b_rgb = -torch.exp(self.A_b_log_rgb.float())
        A_b_event = -torch.exp(self.A_b_log_event.float())
        y_rgb, y_event = mamba_inner_fn_no_out_proj(
            xz_rgb,
            xz_event,
            self.x_proj_peep.weight,
            self.x_proj_Cpeep.weight,
            self.conv1d_rgb.weight,
            self.conv1d_event.weight,
            self.conv1d_rgb.bias,
            self.conv1d_event.bias,
            self.x_proj_rgb.weight,
            self.x_proj_event.weight,
            self.dt_proj_rgb.weight,
            self.dt_proj_event.weight,
            A_rgb,
            A_event,
            None,  # input-dependent B
            None,  # input-dependent B
            None,  # input-dependent C
            None,  # input-dependent C
            self.D_rgb.float(),
            self.D_event.float(),
            delta_bias_rgb=self.dt_proj_rgb.bias.float(),
            delta_bias_event=self.dt_proj_event.bias.float(),
            B_proj_bias_rgb=None,
            B_proj_bias_event=None,
            C_proj_bias_rgb=None,
            C_proj_bias_event=None,
            delta_softplus_rgb=True,
            delta_softplus_event=True,
        )
        y_b_rgb, y_b_event = mamba_inner_fn_no_out_proj(
            xz_rgb.flip([-1]),
            xz_event.flip([-1]),
            self.x_proj_b_peep.weight,
            self.x_proj_b_Cpeep.weight,
            self.conv1d_b_rgb.weight,
            self.conv1d_b_event.weight,
            self.conv1d_b_rgb.bias,
            self.conv1d_b_event.bias,
            self.x_proj_b_rgb.weight,
            self.x_proj_b_event.weight,
            self.dt_proj_b_rgb.weight,
            self.dt_proj_b_event.weight,
            A_b_rgb,
            A_b_event,
            None,
            None,
            None,
            None,
            self.D_b_rgb.float(),
            self.D_b_event.float(),
            self.dt_proj_rgb.bias.float(),
            self.dt_proj_event.bias.float(),
            None,
            None,
            None,
            None,
            True,
            True,
        )
        out_rgb = F.linear(rearrange(y_rgb + y_b_rgb.flip([-1]), "b d l -> b l d") / 2, self.out_proj_rgb.weight,
                           self.out_proj_rgb.bias)
        out_event = F.linear(rearrange(y_event + y_b_event.flip([-1]), "b d l -> b l d") / 2,
                             self.out_proj_event.weight, self.out_proj_event.bias)
        return out_rgb, out_event

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
            self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
            # self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
            self, hidden_states: Tensor, residual=None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)




def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_matrix_mamba = Mamba(dim)

    def forward(self, m1, m2):
        global_f, global_e = self.cross_matrix_mamba(m1, m2)

        return global_f, global_e