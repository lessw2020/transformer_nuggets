import torch
import pytest
from transformer_nuggets.flash import BiasMode, build_alibi_mask, latest_attention


@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(4, 8, 128, 32)])
#@pytest.mark.parametrize("",)
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("bias_choice", [BiasMode.none,]) #  BiasMode.alibi])
@pytest.mark.parametrize("sm_scale", [None, 1])

def test_op(Z, H, N_CTX, D_HEAD, causal, bias_choice, sm_scale, dtype=torch.float16):
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device="cuda")
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )

    if sm_scale is None:
        sm_scale = 1 / (D_HEAD**0.5)
    dout = torch.randn_like(q)

