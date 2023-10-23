import torch
import pytest
from transformer_nuggets.flash import BiasMode, build_alibi_mask, latest_attention, build_causal_attention_mask, build_causal_mask
import time

@pytest.mark.parametrize("Z, H, N_CTX, D_HEAD", [(1, 1, 256, 32)])
#@pytest.mark.parametrize("",)
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("bias_choice", [BiasMode.alibi,]) #  BiasMode.alibi])
@pytest.mark.parametrize("sm_scale", [1,]) # 1])

def test_op(Z, H, N_CTX, D_HEAD, causal, bias_choice, sm_scale, dtype=torch.float16):
    torch.manual_seed(2020)
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

    # ref impl
    # reference implementation
    if bias_choice == BiasMode.rel_pos:
        attn_bias = build_alibi_mask(N_CTX, N_CTX, H, scale=1, causal=causal)
        attn_bias = attn_bias.expand(Z, H, N_CTX, N_CTX).to(q.device).to(q.dtype)
    elif bias_choice == BiasMode.alibi:
        attn_bias = build_alibi_mask(N_CTX, N_CTX, H, scale=None, causal=causal)
        #print(f"ref attn bias {attn_bias=}")
        attn_bias = attn_bias.expand( Z, H, N_CTX, N_CTX).to(q.device).to(q.dtype)
        saved_bias = attn_bias.clone()
        print(f"full ref attn bias {attn_bias=}")
    elif bias_choice == BiasMode.none:
        attn_bias = None
    is_causal = causal if (bias_choice == BiasMode.none) else False

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False):
        ref_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, scale=sm_scale, is_causal=is_causal, attn_mask=attn_bias
        )
    
    # triton impl
    # q, k, v, causal, sm_scale, seq_parallel
    tri_out, tri_mask = latest_attention(q, k, v, causal, sm_scale, bias_choice, True) #  bias_choice, True)
    #tri_out.half()
    # mask_tri = tri_mask.to(torch.float16)
    #print(f"{tri_mask=}")

    #torch.testing.assert_close(ref_out, tri_out, atol=2.5e-1, rtol=0)
    #torch.testing.assert_close(mask, attn_bias)
    #for row in tri_mask[0][0]:
    #    print(f"{row=}\n")
        
    print(f"{tri_mask.shape=}")
    print(f"{attn_bias.shape=}")

    print("N_CTX mask")
    print(f"{attn_bias[0][0][N_CTX-1]=}")
    print(f"{tri_mask[0][0][N_CTX-1]=}")
    print(f" 003 mask")
    print(f"{attn_bias[0][0][3]=}")
    print(f"{tri_mask[0][0][3]=}")
    #print("32 mask")
    #print(f"{attn_bias[0][0][32]=}")
    #print(f"{tri_mask[0][0][32]=}")
    

'''
@pytest.mark.parametrize("N_CTX", [128, 256, 1024, 2048, 4096, 8192, 16384])
def test_mask(N_CTX, ):
    torch.manual_seed(2020)

    print(f"{N_CTX=}")
    start =  time.perf_counter()
    driss_ref = build_causal_mask(N_CTX, N_CTX)
    stop = time.perf_counter()
    dtime = round(stop-start, 5)
    start =  time.perf_counter()
    less_ref = build_causal_attention_mask(N_CTX, N_CTX)
    stop = time.perf_counter()
    ltime = round(stop-start, 5)
    print(f"{dtime=}, {ltime=}")

    torch.testing.assert_close(driss_ref, less_ref, atol=5.5e-2, rtol=0)
'''



