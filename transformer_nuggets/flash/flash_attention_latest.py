"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)

Sequence Parallel implementation inspired by HazyResearch
(see https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/flash_attn_triton.py)
"""

import torch

from triton import cdiv, jit
from triton import language as tl
import triton 
import enum 

def build_causal_mask(seq_len_q, seq_len_kv):
    temp_mask = (
        torch.ones((seq_len_q, seq_len_kv))
        .tril_()
        .bool()
    )
    mask = torch.zeros_like(temp_mask, dtype=torch.float32)
    mask.masked_fill_(temp_mask.logical_not(), float("-inf"))
    return mask

def build_causal_attention_mask(seq_len_q: int, seq_len_kv: int, num_heads: int=0) -> torch.Tensor:
        """builds a generic causal attention mask"""
        causal_mask = torch.triu(
            torch.ones((seq_len_q, seq_len_kv,), dtype=torch.float32) * float("-inf"), diagonal=1
        )
        #attn_mask = causal_mask.repeat(num_heads, 1, 1)
        return causal_mask


@triton.jit
def rel_pos_attention_triton(cur, m, n, head_num, num_heads):
    bias = n - m
    cur = cur + bias
    return cur

class BiasMode(enum.Enum):
    none = 0
    rel_pos = 1
    alibi = 2

@triton.jit
def alibi_attention_triton(cur, m, n, head_num: int, num_heads: int):
    alibi_scale = 1 # tl.math.exp2(-((head_num + 1) * 8.0 / num_heads))
    #tl.device_print("alibi scale ", alibi_scale)
    bias = n-m
    tl.device_print("m, n, bias ", m, n, bias)
    #breakpoint()
    #tl.device_print("m, n, head_num ",m, n, head_num)
    alibi_mask = alibi_scale * bias
    #breakpoint()
    #tl.device_print("alibi mask ", alibi_mask)
    #tl.device_print("m ", m)
    #tl.device_print("alibi_heads ", alibi_scale)
    cur = cur + alibi_mask
    #breakpoint()
    return cur


@jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    L,
    Out, mask_scratch_space, 
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    Z_H_N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BIAS_CHOICE: tl.constexpr, 
    DEBUG_MASK: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    qvk_offset = off_hz * stride_qh
    vk_offset = qvk_offset // stride_qm

    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(BLOCK_DMODEL, Z_H_N_CTX),
        strides=(stride_kk, stride_kn),
        offsets=(0, vk_offset),
        block_shape=(BLOCK_DMODEL, BLOCK_N),
        order=(0, 1)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(vk_offset, 0),
        block_shape=(BLOCK_N, BLOCK_DMODEL),
        order=(1, 0)
    )

    if DEBUG_MASK and BIAS_CHOICE != BiasMode.none:
        mask_block_ptr = tl.make_block_ptr(
            base = mask_scratch_space + off_hz * N_CTX *N_CTX,
            shape = (N_CTX, N_CTX),
            strides = (N_CTX,1),
            offsets=(start_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_N),
            order = (1,0),
        )
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # credits to: Adam P. Goucher (https://github.com/apgoucher):
    # scale sm_scale by 1/log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout

    offs_k = tl.arange(0, BLOCK_DMODEL)
    Q_ptrs = Q + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    q = tl.load(Q_ptrs)

    q = (q * qk_scale).to(K.dtype.element_ty)
    lo = 0
    hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX
    for start_n in range(lo, hi, BLOCK_N):
        # -- load k, v --
        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)
        # -- compute qk ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        tl.device_print("block_m, block_n ", BLOCK_M, BLOCK_N)
        if IS_CAUSAL:
            tl.device_print("qk causal ", qk)
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, -1.0e6)
            tl.device_print("qk after where ", qk)
        qk += tl.dot(q, k, allow_tf32=True)
        # ~~~~~~~~~~~~~ Mask work ~~~~~~~~~~~~~~ 
        if BIAS_CHOICE == BiasMode.alibi:
            qk = alibi_attention_triton(qk, offs_m[:,None], (start_n + offs_n[None,:]),
                                        off_hz % H, H)
        elif BIAS_CHOICE == BiasMode.rel_pos:
            qk = rel_pos_attention_triton(qk, offs_m[:,None], (start_n + offs_n[None,:]),
                                          off_hz % H, H)
        # ---- debug mask - remove for prod
        if DEBUG_MASK and BIAS_CHOICE != BiasMode.none:
            mask = qk - tl.dot(q,k)
            if IS_CAUSAL:
                mask = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), mask, -1.0e6)                                    
            tl.store(mask_block_ptr, mask)
        # ~~~~~~~~~~  End Mask work ~~~~~~~~~~~~
        
        # -- compute scaling constant ---
        m_i_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.math.exp2(m_i - m_i_new)
        p = tl.math.exp2(qk - m_i_new[:, None])
        # -- scale and update acc --
        acc *= alpha[:, None]
        acc += tl.dot(p.to(V.dtype.element_ty), v, allow_tf32=True)
        # -- update m_i and l_i --
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_i_new
        # update pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
    # write back l and m
    acc = acc / l_i[:, None]
    l_ptrs = L + off_hz * N_CTX + offs_m
    tl.store(l_ptrs, m_i + tl.math.log2(l_i))
    # write back O
    O_block_ptr = tl.make_block_ptr(
        base=Out,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_om, stride_on),
        offsets=(vk_offset + start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    # O_ptrs = Out + qvk_offset + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    tl.store(O_block_ptr, acc.to(K.dtype.element_ty))


@jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@jit
def _bwd_kernel_one_col_block(
    Q, K, V, sm_scale, qk_scale,
    Out, DO,
    DQ, DK, DV,
    L,
    D,
    Q_block_ptr, K_block_ptr, V_block_ptr,
    DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,
    stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    Z, H, N_CTX,
    off_hz, start_n, num_block,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    MMA_V3: tl.constexpr
):
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0

    Q_block_ptr = tl.advance(Q_block_ptr, (lo, 0))
    K_block_ptr = tl.advance(K_block_ptr, (start_n * BLOCK_M, 0))
    V_block_ptr = tl.advance(V_block_ptr, (start_n * BLOCK_M, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo, 0))
    DQ_block_ptr = tl.advance(DQ_block_ptr, (lo, 0))
    DK_block_ptr = tl.advance(DK_block_ptr, (start_n * BLOCK_M, 0))
    DV_block_ptr = tl.advance(DV_block_ptr, (start_n * BLOCK_M, 0))

    # initialize row/col offsets
    offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_N)
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    # loop over rows
    for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        q = tl.load(Q_block_ptr)
        # recompute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        if CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), float(0.), float("-inf"))
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dv
        do = tl.load(DO_block_ptr)
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do, allow_tf32=True)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)
        # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp = tl.dot(do, tl.trans(v), allow_tf32=True)
        # compute ds = p * (dp - delta[:, None])
        ds = (p * (dp - Di[:, None]) * sm_scale).to(Q.dtype.element_ty)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q, allow_tf32=True)
        # compute dq
        if not SEQUENCE_PARALLEL:
            dq = tl.load(DQ_block_ptr)
            dq += tl.dot(ds, k, allow_tf32=True)
            tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))
        elif SEQUENCE_PARALLEL:
            if MMA_V3:
                dq = tl.dot(ds, k, allow_tf32=True)
            else:
                # not work with mma v3, becuase M % 64 != 0
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds), allow_tf32=True))
            tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))

        # increment pointers
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
    # write-back
    tl.store(DV_block_ptr, dv.to(V.dtype.element_ty))
    tl.store(DK_block_ptr, dk.to(K.dtype.element_ty))


@jit
def _bwd_kernel(
    # fmt: off
    Q, K, V, sm_scale,
    Out, DO,
    DQ, DK, DV,
    L,
    D,
    stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    Z, H, N_CTX,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    MMA_V3: tl.constexpr
    # fmt: on
):
    qk_scale = sm_scale * 1.44269504
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_qz + off_h * stride_qh
    DQ += off_z * stride_qz + off_h * stride_qh
    DK += off_z * stride_kz + off_h * stride_kh
    DV += off_z * stride_vz + off_h * stride_vh

    if SEQUENCE_PARALLEL:
        DQ += stride_dqa.to(tl.int64) * tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DQ_block_ptr = tl.make_block_ptr(
        base=DQ,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DK_block_ptr = tl.make_block_ptr(
        base=DK,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0)
    )

    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    if not SEQUENCE_PARALLEL:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                Q, K, V, sm_scale, qk_scale, Out, DO,
                DQ, DK, DV,
                L,
                D,
                Q_block_ptr, K_block_ptr, V_block_ptr,
                DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,
                stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,
                stride_kz, stride_kh, stride_kn, stride_kk,
                stride_vz, stride_vh, stride_vn, stride_vk,
                Z, H, N_CTX,
                off_hz, start_n, num_block_n,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                MMA_V3=MMA_V3
            )
    else:
        start_n = tl.program_id(1)
        _bwd_kernel_one_col_block(
            Q, K, V, sm_scale, qk_scale, Out, DO,
            DQ, DK, DV,
            L,
            D,
            Q_block_ptr, K_block_ptr, V_block_ptr,
            DO_block_ptr, DQ_block_ptr, DK_block_ptr, DV_block_ptr,
            stride_dqa, stride_qz, stride_qh, stride_qm, stride_qk,
            stride_kz, stride_kh, stride_kn, stride_kk,
            stride_vz, stride_vh, stride_vn, stride_vk,
            Z, H, N_CTX,
            off_hz, start_n, num_block_n,
            BLOCK_M=BLOCK_M, BLOCK_DMODEL=BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            MMA_V3=MMA_V3
        )


class _attention(torch.autograd.Function):

    @staticmethod
    
    def forward(ctx, q, k, v, causal, sm_scale, bias_choice: BiasMode = BiasMode.none, debug_mask: bool = False, sequence_parallel=False):
        # only support for Ampere now
        capability = torch.cuda.get_device_capability()
        if capability[0] < 8:
            raise RuntimeError("Flash attention currently only supported for compute capability >= 80")
        
        batch_size, num_heads, seq_len_qv, d_head = q.shape
        BLOCK_M = 128
        BLOCK_N = 64
        # shape constraints
        Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
        assert Lq == Lk and Lk == Lv
        assert Lk in {16, 32, 64, 128}
        o = torch.empty_like(q)
        grid = (cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
        L = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        scratch_space = None
        if debug_mask:
            scratch_space = torch.zeros((batch_size, num_heads, seq_len_qv, seq_len_qv), device=q.device, dtype=torch.float32)
        
        num_warps = 4 if Lk <= 64 else 8

        _fwd_kernel[grid](
            q, k, v, sm_scale,
            L,
            o,
            scratch_space,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            q.shape[0] * q.shape[1] * q.shape[2],
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=Lk,
            IS_CAUSAL=causal,
            BIAS_CHOICE=bias_choice.value,
            DEBUG_MASK=debug_mask,
            num_warps=num_warps,
            num_stages=4)

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = sm_scale
        ctx.BLOCK_DMODEL = Lk
        ctx.causal = causal
        ctx.sequence_parallel = sequence_parallel
        return o, scratch_space

    @staticmethod
    def backward(ctx, do):
        capability = torch.cuda.get_device_capability()
        MMA_V3 = capability[0] >= 9
        BLOCK = 128
        q, k, v, o, L = ctx.saved_tensors
        sequence_parallel = ctx.sequence_parallel
        seq_len_kv = k.shape[2]
        do = do.contiguous()
        if sequence_parallel:
            replicas = cdiv(seq_len_kv, BLOCK)
            new_dq_shape = (replicas,) + q.shape
            dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
        else:
            dq = torch.zeros_like(q, dtype=q.dtype)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
        _bwd_preprocess[(cdiv(q.shape[2], BLOCK) * ctx.grid[1], )](
            o, do,
            delta,
            BLOCK_M=BLOCK, D_HEAD=ctx.BLOCK_DMODEL,
        )
        _bwd_kernel[(ctx.grid[1], cdiv(seq_len_kv, BLOCK) if sequence_parallel else 1)](
            q, k, v, ctx.sm_scale,
            o, do,
            dq, dk, dv,
            L,
            delta,
            o.numel(), q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            q.shape[0], q.shape[1], q.shape[2],
            BLOCK_M=BLOCK, BLOCK_N=BLOCK,
            BLOCK_DMODEL=ctx.BLOCK_DMODEL,
            SEQUENCE_PARALLEL=sequence_parallel,
            CAUSAL=ctx.causal,
            MMA_V3=MMA_V3,
            num_warps=8,
            num_stages=1,
        )

        if len(dq.shape) == 5:
            dq = dq.sum(dim=0)
        return dq, dk, dv, None, None, None


latest_attention = _attention.apply