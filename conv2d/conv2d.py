import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M':128, 'BLOCK_N':128, 'BLOCK_K':64, 'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M':128, 'BLOCK_N':64,  'BLOCK_K':64, 'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M':64,  'BLOCK_N':128, 'BLOCK_K':64, 'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
    ],
    key=['GEMM_M','GEMM_N','GEMM_K'],
)
@triton.jit
def _conv2d_fwd(
    out_ptr, in_ptr, w_ptr, bias_ptr,
    apply_bias: tl.constexpr, activation: tl.constexpr,   # 'relu' or 'none'
    N, C, H, W, K, P, Q, R, S,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # ---- program ids & tiling over [M,N] ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)             # [BM]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)             # [BN]

    # Decode offs_m -> (n,p,q) ONCE (hoisted)
    n  = offs_m // (P * Q)
    pq = offs_m %  (P * Q)
    p  = pq // Q
    q  = pq %  Q

    # Precompute the local K-tile pattern once
    k_idx = tl.arange(0, BLOCK_K)
    c_loc = k_idx // (R * S)
    rs    = k_idx %  (R * S)
    r_loc = rs // S
    s_loc = rs %  S

    # A tile spatial positions for this K-slice (independent of k_tile base)
    h_in_base = p[:, None] * stride_h + r_loc[None, :] * dil_h - pad_h
    w_in_base = q[:, None] * stride_w + s_loc[None, :] * dil_w - pad_w

    # Masks that don't change with k_tile
    a_mask_hw = (h_in_base >= 0) & (h_in_base < H) & (w_in_base >= 0) & (w_in_base < W)
    a_mask_nc = (n[:, None] < N) & (c_loc[None, :] < C)
    a_mask = a_mask_hw & a_mask_nc

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ------------- B (weights) block pointer (K-major) -------------
    # Parent tensor: w_ptr with shape [K, C, R, S]
    # We want a [BLOCK_K, BLOCK_N] view where:
    #   rows  traverse the linearized (c,r,s) dimension,
    #   cols  traverse k (=offs_n)
    # For the parent, strides (in elements) are:
    #   stride_k = C*R*S, stride_c = R*S, stride_r = S, stride_s = 1
    stride_k = C * R * S
    stride_c = R * S
    stride_r = S
    stride_s = 1

    # The base offset for the first column is k = offs_n[0]
    base_k = tl.max_contiguous(offs_n, 1)  # hint; first col is contiguous in K dimension
    base_k0 = base_k[0]

    # We build a descriptor whose block_shape = [BLOCK_K, BLOCK_N]
    # shape is the full logical shape [C*R*S, K] (rows, cols)
    # rows stride = 1 over (s,r,c); col stride = stride_k
    w_desc = tl.make_block_ptr(
        base=w_ptr,
        shape=(C * R * S, K),
        strides=(1, stride_k),
        offsets=(0, base_k0),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),  # column-major load for better coalescing on N
    )

    # Hints to help vectorize
    tl.multiple_of(stride_k, 16)
    tl.multiple_of(offs_n, 16)
    tl.max_contiguous(offs_n, BLOCK_N)

    # ----------------- K loop (software pipelined) -----------------
    # NOTE: Triton 3.4 will stage the block-ptr loads in LDS for tl.dot with num_stages>1.
    for k_tile in range(0, GEMM_K, BLOCK_K):
        # A: im2col'ed [BM,BK] slice for this k_tile
        c = c_loc + (k_tile // (R * S)) * 0  # c_loc already enumerates the local slice
        h_in = h_in_base
        w_in = w_in_base

        a_offs = (
            n[:, None] * (C * H * W) +
            c[None, :] * (H * W) +
            h_in * W + w_in
        ).to(tl.int32)
        x = tl.load(in_ptr + a_offs, mask=a_mask, other=0.0)  # [BM,BK], fp16

        # B: advance the descriptor’s row-window to match this k_tile, then load [BK,BN]
        w_desc_k = tl.advance(w_desc, (k_tile, 0))
        # Boundary check on columns (N tail) — rows are handled by K-loop bound
        w = tl.load(w_desc_k, boundary_check=(1,), padding_option='zero')  # [BK,BN], fp16

        # Dot — Triton will keep A/B tiles in LDS/registers and use MFMA
        acc += tl.dot(x, w)

    # Epilogue: bias + activation
    if apply_bias:
        b = tl.load(bias_ptr + offs_n, mask=offs_n < K, other=0.0).to(tl.float32)
        acc += b[None, :]

    if activation == 'relu':
        acc = tl.maximum(acc, 0.0)

    # Store to out [N,K,P,Q]
    out_offs = (
        n[:, None] * (K * P * Q) +
        offs_n[None, :] * (P * Q) +
        p[:, None] * Q + q[:, None]
    ).to(tl.int32)
    o_mask = (
        (n[:, None] < N) & (offs_n[None, :] < K) &
        (p[:, None] < P) & (q[:, None] < Q)
    )
    tl.store(out_ptr + out_offs, acc.to(tl.float16), mask=o_mask)


def conv2d_fwd(x, w, b, stride, padding, dilation, activation='relu'):
    assert x.is_cuda and w.is_cuda
    N, C, H, W = x.shape
    K, Cw, R, S = w.shape
    assert C == Cw
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    P = (H + 2*ph - dh*(R - 1) - 1) // sh + 1
    Q = (W + 2*pw - dw*(S - 1) - 1) // sw + 1

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S

    y = torch.empty((N, K, P, Q), dtype=torch.float16, device=x.device)
    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_M']) *
                         triton.cdiv(GEMM_N, META['BLOCK_N']),)
    apply_bias = b is not None
    b_ptr = b if apply_bias else torch.empty(1, dtype=x.dtype, device=x.device)

    _conv2d_fwd[grid](
        y, x, w, b_ptr,
        apply_bias, activation,
        N, C, H, W, K, P, Q, R, S,
        sh, sw, ph, pw, dh, dw,
        GEMM_M, GEMM_N, GEMM_K,
    )
    return y
