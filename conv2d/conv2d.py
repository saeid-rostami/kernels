import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# =========================
# Helpers
# =========================

def prepack_weights_crs_k(w: torch.Tensor, pad_multiple: int = 64):
    """
    Pack weights [K, C, R, S] -> [CRS_pad, K] with row-major CRS and column-major K.
    Padding makes the reduction dim GEMM_K a multiple of pad_multiple for nicer tiles.
    """
    assert w.is_cuda and w.dtype == torch.float16
    K, C, R, S = w.shape
    CRS = C * R * S
    CRS_pad = ((CRS + pad_multiple - 1) // pad_multiple) * pad_multiple

    # [K, C*R*S] -> [C*R*S, K]
    wp = w.reshape(K, CRS).transpose(0, 1).contiguous()
    if CRS_pad != CRS:
        pad = torch.zeros((CRS_pad - CRS, K), dtype=w.dtype, device=w.device)
        wp = torch.cat([wp, pad], dim=0)  # [CRS_pad, K]
    return wp, CRS_pad


# =========================
# Triton kernel (toggle: packed weights, NHWC input)
# =========================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M':128, 'BLOCK_N':128, 'BLOCK_K':64,  'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M':128, 'BLOCK_N': 64, 'BLOCK_K':64,  'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N':128, 'BLOCK_K':64,  'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K':32,  'GROUP_SIZE_M':8}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M':128, 'BLOCK_N':128, 'BLOCK_K':128, 'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
    ],
    key=['GEMM_M', 'GEMM_N', 'GEMM_K'],
)
@triton.jit
def _conv2d_fwd(
    out_ptr, in_ptr, w_ptr, bias_ptr,
    apply_bias: tl.constexpr, activation: tl.constexpr,   # 'relu' or 'none'
    packed: tl.constexpr, use_nhwc: tl.constexpr,         # toggles
    N, C, H, W, K, P, Q, R, S,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,                              # M = N*P*Q, N = K, K = CRS or CRS_pad
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # MFMA-friendly shapes
    tl.static_assert(BLOCK_M % 16 == 0)
    tl.static_assert(BLOCK_N % 16 == 0)
    tl.static_assert(BLOCK_K % 16 == 0)

    # ---- tile ids over [M, N] ----
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows in GEMM (output pixels)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols in GEMM (output channels)

    # Decode rows -> (n, p, q) once (constant across K-tiles)
    n  = offs_m // (P * Q)
    pq = offs_m %  (P * Q)
    p  = pq // Q
    q  = pq %  Q

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- B (weights) descriptor: rows = GEMM_K, cols = K
    # Packed layout is a contiguous 2D tensor [GEMM_K, K] (row-major):
    #   row stride = K, col stride = 1
    # Original layout [K, C, R, S] viewed as (CRS, K):
    #   row stride = 1 (advance over S/R/C), col stride = C*R*S
    if packed:
        row_stride = K
        col_stride = 1
    else:
        row_stride = 1
        col_stride = C * R * S

    col_start = (pid_n * BLOCK_N).to(tl.int32)  # scalar BN tile start

    w_desc = tl.make_block_ptr(
        base=w_ptr,
        shape=(GEMM_K, K),
        strides=(row_stride, col_stride),
        offsets=(0, col_start),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # ----------------- K loop (robust) -----------------
    for k_tile in range(0, GEMM_K, BLOCK_K):
        offs_k = k_tile + tl.arange(0, BLOCK_K)   # [BK] reduction indices 0..GEMM_K-1

        # Project offs_k into (c,r,s) for reading A (input)
        # Note: even when weights are prepacked, A still uses (c,r,s)
        c  = offs_k // (R * S)
        rs = offs_k %  (R * S)
        r  = rs // S
        s  = rs %  S

        # Input coordinates for this tile
        h_in = p[:, None] * stride_h + r[None, :] * dil_h - pad_h
        w_in = q[:, None] * stride_w + s[None, :] * dil_w - pad_w

        if use_nhwc:
            # x layout: [N, H, W, C]
            a_offs = (
                n[:, None] * (H * W * C) +
                h_in * (W * C) +
                w_in * C +
                c[None, :]
            ).to(tl.int32)
        else:
            # x layout: [N, C, H, W]
            a_offs = (
                n[:, None] * (C * H * W) +
                c[None, :] * (H * W) +
                h_in * W + w_in
            ).to(tl.int32)

        a_mask = (
            (n[:, None] < N) & (c[None, :] < C) &
            (h_in >= 0) & (h_in < H) &
            (w_in >= 0) & (w_in < W)
        )
        A = tl.load(in_ptr + a_offs, mask=a_mask, other=0.0)  # [BM,BK], fp16

        # Weights: advance along rows by k_tile; guard both rows/cols for tails
        w_desc_k = tl.advance(w_desc, (k_tile, 0))
        B = tl.load(w_desc_k, boundary_check=(0, 1), padding_option='zero')  # [BK,BN], fp16

        acc += tl.dot(A, B)

    # ---- Epilogue ----
    if apply_bias:
        b = tl.load(bias_ptr + offs_n, mask=offs_n < K, other=0.0).to(tl.float32)
        acc += b[None, :]
    if activation == 'relu':
        acc = tl.maximum(acc, 0.0)

    # Store [N, K, P, Q] using (n,p,q, k=offs_n)
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


# =========================
# Python wrapper (toggles)
# =========================

def conv2d_fwd(x, w, b=None, stride=(1,1), padding=(0,0), dilation=(1,1),
               activation='relu', layout='nchw', prepack=False, pad_multiple=64):
    """
    x: [N,C,H,W] if layout='nchw', or [N,H,W,C] if layout='nhwc' (fp16, CUDA)
    w: [K,C,R,S] (fp16, CUDA). If prepack=True, packed to [CRS_pad, K].
    b: [K] or None (fp16)
    Returns y: [N,K,P,Q] (fp16)
    """
    assert x.is_cuda and w.is_cuda
    assert x.dtype == torch.float16 and w.dtype == torch.float16
    N, C, H, W = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]) if layout == 'nchw' else (x.shape[0], x.shape[3], x.shape[1], x.shape[2])
    K, Cw, R, S = w.shape
    assert C == Cw, "channel mismatch"
    sh, sw = stride; ph, pw = padding; dh, dw = dilation

    P = (H + 2*ph - dh*(R - 1) - 1) // sh + 1
    Q = (W + 2*pw - dw*(S - 1) - 1) // sw + 1

    GEMM_M = N * P * Q
    GEMM_N = K

    if prepack:
        wp, CRS_pad = prepack_weights_crs_k(w, pad_multiple=pad_multiple)
        w_used = wp
        GEMM_K = CRS_pad
        packed = True
    else:
        w_used = w
        GEMM_K = C * R * S
        packed = False

    # Prepare input pointer per layout
    if layout == 'nhwc':
        if x.stride(-1) != 1:  # ensure contiguous NHWC
            x = x.contiguous()
        x_ptr = x
        use_nhwc = True
    else:
        if x.stride(-1) != 1:  # ensure contiguous NCHW
            x = x.contiguous()
        x_ptr = x
        use_nhwc = False

    y = torch.empty((N, K, P, Q), dtype=torch.float16, device=x.device)
    apply_bias = b is not None
    b_ptr = b if apply_bias else torch.empty(1, dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(GEMM_M, META['BLOCK_M']) *
        triton.cdiv(GEMM_N, META['BLOCK_N']),
    )

    _conv2d_fwd[grid](
        y, x_ptr, w_used, b_ptr,
        apply_bias, activation,
        packed, use_nhwc,
        N, C, H, W, K, P, Q, R, S,
        sh, sw, ph, pw, dh, dw,
        GEMM_M, GEMM_N, GEMM_K,
    )
    return y


# =========================
# Quick checks / benches
# =========================

def _check_case(shape, stride=(1,1), padding=(0,0), dilation=(1,1),
                activation='relu', layout='nchw', prepack=False, pad_multiple=64):
    torch.manual_seed(0)
    dev = torch.device('cuda'); dtype = torch.float16
    N, C, H, W, K, R, S = shape

    if layout == 'nhwc':
        x_nchw = torch.randn(N, C, H, W, dtype=dtype, device=dev)
        x = x_nchw.permute(0, 2, 3, 1).contiguous()  # NHWC for our kernel
    else:
        x = torch.randn(N, C, H, W, dtype=dtype, device=dev)
        x_nchw = x

    w = torch.randn(K, C, R, S, dtype=dtype, device=dev)
    b = torch.randn(K, dtype=dtype, device=dev)

    y = conv2d_fwd(x, w, b, stride, padding, dilation,
                   activation=activation, layout=layout,
                   prepack=prepack, pad_multiple=pad_multiple)

    y_ref = F.conv2d(x_nchw, w, bias=b, stride=stride, padding=padding, dilation=dilation)
    if activation == 'relu':
        y_ref = torch.relu(y_ref)

    ok = torch.allclose(y, y_ref, atol=2e-2, rtol=2e-2)
    print(f"[check] layout={layout:5s} prepack={prepack} shape={shape} -> OK={ok}, "
          f"max_abs={(y-y_ref).abs().max().item():.3e}")


@torch.inference_mode()
def _bench_case(shape, stride=(1,1), padding=(0,0), dilation=(1,1),
                activation='relu', layout='nchw', prepack=False, pad_multiple=64,
                warmup=10, iters=50):
    dev = torch.device('cuda'); dtype = torch.float16
    N, C, H, W, K, R, S = shape

    if layout == 'nhwc':
        x_nchw = torch.randn(N, C, H, W, dtype=dtype, device=dev)
        x = x_nchw.permute(0, 2, 3, 1).contiguous()
    else:
        x = torch.randn(N, C, H, W, dtype=dtype, device=dev)

    w = torch.randn(K, C, R, S, dtype=dtype, device=dev)
    b = torch.randn(K, dtype=dtype, device=dev)

    # warmup
    for _ in range(warmup):
        _ = conv2d_fwd(x, w, b, stride, padding, dilation,
                       activation=activation, layout=layout,
                       prepack=prepack, pad_multiple=pad_multiple)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = conv2d_fwd(x, w, b, stride, padding, dilation,
                       activation=activation, layout=layout,
                       prepack=prepack, pad_multiple=pad_multiple)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters

    P = (H + 2*padding[0] - dilation[0]*(R-1) - 1) // stride[0] + 1
    Q = (W + 2*padding[1] - dilation[1]*(S-1) - 1) // stride[1] + 1
    flops = 2.0 * N * K * P * Q * C * R * S
    print(f"[bench] layout={layout:5s} prepack={prepack} {shape} "
          f"time={dt*1e3:.3f} ms  thru={flops/dt/1e12:.2f} TFLOP/s")


if __name__ == "__main__":
    # Baseline (NCHW, no prepack)
    s = (2, 64, 56, 56, 128, 3, 3)
    _check_case(s, stride=(1,1), padding=(1,1), dilation=(1,1), layout='nchw', prepack=False)
    _bench_case(s,  stride=(1,1), padding=(1,1), dilation=(1,1), layout='nchw', prepack=False)

    # NCHW + prepack (faster on repeated uses of the same weights)
    _check_case(s, stride=(1,1), padding=(1,1), dilation=(1,1), layout='nchw', prepack=True, pad_multiple=64)
    _bench_case(s,  stride=(1,1), padding=(1,1), dilation=(1,1), layout='nchw', prepack=True, pad_multiple=64)

    # NHWC + prepack (favorable A-side coalescing on some GPUs)
    _check_case(s, stride=(1,1), padding=(1,1), dilation=(1,1), layout='nhwc', prepack=True, pad_multiple=64)
    _bench_case(s,  stride=(1,1), padding=(1,1), dilation=(1,1), layout='nhwc', prepack=True, pad_multiple=64)

    # Smaller/taily
    s2 = (1, 13, 31, 29, 17, 3, 3)
    _check_case(s2, stride=(2,2), padding=(1,1), dilation=(1,1), layout='nchw', prepack=True, pad_multiple=64)
    _bench_case(s2,  stride=(2,2), padding=(1,1), dilation=(1,1), layout='nchw', prepack=True, pad_multiple=64)
