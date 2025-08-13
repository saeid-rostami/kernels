import time
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# =========================
# Conv2D Forward (NCHW) -> [N, K, P, Q]
# Weights [K, C, R, S], optional bias [K]
# Robust across configs: no alignment assumptions; LDS tiling for B via block_ptr.
# =========================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M':128, 'BLOCK_N':128, 'BLOCK_K':64, 'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M':128, 'BLOCK_N': 64, 'BLOCK_K':64, 'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N':128, 'BLOCK_K':64, 'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K':32, 'GROUP_SIZE_M':8}, num_warps=4, num_stages=3),
    ],
    key=['GEMM_M', 'GEMM_N', 'GEMM_K'],
)
@triton.jit
def _conv2d_fwd(
    out_ptr, in_ptr, w_ptr, bias_ptr,
    apply_bias: tl.constexpr, activation: tl.constexpr,  # 'relu' or 'none'
    N, C, H, W, K, P, Q, R, S,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # MFMA-friendly tiles (safe & performant)
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # rows (M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # cols (N)

    # Decode rows -> (n, p, q) once (constant across K-tiles)
    n  = offs_m // (P * Q)
    pq = offs_m %  (P * Q)
    p  = pq // Q
    q  = pq %  Q

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- B (weights) block pointer: rows=(C*R*S), cols=K ----
    stride_k = C * R * S
    tl.multiple_of(stride_k, 16)
    tl.multiple_of(offs_n, 16)
    tl.max_contiguous(offs_n, BLOCK_N)

    # Use a SCALAR column start for this program's BN tile
    col_start = (pid_n * BLOCK_N).to(tl.int32)

    w_desc = tl.make_block_ptr(
        base=w_ptr,
        shape=(C * R * S, K),          # rows, cols
        strides=(1, stride_k),         # row stride over s/r/c, col stride over K
        offsets=(0, col_start),        # <-- scalar start column
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    # ----------------- K loop (robust for any R,S and BLOCK_K) -----------------
    for k_tile in range(0, GEMM_K, BLOCK_K):
        # Linear K indices for this tile and their (c,r,s) projection
        offs_k = k_tile + tl.arange(0, BLOCK_K)
        c  = offs_k // (R * S)
        rs = offs_k %  (R * S)
        r  = rs // S
        s  = rs %  S

        # Im2col addresses for A tile [BLOCK_M, BLOCK_K]
        h_in = p[:, None] * stride_h + r[None, :] * dil_h - pad_h
        w_in = q[:, None] * stride_w + s[None, :] * dil_w - pad_w

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

        # Weights: advance descriptor along row dimension by k_tile; guard rows+cols
        w_desc_k = tl.advance(w_desc, (k_tile, 0))
        B = tl.load(w_desc_k, boundary_check=(0, 1), padding_option='zero')  # [BK,BN], fp16

        # Dot — Triton 3.4 will stage tiles into LDS (num_stages>1) and use MFMA
        acc += tl.dot(A, B)

    # ---- Epilogue ----
    if apply_bias:
        b = tl.load(bias_ptr + offs_n, mask=offs_n < K, other=0.0).to(tl.float32)
        acc += b[None, :]

    if activation == 'relu':
        acc = tl.maximum(acc, 0.0)

    # Store [N, K, P, Q]
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


def conv2d_fwd(x, w, b=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), activation='relu'):
    """
    x: [N,C,H,W] (fp16), w: [K,C,R,S] (fp16), b: [K] or None (fp16)
    returns y: [N,K,P,Q] (fp16)
    """
    # Sanity / dtypes
    assert x.is_cuda and w.is_cuda, "use CUDA/HIP tensors"
    assert x.dtype == torch.float16 and w.dtype == torch.float16, "kernel expects fp16"
    N, C, H, W = x.shape
    K, Cw, R, S = w.shape
    assert C == Cw, "channel mismatch"

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation

    P = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S

    y = torch.empty((N, K, P, Q), dtype=torch.float16, device=x.device)

    grid = lambda META: (
        triton.cdiv(GEMM_M, META['BLOCK_M']) *
        triton.cdiv(GEMM_N, META['BLOCK_N']),
    )

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


# =========================
# Quick correctness + bench
# =========================
def _check_once(shape, stride=(1, 1), padding=(0, 0), dilation=(1, 1), activation='relu'):
    torch.manual_seed(0)
    dev = torch.device('cuda')  # ROCm builds also use 'cuda'
    dtype = torch.float16
    N, C, H, W, K, R, S = shape
    x = torch.randn(N, C, H, W, dtype=dtype, device=dev)
    w = torch.randn(K, C, R, S, dtype=dtype, device=dev)
    b = torch.randn(K, dtype=dtype, device=dev)

    y = conv2d_fwd(x, w, b, stride, padding, dilation, activation)
    y_ref = F.conv2d(x, w, bias=b, stride=stride, padding=padding, dilation=dilation)
    if activation == 'relu':
        y_ref = torch.relu(y_ref)

    ok = torch.allclose(y, y_ref, atol=2e-2, rtol=2e-2)
    max_abs = (y - y_ref).abs().max().item()
    max_rel = ((y - y_ref).abs() / (y_ref.abs() + 1e-6)).max().item()
    print(f"[check] {shape} stride={stride} pad={padding} dil={dilation} -> "
          f"OK={ok} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")


@torch.inference_mode()
def _bench(shape, stride=(1, 1), padding=(0, 0), dilation=(1, 1), activation='relu', warmup=10, iters=50):
    dev = torch.device('cuda')
    dtype = torch.float16
    N, C, H, W, K, R, S = shape
    x = torch.randn(N, C, H, W, dtype=dtype, device=dev)
    w = torch.randn(K, C, R, S, dtype=dtype, device=dev)
    b = torch.randn(K, dtype=dtype, device=dev)

    # warmup (JIT + caches)
    for _ in range(warmup):
        _ = conv2d_fwd(x, w, b, stride, padding, dilation, activation)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        _ = conv2d_fwd(x, w, b, stride, padding, dilation, activation)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters

    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    P = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1
    flops = 2.0 * N * K * P * Q * C * R * S
    print(f"[bench] {shape} stride={stride} pad={padding} dil={dilation} "
          f"time={dt*1e3:.3f} ms  thru={flops/dt/1e12:.2f} TFLOP/s")


if __name__ == "__main__":
    # ResNet-like
    s1 = (2, 64, 56, 56, 128, 3, 3)
    _check_once(s1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), activation='relu')
    _bench(s1,  stride=(1, 1), padding=(1, 1), dilation=(1, 1), activation='relu')

    # Odd/tail cases (channels/size/stride)
    s2 = (1, 13, 31, 29, 17, 3, 3)
    _check_once(s2, stride=(2, 2), padding=(1, 1), dilation=(1, 1), activation='relu')
    _bench(s2,  stride=(2, 2), padding=(1, 1), dilation=(1, 1), activation='relu')
