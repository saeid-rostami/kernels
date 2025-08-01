import triton
import triton.language as tl
import torch


'''
    output_ptr -> pointer to the output tensor [N, K, P, Q] (in NCHW layout)
    input_ptr -> pointer to the input tensor [N, C, H, W]
    weight_ptr -> pointer to the weights [K, C, R, S]
    bias_ptr -> pointer to the bias tensor [K]

    Input shape:
        N -> batch size
        C -> number of input channels
        H -> height of each input feature map
        W -> width of each input feature map

    Output shape:
        N -> batch size
        K -> number of output channels, number of conv filters to apply
        P -> output height of feature map
        Q -> output width of feature map

    Weigh/Filter shape:
        K -> number of output channels (Number of feature map produced by the conv layer)
        C -> number of input channels (Number of feature map coming into the conv layer)
        R -> filter height
        S -> filter width

    P and Q are computed from H, W, R, S, stride, padding and dilation
    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // stride_h + 1 
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // stride_w + 1

    stride+h, stride_w -> how far the window jumps vertically/horizontally
    pad_h, pad_w -> padding on top/bottom and left/right
    dil_h, dil_w -> dilation between kernel elements 

    GEMM Dimensions
    GEMM_M, GEMM_N, GEMM_K
    Y = A @ B
    [M, K] @ [K, N] -> [M, N]

    GEMM_M -> rows of output matrix N*P*Q
    GEMM_N -> Columns of output matrix K
    GEMM_K -> Reduction dimension C*R*S

    A -> [N*P*Q C*R*S]
    B -> [C*R*C K]
    C -> [N*P*Q K]

    Each row of A -> one input patch
    Each column of B -> one filter
    Each column of C -> output of one filter at all locations

    BLOCK_M -> Number of output pixels (row of GEMM) processed per block
    BLOCK_N -> Number of output channels (Columns of GEMM) per block
    BLOCK_K -> How much of the reduction processed per tile 

'''

@triton.autotune(
        configs =  [
            triton.Config (
                {'BLOCK_M':128, 'BLOCK_N':64, 'BLOCK_K':64, 'GROUP_SIZE_M':8},
                num_stages = 3, num_warps = 4
            ),
        ],
        key = ['GEMM_M', 'GEMM_N', 'GEMM_K']
)

@triton.jit
def _conv2d_fwd(
    output_ptr, input_ptr, weight_ptr, bias_ptr, 
    apply_bias: tl.constexpr,
    N, C, H, W, K, P, Q, R, S,
    stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
    GEMM_M, GEMM_N, GEMM_K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(GEMM_M, BLOCK_M)
    num_pid_n = tl.cdiv(GEMM_N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n

    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype = tl.float32)

    for k_tile in range(0, GEMM_K, BLOCK_K):
        offs_k = k_tile + tl.arange(0, BLOCK_K)

        #compute maping A[offs_m, offs_k], B[offs_k, offs_n]
        n = offs_m // (P * Q)
        pq = offs_m % (P * Q)
        p = pq // Q
        q = pq % Q

        c = offs_k // (R * S)
        rs = offs_k % (R * S)
        r = rs // S
        s = rs % S

        h_in = p[:, None] * stride_h + r[None, :] * dil_h - pad_h
        w_in = q[:, None] * stride_w + s[None, :] * dil_w - pad_w

        input_offsets = (
            n[:, None] * C * H * W +
            c[None, :] * H * W +
            h_in * W + w_in
        )
        mask_x = (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W) & (c[None, :] < C)
        x = tl.load(input_ptr + input_offsets, mask=mask_x, other = 0.0).to(tl.float16)

        k = offs_n
        weight_offsets = (
            k[None, :] * C * R * S +
            c[:, None] * R * S +
            r[:, None] * S +
            s[:, None]
        )
        mask_w = (c[:, None] < C) & (r[:, None] < R) & (s[:, None] < S) & (k[None, :] < K)
        w = tl.load(weight_ptr + weight_offsets, mask=mask_w, other = 0.0).to(tl.float16)

        acc += tl.dot(x, w)
    

    # Apply Bias
    if apply_bias:
        bias_offsets = (offs_n[None, :]).to(tl.int32)
        bias = tl.load(bias_ptr + bias_offsets, mask=offs_n[None, :] < K, other = 0.0)
        acc += bias
    
    #ReLu activation
    acc = tl.maximum(acc, 0.0)

    #store to the output
    n = offs_m // (P * Q)
    pq = offs_m % (P * Q)
    p = pq // Q
    q = pq % Q
    k = offs_n

    output_ptr = (
        output_ptr + n[:, None] * K * P * Q +
        k[None, :] * P * Q +
        p[:, None] * Q +
        q[:, None]
    )

    mask_o = (n[:, None] < N) & (k[None, :] < K) & (p[:, None] < P) & (q[:, None] < Q)
    tl.store(output_ptr, acc.to(tl.float16), mask=mask_o)


def conv2d_fwd(input, weight, bias, stride, padding, dilation) :
    N, C, H, W = input.shape
    K, _, R, S = weight.shape
    str_h, str_w = stride
    pad_h, pad_w = padding
    dil_h, dil_w = dilation

    P = (H + 2 * pad_h - dil_h * (R - 1) - 1) // str_h + 1
    Q = (W + 2 * pad_w - dil_w * (S - 1) - 1) // str_w + 1

    GEMM_M = N * P * Q
    GEMM_N = K
    GEMM_K = C * R * S

    out = torch.empty((N, K, P, Q), dtype = torch.float16, device = input.device)

    grid = lambda META: (triton.cdiv(GEMM_M, META['BLOCK_M']) * 
                        triton.cdiv(GEMM_N, META['BLOCK_N']),)
    
    apply_bias = bias is not None
    bias_ptr = bias if apply_bias else torch.empty(1, dtype=input.dtype, device=input.device)

    _conv2d_fwd[grid](
        out, input, weight, bias_ptr, apply_bias,
        N, C, H, W, K, P, Q, R, S,
        str_h, str_w, pad_h, pad_w, dil_h, dil_w,
        GEMM_M, GEMM_N, GEMM_K
    )

    return out


x = torch.randn(2, 64, 56, 56, dtype=torch.float16, device='cuda')
w = torch.randn(128, 64, 3, 3, dtype=torch.float16, device='cuda')
b = torch.randn(128, dtype=torch.float16, device='cuda')

y = conv2d_fwd(x, w, b, stride=(1, 1), padding=(1, 1), dilation=(1, 1))
print(y.shape)
