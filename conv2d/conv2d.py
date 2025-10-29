#!/usr/bin/env python3

from __future__ import annotations
import argparse, random, statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


def dynamic_conv_tolerances(dtype: torch.dtype, K_red: int, ref: torch.Tensor):
    eps = {torch.float16: 2**-10, torch.bfloat16: 2**-7, torch.float32: 2**-23}.get(dtype, 2**-10)
    rtol = 6e-3 if K_red < 1024 else (8e-3 if K_red < 4096 else 1.2e-2)
    ref32 = ref.float()
    scale = ref32.abs().median().item() if ref32.numel() else 1.0
    atol = max(1e-5, 3.0 * eps * (K_red ** 0.5) * max(scale, 1.0))
    return rtol, atol

def _bench_ms(fn, warmup=10, rep=40):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total = 0.0
    for _ in range(rep):
        if torch.cuda.is_available():
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            total += s.elapsed_time(e)
        else:
            import timeit
            total += timeit.timeit(fn, number=1) * 1e3
    return total / rep

def flops_conv(N, C, H, W, K_out, R, S, P, Q):
    return 2.0 * N * P * Q * K_out * C * R * S

def _out_hw(H, W, R, S, stride, padding, dilation):
    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    P = (H + 2*ph - dh*(R-1) - 1)//sh + 1
    Q = (W + 2*pw - dw*(S-1) - 1)//sw + 1
    return P, Q


_PACK_CACHE: Dict[Tuple[int, Tuple[int, ...], torch.dtype, int, int], Tuple[torch.Tensor, Tuple[int,int]]] = {}

def _storage_ptr(t: torch.Tensor) -> int:
    return t.untyped_storage().data_ptr() if hasattr(t, "untyped_storage") else t.storage().data_ptr()

def prepack_oihw_to_kmajor(w_oihw: torch.Tensor, block_k: int = 64):
    K_out, C, R, S = w_oihw.shape
    K_red = C*R*S
    K_pad = ((K_red + block_k - 1)//block_k) * block_k
    w_rs = w_oihw.reshape(K_out, K_red)
    if K_pad != K_red:
        pad = torch.zeros((K_out, K_pad - K_red), device=w_oihw.device, dtype=w_oihw.dtype)
        w_rs = torch.cat([w_rs, pad], dim=1)
    return w_rs.contiguous(), (K_out, K_pad)

def get_or_make_weight_pack(w_oihw: torch.Tensor, block_k: int = 64):
    key = (_storage_ptr(w_oihw), tuple(w_oihw.shape), w_oihw.dtype, block_k, int(getattr(w_oihw, "_version", 0)))
    item = _PACK_CACHE.get(key)
    if item is None:
        item = prepack_oihw_to_kmajor(w_oihw, block_k)
        _PACK_CACHE.clear()
        _PACK_CACHE[key] = item
    return item


if triton is not None:

    @triton.jit
    def _tanh(x):
        x = tl.minimum(tl.maximum(x, -10.0), 10.0)
        e2x = tl.exp(2 * x)
        return (e2x - 1) / (e2x + 1)

    AUTOTUNE_CONFIGS = [
        # 1x1 / GEMM-like (large K_out, large reduction)
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 64, "GROUP_SIZE_M": 4}, num_warps=8, num_stages=3),
        # 3x3 @ 56x56 family (more mem pressure)
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_SIZE_M": 8}, num_warps=8, num_stages=3),
        # fallback small-K
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32, "GROUP_SIZE_M": 4}, num_warps=4, num_stages=3),
    ]

    @triton.autotune(
        configs=AUTOTUNE_CONFIGS,
        key=["N","C","H","W_in","K_out","R","S","stride_h","stride_w","pad_h","pad_w","dil_h","dil_w"]
    )
    @triton.jit
    def _conv2d_kernel(
        X, WK, BIAS, Y,
        N, C, H, W_in, K_out, R, S, P, Q,
        stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
        stride_x_n, stride_x_c, stride_x_h, stride_x_w,
        stride_wk_kred, stride_wk_kout,
        stride_y_n, stride_y_k, stride_y_p, stride_y_q,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, HAS_BIAS: tl.constexpr, ACT_TYPE: tl.constexpr,
    ):

        pid  = tl.program_id(axis=0)
        nprog = tl.num_programs(0)  

        num_pid_m = tl.cdiv(N * P * Q, BLOCK_M)
        num_pid_n = tl.cdiv(K_out,     BLOCK_N)
        total_tiles = num_pid_m * num_pid_n

        tile_id = pid
        while tile_id < total_tiles:
            pid_m = tile_id // num_pid_n
            pid_n = tile_id %  num_pid_n

            offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            offs_k = tl.arange(0, BLOCK_K)

            n_idx = offs_m[:, None] // (P * Q)
            pq    = offs_m[:, None] %  (P * Q)
            p_idx = pq // Q
            q_idx = pq %  Q

            Y_ptrs = (
                Y
                + n_idx * stride_y_n
                + offs_n[None, :] * stride_y_k
                + p_idx * stride_y_p
                + q_idx * stride_y_q
            )

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            K_red = C * R * S

            k0 = 0
            if K_red > 0:
                offs_k = tl.arange(0, BLOCK_K)
                kred   = k0 + offs_k
                k_mask = kred < K_red

                WK_ptrs = WK + kred[:, None] * stride_wk_kred + offs_n[None, :] * stride_wk_kout
                w_mask  = k_mask[:, None] & (offs_n[None, :] < K_out)
                w_cur   = tl.load(WK_ptrs, mask=w_mask, other=0.0)

                c  = kred // (R * S)
                rs = kred %  (R * S)
                r  = rs // S
                s  = rs %  S
                oh = p_idx * stride_h - pad_h + r * dil_h
                ow = q_idx * stride_w - pad_w + s * dil_w
                X_ptrs = X + n_idx * stride_x_n + c * stride_x_c + oh * stride_x_h + ow * stride_x_w
                x_mask = (n_idx < N) & (oh >= 0) & (ow >= 0) & (oh < H) & (ow < W_in) & k_mask[None, :]
                x_cur  = tl.where(x_mask, tl.load(X_ptrs, mask=x_mask, other=0.0), 0.0)

                k0 += BLOCK_K

                while k0 < K_red:
                    kred_n   = k0 + offs_k
                    k_mask_n = kred_n < K_red

                    WK_ptrs_n = WK + kred_n[:, None] * stride_wk_kred + offs_n[None, :] * stride_wk_kout
                    w_mask_n  = k_mask_n[:, None] & (offs_n[None, :] < K_out)
                    w_nxt     = tl.load(WK_ptrs_n, mask=w_mask_n, other=0.0)

                    c_n  = kred_n // (R * S)
                    rs_n = kred_n %  (R * S)
                    r_n  = rs_n // S
                    s_n  = rs_n %  S
                    oh_n = p_idx * stride_h - pad_h + r_n * dil_h
                    ow_n = q_idx * stride_w - pad_w + s_n * dil_w
                    X_ptrs_n = X + n_idx * stride_x_n + c_n * stride_x_c + oh_n * stride_x_h + ow_n * stride_x_w
                    x_mask_n = (n_idx < N) & (oh_n >= 0) & (ow_n >= 0) & (oh_n < H) & (ow_n < W_in) & k_mask_n[None, :]
                    x_nxt    = tl.where(x_mask_n, tl.load(X_ptrs_n, mask=x_mask_n, other=0.0), 0.0)

                    acc += tl.dot(x_cur, w_cur)

                    x_cur, w_cur = x_nxt, w_nxt
                    k0 += BLOCK_K

                acc += tl.dot(x_cur, w_cur)

            if HAS_BIAS:
                b = tl.load(BIAS + offs_n, mask=offs_n < K_out, other=0.0)
                acc += b[None, :]

            if ACT_TYPE == 1:
                acc = tl.maximum(acc, 0)
            elif ACT_TYPE == 2:
                acc = tl.minimum(tl.maximum(acc, 0), 6)
            elif ACT_TYPE == 3:
                acc = 0.5 * acc * (1.0 + _tanh(0.7978845608 * (acc + 0.044715 * acc * acc * acc)))

            tl.store(
                Y_ptrs,
                acc,
                mask=(n_idx < N) & (p_idx < P) & (q_idx < Q) & (offs_n[None, :] < K_out),
            )

            tile_id += nprog


def _launch_common(x, w_k, bias_fp32, y, N, C, H, W_in, K_out, R, S, P, Q,
                   stride, padding, dilation, out_dtype, block_k, activation,
                   y_layout: str = "nchw"):
    if triton is None:
        raise RuntimeError("Triton not available")

    sh, sw = stride; ph, pw = padding; dh, dw = dilation
    sx_n, sx_c, sx_h, sx_w = x.stride()

    if y_layout == "nchw":
        sy_n, sy_k, sy_p, sy_q = y.stride()
    elif y_layout == "nhwc":
        s_n, s_p, s_q, s_k = y.stride()
        sy_n, sy_k, sy_p, sy_q = s_n, s_k, s_p, s_q
    else:
        raise ValueError("bad y_layout")

    swk_kout, swk_kred = w_k.stride()
    stride_wk_kred, stride_wk_kout = swk_kred, swk_kout

    def grid(meta):
        BM = meta["BLOCK_M"]; BN = meta["BLOCK_N"]
        return (triton.cdiv(N*P*Q, BM) * triton.cdiv(K_out, BN),)

    ACT_MAP = {"none":0, "relu":1, "relu6":2, "gelu":3}
    bias_arg = bias_fp32 if bias_fp32 is not None else w_k.new_empty(1)

    _conv2d_kernel[grid](
        x, w_k, bias_arg, y,
        N, C, H, W_in, K_out, R, S, P, Q,
        sh, sw, ph, pw, dh, dw,
        sx_n, sx_c, sx_h, sx_w,
        stride_wk_kred, stride_wk_kout,
        sy_n, sy_k, sy_p, sy_q,
        HAS_BIAS=1 if bias_fp32 is not None else 0,
        ACT_TYPE=ACT_MAP.get(activation, 0),
    )

def apply_activation(y: torch.Tensor, activation: str):
    if activation == "relu":  return F.relu(y)
    if activation == "relu6": return torch.clamp_min(torch.clamp_max(y, 6), 0)
    if activation == "gelu":  return F.gelu(y, approximate="tanh")
    return y

def conv2d_nchw(x, w_oihw, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1),
                 activation="none", out_dtype=torch.float16, block_k=64):
    assert x.is_cuda and w_oihw.is_cuda
    N,C,H,W_in = x.shape
    K_out,Cw,R,S = w_oihw.shape
    assert Cw == C
    P,Q = _out_hw(H, W_in, R, S, stride, padding, dilation)
    y = torch.empty((N, K_out, P, Q), device=x.device, dtype=out_dtype)
    w_k, _ = get_or_make_weight_pack(w_oihw.contiguous(), block_k)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    _launch_common(x, w_k, bias_fp32, y, N, C, H, W_in, K_out, R, S, P, Q,
                   stride, padding, dilation, out_dtype, block_k, activation, y_layout="nchw")
    return y

def conv2d_nchw_channels_last(x, w_oihw, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1),
                              activation="none", out_dtype=torch.float16, block_k=64):
    assert x.is_cuda and w_oihw.is_cuda
    x = x.to(memory_format=torch.channels_last)
    N,C,H,W_in = x.shape
    K_out,Cw,R,S = w_oihw.shape
    assert Cw == C
    P,Q = _out_hw(H, W_in, R, S, stride, padding, dilation)
    y = torch.empty((N, P, Q, K_out), device=x.device, dtype=out_dtype).contiguous()
    w_k, _ = get_or_make_weight_pack(w_oihw.contiguous(), block_k)
    bias_fp32 = bias.float().contiguous() if bias is not None else None
    _launch_common(x, w_k, bias_fp32, y, N, C, H, W_in, K_out, R, S, P, Q,
                   stride, padding, dilation, out_dtype, block_k, activation, y_layout="nhwc")
    return y


@dataclass
class TestResult:
    name: str
    passed: bool
    max_abs_error: float
    rel_error: float
    message: str = ""

class TestSuite:
    def __init__(self, device: str, dtype: torch.dtype, verbose=True, bench_enabled=False,
                 print_shapes=True, fp32_audit=False):
        self.device = torch.device(device)
        self.dtype = dtype
        self.verbose = verbose
        self.bench_enabled = bench_enabled
        self.print_shapes = print_shapes
        self.fp32_audit = fp32_audit
        self.results: List[TestResult] = []
        self.bench_records: List[Dict[str, float]] = []
        self.total_flops_tri = 0.0
        self.total_time_tri  = 0.0
        self.total_flops_th  = 0.0
        self.total_time_th   = 0.0

    def check_close(self, name: str, got: torch.Tensor, ref: torch.Tensor, K_red: Optional[int]=None,
                    rtol: Optional[float]=None, atol: Optional[float]=None) -> TestResult:
        got32 = got.float(); ref32 = ref.float()
        diff = (got32 - ref32).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        rel = max_abs / (float(ref32.abs().max().item()) + 1e-6)
        if rtol is None or atol is None:
            K_est = int(K_red) if K_red is not None else 1024
            rtol_calc, atol_calc = dynamic_conv_tolerances(self.dtype, K_est, ref32)
            rtol = rtol if rtol is not None else rtol_calc
            atol = atol if atol is not None else atol_calc
        try:
            torch.testing.assert_close(got32, ref32, rtol=rtol, atol=atol)
            passed = True; msg = "OK"
        except AssertionError as e:
            passed = False; msg = str(e).split("\n")[0]
        res = TestResult(name, passed, max_abs, rel, msg)
        self.results.append(res)
        if self.verbose:
            mark = "✓" if passed else "✗"
            print(f"  {mark} {name:<40} | max_abs={max_abs:.3e} rel={rel:.3e}")
        return res

    def add_bench(self, name: str, flops: float, ms_tri: float, ms_th: float):
        tf_tri = flops/(ms_tri*1e-3)/1e12
        tf_th  = flops/(ms_th *1e-3)/1e12
        self.bench_records.append({
            "name": name, "ms_tri": ms_tri, "tflops_tri": tf_tri,
            "ms_torch": ms_th, "tflops_torch": tf_th
        })
        self.total_flops_tri += flops; self.total_time_tri += ms_tri*1e-3
        self.total_flops_th  += flops; self.total_time_th  += ms_th *1e-3
        print(f"      Bench: Triton {ms_tri:6.3f} ms | {tf_tri:6.2f} TF/s | Torch {ms_th:6.3f} ms | {tf_th:6.2f} TF/s")

    def summary(self) -> bool:
        total = len(self.results); passed = sum(r.passed for r in self.results)
        print("\n" + "="*80)
        print(f"TEST SUMMARY: {passed}/{total} passed")
        print("="*80)
        if passed < total:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  ✗ {r.name}")
                    print(f"    max_abs={r.max_abs_error:.3e}, rel={r.rel_error:.3e}")
                    if r.message:
                        print(f"    {r.message}")
        if self.bench_enabled and self.bench_records:
            print("\n" + "-"*80)
            print("BENCHMARK SUMMARY (whole image)")
            print("-"*80)
            tri_tf = [r["tflops_tri"] for r in self.bench_records]
            th_tf  = [r["tflops_torch"] for r in self.bench_records]
            tri_ms = [r["ms_tri"] for r in self.bench_records]
            th_ms  = [r["ms_torch"] for r in self.bench_records]
            def s(v):   return f"mean={statistics.mean(v):6.2f} | median={statistics.median(v):6.2f}"
            def s_ms(v): return f"mean={statistics.mean(v):7.3f} | median={statistics.median(v):7.3f}"
            print(f"Triton TFLOPS : {s(tri_tf)}")
            print(f"Torch  TFLOPS : {s(th_tf)}")
            print(f"Triton ms     : {s_ms(tri_ms)}")
            print(f"Torch  ms     : {s_ms(th_ms)}")
            print("\nTOTAL (end-to-end) EFFECTIVE TFLOPS")
            eff_tri = self.total_flops_tri / max(self.total_time_tri, 1e-12) / 1e12
            eff_th  = self.total_flops_th  / max(self.total_time_th,  1e-12) / 1e12
            print(f"  Triton : {eff_tri:6.2f} TF/s   (sum FLOPs / sum time)")
            print(f"  Torch  : {eff_th:6.2f} TF/s   (sum FLOPs / sum time)")
        return passed == total

# ======================== Tests ========================

def get_edge_case_shapes():
    return [
        (1,3,7,7,8, 3,3,(1,1),(1,1),(1,1),"3x3 same padding"),
        (1,3,8,8,16,1,1,(1,1),(0,0),(1,1),"1x1 stride1"),
        (2,16,32,32,32,3,3,(2,2),(1,1),(1,1),"stride2"),
        (2,32,17,23,64,5,5,(2,2),(2,2),(1,1),"odd dims + pad"),
        (4,64,28,28,128,3,3,(1,1),(0,0),(2,2),"dilation2"),
        (2,512,7,7,1024,1,1,(1,1),(0,0),(1,1),"1x1 large channels"),
    ]

def run_bench_case(suite, x, w, b, stride, padding, dilation, activation, name):
    N,C,H,W = x.shape; K_out,_,R,S = w.shape
    P,Q = _out_hw(H, W, R, S, stride, padding, dilation)
    total_flops = flops_conv(N,C,H,W,K_out,R,S,P,Q)

    def fn_torch():
        _ = F.conv2d(x, w, b.to(dtype=suite.dtype) if b is not None else None,
                     stride=stride, padding=padding, dilation=dilation)

    def fn_triton():
        _ = conv2d_nchw(x, w, b, stride, padding, dilation, activation="none", out_dtype=suite.dtype)

    ms_tri = _bench_ms(fn_triton, warmup=5, rep=30)
    ms_th  = _bench_ms(fn_torch,   warmup=5, rep=30)
    suite.add_bench(name, total_flops, ms_tri, ms_th)

def test_edge_cases(suite: TestSuite, activation: str = "none"):
    print("\n" + "="*80); print(f"EDGE CASE TESTS (activation={activation})"); print("="*80)
    for N,C,H,W,K_out,R,S,stride,padding,dilation,desc in get_edge_case_shapes():
        P,Q = _out_hw(H,W,R,S,stride,padding,dilation)
        if P < 1 or Q < 1: continue
        x = torch.randn((N,C,H,W), device=suite.device, dtype=suite.dtype)
        w = torch.randn((K_out,C,R,S), device=suite.device, dtype=suite.dtype)
        b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
        y_ref = apply_activation(F.conv2d(x,w,b.to(dtype=suite.dtype), stride=stride, padding=padding, dilation=dilation), activation)
        y_tri = conv2d_nchw(x,w,b,stride,padding,dilation,activation=activation,out_dtype=suite.dtype)
        if suite.print_shapes:
            print(f"    {desc}: X{tuple(x.shape)} W{tuple(w.shape)} -> Y{tuple(y_tri.shape)}")
        suite.check_close(f"{desc} / NCHW", y_tri, y_ref, K_red=C*R*S)
        if suite.bench_enabled:
            run_bench_case(suite, x,w,b, stride,padding,dilation, activation, f"{desc}")

def test_random_fuzzing(suite: TestSuite, num_tests=200, activation="none"):
    print("\n" + "="*80); print(f"RANDOM FUZZING TESTS (n={num_tests}, activation={activation})"); print("="*80)
    for i in range(num_tests):
        N = random.randint(1,8)
        C = random.choice([1,3,16,32,64,128,256])
        H = random.randint(4,64); W = random.randint(4,64)
        K_out = random.choice([16,32,64,128,256])
        R = random.randint(1, min(7,H)); S = random.randint(1, min(7,W))
        sh = random.randint(1,3); sw = random.randint(1,3)
        ph = random.randint(0,R//2); pw = random.randint(0,S//2)
        dh = random.randint(1,2); dw = random.randint(1,2)
        P,Q = _out_hw(H,W,R,S,(sh,sw),(ph,pw),(dh,dw))
        if P < 1 or Q < 1: continue
        try:
            x = torch.randn((N,C,H,W), device=suite.device, dtype=suite.dtype)
            w = torch.randn((K_out,C,R,S), device=suite.device, dtype=suite.dtype)
            b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
            y_ref = apply_activation(F.conv2d(x,w,b.to(dtype=suite.dtype), stride=(sh,sw), padding=(ph,pw), dilation=(dh,dw)), activation)
            y_tri = conv2d_nchw(x,w,b,(sh,sw),(ph,pw),(dh,dw), activation=activation, out_dtype=suite.dtype)
            if suite.print_shapes:
                print(f"    Rand[{i}]: X{tuple(x.shape)} W{tuple(w.shape)} -> Y{tuple(y_tri.shape)}")
            suite.check_close(f"Random[{i}] ({N},{C},{H},{W})->({N},{K_out},{P},{Q})", y_tri, y_ref, K_red=C*R*S)
            if suite.fp32_audit:
                y_ref32 = F.conv2d(x.float(), w.float(), b.float(), stride=(sh,sw), padding=(ph,pw), dilation=(dh,dw))
                y_tri32 = conv2d_nchw(x.float(), w.float(), b.float(), (sh,sw), (ph,pw), (dh,dw), activation="none", out_dtype=torch.float32)
                suite.check_close(f"Random[{i}] (FP32 compare)", y_tri32, y_ref32, K_red=C*R*S, rtol=1e-6, atol=1e-6)
        except Exception as e:
            print(f"  ✗ Random[{i}] EXCEPTION: {str(e)[:120]}")
            suite.results.append(TestResult(f"Random[{i}]", False, float("inf"), float("inf"), str(e)))

def _resolve_torchvision_weights(tvm, model_name: str, use_pretrained: bool):
    if not use_pretrained:
        return None
    try:
        from torchvision.models import get_model_weights  
        ws = get_model_weights(model_name)
        if ws is not None and len(ws):
            return getattr(ws, "DEFAULT", next(iter(ws)))
    except Exception:
        pass
    aliases = {
        "resnet18":"ResNet18_Weights","resnet34":"ResNet34_Weights","resnet50":"ResNet50_Weights",
        "resnet101":"ResNet101_Weights","resnet152":"ResNet152_Weights",
        "wide_resnet50_2":"Wide_ResNet50_2_Weights","wide_resnet101_2":"Wide_ResNet101_2_Weights",
        "mobilenet_v2":"MobileNet_V2_Weights","mobilenet_v3_small":"MobileNet_V3_Small_Weights",
        "mobilenet_v3_large":"MobileNet_V3_Large_Weights",
        "efficientnet_b0":"EfficientNet_B0_Weights","efficientnet_b1":"EfficientNet_B1_Weights",
    }
    enum_name = aliases.get(model_name)
    if enum_name and hasattr(tvm, enum_name):
        enum = getattr(tvm, enum_name)
        return getattr(enum, "DEFAULT", (next(iter(enum)) if len(enum) else None))
    # Fuzzy fallback
    norm = model_name.replace("_","").lower()
    for attr in dir(tvm):
        if attr.endswith("_Weights"):
            cand = attr.replace("_","").lower()
            if norm in cand:
                enum = getattr(tvm, attr)
                return getattr(enum, "DEFAULT", (next(iter(enum)) if len(enum) else None))
    return None

def test_models(suite: TestSuite, activation: str = "none", models: Optional[str] = None, num_layers: int = 5, pretrained: bool = False):
    try:
        from torchvision import models as tvm  # type: ignore
    except Exception:
        print("  (skip model tests: torchvision not available)")
        return

    print("\n" + "="*80)
    print("======================== MODEL TESTING =========================")
    print("="*80)

    model_names = [m.strip() for m in (models.split(",") if models else ["resnet18","resnet50"])]

    for name in model_names:
        weights = _resolve_torchvision_weights(tvm, name, pretrained)
        if weights is not None:
            print(f"  (using torchvision pretrained weights for {name}: {weights})")
        # Construct
        try:
            net = getattr(tvm, name)(weights=weights).to(device=suite.device).eval()
        except Exception as e:
            print(f"  (skip {name}: could not construct model: {e})")
            continue

        conv_layers: List[torch.nn.Conv2d] = []
        def hook(mod, inp, out):
            if isinstance(mod, torch.nn.Conv2d):
                conv_layers.append(mod)
        regs = []
        net.apply(lambda m: regs.append(m.register_forward_hook(hook)))
        with torch.no_grad():
            dummy = torch.randn(1,3,224,224, device=suite.device)
            net(dummy)
        for h in regs: h.remove()

        if not conv_layers:
            print(f"  (skip {name}: no Conv2d layers)")
            continue

        limit = min(num_layers, len(conv_layers))
        for li, conv in enumerate(conv_layers[:limit]):
            N = 2; H=W=56
            C = conv.in_channels
            stride   = tuple(conv.stride)   if isinstance(conv.stride, tuple)   else (conv.stride, conv.stride)
            padding  = tuple(conv.padding)  if isinstance(conv.padding, tuple)  else (conv.padding, conv.padding)
            dilation = tuple(conv.dilation) if isinstance(conv.dilation, tuple) else (conv.dilation, conv.dilation)
            K_out = conv.out_channels
            if isinstance(conv.kernel_size, tuple): R,S = conv.kernel_size
            else: R = S = conv.kernel_size

            x = torch.randn((N,C,H,W), device=suite.device, dtype=suite.dtype)
            w = conv.weight.to(dtype=suite.dtype, device=suite.device)
            b = (conv.bias.to(dtype=suite.dtype, device=suite.device)
                 if conv.bias is not None else torch.zeros((K_out,), device=suite.device, dtype=suite.dtype))

            y_ref = F.conv2d(x,w,b, stride=stride, padding=padding, dilation=dilation)
            y_tri = conv2d_nchw(x,w,b, stride,padding,dilation, activation="none", out_dtype=suite.dtype)

            if suite.print_shapes:
                P,Q = _out_hw(H,W,R,S, stride,padding,dilation)
                print(f"    {name} L{li}: X{tuple(x.shape)} W{tuple(w.shape)} -> Y{(N,K_out,P,Q)}")

            suite.check_close(f"{name} L{li}", y_tri, y_ref, K_red=C*R*S)

            if suite.fp32_audit:
                y_ref32 = F.conv2d(x.float(), w.float(), b.float(), stride=stride, padding=padding, dilation=dilation)
                y_tri32 = conv2d_nchw(x.float(), w.float(), b.float(), stride, padding, dilation, activation="none", out_dtype=torch.float32)
                suite.check_close(f"{name} L{li} (FP32 compare)", y_tri32, y_ref32, K_red=C*R*S, rtol=1e-6, atol=1e-6)

            if suite.bench_enabled:
                run_bench_case(suite, x,w,b, stride,padding,dilation, "none", f"{name} L{li}")


def pick_torch_dtype(d: str):
    if d == "fp16": return torch.float16
    if d == "bf16": return torch.bfloat16
    raise ValueError(d)

def main():
    p = argparse.ArgumentParser(description="Conv2D Triton test + bench suite (RDNA4 persistent + double-buffer)")
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16","bf16"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--test-mode", type=str, required=True,
                   choices=["edge","random","stability","activations","models","all"])
    p.add_argument("--num-random", type=int, default=200)
    p.add_argument("--models", type=str, default=None)
    p.add_argument("--num-layers", type=int, default=5, help="max conv layers per model for --test-mode models")
    p.add_argument("--pretrained", action="store_true", help="use torchvision pretrained weights (if available)")
    p.add_argument("--bench", action="store_true", help="print TFLOPS per case and summary")
    p.add_argument("--no-print-shapes", action="store_true", help="disable per-case shape prints")
    p.add_argument("--fp32-audit", action="store_true", help="add FP32 reference comparisons")
    args = p.parse_args()

    torch.set_grad_enabled(False)
    device = args.device
    dtype = pick_torch_dtype(args.dtype)
    backend = "CUDA" if torch.version.cuda is not None else "HIP"
    print(f"Backend: {backend} | torch device: {device} | dtype: {dtype}")

    suite = TestSuite(device=device, dtype=dtype, bench_enabled=args.bench,
                      print_shapes=not args.no_print_shapes, fp32_audit=args.fp32_audit)

    if args.test_mode in ("edge","all"):
        test_edge_cases(suite)
    if args.test_mode in ("random","all"):
        test_random_fuzzing(suite, num_tests=args.num_random)
    if args.test_mode in ("stability","all"):
        test_random_fuzzing(suite, num_tests=20)
    if args.test_mode in ("activations","all"):
        N,C,H,W = 2,16,32,32
        K_out,R,S = 32,3,3
        stride,padding,dilation = (1,1),(1,1),(1,1)
        x = torch.randn((N,C,H,W), device=device, dtype=dtype)
        w = torch.randn((K_out,C,R,S), device=device, dtype=dtype)
        b = torch.randn((K_out,), device=device, dtype=dtype)
        for act in ["none","relu","relu6","gelu"]:
            y_ref = apply_activation(F.conv2d(x,w,b.to(dtype), stride=stride, padding=padding, dilation=dilation), act)
            y_tri = conv2d_nchw(x,w,b, stride,padding,dilation, activation=act, out_dtype=dtype)
            suite.check_close(f"act={act} / NCHW", y_tri, y_ref, K_red=C*R*S)
            if args.bench:
                run_bench_case(suite, x,w,b, stride,padding,dilation, act, f"act={act}")
    if args.test_mode in ("models","all"):
        test_models(suite, models=args.models, num_layers=args.num_layers, pretrained=args.pretrained)

    ok = suite.summary()
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
