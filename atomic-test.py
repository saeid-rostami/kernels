# test.py
import sys
import numpy as np
import torch
import triton
import triton.language as tl
import pytest
from numpy.random import RandomState


FORCED_CTAS = 1
DTYPE_NAME = "float16"          
TORCH_DTYPE = torch.float16
TL_DTYPE    = tl.float16


def to_triton(np_array: np.ndarray, device: torch.device) -> torch.Tensor:
    t = torch.as_tensor(np_array)
    return t.to(device=device, dtype=TORCH_DTYPE)

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


@pytest.mark.parametrize(
    "num_ctas, dtype_x_str, check_return_val",
    [(FORCED_CTAS, DTYPE_NAME, rv) for rv in [True]],
)
def test_scalar_atomic_rmw(num_ctas, dtype_x_str, check_return_val):
   
    assert num_ctas == 1
    assert dtype_x_str == "float16"
    print("TEST START")

    @triton.jit
    def _scalar_atomic_add_kernel(
        Z, A, B, OLD,
        DTYPE: tl.constexpr,
        RETURN_VAL: tl.constexpr,
    ):
        pid = tl.program_id(0)

       
        a = tl.load(A)
        b = tl.load(B)

       
        a = a.to(tl.float16)
        b = b.to(tl.float16)

        val = a + b
        old = tl.atomic_add(Z, val)

       
        if RETURN_VAL:
            tl.store(OLD, old, mask=pid == 0)

    
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rs = RandomState(17)
    
    def rand_scalar_fp16() -> np.ndarray:
        
        v = rs.uniform(-1.0, 1.0)
        return np.array(v, dtype=np.float16)

    a  = rand_scalar_fp16()
    b  = rand_scalar_fp16()
    z0 = rand_scalar_fp16() 
    
    z_ref_f32 = z0.astype(np.float32) + (a.astype(np.float32) + b.astype(np.float32)) * 1
    z_ref  = z_ref_f32.astype(np.float16)
    old_ref = z0.astype(np.float16)

    
    Z_tri   = to_triton(np.array([z0], dtype=np.float16), device=dev)
    A_tri   = to_triton(np.array([a],  dtype=np.float16), device=dev)
    B_tri   = to_triton(np.array([b],  dtype=np.float16), device=dev)
    OLD_tri = to_triton(np.array([0],  dtype=np.float16), device=dev)

    
    _scalar_atomic_add_kernel[(1,)](
        Z_tri, A_tri, B_tri, OLD_tri,
        TL_DTYPE,
        check_return_val,
        num_ctas=1,
    )

    
    z_out = to_numpy(Z_tri)
    np.testing.assert_allclose(z_ref.reshape(1), z_out, rtol=1e-3, atol=1e-3)
    if check_return_val:
        np.testing.assert_equal(old_ref.reshape(1), to_numpy(OLD_tri))


if __name__ == "__main__":
    sys.exit(pytest.main(["-q", __file__, "-k", "scalar_atomic_rmw"]))
