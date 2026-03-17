"""
FFT Compatibility Tests — Validates that the limiter does not break rocFFT JIT compilation.

Regression tests for: logging::init() during .init_array corrupts HIP/ROCr internal state,
causing rocFFT's runtime kernel compilation to fail with HIPFFT_PARSE_ERROR.

Key behaviors under test:
  - hipfft C2C (forward + inverse) succeeds with limiter loaded
  - hipfft R2C/C2R transform succeeds with limiter loaded (different JIT path)
  - hipfft 2D transform succeeds with limiter loaded (different plan type)
  - hipfft batch transform succeeds with limiter loaded
  - hipRTC kernel compilation succeeds with limiter loaded
  - FFT works after memory pressure (alloc/free cycle before FFT)
"""

import ctypes
import ctypes.util

import pytest

from conftest import requires_gpu

# hipfftType constants — shared across subprocess scripts via f-string interpolation.
_HIPFFT_C2C = 0x29
_HIPFFT_R2C = 0x2A


def _lib_loadable(name: str) -> bool:
    """Check if a shared library is available without loading it into this process.

    Uses ctypes.util.find_library (which searches ldconfig cache / LD_LIBRARY_PATH)
    with a RTLD_NOLOAD fallback. Avoids side-effects from actually loading HIP/ROCm
    libraries into the pytest collector process.
    """
    # find_library expects the short name (e.g., "hipfft" not "libhipfft.so")
    short_name = name.removeprefix("lib").removesuffix(".so")
    if ctypes.util.find_library(short_name) is not None:
        return True
    # Fallback: try RTLD_NOLOAD (succeeds only if already loaded by another lib)
    try:
        ctypes.CDLL(name, mode=ctypes.RTLD_NOLOAD)
        return True
    except OSError:
        return False


requires_hipfft = pytest.mark.skipif(
    not _lib_loadable("libhipfft.so"), reason="libhipfft.so not available"
)
requires_hiprtc = pytest.mark.skipif(
    not _lib_loadable("libhiprtc.so"), reason="libhiprtc.so not available"
)

pytestmark = requires_gpu


class TestFFTWithLimiter:
    """Verify rocFFT JIT compilation works with the limiter loaded via LD_PRELOAD.

    rocFFT compiles GPU kernels at runtime (via hipRTC). This is sensitive to
    HIP/ROCr internal state corruption. The limiter's .init_array constructor
    must not interfere with this process.
    """

    @requires_hipfft
    def test_fft_1d_c2c(self, cts):
        """1D complex-to-complex FFT (forward + inverse) — primary regression test."""
        result = cts.run_hip_test(f"""
            import ctypes
            from ctypes import c_void_p, c_size_t, byref

            hip = ctypes.CDLL("libamdhip64.so")
            fft = ctypes.CDLL("libhipfft.so")

            N = 16
            buf_bytes = N * 2 * 4  # complex64 = 2 x float32

            d_buf = c_void_p()
            assert hip.hipMalloc(byref(d_buf), c_size_t(buf_bytes)) == 0, "hipMalloc failed"

            plan = c_void_p()
            rc = fft.hipfftPlan1d(byref(plan), N, {_HIPFFT_C2C}, 1)
            assert rc == 0, f"hipfftPlan1d failed: {{rc}}"

            # Forward (exercises JIT compilation — the original failure point)
            rc = fft.hipfftExecC2C(plan, d_buf, d_buf, -1)
            assert rc == 0, f"forward FFT failed: {{rc}}"

            # Inverse (exercises a second JIT path)
            rc = fft.hipfftExecC2C(plan, d_buf, d_buf, 1)
            assert rc == 0, f"inverse FFT failed: {{rc}}"

            assert hip.hipDeviceSynchronize() == 0
            fft.hipfftDestroy(plan)
            hip.hipFree(d_buf)
            print("FFT_C2C_OK")
        """)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "FFT_C2C_OK" in result.stdout, f"FFT C2C failed:\n{result.output}"

    @requires_hipfft
    def test_fft_1d_r2c(self, cts):
        """1D real-to-complex FFT — exercises a different rocFFT JIT kernel path."""
        result = cts.run_hip_test(f"""
            import ctypes
            from ctypes import c_void_p, c_size_t, byref

            hip = ctypes.CDLL("libamdhip64.so")
            fft = ctypes.CDLL("libhipfft.so")

            N = 32
            input_bytes = N * 4                  # N real float32
            output_bytes = (N // 2 + 1) * 2 * 4  # (N/2+1) complex64

            d_input = c_void_p()
            d_output = c_void_p()
            assert hip.hipMalloc(byref(d_input), c_size_t(input_bytes)) == 0
            assert hip.hipMalloc(byref(d_output), c_size_t(output_bytes)) == 0

            plan = c_void_p()
            rc = fft.hipfftPlan1d(byref(plan), N, {_HIPFFT_R2C}, 1)
            assert rc == 0, f"hipfftPlan1d R2C failed: {{rc}}"

            rc = fft.hipfftExecR2C(plan, d_input, d_output)
            assert rc == 0, f"hipfftExecR2C failed: {{rc}}"

            assert hip.hipDeviceSynchronize() == 0
            fft.hipfftDestroy(plan)
            hip.hipFree(d_input)
            hip.hipFree(d_output)
            print("FFT_R2C_OK")
        """)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "FFT_R2C_OK" in result.stdout, f"FFT R2C failed:\n{result.output}"

    @requires_hipfft
    def test_fft_2d(self, cts):
        """2D FFT — exercises hipfftPlan2d, a different plan type and JIT path."""
        result = cts.run_hip_test(f"""
            import ctypes
            from ctypes import c_void_p, c_size_t, byref

            hip = ctypes.CDLL("libamdhip64.so")
            fft = ctypes.CDLL("libhipfft.so")

            NX, NY = 16, 16
            buf_bytes = NX * NY * 2 * 4  # complex64

            d_buf = c_void_p()
            assert hip.hipMalloc(byref(d_buf), c_size_t(buf_bytes)) == 0

            plan = c_void_p()
            rc = fft.hipfftPlan2d(byref(plan), NX, NY, {_HIPFFT_C2C})
            assert rc == 0, f"hipfftPlan2d failed: {{rc}}"

            rc = fft.hipfftExecC2C(plan, d_buf, d_buf, -1)
            assert rc == 0, f"2D FFT exec failed: {{rc}}"

            assert hip.hipDeviceSynchronize() == 0
            fft.hipfftDestroy(plan)
            hip.hipFree(d_buf)
            print("FFT_2D_OK")
        """)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "FFT_2D_OK" in result.stdout, f"FFT 2D failed:\n{result.output}"

    @requires_hipfft
    def test_fft_batch(self, cts):
        """Batch FFT (multiple transforms) — exercises more of rocFFT's JIT paths."""
        result = cts.run_hip_test(f"""
            import ctypes
            from ctypes import c_void_p, c_size_t, byref

            hip = ctypes.CDLL("libamdhip64.so")
            fft = ctypes.CDLL("libhipfft.so")

            N = 32
            batch = 4
            buf_bytes = N * batch * 2 * 4  # complex64

            d_buf = c_void_p()
            assert hip.hipMalloc(byref(d_buf), c_size_t(buf_bytes)) == 0

            plan = c_void_p()
            rc = fft.hipfftPlan1d(byref(plan), N, {_HIPFFT_C2C}, batch)
            assert rc == 0, f"hipfftPlan1d batch failed: {{rc}}"

            rc = fft.hipfftExecC2C(plan, d_buf, d_buf, -1)
            assert rc == 0, f"batch FFT exec failed: {{rc}}"

            assert hip.hipDeviceSynchronize() == 0
            fft.hipfftDestroy(plan)
            hip.hipFree(d_buf)
            print("FFT_BATCH_OK")
        """)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "FFT_BATCH_OK" in result.stdout, f"FFT batch failed:\n{result.output}"

    @requires_hipfft
    def test_fft_after_memory_pressure(self, cts):
        """FFT after alloc/free cycles — ensures deferred init survives memory pressure.

        Allocates ~75% of the 1 GiB limit across two rounds of alloc/free to exercise
        the limiter's reserve/free accounting before attempting FFT JIT compilation.
        """
        result = cts.run_hip_test(f"""
            import ctypes
            from ctypes import c_void_p, c_size_t, byref

            hip = ctypes.CDLL("libamdhip64.so")
            fft = ctypes.CDLL("libhipfft.so")

            CHUNK = 96 * 1024 * 1024  # 96 MiB per allocation

            # Two rounds of alloc/free to stress the accounting paths
            for round_num in range(2):
                ptrs = []
                for _ in range(4):
                    p = c_void_p()
                    assert hip.hipMalloc(byref(p), c_size_t(CHUNK)) == 0, \\
                        f"hipMalloc failed in round {{round_num}}"
                    ptrs.append(p)
                # ~384 MiB allocated (of 1 GiB limit)
                for p in ptrs:
                    assert hip.hipFree(p) == 0

            # Now run FFT — rocFFT JIT must still work after the limiter has
            # processed multiple reserve/free cycles
            N = 16
            d_buf = c_void_p()
            assert hip.hipMalloc(byref(d_buf), c_size_t(N * 2 * 4)) == 0

            plan = c_void_p()
            rc = fft.hipfftPlan1d(byref(plan), N, {_HIPFFT_C2C}, 1)
            assert rc == 0, f"hipfftPlan1d failed: {{rc}}"

            rc = fft.hipfftExecC2C(plan, d_buf, d_buf, -1)
            assert rc == 0, f"FFT exec after pressure failed: {{rc}}"

            assert hip.hipDeviceSynchronize() == 0
            fft.hipfftDestroy(plan)
            hip.hipFree(d_buf)
            print("FFT_PRESSURE_OK")
        """)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "FFT_PRESSURE_OK" in result.stdout, f"FFT after pressure failed:\n{result.output}"


class TestHipRTCWithLimiter:
    """Verify hipRTC runtime compilation works with the limiter loaded.

    hipRTC is the underlying mechanism rocFFT uses for JIT kernel compilation.
    Testing it directly provides a more targeted regression test.
    """

    @requires_hiprtc
    def test_hiprtc_compile_simple_kernel(self, cts):
        """Compile a trivial HIP kernel via hipRTC with the limiter loaded."""
        result = cts.run_hip_test("""
            import ctypes
            from ctypes import c_void_p, c_size_t, byref

            rtc = ctypes.CDLL("libhiprtc.so")

            source = b'''
            extern "C" __global__ void vector_add(float* a, float* b, float* c, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) c[i] = a[i] + b[i];
            }
            '''

            prog = c_void_p()
            rc = rtc.hiprtcCreateProgram(byref(prog), source, b"vector_add.hip", 0, None, None)
            assert rc == 0, f"hiprtcCreateProgram failed: {rc}"

            rc = rtc.hiprtcCompileProgram(prog, 0, None)
            if rc != 0:
                log_size = c_size_t()
                rtc.hiprtcGetProgramLogSize(prog, byref(log_size))
                log_buf = ctypes.create_string_buffer(log_size.value)
                rtc.hiprtcGetProgramLog(prog, log_buf)
                assert False, f"hiprtcCompileProgram failed ({rc}): {log_buf.value.decode()}"

            code_size = c_size_t()
            rc = rtc.hiprtcGetCodeSize(prog, byref(code_size))
            assert rc == 0 and code_size.value > 0, f"No code generated (size={code_size.value})"

            rtc.hiprtcDestroyProgram(byref(prog))
            print(f"HIPRTC_OK code_size={code_size.value}")
        """)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "HIPRTC_OK" in result.stdout, f"hipRTC compilation failed:\n{result.output}"

    @requires_hiprtc
    def test_hiprtc_compile_and_launch(self, cts):
        """Compile via hipRTC, load module, and launch kernel — full JIT pipeline."""
        result = cts.run_hip_test("""
            import ctypes
            from ctypes import c_void_p, c_size_t, c_int, c_float, byref

            hip = ctypes.CDLL("libamdhip64.so")
            rtc = ctypes.CDLL("libhiprtc.so")

            source = b'''
            extern "C" __global__ void fill_kernel(float* data, float val, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) data[i] = val;
            }
            '''

            # Compile
            prog = c_void_p()
            assert rtc.hiprtcCreateProgram(byref(prog), source, b"fill.hip", 0, None, None) == 0
            rc = rtc.hiprtcCompileProgram(prog, 0, None)
            assert rc == 0, f"compile failed: {rc}"

            # Get compiled code
            code_size = c_size_t()
            rtc.hiprtcGetCodeSize(prog, byref(code_size))
            code = ctypes.create_string_buffer(code_size.value)
            rtc.hiprtcGetCode(prog, code)
            rtc.hiprtcDestroyProgram(byref(prog))

            # Load module and get kernel function
            module = c_void_p()
            rc = hip.hipModuleLoadData(byref(module), code)
            assert rc == 0, f"hipModuleLoadData failed: {rc}"

            kernel = c_void_p()
            rc = hip.hipModuleGetFunction(byref(kernel), module, b"fill_kernel")
            assert rc == 0, f"hipModuleGetFunction failed: {rc}"

            # Allocate device memory and launch
            N = 256
            d_data = c_void_p()
            assert hip.hipMalloc(byref(d_data), c_size_t(N * 4)) == 0

            FILL_VALUE = 42.0
            val = c_float(FILL_VALUE)
            n = c_int(N)

            args = (c_void_p * 3)(
                ctypes.cast(byref(d_data), c_void_p),
                ctypes.cast(byref(val), c_void_p),
                ctypes.cast(byref(n), c_void_p),
            )

            rc = hip.hipModuleLaunchKernel(
                kernel,
                1, 1, 1,       # grid
                256, 1, 1,     # block
                0,             # shared mem
                None,          # stream
                args,          # args
                None,          # extra
            )
            assert rc == 0, f"hipModuleLaunchKernel failed: {rc}"
            assert hip.hipDeviceSynchronize() == 0

            # Read back and verify — exact float equality is correct here because
            # the kernel writes a literal float with no arithmetic (IEEE 754 exact).
            h_data = (c_float * N)()
            rc = hip.hipMemcpy(h_data, d_data, c_size_t(N * 4), 2)  # hipMemcpyDeviceToHost = 2
            assert rc == 0, f"hipMemcpy failed: {rc}"

            bad = [(i, h_data[i]) for i in range(N) if h_data[i] != FILL_VALUE]
            assert not bad, f"Kernel output mismatch at {len(bad)} indices, first 5: {bad[:5]}"

            hip.hipFree(d_data)
            hip.hipModuleUnload(module)
            print("HIPRTC_LAUNCH_OK")
        """)
        assert result.succeeded, f"Subprocess failed:\n{result.output}"
        assert "HIPRTC_LAUNCH_OK" in result.stdout, f"hipRTC launch failed:\n{result.output}"
