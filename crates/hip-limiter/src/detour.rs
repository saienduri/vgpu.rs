pub(crate) mod mem;

/// Get the current HIP device and execute an expression with the limiter and device index.
/// Returns Result<T, Error> where T is the return type of the expression.
#[macro_export]
macro_rules! with_device {
    ($expr:expr) => {{
        let mut device: $crate::hiplib::HipDevice = 0;
        let hip_result = unsafe { ($crate::hiplib::hiplib().hip_get_device)(&mut device) };

        if hip_result != $crate::hiplib::HIP_SUCCESS {
            Err($crate::limiter::Error::Hip(hip_result))
        } else {
            match $crate::GLOBAL_LIMITER.get() {
                Some(limiter) => match limiter.device_index_by_hip_device(device) {
                    Ok(device_index) => Ok($expr(limiter, device_index)),
                    Err(error) => Err(error),
                },
                None => {
                    $crate::report_limiter_not_initialized();
                    Err($crate::limiter::Error::LimiterNotInitialized)
                }
            }
        }
    }};
}
