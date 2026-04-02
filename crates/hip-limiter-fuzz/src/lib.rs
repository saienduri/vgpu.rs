use std::sync::atomic::{AtomicU64, Ordering};

pub mod simulated_limiter;
pub mod simulated_pod;

// Re-export for existing tests.
pub use simulated_limiter::{MultiDeviceSimulatedLimiter, SimulatedLimiter};
pub use simulated_pod::SimulatedPod;

/// Atomically subtract `size` from `counter`, clamping at zero.
///
/// Uses a CAS loop to prevent underflow wrapping. Mirrors
/// `saturating_fetch_sub_pod_memory_used` on `SharedDeviceInfoV2`.
pub(crate) fn saturating_fetch_sub(counter: &AtomicU64, size: u64) {
    loop {
        let current = counter.load(Ordering::Acquire);
        let new_value = current.saturating_sub(size);
        if counter
            .compare_exchange_weak(current, new_value, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            break;
        }
    }
}
