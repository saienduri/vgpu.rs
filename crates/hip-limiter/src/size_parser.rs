/// Parses a human-readable memory size string into bytes.
///
/// Supports:
/// - Plain bytes: "137438953472"
/// - SI suffixes (powers of 1000): "126G", "126GB", "126M", "126MB"
/// - Binary suffixes (powers of 1024): "126GiB", "126MiB"
/// - Fractional values with suffixes: "1.5G", "0.5GiB"
/// - Case-insensitive
///
/// Returns `None` for zero, overflow, or unparseable input.
pub(crate) fn parse_memory_limit(s: &str) -> Option<u64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }

    // Find where the numeric part ends and the suffix begins.
    // Characters like '-' are not digits, so negative inputs produce an empty
    // numeric part which fails to parse below.
    let numeric_end = s
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(s.len());

    let (num_str, suffix) = s.split_at(numeric_end);
    let suffix = suffix.trim().to_lowercase();

    let multiplier: u64 = match suffix.as_str() {
        "" | "b" => 1,
        "k" | "kb" => 1_000,
        "kib" => 1_024,
        "m" | "mb" => 1_000_000,
        "mib" => 1_048_576,
        "g" | "gb" => 1_000_000_000,
        "gib" => 1_073_741_824,
        "t" | "tb" => 1_000_000_000_000,
        "tib" => 1_099_511_627_776,
        _ => return None,
    };

    // For integer inputs with multiplier=1 (plain bytes or "B" suffix),
    // parse as u64 directly to avoid f64 precision loss on large values.
    let is_integer = !num_str.contains('.');
    if is_integer && multiplier == 1 {
        return num_str.parse::<u64>().ok().filter(|&v| v > 0);
    }

    // For suffixed or fractional values, use f64 (precision is fine because
    // the numeric part is small — the multiplier carries the magnitude).
    let value: f64 = num_str.parse().ok()?;
    if value <= 0.0 || !value.is_finite() {
        return None;
    }

    let bytes = value * multiplier as f64;
    if bytes > u64::MAX as f64 || bytes < 1.0 {
        return None;
    }

    Some(bytes as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plain_bytes() {
        assert_eq!(parse_memory_limit("137438953472"), Some(137438953472));
        assert_eq!(parse_memory_limit("1024"), Some(1024));
        assert_eq!(parse_memory_limit("1024B"), Some(1024));
    }

    #[test]
    fn test_plain_bytes_exact_precision() {
        // Plain-byte inputs use u64 parsing, so large values are exact
        assert_eq!(
            parse_memory_limit("9007199254740993"),
            Some(9_007_199_254_740_993)
        ); // 2^53 + 1
    }

    #[test]
    fn test_si_suffixes() {
        assert_eq!(parse_memory_limit("126G"), Some(126_000_000_000));
        assert_eq!(parse_memory_limit("126GB"), Some(126_000_000_000));
        assert_eq!(parse_memory_limit("512M"), Some(512_000_000));
        assert_eq!(parse_memory_limit("512MB"), Some(512_000_000));
        assert_eq!(parse_memory_limit("1T"), Some(1_000_000_000_000));
        assert_eq!(parse_memory_limit("1TB"), Some(1_000_000_000_000));
        assert_eq!(parse_memory_limit("100K"), Some(100_000));
        assert_eq!(parse_memory_limit("100KB"), Some(100_000));
    }

    #[test]
    fn test_binary_suffixes() {
        assert_eq!(parse_memory_limit("126GiB"), Some(126 * 1_073_741_824));
        assert_eq!(parse_memory_limit("512MiB"), Some(512 * 1_048_576));
        assert_eq!(parse_memory_limit("1TiB"), Some(1_099_511_627_776));
        assert_eq!(parse_memory_limit("100KiB"), Some(100 * 1_024));
    }

    #[test]
    fn test_case_insensitive() {
        assert_eq!(parse_memory_limit("126g"), Some(126_000_000_000));
        assert_eq!(parse_memory_limit("126gb"), Some(126_000_000_000));
        assert_eq!(parse_memory_limit("126gib"), Some(126 * 1_073_741_824));
        assert_eq!(parse_memory_limit("126GIB"), Some(126 * 1_073_741_824));
    }

    #[test]
    fn test_fractional() {
        assert_eq!(parse_memory_limit("1.5G"), Some(1_500_000_000));
        assert_eq!(parse_memory_limit("0.5GiB"), Some(536_870_912));
    }

    #[test]
    fn test_invalid() {
        assert_eq!(parse_memory_limit(""), None);
        assert_eq!(parse_memory_limit("0G"), None);
        assert_eq!(parse_memory_limit("-1G"), None);
        assert_eq!(parse_memory_limit("abc"), None);
        assert_eq!(parse_memory_limit("126X"), None);
        assert_eq!(parse_memory_limit("G"), None);
        assert_eq!(parse_memory_limit("0"), None);
        assert_eq!(parse_memory_limit("0B"), None);
    }

    #[test]
    fn test_whitespace() {
        assert_eq!(parse_memory_limit("  126G  "), Some(126_000_000_000));
        assert_eq!(parse_memory_limit("126 G"), Some(126_000_000_000));
    }
}
