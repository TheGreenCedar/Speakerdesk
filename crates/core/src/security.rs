/// Header required on POST, PUT, PATCH, and DELETE.
pub const TOKEN_HEADER: &str = "X-Speakerdesk-Token";

pub const CONTENT_SECURITY_POLICY: &str = "\
default-src 'self'; style-src 'self'; script-src 'self'; media-src 'self'; \
img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'";

pub const X_CONTENT_TYPE_OPTIONS: &str = "nosniff";
pub const REFERRER_POLICY: &str = "no-referrer";
pub const CACHE_CONTROL: &str = "no-store";

pub const TRUSTED_HOSTS: &[&str] = &["localhost", "127.0.0.1", "[::1]"];

pub const MAX_UPLOAD_BYTES: usize = 256 * 1024 * 1024;

pub fn host_allowed(host: &str) -> bool {
    let name = if host.starts_with('[') {
        match host.find(']') {
            Some(end) => &host[..=end],
            None => host,
        }
    } else {
        host.split_once(':').map(|(name, _)| name).unwrap_or(host)
    };
    TRUSTED_HOSTS.contains(&name)
}

/// Constant-time compare for the urlsafe session token.
pub fn tokens_match(expected: &[u8], provided: &[u8]) -> bool {
    let mut diff = expected.len() ^ provided.len();
    let length = expected.len().max(provided.len());
    for index in 0..length {
        let left = expected.get(index).copied().unwrap_or(0);
        let right = provided.get(index).copied().unwrap_or(0);
        diff |= usize::from(left ^ right);
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::{CONTENT_SECURITY_POLICY, host_allowed, tokens_match};

    #[test]
    fn token_compare_rejects_a_changed_byte() {
        assert!(tokens_match(b"abc", b"abc"));
        assert!(!tokens_match(b"abc", b"abd"));
        assert!(!tokens_match(b"abc", b"ab"));
    }

    #[test]
    fn host_allowlist_matches_v0_6_6() {
        assert!(host_allowed("localhost"));
        assert!(host_allowed("127.0.0.1:8790"));
        assert!(host_allowed("[::1]"));
        assert!(host_allowed("[::1]:8790"));
        assert!(!host_allowed("example.com"));
    }

    #[test]
    fn csp_is_the_v0_6_6_policy() {
        assert!(CONTENT_SECURITY_POLICY.contains("frame-ancestors 'none'"));
        assert!(CONTENT_SECURITY_POLICY.contains("base-uri 'none'"));
    }
}
