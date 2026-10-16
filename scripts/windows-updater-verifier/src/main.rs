//! Build-only verification of the same key/global signature/version contract as the updater.
use base64::{engine::general_purpose::STANDARD, Engine};
use minisign_verify::{PublicKey, Signature};
use serde_json::Value;
use std::{env, fs, io::Read, path::Path};

fn bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let source = fs::File::open(path)?;
    let size = source.metadata()?.len();
    if size == 0 || size > maximum {
        return Err("Updater verification input exceeds its bounded size".into());
    }
    let mut value = Vec::with_capacity(size as usize);
    source.take(maximum + 1).read_to_end(&mut value)?;
    if value.len() as u64 != size {
        return Err("Input changed while reading".into());
    }
    Ok(value)
}

fn signed_version(comment: &str, expected: &str) -> bool {
    let versions: Vec<_> = comment
        .split('\t')
        .filter_map(|v| v.strip_prefix("version:"))
        .collect();
    versions == [expected]
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("Usage: verifier installer signature tauri-config".into());
    }
    let config: Value = serde_json::from_slice(&bounded(Path::new(&args[2]), 1024 * 1024)?)?;
    let wrapped = bounded(Path::new(&args[1]), 8192)?;
    let data = bounded(Path::new(&args[0]), 256 * 1024 * 1024)?;
    verify(&data, std::str::from_utf8(&wrapped)?, &config)?;
    println!("Verified final installer signature, configured public key and signed app version.");
    Ok(())
}

fn verify(data: &[u8], wrapped: &str, config: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let policy = &config["plugins"]["updater"];
    if policy["requireSignedVersion"] != true || policy["allowDowngrades"] == true {
        return Err("Compiled updater version policy differs".into());
    }
    let key = STANDARD.decode(
        policy["pubkey"]
            .as_str()
            .ok_or("Missing updater public key")?
            .trim(),
    )?;
    let key = PublicKey::decode(std::str::from_utf8(&key)?)?;
    let signature = STANDARD.decode(wrapped.trim())?;
    let signature = Signature::decode(std::str::from_utf8(&signature)?)?;
    key.verify(data, &signature, true)?;
    // The trusted comment is meaningful only after global signature verification.
    if !signed_version(
        signature.trusted_comment(),
        config["version"].as_str().ok_or("Missing version")?,
    ) {
        return Err("Updater signature is not bound to this exact app version".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn final_bytes_key_and_authenticated_version_are_all_required() {
        // Public-only fixture from a disposable OpenSSL Ed25519 key, destroyed
        // after signing this fixed message and its Minisign trusted comment.
        let key = "RWR0ZXN0LWtlebidQBLAi8eQ4bJZu5MYWlp2yDxzt3cAtUZVaYubefdR";
        let raw = concat!("untrusted comment: disposable test signature\n",
            "RUR0ZXN0LWtleQt96XcBaVUanyuXqLagZVdl89zGz/9aFU9rR2bB54NqiVg4dVs2y8H6MPojqH5JMSUXmFzw4/sKJHueHmKEXA0=\n",
            "trusted comment: timestamp:1\tfile:Speakerdesk_0.7.5_x64-setup.exe\tversion:0.7.5\n",
            "9EDW2uGxrMvxhNLbbv8VsPjiLkr72auwTxkbGTiUybeWl36sPvIqME8cD9Fhd9Mtgzwuvsuct8jzPEFlNV+0CQ==\n");
        let data = b"final-authenticode-installer-test-bytes";
        let mut config = serde_json::json!({"version":"0.7.5","plugins":{"updater":{
            "pubkey":STANDARD.encode(format!("untrusted comment: disposable test key\n{key}")),
            "requireSignedVersion":true,"allowDowngrades":false}}});
        verify(data, &STANDARD.encode(raw), &config).unwrap();
        let mut changed = data.to_vec();
        changed[0] ^= 1;
        assert!(verify(&changed, &STANDARD.encode(raw), &config).is_err());
        // Keep the key ID: this must fail cryptography, not merely key lookup.
        let mut wrong_key = STANDARD.decode(key).unwrap();
        wrong_key[10] ^= 1;
        config["plugins"]["updater"]["pubkey"] = Value::String(STANDARD.encode(format!(
            "untrusted comment: wrong test key\n{}",
            STANDARD.encode(wrong_key)
        )));
        assert!(verify(data, &STANDARD.encode(raw), &config).is_err());
        config["plugins"]["updater"]["pubkey"] = Value::String(
            STANDARD.encode(format!("untrusted comment: disposable test key\n{key}")),
        );
        config["version"] = Value::String("0.7.6".into());
        assert!(verify(data, &STANDARD.encode(raw), &config).is_err());
        // Faking a matching version cannot bypass the authenticated comment.
        assert!(verify(
            data,
            &STANDARD.encode(raw.replace("version:0.7.5", "version:0.7.6")),
            &config
        )
        .is_err());
    }

    #[test]
    fn rejects_missing_crossed_or_ambiguous_signed_versions() {
        assert!(signed_version(
            "timestamp:1\tfile:setup.exe\tversion:0.7.5",
            "0.7.5"
        ));
        for comment in [
            "timestamp:1\tfile:setup.exe",
            "version:0.7.4",
            "file:version:0.7.5",
            "version:0.7.5\tversion:0.7.4",
        ] {
            assert!(!signed_version(comment, "0.7.5"));
        }
    }
}
