use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::IdError;

/// 32-character lowercase hex id, matching `uuid.uuid4().hex`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct JobId(String);

impl JobId {
    pub fn parse(value: impl AsRef<str>) -> Result<Self, IdError> {
        let value = value.as_ref();
        if value.len() == 32
            && value
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            Ok(Self(value.to_string()))
        } else {
            Err(IdError::JobId)
        }
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for JobId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for JobId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::parse(value).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::JobId;

    #[test]
    fn accepts_uuid4_hex() {
        let id = JobId::parse("0123456789abcdef0123456789abcdef").unwrap();
        assert_eq!(id.as_str(), "0123456789abcdef0123456789abcdef");
    }

    #[test]
    fn rejects_uppercase_and_short_ids() {
        assert!(JobId::parse("0123456789ABCDEF0123456789ABCDEF").is_err());
        assert!(JobId::parse("abc").is_err());
    }
}
