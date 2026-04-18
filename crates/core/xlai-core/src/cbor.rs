//! CBOR encode/decode helpers for IPC-sized `xlai_core` payloads.

use serde::Serialize;
use serde::de::DeserializeOwned;
use std::io::Cursor;

/// Serialize `value` to canonical CBOR bytes.
pub fn to_cbor_vec<T: Serialize>(value: &T) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(|e| e.to_string())?;
    Ok(buf)
}

/// Deserialize `T` from a CBOR byte slice.
pub fn from_cbor_slice<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    ciborium::from_reader(Cursor::new(bytes)).map_err(|e| e.to_string())
}
