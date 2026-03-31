use std::ffi::c_char;

static XLAI_VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// Returns the current `xlai` package version as a static C string.
///
/// The returned pointer is valid for the lifetime of the process and must not
/// be freed by the caller.
#[unsafe(no_mangle)]
pub extern "C" fn xlai_version() -> *const c_char {
    XLAI_VERSION.as_ptr().cast()
}

#[cfg(test)]
mod tests {
    use std::ffi::CStr;

    use super::xlai_version;

    #[test]
    fn xlai_version_matches_cargo_package_version() {
        let version = unsafe {
            // SAFETY: `xlai_version` returns a non-null pointer to a
            // static, NUL-terminated string that lives for the process lifetime.
            CStr::from_ptr(xlai_version())
        };

        assert_eq!(version.to_str().ok(), Some(env!("CARGO_PKG_VERSION")));
    }
}
