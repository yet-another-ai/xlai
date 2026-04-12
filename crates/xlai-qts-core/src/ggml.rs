//! Standalone GGML FFI from `xlai_sys_ggml` (QTS talker).

pub use xlai_sys_ggml as sys;

#[cfg(test)]
mod tests {
    use super::sys;

    #[test]
    fn smoke_add_graph() {
        unsafe {
            let params = sys::ggml_init_params {
                mem_size: 16 * 1024 * 1024,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: false,
            };
            let ctx = sys::ggml_init(params);
            assert!(!ctx.is_null());

            let a = sys::ggml_new_tensor_1d(ctx, sys::ggml_type_GGML_TYPE_F32, 1);
            let b = sys::ggml_new_tensor_1d(ctx, sys::ggml_type_GGML_TYPE_F32, 1);
            assert!(!sys::ggml_set_f32(a, 2.0).is_null());
            assert!(!sys::ggml_set_f32(b, 3.0).is_null());
            let sum = sys::ggml_add(ctx, a, b);
            let gf = sys::ggml_new_graph(ctx);
            sys::ggml_build_forward_expand(gf, sum);
            let st = sys::ggml_graph_compute_with_ctx(ctx, gf, 1);
            assert_eq!(st, sys::ggml_status_GGML_STATUS_SUCCESS);
            assert!((sys::ggml_get_f32_1d(sum, 0) - 5.0).abs() < 1e-5);

            sys::ggml_free(ctx);
        }
    }
}
