use crate::LContextConfig;
use std::path::{Path, PathBuf};
use std::ptr;

impl LContextConfig {
    pub fn new<T: AsRef<Path>>(path: T) -> LContextConfig {
        unsafe {
            Self::from_native_ptr(path, llama_cpp_sys::llama_context_default_params())
        }
    }

    pub(crate) fn from_native_ptr<T: AsRef<Path>>(model_path: T, params: llama_cpp_sys::llama_context_params) -> Self {
        Self {
            model_path: PathBuf::from(model_path.as_ref()),
            n_ctx: params.n_ctx,
            n_gpu_layers: params.n_gpu_layers,
            seed: params.seed,
            f16_kv: params.f16_kv,
            logits_all: params.logits_all,
            vocab_only: params.vocab_only,
            use_mlock: params.use_mlock,
            use_mmap: params.use_mmap,
            embedding: params.embedding,
        }
    }

    pub(crate) unsafe fn native_ptr(&mut self) ->  llama_cpp_sys::llama_context_params {
        llama_cpp_sys::llama_context_params {
            n_ctx: self.n_ctx,
            n_gpu_layers: self.n_gpu_layers,
            seed: self.seed,
            f16_kv: self.f16_kv,
            logits_all: self.logits_all,
            vocab_only: self.vocab_only,
            use_mmap: self.use_mmap,
            use_mlock: self.use_mlock,
            embedding: self.embedding,
            progress_callback: None,
            progress_callback_user_data: ptr::null_mut(),
        }
    }
}
