use std::collections::HashMap;
use llama_cpp_sys;
use std::path::PathBuf;

mod llama_context;
mod llama_context_config;
mod llama_error;
mod llama_sample_params;
mod llama_token;
mod llama_token_sequence;

pub use self::llama_error::LError;

/// You construct a context using these parameters
pub struct LContextConfig {
    model_path: PathBuf,
    pub n_ctx: i32,
    pub n_gpu_layers: i32,
    pub seed: i32,
    pub f16_kv: bool,
    pub logits_all: bool,
    pub vocab_only: bool,
    pub use_mlock: bool,
    pub use_mmap: bool,
    pub embedding: bool,
}

/// Parameters for sampling the context
#[derive(Clone, Debug)]
pub struct LSampleParams {
    pub n_threads: i32,
    pub n_tok_predict: i32,
    pub logit_bias: HashMap<i32, f32>,
    pub top_k: i32,
    pub top_p: f32,
    pub tfs_z: f32,
    pub typical_p: f32,
    pub temp: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub mirostat: i32,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub penalize_nl: bool,
    pub stop_sequence: Vec<String>,
}

/// A context contains the loaded model
pub struct LContext {
    pub(crate) config: LContextConfig,
    pub(crate) ctx: *mut llama_cpp_sys::llama_context,
}

/// A text sequence is represented as a sequence of tokens for inference.
/// A `Context` can convert a token into the associated text sequence.
#[derive(Clone)]
pub enum LToken {
    BeginningOfStream,
    EndOfStream,
    Token(llama_cpp_sys::llama_token),
}

/// A set of tokens representing a block of text.
#[derive(Clone)]
pub struct LTokenSequence {
    pub(crate) tokens: Vec<llama_cpp_sys::llama_token>,
}
