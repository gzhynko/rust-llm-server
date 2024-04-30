use crate::domain::LTokenSequence;
use crate::{LContext, LContextConfig, LError, LSampleParams, LToken};
use std::ffi::{c_char, CString};

impl LContext {
    pub fn new(mut config: LContextConfig) -> Result<LContext, LError> {
        let model_path = config.model_path.to_string_lossy();
        let model_path_c = CString::new(model_path.as_ref())?;
        let context = unsafe {
            let params = config.native_ptr();
            let ctx = llama_cpp_sys::llama_init_from_file(model_path_c.as_ptr(), params);

            LContext { config, ctx }
        };

        Ok(context)
    }

    /// Convert a string into a token sequence object.
    pub fn tokenize(&self, value: &str, add_bos: bool) -> Result<LTokenSequence, LError> {
        let mut tokens = LTokenSequence::new();

        // We need to allocate enough space for the entire value to fit into the token space.
        // Since a token can be 0..n in length, we allocate the maximum possible length and
        // shrink afterwards.
        tokens.resize(value.bytes().len() + add_bos as usize);

        // Use the context to generate tokens for the input sequence.
        unsafe {
            let ctx = self.native_ptr();
            let value_c = CString::new(value)?;
            let token_count = llama_cpp_sys::llama_tokenize(
                ctx,
                value_c.as_ptr() as *const c_char,
                tokens.native_mut_ptr(),
                tokens.capacity() as i32,
                add_bos,
            );
            if token_count < 0 {
                return Err(LError::TokenizationError(format!("failed to tokenize string; context returned {} tokens for a string of length {}", token_count, value.len())));
            }

            // Shrink buffer to actual token count
            tokens.resize(token_count as usize);
        };

        Ok(tokens)
    }

    pub fn llama_get_logits_as_slice(&self, n_tokens: usize, n_vocab: usize) -> Vec<f32> {
        let len = n_tokens * n_vocab;
        unsafe { std::slice::from_raw_parts_mut(llama_cpp_sys::llama_get_logits(self.ctx), len) }.to_vec()
    }

    pub fn llama_n_vocab(&self) -> i32 {
        unsafe { llama_cpp_sys::llama_n_vocab(self.ctx) }
    }

    /// Load a sequence of tokens into the context
    pub fn load_prompt(
        &mut self,
        prompt: &LTokenSequence,
        params: &LSampleParams,
    ) -> Result<(), LError> {
        self.eval(prompt, prompt.len() as i32, 0, params)
    }

    pub fn sample(
        &self,
        n_ctx: i32,
        seq: &LTokenSequence,
        params: &LSampleParams,
    ) -> Result<LToken, LError> {
        let top_k = if params.top_k <= 0 {
            self.llama_n_vocab()
        } else {
            params.top_k
        };
        let repeat_last_n = if params.repeat_last_n < 0 {
            n_ctx
        } else {
            params.repeat_last_n
        };

        let n_vocab = self.llama_n_vocab() as usize;
        // only get the last row, as the sample only requires this.
        let mut logits = self.llama_get_logits_as_slice(1, n_vocab);

        params
            .logit_bias
            .iter()
            .for_each(|(k, v)| logits[*k as usize] += v);
        let mut candidates: Vec<llama_cpp_sys::llama_token_data> = Vec::with_capacity(n_vocab);
        (0..n_vocab).for_each(|i| {
            candidates.push(llama_cpp_sys::llama_token_data {
                id: i as i32,
                logit: logits[i],
                p: params.top_p,
            })
        });
        let mut candidates_p = llama_cpp_sys::llama_token_data_array {
            data: candidates.as_mut_ptr(),
            size: candidates.len(),
            sorted: false,
        };
        let nl_logit = logits[unsafe { llama_cpp_sys::llama_token_nl() } as usize];
        let last_n_repeat = i32::min(i32::min(seq.len() as i32, repeat_last_n), n_ctx) as usize;

        unsafe {
            llama_cpp_sys::llama_sample_repetition_penalty(
                self.ctx,
                &mut candidates_p,
                seq.native_ptr_offset((seq.len() - last_n_repeat) as usize),
                last_n_repeat,
                params.repeat_penalty,
            )
        };
        unsafe {
            llama_cpp_sys::llama_sample_frequency_and_presence_penalties(
                self.ctx,
                &mut candidates_p,
                seq.native_ptr_offset((seq.len() - last_n_repeat) as usize),
                last_n_repeat,
                params.frequency_penalty,
                params.presence_penalty,
            )
        };
        if !params.penalize_nl {
            logits[unsafe { llama_cpp_sys::llama_token_nl() as usize }] = nl_logit;
        }

        let id = if params.temp <= 0.0 {
            // Greedy sampling
            unsafe { llama_cpp_sys::llama_sample_token_greedy(self.ctx, &mut candidates_p) }
        } else if params.mirostat == 1 {
            let mut mirostat_mu = 2.0 * params.mirostat_tau;
            let mirostat_m = 100_i32;
            unsafe { llama_cpp_sys::llama_sample_temperature(self.ctx, &mut candidates_p, params.temp) };
            unsafe {
                llama_cpp_sys::llama_sample_token_mirostat(
                    self.ctx,
                    &mut candidates_p,
                    params.mirostat_tau,
                    params.mirostat_eta,
                    mirostat_m,
                    &mut mirostat_mu,
                )
            }
        } else if params.mirostat == 2 {
            let mut mirostat_mu = 2.0 * params.mirostat_tau;
            unsafe { llama_cpp_sys::llama_sample_temperature(self.ctx, &mut candidates_p, params.temp) };
            unsafe {
                llama_cpp_sys::llama_sample_token_mirostat_v2(
                    self.ctx,
                    &mut candidates_p,
                    params.mirostat_tau,
                    params.mirostat_eta,
                    &mut mirostat_mu,
                )
            }
        } else {
            // Temperature sampling
            unsafe { llama_cpp_sys::llama_sample_top_k(self.ctx, &mut candidates_p, top_k, 1) };
            unsafe { llama_cpp_sys::llama_sample_tail_free(self.ctx, &mut candidates_p, params.tfs_z, 1) };
            unsafe { llama_cpp_sys::llama_sample_typical(self.ctx, &mut candidates_p, params.typical_p, 1) };
            unsafe { llama_cpp_sys::llama_sample_top_p(self.ctx, &mut candidates_p, params.top_p, 1) };
            unsafe { llama_cpp_sys::llama_sample_temperature(self.ctx, &mut candidates_p, params.temp) };
            unsafe { llama_cpp_sys::llama_sample_token(self.ctx, &mut candidates_p) }
        };

        Ok(LToken::Token(id))
    }

    pub fn eval(
        &self,
        seq: &LTokenSequence,
        n_tokens: i32,
        n_past: i32,
        params: &LSampleParams,
    ) -> Result<(), LError> {
        let res =
            unsafe { llama_cpp_sys::llama_eval(self.ctx, seq.native_ptr(), n_tokens, n_past, params.n_threads) };
        if res == 0 {
            Ok(())
        } else {
            Err(LError::EvalError(res))
        }
    }

    pub fn free_native_ctx(&self) {
        unsafe {
            llama_cpp_sys::llama_free(self.ctx);
        }
    }

    pub(crate) unsafe fn native_ptr(&self) -> *mut llama_cpp_sys::llama_context {
        self.ctx
    }
}
