use crate::{LContext, LError, LSampleParams, LToken};

pub struct LGenerator {
    context: LContext,
}

impl LGenerator {
    pub fn new(context: LContext) -> LGenerator {
        LGenerator { context }
    }

    fn generate_no_op(_value: &str) -> bool { false }

    pub fn generate(&mut self, prompt: &str, params: LSampleParams) -> Result<String, LError> {
        self.generate_internal(prompt, params, LGenerator::generate_no_op)
    }

    pub fn generate_incremental(
        &mut self,
        prompt: &str,
        params: LSampleParams,
        callback: impl FnMut(&str) -> bool,
    ) -> Result<String, LError> {
        self.generate_internal(prompt, params, callback)
    }

    pub fn generate_internal(
        &mut self,
        prompt: &str,
        params: LSampleParams,
        mut callback: impl FnMut(&str) -> bool,
    ) -> Result<String, LError> {
        let context_size = self.context.config.n_ctx as usize;

        // Load prompt
        let tokenized_input = self.context.tokenize(prompt, true)?;

        // Tokenize the first stop sequence (TODO: support multiple stop sequences)
        let tokenized_stop_prompt = self.context.tokenize(
            params
                .stop_sequence
                .first()
                .map(|x| x.as_str())
                .unwrap_or("\n\n"),
            false,
        ).unwrap();

        // Feed the prompt
        self.context
            .load_prompt(&tokenized_input, &params).unwrap();

        // Embd contains the prompt and the completion. The longer the prompt, the shorter the completion.
        let mut embd = tokenized_input.clone();

        let mut token_strings = Vec::new();
        let mut n_remaining = context_size - tokenized_input.len();
        let mut n_used = tokenized_input.len() as i32 - 1;
        let stop_sequence_i = 0;
        embd.resize(context_size);
        while n_remaining > 0 {
            // Sample result
            let token = self
                .context
                .sample(context_size as i32, &embd, &params)?;

            n_used += 1;
            n_remaining -= 1;
            embd.tokens[n_used as usize] = unsafe { token.native_value() };
            if matches!(token, LToken::EndOfStream) {
                break;
            }

            self.context.eval(&embd.slice(n_used as usize), 1, n_used, &params).unwrap();

            // Incremental completion callback
            if token.has_str_value() {
                let token_string = token.as_string(&self.context)?;
                if token_string.contains("#") {
                    break;
                }

                let should_terminate = callback(&token_string);
                if should_terminate {
                    break;
                }
                token_strings.push(token_string);
            }
        }

        // print timings
        unsafe { llama_cpp_sys::llama_print_timings(self.context.ctx); }

        // Convert token stream back into a string
        Ok(token_strings.join(""))
    }

    pub fn free(self) {
        self.context.free_native_ctx();
        drop(self);
    }
}
