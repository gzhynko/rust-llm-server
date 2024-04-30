use crate::LSampleParams;

impl Default for LSampleParams {
    fn default() -> Self {
        LSampleParams {
            n_threads: 1,
            n_tok_predict: 0,
            logit_bias: Default::default(),
            top_k: 40,
            top_p: 0.95f32,
            tfs_z: 1.0,
            typical_p: 1.0,
            temp: 0.8f32,
            repeat_penalty: 1.1f32,
            repeat_last_n: 64,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            mirostat: 0, // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
            mirostat_tau: 5.0, // target entropy
            mirostat_eta: 0.1, // learning rate
            penalize_nl: true, // consider newlines as a repeatable token
            stop_sequence: vec![],
        }
    }
}
