/*
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use llm::KnownModel;
use rust_llm_server_common::{GenerationResults};

#[derive(Default)]
pub(crate) struct GenerationState {
    pub(crate) should_terminate: bool,
    pub(crate) is_generating: bool,
    pub(crate) generated_lines: Vec<String>,
}

pub(crate) struct LlmRunner {
    model: llm::models::Llama,
}

impl LlmRunner {
    pub(crate) fn new() -> Self {
        // load a GGML model from disk
        let llama = llm::load::<llm::models::Llama>(
            // path to GGML file
            std::path::Path::new("../models/wizard-vicuna-uncensored-7b/ggml-model.q4_0.bin"),
            llm::TokenizerSource::Embedded,
            // llm::ModelParameters
            llm::ModelParameters::default(),
            // load progress callback
            llm::load_progress_callback_stdout
        )
            .unwrap_or_else(|err| panic!("Failed to load model: {err}"));

        Self {
            model: llama,
        }
    }

    pub(crate) fn run(&self, prompt: String, gen_state: Arc<Mutex<GenerationState>>) -> GenerationResults {
        // start the inference session
        let mut current_line = String::new();
        let mut inference_session = self.model.start_session(Default::default());
        let res = inference_session.infer::<Infallible>(
            &self.model,
            &mut rand::thread_rng(),
            &llm::InferenceRequest {
                prompt: (&prompt).into(),
                parameters: &llm::InferenceParameters::default(),
                play_back_previous_tokens: false,
                maximum_token_count: None,
            },
            &mut Default::default(),
            |r| match r {
                llm::InferenceResponse::InferredToken(t) => {
                    print!("{}", t);
                    if t.contains("\n") && !current_line.is_empty() {
                        // acquire the lock, trim the line, and push it to the array
                        let mut gen_state_lock = gen_state.lock().unwrap();
                        gen_state_lock.generated_lines.push(current_line.trim().to_string());
                        current_line = String::new();
                    } else {
                        // remove all newlines, hashtags (the ai sometimes adds them for some reason),
                        // and add the new token to the current line
                        current_line += &t.replace("#", "").replace("\n", "");
                    }

                    Ok(llm::InferenceFeedback::Continue)
                },
                _ => {
                    Ok(llm::InferenceFeedback::Continue)
                },
            },
        );

        match res {
            Ok(result) => {
                // we are done generating this topic!
                let mut gen_state_lock = gen_state.lock().unwrap();
                gen_state_lock.generated_lines.push(current_line.to_string());
                gen_state_lock.is_generating = false;

                return GenerationResults {
                    was_terminated: false,
                    full_generated_lines: gen_state_lock.generated_lines.clone(),
                    feed_prompt_dur_ms: result.feed_prompt_duration.as_millis(),
                    predict_dur_ms: result.predict_duration.as_millis(),
                    predict_tokens: result.predict_tokens,
                }
            },
            Err(err) => panic!("Dialogue generation error: {}", err),
        }
    }
}
*/
