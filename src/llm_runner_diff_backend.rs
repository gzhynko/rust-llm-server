use std::io::Write;
use std::sync::{Arc, Mutex};
use llama_cpp_rs::{LContext, LContextConfig, LGenerator, LGeneratorParams, LSampleParams};
use rust_llm_server_common::GenerationResults;

#[derive(Default)]
pub(crate) struct GenerationState {
    pub(crate) should_terminate: bool,
    pub(crate) is_generating: bool,
    pub(crate) generated_lines: Vec<String>,
}

pub(crate) struct LlmRunner {
}

impl LlmRunner {
    pub(crate) fn new() -> Self {
        Self {
        }
    }

    pub(crate) fn run(&self, prompt: String, gen_state: Arc<Mutex<GenerationState>>) -> GenerationResults {
        let mut config = LContextConfig::new("models/wizard-vicuna-uncensored-7b/Wizard-Vicuna-7B-Uncensored.Q3_K_M.gguf");
        config.n_ctx = 1024;
        config.seed = rand::random::<u32>();

        let context = LContext::new(config).unwrap();
        let mut generator = LGenerator::new(context);

        let mut current_line = String::new();
        generator
            .generate_incremental(
                &prompt,
                LGeneratorParams {
                    worker_thread_count: 8,
                    sample_params: LSampleParams {
                        top_k: 40,
                        top_p: 0.75,
                        repeat_penalty: 1.1,
                        temp: 0.25,
                        repeat_history_length: 64,
                        ..LSampleParams::default()
                    },
                    generate_tokens: 1024,
                },
                |generated| {
                    let t = generated[generated.len() - 1].as_str();
                    let mut gen_state_lock = gen_state.lock().unwrap();
                    if gen_state_lock.should_terminate {
                        return false;
                    }
                    print!("{t}");
                    std::io::stdout().flush().unwrap();

                    if t.contains(&"\n".to_string()) && !current_line.is_empty() {
                        // trim the line and push it to the array
                        gen_state_lock.generated_lines.push(current_line.trim().to_string());
                        current_line = String::new();
                    } else {
                        // remove all newlines, hashtags (the ai sometimes adds them for some reason),
                        // and add the new token to the current line
                        current_line += &t.replace("#", "").replace("\n", "");
                    }

                    true
                },
            )
            .unwrap();

        // add the rest of the generated stuff as a new line and end the execution
        let mut gen_state_lock = gen_state.lock().unwrap();
        if gen_state_lock.should_terminate {
            gen_state_lock.should_terminate = false;
            gen_state_lock.is_generating = false;
            println!("terminated text gen");

            GenerationResults {
                was_terminated: true,
                full_generated_lines: gen_state_lock.generated_lines.clone(),
                feed_prompt_dur_ms: 0,
                predict_dur_ms: 0,
                predict_tokens: 0,
            }
        } else {
            gen_state_lock.generated_lines.push(current_line.trim().to_string());

            GenerationResults {
                was_terminated: false,
                full_generated_lines: gen_state_lock.generated_lines.clone(),
                feed_prompt_dur_ms: 0,
                predict_dur_ms: 0,
                predict_tokens: 0,
            }
        }
    }
}

