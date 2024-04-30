use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub enum Message {
    // from client to server
    GeneratePrompt(String),
    RequestCurrentGeneratedLines,

    // from server to client
    GenerationDone(GenerationResults),
    CurrentGeneratedLinesResponse(Vec<String>)
}

#[derive(Serialize, Deserialize)]
pub struct GenerationResults {
    pub was_terminated: bool,
    pub full_generated_lines: Vec<String>,
    pub feed_prompt_dur_ms: u128,
    pub predict_dur_ms: u128,
    pub predict_tokens: usize,
}

impl GenerationResults {
    pub fn create_inference_stats_array(&self, total_topics_gen: i32) -> Vec<f32> {
        let mut res = Vec::new();
        res.push(total_topics_gen as f32);
        res.push(self.feed_prompt_dur_ms as f32);
        res.push(self.predict_dur_ms as f32);
        res.push(self.predict_dur_ms as f32 / self.predict_tokens as f32);

        res
    }
}
