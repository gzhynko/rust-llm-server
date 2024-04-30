use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use message_io::network::{Endpoint, NetEvent, Transport};
use message_io::node;
use rust_llm_server_common::{Message, GenerationResults};

use crate::llm_runner_diff_backend::{GenerationState, LlmRunner};

mod llm_runner;
mod llm_runner_diff_backend;

enum LlmServerMessage {
    // from server to llm runner
    GeneratePrompt(String),
    // from llm runner to server
    PromptDone(GenerationResults),
}

fn main() {
    let gen_state = Arc::new(Mutex::new(GenerationState::default()));

    let (llm_tx, llm_rx) = channel::<LlmServerMessage>();
    let (serv_tx, serv_rx) = channel::<LlmServerMessage>();

    run_server(Arc::clone(&gen_state), serv_rx, llm_tx);
    run_llm_model(Arc::clone(&gen_state), llm_rx, serv_tx);

    loop {}
}

fn run_llm_model(gen_state: Arc<Mutex<GenerationState>>, rx: Receiver<LlmServerMessage>, tx: Sender<LlmServerMessage>) {
    let runner = LlmRunner::new();
    thread::spawn(move || {
        loop {
            let block = rx.recv().unwrap();
            match block {
                LlmServerMessage::GeneratePrompt(prompt) => {
                    println!("received prompt request: {}", prompt);

                    let mut gen_state_lock = gen_state.lock().unwrap();
                    gen_state_lock.is_generating = true;
                    gen_state_lock.generated_lines = Vec::new();
                    drop(gen_state_lock);

                    // generate the thing!
                    let gen_res = runner.run(prompt, Arc::clone(&gen_state));

                    let mut gen_state_lock = gen_state.lock().unwrap();
                    gen_state_lock.is_generating = false;
                    drop(gen_state_lock);

                    tx.send(LlmServerMessage::PromptDone(gen_res)).unwrap();
                },
                _ => {}
            }
        }
    });
}

fn run_server(gen_state: Arc<Mutex<GenerationState>>, rx: Receiver<LlmServerMessage>, tx: Sender<LlmServerMessage>) {
    let (handler, node_listener) = node::split::<()>();

    let listen_addr = "10.140.153.73:5341";
    handler.network().listen(Transport::FramedTcp, listen_addr).unwrap();

    println!("Llm server running at {}", listen_addr);

    let client_endpoint = Arc::new(Mutex::new(Option::<Endpoint>::None));
    // set up the llm comm loop
    let handler_llm_loop = handler.clone();
    let client_endpoint_llm_loop = client_endpoint.clone();
    let gen_state_llm_loop = gen_state.clone();
    thread::spawn(move || {
        loop {
            let block = rx.recv().unwrap();
            match block {
                LlmServerMessage::PromptDone(gen_res) => {
                    let client_endpoint_lock = client_endpoint_llm_loop.lock().unwrap();
                    if client_endpoint_lock.is_none() {
                        println!("client endpoint is none");
                        return;
                    }

                    let mut gen_state_lock = gen_state_llm_loop.lock().unwrap();
                    gen_state_lock.is_generating = false;
                    gen_state_lock.generated_lines = Vec::new();

                    if gen_res.was_terminated {
                        return;
                    }

                    let message = Message::GenerationDone(gen_res);
                    let output_data = bincode::serialize(&message).unwrap();
                    handler_llm_loop.network().send(client_endpoint_lock.unwrap(), &output_data);
                },
                _ => {}
            }
        }
    });

    // set up the msg receiving loop
    let handler_server_loop = handler.clone();
    let client_endpoint_server_loop = client_endpoint.clone();
    let gen_state_server_loop = gen_state.clone();
    thread::spawn(move || {
        node_listener.for_each(move |event| match event.network() {
            NetEvent::Accepted(endpoint, _) => {
                let mut client_endpoint_lock = client_endpoint_server_loop.lock().unwrap();
                let _ = std::mem::replace(&mut *client_endpoint_lock, Some(endpoint));
                println!("client connected");
            }
            NetEvent::Message(endpoint, data) => {
                let message: Message = bincode::deserialize(&data).unwrap();
                match message {
                    Message::GeneratePrompt(prompt_str) => {
                        let gen_state_lock = gen_state_server_loop.lock().unwrap();
                        if gen_state_lock.is_generating {
                            if gen_state_lock.should_terminate {
                                println!("still terminating previous prompt")
                            } else {
                                println!("unable to generate new prompt: still generating");
                            }
                            return;
                        }

                        tx.send(LlmServerMessage::GeneratePrompt(prompt_str)).unwrap();
                    },
                    Message::RequestCurrentGeneratedLines => {
                        let gen_state_lock = gen_state.lock().unwrap();

                        let message = Message::CurrentGeneratedLinesResponse(gen_state_lock.generated_lines.clone());
                        let output_data = bincode::serialize(&message).unwrap();
                        handler_server_loop.network().send(endpoint, &output_data);
                    },
                    _ => {
                        println!("unexpected message type received")
                    }
                }
            }
            NetEvent::Disconnected(_endpoint) => {
                let mut gen_state_lock = gen_state.lock().unwrap();
                if gen_state_lock.is_generating {
                    println!("client disconnected, terminating current text gen");
                    gen_state_lock.should_terminate = true;
                } else {
                    println!("client disconnected");
                }
            }
            _ => {}
        })
    });
}
