//! Deterministic helper protocol replay for mounted-host tests. No audio device is used.
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, Write},
    path::PathBuf,
    sync::mpsc,
    time::Duration,
};

fn emit(value: Value) {
    println!("{value}");
    std::io::stdout().flush().unwrap();
}
fn audio(start: u32, end: u32, sources: &[&str]) {
    for position in (start..end).step_by(4000) {
        let stop = (position + 4000).min(end);
        for source in sources {
            let scale = if *source == "system"
                && std::path::Path::new(&std::env::args().nth(2).unwrap())
                    .join("distinct-system")
                    .exists()
            {
                0.35
            } else {
                0.2
            };
            let bytes: Vec<u8> = (position..stop)
                .flat_map(|i| {
                    let v = scale * (i as f32 * 0.071).sin();
                    v.to_le_bytes()
                })
                .collect();
            emit(
                json!({"type":"audio","source":source,"time":f64::from(position)/16000.,"rate":16000,"channels":1,"pcm":STANDARD.encode(bytes)}),
            );
        }
        emit(json!({"type":"clock","time":f64::from(stop)/16000.}));
    }
}
pub fn run() {
    let root = PathBuf::from(std::env::args_os().nth(2).unwrap());
    fs::write(root.join("capture-pid"), std::process::id().to_string()).unwrap();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if tx
                .send(serde_json::from_str::<Value>(&line).unwrap())
                .is_err()
            {
                break;
            }
        }
    });
    let start = rx.recv().unwrap();
    assert_eq!(start["type"], "start");
    let dual = start["microphone"] == true && start["system"] == true;
    let receipt = if dual && start["capture_export_contract"].is_string() {
        let mut receipt: Value =
            serde_json::from_slice(&fs::read(root.join("receipt.json")).unwrap()).unwrap();
        receipt["job_id"] = start["job_id"].clone();
        emit(receipt.clone());
        Some(receipt)
    } else {
        None
    };
    emit(json!({"type":"recording"}));
    let sources: Vec<_> = if dual {
        vec!["microphone", "system"]
    } else if start["microphone"] == true {
        vec!["microphone"]
    } else {
        vec!["system"]
    };
    let mut endpoint = 0;
    let mut stage = 0;
    let mut paused = false;
    loop {
        if !paused && root.join(format!("stage-{}", stage + 1)).exists() {
            stage += 1;
            let samples = if stage == 1 {
                fs::read_to_string(root.join("first-stage-samples"))
                    .ok()
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(112000)
            } else {
                112000
            };
            let target = endpoint + samples;
            audio(endpoint, target, &sources);
            endpoint = target;
            fs::write(root.join(format!("sent-{stage}")), endpoint.to_string()).unwrap();
        }
        if root.join("fail-capture").exists() {
            emit(json!({"type":"error","error":"Replay capture failed"}));
            return;
        }
        match rx.recv_timeout(Duration::from_millis(5)) {
            Ok(command) => match command["type"].as_str().unwrap() {
                "pause" => {
                    paused = true;
                    emit(json!({"type":"paused","time":f64::from(endpoint)/16000.}));
                }
                "resume" => {
                    fs::write(root.join("resume-started"), b"replay").unwrap();
                    while root.join("hold-resume").exists() && !root.join("release-resume").exists()
                    {
                        match rx.recv_timeout(Duration::from_millis(5)) {
                            Ok(command) if command["type"] == "stop" => {
                                if root.join("late-recording").exists() {
                                    if let Some(receipt) = &receipt {
                                        emit(receipt.clone());
                                    }
                                    emit(json!({"type":"recording"}));
                                    audio(endpoint, endpoint + 16000, &sources);
                                    endpoint += 16000;
                                    fs::write(root.join("late-recording-sent"), b"replay").unwrap();
                                    while !root.join("release-stop").exists() {
                                        std::thread::sleep(Duration::from_millis(5));
                                    }
                                }
                                emit(json!({"type":"stopped","time":f64::from(endpoint)/16000.}));
                                return;
                            }
                            Ok(other) => panic!("unexpected pending-resume command {other}"),
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                            Err(_) => return,
                        }
                    }
                    if root.join("fail-resume").exists() {
                        emit(
                            json!({"type":"error","error":"Replay resumed microphone device changed"}),
                        );
                        return;
                    }
                    if let Some(receipt) = &receipt {
                        emit(receipt.clone());
                    }
                    paused = false;
                    emit(json!({"type":"recording"}));
                }
                "stop" => {
                    if root.join("stall-stop").exists() {
                        loop {
                            std::thread::park();
                        }
                    }
                    if !paused {
                        audio(endpoint, endpoint + 513, &sources);
                        endpoint += 513;
                    }
                    emit(json!({"type":"stopped","time":f64::from(endpoint)/16000.}));
                    return;
                }
                other => panic!("unexpected replay command {other}"),
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(_) => return,
        }
    }
}
