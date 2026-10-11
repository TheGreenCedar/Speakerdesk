use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use crate::dispatch;
use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{Envelope, EnvelopeBody, Request, Response};
pub use speakerdesk_core::ipc::{read_envelope, write_envelope};

type Active = Arc<Mutex<Option<(u64, Arc<AtomicBool>)>>>;

pub fn serve(model_root: std::path::PathBuf) -> std::io::Result<()> {
    #[cfg(target_os = "macos")]
    {
        let mut runtime = dispatch::Runtime::new(model_root).map_err(std::io::Error::other)?;
        serve_with(
            std::io::stdin(),
            std::io::stdout(),
            move |request, cancel| runtime.execute(request, cancel),
        )
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = model_root;
        Err(std::io::Error::other(ChildError::MetalUnavailable))
    }
}

pub fn serve_on<R: Read + Send, W: Write + Send>(reader: R, writer: W) -> std::io::Result<()> {
    serve_with(reader, writer, dispatch::dispatch)
}

fn serve_with<R: Read + Send, W: Write + Send>(
    mut reader: R,
    writer: W,
    mut execute: impl FnMut(Request, &AtomicBool) -> Result<Response, ChildError>,
) -> std::io::Result<()> {
    let writer = Arc::new(Mutex::new(writer));
    let active: Active = Arc::new(Mutex::new(None));
    std::thread::scope(|scope| {
        let (send, receive) = mpsc::sync_channel::<(u64, Request, Arc<AtomicBool>)>(1);
        let output = writer.clone();
        let worker_active = active.clone();
        let control = scope.spawn(move || {
            let result = (|| {
                let mut last_id = 0;
                loop {
                    let envelope = match read_envelope(&mut reader) {
                        Ok(envelope) => envelope,
                        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                            break Ok(());
                        }
                        Err(error) => break Err(error),
                    };
                    if envelope.request_id <= last_id {
                        break Err(std::io::Error::other("non-monotonic request id"));
                    }
                    last_id = envelope.request_id;
                    match envelope.body {
                        EnvelopeBody::Request(Request::Cancel { request_id }) => {
                            if let Some((id, flag)) = &*active.lock().expect("active operation")
                                && *id == request_id
                            {
                                flag.store(true, Ordering::Release);
                            }
                            reply(&writer, envelope.request_id, Ok(Response::Ack))?;
                        }
                        EnvelopeBody::Request(Request::Shutdown) => {
                            reply(&writer, envelope.request_id, Ok(Response::Ack))?;
                            break Ok(());
                        }
                        EnvelopeBody::Request(request) => {
                            let flag = Arc::new(AtomicBool::new(false));
                            let mut current = active.lock().expect("active operation");
                            if current.is_some() {
                                reply(
                                    &writer,
                                    envelope.request_id,
                                    Err(ChildError::failed("inference already active", true)),
                                )?;
                                continue;
                            }
                            *current = Some((envelope.request_id, flag.clone()));
                            send.try_send((envelope.request_id, request, flag))
                                .map_err(std::io::Error::other)?;
                        }
                        EnvelopeBody::Response(_) => {
                            break Err(std::io::Error::other("child received a response"));
                        }
                    }
                }
            })();
            if let Some((_, flag)) = &*active.lock().expect("active operation") {
                flag.store(true, Ordering::Release);
            }
            drop(send);
            result
        });
        let result = (|| -> std::io::Result<()> {
            for (id, request, cancelled) in receive {
                let vad_open = matches!(request, Request::VadOpen { .. });
                let diar_open = matches!(request, Request::DiarOpen { .. });
                let session = match &request {
                    Request::VadFeed { handle, .. } => Some(Request::VadClose { handle: *handle }),
                    Request::DiarFeed { handle, .. } => {
                        Some(Request::DiarClose { handle: *handle })
                    }
                    _ => None,
                };
                let mut result = execute(request, &cancelled);
                if cancelled.load(Ordering::Acquire) {
                    // A cancelled feed invalidates its session, leaving unrelated live state alone.
                    let close = match &result {
                        Ok(Response::Handle(handle)) if vad_open => {
                            Some(Request::VadClose { handle: *handle })
                        }
                        Ok(Response::Handle(handle)) if diar_open => {
                            Some(Request::DiarClose { handle: *handle })
                        }
                        _ => session,
                    };
                    if let Some(close) = close {
                        execute(close, &AtomicBool::new(false)).map_err(std::io::Error::other)?;
                    }
                    result = Err(ChildError::Cancelled);
                }
                *worker_active.lock().expect("active operation") = None;
                reply(&output, id, result)?;
            }
            Ok(())
        })();
        control
            .join()
            .map_err(|_| std::io::Error::other("control reader panicked"))??;
        result
    })
}

fn reply(
    writer: &Mutex<impl Write>,
    request_id: u64,
    result: Result<Response, ChildError>,
) -> std::io::Result<()> {
    write_envelope(
        &mut *writer.lock().expect("protocol writer"),
        Envelope {
            request_id,
            body: EnvelopeBody::Response(result),
        },
    )
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::os::unix::net::UnixStream;
    use std::thread;

    use speakerdesk_core::ipc::{Envelope, EnvelopeBody, Request, Response};

    use super::{read_envelope, serve_on, write_envelope};

    #[test]
    fn hello_and_shutdown_round_trip() {
        let (child, mut parent) = UnixStream::pair().unwrap();
        let child_read = child.try_clone().unwrap();
        let worker = thread::spawn(move || serve_on(child_read, child));
        write_envelope(
            &mut parent,
            Envelope {
                request_id: 7,
                body: EnvelopeBody::Request(Request::Hello {
                    build_id: "test".into(),
                }),
            },
        )
        .unwrap();
        let ready = read_envelope(&mut parent).unwrap();
        match ready.body {
            EnvelopeBody::Response(Ok(Response::Ready {
                build_id,
                device,
                models,
            })) => {
                assert_eq!(build_id, "test");
                assert_eq!(device, "unavailable");
                assert!(models.is_empty());
                assert_eq!(ready.request_id, 7);
            }
            other => panic!("unexpected {other:?}"),
        }
        write_envelope(
            &mut parent,
            Envelope {
                request_id: 8,
                body: EnvelopeBody::Request(Request::Shutdown),
            },
        )
        .unwrap();
        let ack = read_envelope(&mut parent).unwrap();
        assert!(matches!(
            ack.body,
            EnvelopeBody::Response(Ok(Response::Ack))
        ));
        worker.join().unwrap().unwrap();
    }

    #[test]
    fn stale_cancel_does_not_poison_the_next_request() {
        let (child, mut parent) = UnixStream::pair().unwrap();
        let child_read = child.try_clone().unwrap();
        let worker = thread::spawn(move || serve_on(child_read, child));
        write_envelope(
            &mut parent,
            Envelope {
                request_id: 1,
                body: EnvelopeBody::Request(Request::Cancel { request_id: 99 }),
            },
        )
        .unwrap();
        assert!(matches!(
            read_envelope(&mut parent).unwrap().body,
            EnvelopeBody::Response(Ok(Response::Ack))
        ));
        write_envelope(
            &mut parent,
            Envelope {
                request_id: 2,
                body: EnvelopeBody::Request(Request::VadOpen { conditioning: None }),
            },
        )
        .unwrap();
        assert!(matches!(
            read_envelope(&mut parent).unwrap().body,
            EnvelopeBody::Response(Err(speakerdesk_core::error::ChildError::Unimplemented))
        ));
        write_envelope(
            &mut parent,
            Envelope {
                request_id: 3,
                body: EnvelopeBody::Request(Request::Shutdown),
            },
        )
        .unwrap();
        read_envelope(&mut parent).unwrap();
        worker.join().unwrap().unwrap();
    }

    #[test]
    fn targeted_cancel_reaches_a_running_executor_and_invalidates_only_its_feed() {
        use speakerdesk_core::error::ChildError;
        use speakerdesk_core::ipc::Handle;
        use std::sync::atomic::Ordering;
        use std::sync::{Arc, Mutex, mpsc};
        let (child, mut parent) = UnixStream::pair().unwrap();
        let input = child.try_clone().unwrap();
        let (started, wait) = mpsc::channel();
        let closed = Arc::new(Mutex::new(Vec::new()));
        let observed = closed.clone();
        let worker = thread::spawn(move || {
            // Native model objects stay on their construction thread without unsafe Send.
            let owner = std::rc::Rc::new(thread::current().id());
            super::serve_with(input, child, move |request, cancel| match request {
                Request::VadFeed { .. } => {
                    assert_eq!(thread::current().id(), *owner);
                    started.send(()).unwrap();
                    while !cancel.load(Ordering::Acquire) {
                        thread::yield_now();
                    }
                    Err(ChildError::Cancelled)
                }
                Request::VadClose { handle } => {
                    observed.lock().unwrap().push(handle.session);
                    Ok(Response::Ack)
                }
                _ => Err(ChildError::Unimplemented),
            })
        });
        write_envelope(
            &mut parent,
            Envelope {
                request_id: 1,
                body: EnvelopeBody::Request(Request::VadFeed {
                    handle: Handle {
                        generation: 0,
                        session: 7,
                    },
                    pcm: vec![0.0],
                    last: false,
                }),
            },
        )
        .unwrap();
        wait.recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        write_envelope(
            &mut parent,
            Envelope {
                request_id: 2,
                body: EnvelopeBody::Request(Request::Cancel { request_id: 1 }),
            },
        )
        .unwrap();
        let mut replies = [
            read_envelope(&mut parent).unwrap(),
            read_envelope(&mut parent).unwrap(),
        ];
        replies.sort_by_key(|reply| reply.request_id);
        assert_eq!(
            replies[0].body,
            EnvelopeBody::Response(Err(ChildError::Cancelled))
        );
        assert_eq!(replies[1].body, EnvelopeBody::Response(Ok(Response::Ack)));
        assert_eq!(*closed.lock().unwrap(), vec![7]);
        write_envelope(
            &mut parent,
            Envelope {
                request_id: 3,
                body: EnvelopeBody::Request(Request::Shutdown),
            },
        )
        .unwrap();
        read_envelope(&mut parent).unwrap();
        worker.join().unwrap().unwrap();
    }

    #[test]
    fn rejects_partial_frames_instead_of_treating_them_as_clean_eof() {
        for bytes in [vec![0], vec![0, 0, 0, 3, 1]] {
            let error = read_envelope(&mut Cursor::new(bytes)).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        }
    }

    #[test]
    fn rejects_an_oversized_length_prefix() {
        let mut cursor = Cursor::new(u32::MAX.to_be_bytes().to_vec());
        let error = read_envelope(&mut cursor).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }
}
