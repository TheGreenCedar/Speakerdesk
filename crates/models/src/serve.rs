use std::io::{Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use speakerdesk_core::error::ChildError;
use speakerdesk_core::ipc::{Envelope, EnvelopeBody, MAX_FRAME_BYTES, Request};

use crate::dispatch;
use crate::metallib;

pub fn serve() -> std::io::Result<()> {
    serve_on(std::io::stdin(), std::io::stdout())
}

pub fn serve_on<R, W>(mut reader: R, mut writer: W) -> std::io::Result<()>
where
    R: Read,
    W: Write,
{
    let _ = metallib::prepare(None);
    let cancel = Arc::new(AtomicBool::new(false));
    loop {
        let envelope = match read_envelope(&mut reader) {
            Ok(envelope) => envelope,
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error),
        };
        match envelope.body {
            EnvelopeBody::Request(Request::Cancel { .. }) => {
                cancel.store(true, Ordering::Relaxed);
                write_envelope(
                    &mut writer,
                    Envelope {
                        request_id: envelope.request_id,
                        body: EnvelopeBody::Response(Ok(speakerdesk_core::ipc::Response::Ack)),
                    },
                )?;
            }
            EnvelopeBody::Request(Request::Shutdown) => {
                write_envelope(
                    &mut writer,
                    Envelope {
                        request_id: envelope.request_id,
                        body: EnvelopeBody::Response(dispatch::dispatch(
                            Request::Shutdown,
                            &cancel,
                        )),
                    },
                )?;
                return Ok(());
            }
            EnvelopeBody::Request(request) => {
                let result = dispatch::dispatch(request, &cancel);
                if matches!(result, Err(ChildError::Cancelled)) {
                    cancel.store(false, Ordering::Relaxed);
                }
                write_envelope(
                    &mut writer,
                    Envelope {
                        request_id: envelope.request_id,
                        body: EnvelopeBody::Response(result),
                    },
                )?;
            }
            EnvelopeBody::Response(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "child received a response frame",
                ));
            }
        }
    }
}

pub fn read_envelope<R: Read>(reader: &mut R) -> std::io::Result<Envelope> {
    let mut length_bytes = [0u8; 4];
    reader.read_exact(&mut length_bytes)?;
    let length = u32::from_be_bytes(length_bytes) as usize;
    if length > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame exceeds 64 MiB",
        ));
    }
    let mut payload = vec![0u8; length];
    reader.read_exact(&mut payload)?;
    postcard::from_bytes(&payload)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))
}

pub fn write_envelope<W: Write>(writer: &mut W, envelope: Envelope) -> std::io::Result<()> {
    let payload = postcard::to_allocvec(&envelope)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;
    if payload.len() > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame exceeds 64 MiB",
        ));
    }
    let length = u32::try_from(payload.len())
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.to_string()))?;
    writer.write_all(&length.to_be_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(())
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
            EnvelopeBody::Response(Ok(Response::Ready { build_id, device })) => {
                assert_eq!(build_id, "test");
                assert_eq!(device, "unavailable");
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
    fn rejects_an_oversized_length_prefix() {
        let mut cursor = Cursor::new(u32::MAX.to_be_bytes().to_vec());
        let error = read_envelope(&mut cursor).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    }
}
