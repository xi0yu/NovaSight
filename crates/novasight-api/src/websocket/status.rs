use axum::extract::ws::Message;
use futures_util::{Sink, SinkExt, Stream, StreamExt};

/// Send one status frame without letting a blocked client prevent close handling.
pub(crate) async fn send_while_receiving<S, R, E>(
    outbound: &mut S,
    inbound: &mut R,
    message: Message,
) -> bool
where
    S: Sink<Message> + Unpin,
    R: Stream<Item = Result<Message, E>> + Unpin,
{
    let send = outbound.send(message);
    tokio::pin!(send);

    loop {
        tokio::select! {
            result = &mut send => return result.is_ok(),
            incoming = inbound.next() => {
                match incoming {
                    Some(Ok(Message::Close(_))) | Some(Err(_)) | None => return false,
                    Some(Ok(_)) => {}
                }
            }
        }
    }
}
