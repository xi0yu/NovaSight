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

#[cfg(test)]
mod tests {
    use std::{
        future::Future,
        pin::Pin,
        task::{Context, Poll},
        time::Duration,
    };

    use futures_util::{Sink, Stream};
    use tokio::sync::oneshot;

    use super::{Message, send_while_receiving};

    struct PendingSink {
        polled_tx: Option<oneshot::Sender<()>>,
    }

    impl Sink<Message> for PendingSink {
        type Error = ();

        fn poll_ready(
            mut self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            if let Some(polled_tx) = self.polled_tx.take() {
                let _ = polled_tx.send(());
            }
            Poll::Pending
        }

        fn start_send(self: Pin<&mut Self>, _item: Message) -> Result<(), Self::Error> {
            unreachable!("pending sink cannot accept a frame")
        }

        fn poll_flush(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Pending
        }

        fn poll_close(
            self: Pin<&mut Self>,
            _context: &mut Context<'_>,
        ) -> Poll<Result<(), Self::Error>> {
            Poll::Ready(Ok(()))
        }
    }

    struct CloseAfterSendPending {
        send_polled_rx: oneshot::Receiver<()>,
    }

    impl Stream for CloseAfterSendPending {
        type Item = Result<Message, ()>;

        fn poll_next(
            mut self: Pin<&mut Self>,
            context: &mut Context<'_>,
        ) -> Poll<Option<Self::Item>> {
            match Pin::new(&mut self.send_polled_rx).poll(context) {
                Poll::Ready(_) => Poll::Ready(Some(Ok(Message::Close(None)))),
                Poll::Pending => Poll::Pending,
            }
        }
    }

    #[tokio::test]
    async fn blocked_outbound_send_still_observes_client_close() {
        let (send_polled_tx, send_polled_rx) = oneshot::channel();
        let mut sink = PendingSink {
            polled_tx: Some(send_polled_tx),
        };
        let mut incoming = CloseAfterSendPending { send_polled_rx };

        let connected = tokio::time::timeout(
            Duration::from_secs(1),
            send_while_receiving(&mut sink, &mut incoming, Message::Text("blocked".into())),
        )
        .await
        .expect("close must interrupt blocked outbound send");

        assert!(!connected);
    }
}
