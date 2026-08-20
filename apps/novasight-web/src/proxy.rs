use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{Request, Response, StatusCode, header};
use hyper::client::conn::http1;
use hyper::upgrade;
use hyper_util::rt::TokioIo;
use tokio::io::AsyncWriteExt;
use tokio::net::UnixStream;

#[derive(Clone)]
pub struct DaemonProxy {
    socket: PathBuf,
}

impl DaemonProxy {
    pub fn new(socket: impl Into<PathBuf>) -> Self {
        Self {
            socket: socket.into(),
        }
    }

    pub fn socket(&self) -> &Path {
        &self.socket
    }

    pub async fn healthy(&self) -> bool {
        let request = Request::builder()
            .uri("/healthz")
            .header(header::HOST, "localhost")
            .body(Body::empty());
        match request {
            Ok(request) => self
                .forward(request, None)
                .await
                .is_ok_and(|response| response.status() == StatusCode::OK),
            Err(_) => false,
        }
    }

    pub async fn forward(
        &self,
        mut request: Request<Body>,
        session_expires_at: Option<u64>,
    ) -> Result<Response<Body>, ProxyError> {
        let is_upgrade = request
            .headers()
            .get(header::UPGRADE)
            .is_some_and(|value| !value.is_empty());
        let downstream_upgrade = is_upgrade.then(|| upgrade::on(&mut request));
        sanitize_request_headers(request.headers_mut(), is_upgrade);
        request
            .headers_mut()
            .insert(header::HOST, "localhost".parse().expect("static host"));

        let stream =
            tokio::time::timeout(Duration::from_secs(3), UnixStream::connect(&self.socket))
                .await
                .map_err(|_| ProxyError::ConnectTimeout)?
                .map_err(ProxyError::Connect)?;
        let (mut sender, connection) = http1::handshake(TokioIo::new(stream))
            .await
            .map_err(ProxyError::Handshake)?;
        tokio::spawn(async move {
            if let Err(error) = connection.with_upgrades().await {
                tracing::debug!(%error, "daemon IPC connection ended");
            }
        });
        let mut response = sender
            .send_request(request)
            .await
            .map_err(ProxyError::Request)?;
        let upstream_upgrade = is_upgrade.then(|| upgrade::on(&mut response));
        sanitize_response_headers(response.headers_mut(), is_upgrade);
        let response = response.map(Body::new);

        if let (Some(downstream), Some(upstream)) = (downstream_upgrade, upstream_upgrade) {
            tokio::spawn(async move {
                let relay = async {
                    let downstream = downstream.await.map_err(ProxyError::Upgrade)?;
                    let upstream = upstream.await.map_err(ProxyError::Upgrade)?;
                    let mut downstream = TokioIo::new(downstream);
                    let mut upstream = TokioIo::new(upstream);
                    if let Some(expires_at) = session_expires_at {
                        let now = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .map_or(0, |duration| duration.as_secs());
                        let mut copy = Box::pin(tokio::io::copy_bidirectional(
                            &mut downstream,
                            &mut upstream,
                        ));
                        tokio::select! {
                            result = copy.as_mut() => {
                                result.map_err(ProxyError::Relay)?;
                            }
                            () = tokio::time::sleep(Duration::from_secs(expires_at.saturating_sub(now))) => {
                                drop(copy);
                                let reason = b"web session expired";
                                let mut frame = Vec::with_capacity(4 + reason.len());
                                frame.extend_from_slice(&[0x88, (2 + reason.len()) as u8]);
                                frame.extend_from_slice(&4403_u16.to_be_bytes());
                                frame.extend_from_slice(reason);
                                downstream.write_all(&frame).await.map_err(ProxyError::Relay)?;
                                downstream.flush().await.map_err(ProxyError::Relay)?;
                                downstream.shutdown().await.map_err(ProxyError::Relay)?;
                            }
                        }
                    } else {
                        tokio::io::copy_bidirectional(&mut downstream, &mut upstream)
                            .await
                            .map_err(ProxyError::Relay)?;
                    }
                    Ok::<(), ProxyError>(())
                };
                if let Err(error) = relay.await {
                    tracing::debug!(%error, "daemon WebSocket relay ended");
                }
            });
        }
        Ok(response)
    }
}

fn sanitize_request_headers(headers: &mut axum::http::HeaderMap, is_upgrade: bool) {
    for name in [
        header::AUTHORIZATION,
        header::COOKIE,
        header::ORIGIN,
        header::REFERER,
        header::FORWARDED,
        header::PROXY_AUTHORIZATION,
        header::PROXY_AUTHENTICATE,
        header::TRAILER,
    ] {
        headers.remove(name);
    }
    for name in [
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-proto",
        "x-real-ip",
        "x-novasight-csrf",
    ] {
        headers.remove(name);
    }
    if !is_upgrade {
        headers.remove(header::CONNECTION);
        headers.remove(header::UPGRADE);
        headers.remove(header::TE);
        headers.remove(header::TRANSFER_ENCODING);
    }
}

fn sanitize_response_headers(headers: &mut axum::http::HeaderMap, is_upgrade: bool) {
    headers.remove(header::SET_COOKIE);
    headers.remove(header::PROXY_AUTHENTICATE);
    headers.remove(header::TRAILER);
    if !is_upgrade {
        headers.remove(header::CONNECTION);
        headers.remove(header::UPGRADE);
        headers.remove(header::TE);
        headers.remove(header::TRANSFER_ENCODING);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProxyError {
    #[error("daemon IPC connection timed out")]
    ConnectTimeout,
    #[error("failed to connect to daemon IPC: {0}")]
    Connect(#[source] std::io::Error),
    #[error("failed to establish daemon HTTP transport: {0}")]
    Handshake(#[source] hyper::Error),
    #[error("daemon request failed: {0}")]
    Request(#[source] hyper::Error),
    #[error("WebSocket upgrade failed: {0}")]
    Upgrade(#[source] hyper::Error),
    #[error("WebSocket relay failed: {0}")]
    Relay(#[source] std::io::Error),
}
