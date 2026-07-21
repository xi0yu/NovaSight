use std::time::Duration;

use axum::{
    body::Body,
    http::{
        Request, StatusCode,
        header::{
            ACCESS_CONTROL_ALLOW_CREDENTIALS, ACCESS_CONTROL_ALLOW_HEADERS,
            ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
            ACCESS_CONTROL_REQUEST_HEADERS, ACCESS_CONTROL_REQUEST_METHOD, ORIGIN,
        },
    },
};
use novasight_api::{ApiState, build_router};
use novasight_core::{RuntimeDependencies, RuntimeManager};
use tower::ServiceExt;

fn app() -> axum::Router {
    let runtime = RuntimeManager::spawn(RuntimeDependencies::replay_fixture(Duration::ZERO));
    build_router(ApiState::new(runtime))
}

#[tokio::test]
async fn studio_and_loopback_origins_are_allowed_without_a_wildcard() {
    let allowed_origins = [
        "http://tauri.localhost",
        "https://tauri.localhost",
        "tauri://localhost",
        "http://localhost",
        "https://localhost:43123",
        "http://127.0.0.1:5174",
        "https://127.0.0.1",
        "http://[::1]:62000",
        "https://[::1]",
    ];

    for origin in allowed_origins {
        let response = app()
            .oneshot(
                Request::get("/healthz")
                    .header(ORIGIN, origin)
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK, "origin {origin}");
        assert_eq!(
            response
                .headers()
                .get(ACCESS_CONTROL_ALLOW_ORIGIN)
                .expect("allowed origin header"),
            origin,
            "origin {origin}"
        );
        assert_ne!(
            response.headers().get(ACCESS_CONTROL_ALLOW_ORIGIN).unwrap(),
            "*"
        );
        assert_eq!(
            response
                .headers()
                .get(ACCESS_CONTROL_ALLOW_CREDENTIALS)
                .expect("credentials header"),
            "true"
        );
    }
}

#[tokio::test]
async fn allowed_preflight_mirrors_the_requested_method_and_headers() {
    let response = app()
        .oneshot(
            Request::options("/api/runtime/start")
                .header(ORIGIN, "http://localhost:43123")
                .header(ACCESS_CONTROL_REQUEST_METHOD, "POST")
                .header(ACCESS_CONTROL_REQUEST_HEADERS, "x-novasight-test")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(ACCESS_CONTROL_ALLOW_ORIGIN)
            .expect("allowed origin"),
        "http://localhost:43123"
    );
    assert_eq!(
        response
            .headers()
            .get(ACCESS_CONTROL_ALLOW_CREDENTIALS)
            .expect("credentials"),
        "true"
    );
    assert_eq!(
        response
            .headers()
            .get(ACCESS_CONTROL_ALLOW_METHODS)
            .expect("allowed method"),
        "POST"
    );
    assert_eq!(
        response
            .headers()
            .get(ACCESS_CONTROL_ALLOW_HEADERS)
            .expect("allowed headers"),
        "x-novasight-test"
    );
}

#[tokio::test]
async fn non_loopback_origins_receive_no_cors_authorization() {
    let disallowed_origins = [
        "https://example.com",
        "http://localhost.example.com:5174",
        "http://192.168.1.10:5174",
        "tauri://example.com",
        "file://localhost",
    ];

    for origin in disallowed_origins {
        let response = app()
            .oneshot(
                Request::get("/healthz")
                    .header(ORIGIN, origin)
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");

        assert_eq!(response.status(), StatusCode::OK, "origin {origin}");
        assert!(
            response
                .headers()
                .get(ACCESS_CONTROL_ALLOW_ORIGIN)
                .is_none(),
            "origin {origin} must not be authorized"
        );
    }
}
