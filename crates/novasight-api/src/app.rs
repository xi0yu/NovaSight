use axum::http::HeaderValue;
use tower_http::cors::{AllowHeaders, AllowMethods, AllowOrigin, CorsLayer};

pub(crate) fn studio_cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| {
            is_allowed_studio_origin(origin)
        }))
        .allow_credentials(true)
        .allow_methods(AllowMethods::mirror_request())
        .allow_headers(AllowHeaders::mirror_request())
}

fn is_allowed_studio_origin(origin: &HeaderValue) -> bool {
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    if matches!(
        origin,
        "http://tauri.localhost" | "https://tauri.localhost" | "tauri://localhost"
    ) {
        return true;
    }

    let authority = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"));
    authority.is_some_and(is_loopback_authority)
}

fn is_loopback_authority(authority: &str) -> bool {
    ["localhost", "127.0.0.1", "[::1]"].into_iter().any(|host| {
        authority == host
            || authority.strip_prefix(host).is_some_and(|suffix| {
                let port = suffix.strip_prefix(':').unwrap_or_default();
                !port.is_empty() && port.bytes().all(|byte| byte.is_ascii_digit())
            })
    })
}
