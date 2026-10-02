use std::sync::LazyLock;

use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub struct NetworkEndpoint {
    pub host: String,
    pub port: u16,
}

#[derive(Clone, Debug, Deserialize)]
pub struct StudioEndpointContract {
    pub studio: NetworkEndpoint,
    pub frontend_development_api: NetworkEndpoint,
}

static STUDIO_ENDPOINTS: LazyLock<StudioEndpointContract> = LazyLock::new(|| {
    let contract: StudioEndpointContract =
        serde_json::from_str(include_str!("../../../deploy/studio-endpoints.json"))
            .expect("deploy/studio-endpoints.json must contain a valid endpoint contract");
    assert!(
        !contract.studio.host.is_empty() && contract.studio.port != 0,
        "Studio endpoint must have a host and non-zero port"
    );
    assert!(
        !contract.frontend_development_api.host.is_empty()
            && contract.frontend_development_api.port != 0,
        "frontend development API endpoint must have a host and non-zero port"
    );
    contract
});

pub fn studio_endpoint_contract() -> &'static StudioEndpointContract {
    &STUDIO_ENDPOINTS
}
