use novasight_core::RuntimeHandle;

#[derive(Clone)]
pub struct ApiState {
    pub(crate) runtime: RuntimeHandle,
}

impl ApiState {
    pub fn new(runtime: RuntimeHandle) -> Self {
        Self { runtime }
    }
}
