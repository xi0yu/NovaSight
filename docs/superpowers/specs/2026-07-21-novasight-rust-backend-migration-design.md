# NovaSight Rust 后端完整迁移设计

状态：已批准设计  
日期：2026-07-21  
目标入口：`rust/bins/relink_server`  
适用范围：现有 Python/FastAPI 在线后端、DeepStream 实时链、控制算法、设备输出、配置、模型与 Studio 控制面

## 1. 决策摘要

NovaSight 采用分阶段完整迁移。最终 `relink_server` 是唯一生产入口并独占 HTTP/WS、Runtime、DeepStream、targeting、control、device、配置与模型在线状态。旧 Python/C++ 源码和测试全部保留；Python 仅作为由 Rust 受控启动的离线模型转换、inspection、probe 和 doctor 工具，不作为常驻在线 sidecar。

生产实现整合进现有根 `rust/` workspace，不继续扩展 `base/relink_server` 原型。`base/relink_server` 原样保留作为历史设计和依赖验证参考；根 workspace 只保留一个 `Cargo.lock`。

不按 Python 文件逐行翻译。目标是 Jetson-first 的深模块化单体：控制面通过少量 typed command 操作 `RuntimeManager`；每次启动创建一个拥有独立 `RuntimeEpoch` 的 `RuntimeSession`；frame、DetectionBatch、DeviceCommand 和 OperationalSnapshot 使用 latest-only 语义；API 只发送命令或读取不可变快照。

## 2. 现状与问题

### 2.1 当前主要职责

| 位置 | 当前职责 |
|---|---|
| `novasight/main.py` | 服务启动、CLI、配置、Uvicorn、硬件 doctor、smoke、报告校验 |
| `novasight/api/app.py` | FastAPI composition、生命周期、middleware、路由注册 |
| `novasight/api/routes_*.py` | Runtime、配置、采集、模型、执行器、License、准星、motion、HTTP/WS DTO |
| `novasight/api/routes_models.py` | HTTP、上传、扫描、hash、构建、发布、回滚、运行时模型切换 |
| `novasight/runtime/service.py` | freshness、geometry、tracking、selection、control、trigger、output、trace、状态投影 |
| `novasight/runtime/pipeline.py` | 普通采集/推理/control worker 生命周期 |
| `novasight/deepstream/runtime_pipeline.py` | DeepStream DetectionBatch/control worker 生命周期 |
| `novasight/deepstream/backend.py` | GStreamer/DeepStream、pad probe、preview、时间映射、恢复、遥测 |
| `novasight/config/runtime.py` | 配置类型、历史兼容、校验、序列化和保存 |
| `novasight/model_registry/store.py` | SQLite model project/version/artifact/job/deployment |
| `novasight/control/` | targeting 后控制算法、recoil、scheduler、humanized motion |
| `novasight/executors/` | kmNet、命令调度、按钮和诊断 |
| `native/deepstream-bridge/` | DeepStream vendor metadata 到固定布局 C ABI |
| `rust/crates/novasight-deepstream-bridge/` | C ABI Rust 类型、布局校验和安全包装 |

### 2.2 结构问题

- `RuntimeService` 同时协调算法、资源、设备副作用和状态展示，无法独立替换或测试。
- FastAPI handler 通过 `app.state` 直接操作活对象，没有稳定应用 interface。
- 普通 pipeline 与 DeepStream pipeline 重复 worker、control loop、stop/fault 和统计逻辑。
- 热路径大量使用 `Any`、`dict[str, Any]`、字符串状态和任意 metadata，关键 invariant 只能靠约定。
- `main.py` 混合常驻服务与诊断工具；`routes_models.py` 混合 HTTP、文件、SQLite、job 和 Runtime transaction。
- 同步 GStreamer/TensorRT/kmNet/SQLite 操作缺少可强制的线程所有权。
- 状态由多个 mutable dict 拼装，故障状态与 UI 投影可能不一致。
- `base/relink_server` 的 `/api/health:5656` 与当前 `/healthz:5174`、`/api/...`、`/ws/status` 产品契约不一致。

## 3. 目标 workspace

```text
rust/
├── Cargo.toml
├── Cargo.lock
├── config/
│   └── novasightd.example.yaml
├── bins/
│   └── relink_server/
│       ├── Cargo.toml
│       └── src/
│           ├── main.rs
│           ├── bootstrap.rs
│           └── shutdown.rs
└── crates/
    ├── novasight-core/
    │   └── src/
    │       ├── lib.rs
    │       ├── error.rs
    │       ├── ports.rs
    │       ├── runtime/
    │       ├── perception/
    │       ├── targeting/
    │       ├── control/
    │       ├── output/
    │       ├── config/
    │       └── telemetry/
    ├── novasight-api/
    │   └── src/
    │       ├── lib.rs
    │       ├── app.rs
    │       ├── state.rs
    │       ├── error.rs
    │       ├── routes/
    │       ├── dto/
    │       ├── middleware/
    │       └── websocket/
    ├── novasight-store/
    │   └── src/
    │       ├── lib.rs
    │       ├── config/
    │       ├── sqlite/
    │       ├── model/
    │       ├── license/
    │       └── jobs/
    ├── novasight-platform-jetson/
    │   └── src/
    │       ├── lib.rs
    │       ├── clock.rs
    │       ├── systemd.rs
    │       ├── gstreamer/
    │       ├── deepstream/
    │       ├── tensorrt/
    │       ├── cuda/
    │       └── kmnet/
    └── novasight-deepstream-bridge/
```

### 3.1 Crate 单一职责

- `novasight-core`：只因 NovaSight 业务规则变化而变化。禁止依赖 Axum、SQLite、GStreamer、CUDA、TensorRT 和 kmNet。
- `novasight-api`：只因 HTTP/WS 契约、DTO、鉴权和 compatibility projection 变化而变化。
- `novasight-store`：只因 YAML/SQLite/License/模型目录、schema migration 和离线 job 协议变化而变化。
- `novasight-platform-jetson`：只因 Jetson、GStreamer、DeepStream、CUDA、TensorRT、kmNet 或 systemd adapter 变化而变化。所有 vendor `unsafe` 在此 crate 或专用 bridge crate 内。
- `relink_server`：唯一 composition root；只负责加载配置、执行 migration、创建 adapter、启动 API/Runtime 和协调 shutdown，不包含业务判断。

### 3.2 依赖规则

```text
relink_server binary
  ├── novasight-api
  ├── novasight-store
  ├── novasight-platform-jetson
  └── novasight-core

novasight-api -------------> novasight-core public handles/types
novasight-store -----------> novasight-core persistence contracts
novasight-platform-jetson -> novasight-core ports
novasight-core ------------> no infrastructure crates
```

`novasight-api` 不持有 SQLite connection、GStreamer、TensorRT 或 kmNet 实例。`store` 与 `platform` 互不依赖。只有确实具有两个及以上 adapter 的 seam 才定义 trait。

## 4. Core 深模块

### 4.1 Runtime

职责：Run Intent、Runtime Phase、Runtime Epoch、start/stop/fault/standby、RuntimeSession 生命周期和资源独占。

状态：`Stopped -> Starting -> Running -> Stopping -> Stopped`，运行中可进入 `Standby`，任意启动/运行阶段可进入 `Faulted`。每次成功 start 创建新 Epoch；旧 Epoch 的 frame、batch、command 和 receipt 永不进入新 Session。

外部 interface 仅暴露 `RuntimeHandle` 的 typed commands、queries 和 snapshot subscription，不暴露 worker。

### 4.2 Perception

职责：FrameStamp、DetectionBatch 以及 epoch、generation、ordering、clock、age、coordinate space 和 candidate bound admission。DeepStream 过渡阶段不负责 decode/NMS；直接 TensorRT 阶段再增加 parser/NMS adapter。

### 4.3 Targeting

职责：候选过滤、association、Track 生命周期、Kalman state、FOV、类别优先级、sticky bias 和 switch hysteresis。Tracker 与 Selector 是一个深模块的内部实现。输出一个 `SelectedTarget` 或 typed no-target reason。

### 4.4 Control

职责：将 SelectedTarget、geometry、trigger 和 timing 组成 `ControlObservation`；执行算法状态、prediction、FAR/NEAR、Atan、quantization、residual 和 reset edge；输出完整 `ControlDecision`。不调用设备。

### 4.5 Output

职责：保存唯一 `LatestCommand`；发送前复查 epoch、generation、expiry、Output Gate 和实时 trigger；限制设备范围；调用 `PointerDevice`；记录 `DeviceReceipt`。新 observation 可覆盖尚未发送的旧命令。

### 4.6 Telemetry

热路径只写 typed counters/events，不构造 JSON。`SnapshotHub` 以 5–10Hz 汇总不可变 `Arc<OperationalSnapshot>`。HTTP/WS 只读取 snapshot；慢客户端容量为 1，不反压 Runtime。

## 5. Seams 与 adapters

| Seam | Production adapters | Test/transition adapters |
|---|---|---|
| `PerceptionSource` | `DeepStreamObjectMetaSource`、后续 `LatestFrameTensorRtSource` | `ReplayPerceptionSource` |
| `PointerDevice` | `KmNetDevice` 或 `KmNetHostClient` | `DryRunDevice`、`RecordingDevice` |
| `Clock` | `SystemMonotonicClock` | `ManualClock` |
| `ConfigRepository` | `YamlConfigRepository` | in-memory repository |
| `ModelCatalog` | `SqliteModelCatalog` | in-memory catalog |
| `LicenseRepository` | `FileLicenseRepository` | in-memory repository |

kmNet vendor driver 只有在 Jetson 实验证明可控时才内联；否则使用独立 `novasight-kmnet-host` helper process。无论哪种 adapter，DeviceLane 的 interface 不变。

## 6. 数据流和线程所有权

```text
Axum command
-> RuntimeHandle
-> RuntimeManager actor
-> RuntimeSession(epoch)

DeepStream C ABI
-> typed DetectionBatch
-> PerceptionAdmission
-> TargetingCore
-> ControlCore
-> LatestCommandSlot
-> DeviceLane
-> kmNet

typed counters/events
-> SnapshotHub
-> immutable OperationalSnapshot
-> HTTP/WS projection
```

Tokio 只承担 Axum、WebSocket、RuntimeManager command actor、低频 snapshot 和异步 job supervision。禁止在 Tokio worker 上执行 GStreamer state wait、CUDA synchronize、TensorRT enqueue、kmNet driver 或同步 SQLite。

专用 OS thread 分别拥有 GStreamer/DeepStream main loop、Perception worker、1–10ms Control cadence、DeviceLane 和 StoreExecutor。vendor context 有唯一 owner，停止时按资源依赖逆序 join/drop。

控制命令使用 bounded channel。frame、DetectionBatch、DeviceCommand 和 OperationalSnapshot 使用 latest-only slot/watch。禁止无界队列；慢 API、WS、日志或 store consumer 不得反压实时线程。

## 7. 配置设计

### 7.1 AppConfig

Rust 使用 `serde` typed config，不使用 Python/Pydantic。保留现有 `config/novasight.yaml`，增加显式 `schema_version` 与 `revision`。产品 section 至少为：

- `server`
- `capture`
- `inference`
- `targeting`
- `control`
- `device`
- `telemetry`
- `power`
- `license_policy`
- `paths`

CLI 通过 `--config` 指定配置，默认仍为 `config/novasight.yaml`。环境变量只覆盖部署参数，例如 bind、data directory、native library 和 Python executable。产品行为参数必须来自版本化 YAML，不散落为 handler 或 worker 中的硬编码值。仓库提交 `rust/config/novasightd.example.yaml`。

### 7.2 Desired 与 Effective

`DesiredConfiguration` 是已验证并持久化的用户意图；`EffectiveConfiguration` 是当前 Runtime Epoch 已安装的 revision。两者必须分别进入 OperationalSnapshot，避免“保存成功”被误报为“在线链已生效”。

### 7.3 配置事务

```text
PUT /api/config + expected_revision
-> parse current/legacy DTO
-> validate typed AppConfig
-> PlanConfigChange: Hot | SessionRestart | ProcessRestart | Rejected
-> RuntimeManager prepare(operation_id)
-> apply + ready gate
-> atomic persist: temp + fsync + rename
-> publish Desired/Effective revision
-> operation receipt
```

失败时回滚 runtime change，保留上一份 YAML 和 Effective revision。revision 冲突返回 HTTP 409，不允许 last-write-wins。

## 8. 持久化与离线 job

- `YamlConfigRepository`：读取现有 YAML，按 schema version 幂等逐级迁移，原子写入。
- `SqliteModelCatalog`：原地读取 `data/novasight.db`，通过 migration table 管理 schema，继续维护 project/version/artifact/job/deployment。
- `FileLicenseRepository`：兼容 `data/license.json`，只暴露 typed LicenseStatus。
- SQLite 使用 `rusqlite` 和专用 StoreExecutor/阻塞线程；Axum handler 不持有 connection，也不在 Tokio worker 上执行 SQL。

Rust 只通过 allowlist job 类型启动 Python：固定 executable、固定脚本、隔离 working directory、显式参数、timeout、输出上限、取消与退出码。禁止接收任意 shell command。离线 job 产出 immutable artifact/profile 后才进入 ModelCatalog transaction。

## 9. Deployment 切换事务

```text
validate ModelArtifact fingerprint and runtime contract
-> close old epoch Output Gate lane
-> prepare candidate RuntimeSession
-> start perception and wait first Fresh Batch ready gate
-> commit Deployment + Effective revision in SQLite transaction
-> atomically publish new RuntimeEpoch snapshot
-> stop/drop old session
```

任何失败都销毁 candidate 并恢复旧 Session/Deployment，不留下半加载 engine，不谎报 running。

## 10. 错误契约

统一错误包含：

```rust
AppError {
    code: ErrorCode,
    subsystem: Subsystem,
    severity: Severity,
    recoverability: Recoverability,
    epoch: Option<RuntimeEpoch>,
    operation_id: Option<OperationId>,
    message: String,
    source: Option<ErrorSource>,
}
```

API 只负责 `AppError -> HTTP/WS DTO` 映射。热路径 fault 必须先关闭 Output Gate、更新 Runtime Phase，再异步记录详情。日志不是业务状态真相。

## 11. API 与 Studio 兼容

必须保留现有前端依赖的关键路径和 DTO 语义：

- `/healthz`
- `/api/runtime/state`
- `/api/runtime/start`
- `/api/runtime/stop`
- `/api/config`
- `/api/config/schema`
- `/api/capture/*`
- `/api/models/*`
- `/api/executors/*`
- `/api/license*`
- `/api/crosshair/*`
- `/api/motion/*`
- `/ws/status`

Axum handler 只做 DTO parse、handle command/query 和 compatibility projection。现有 HTTP 状态、必需字段和前端消费语义保持；operation/revision 只能以向后兼容字段加入。API 无权获得 platform 活对象。

## 12. 分阶段迁移

### Phase 0：契约与基线

冻结 API/config fixtures、Runtime trace、时钟和坐标语义；采集真实静态目标、运动目标、切换/丢失 fixture；记录 Jetson 60 秒 P50/P95/P99、drop、RSS、CPU、温度和 clock source。

### Phase 1：Workspace 与回放纵向切片

建立单一 workspace、core/api/store/platform/binary 最小实现，完成：

```text
ReplaySource -> Perception -> Targeting -> Control
-> LatestCommand -> RecordingDevice -> OperationalSnapshot
```

兼容 `/healthz`、runtime state/start/stop 和 `/ws/status`，默认 Output Gate 关闭。

### Phase 2：纯算法核心

迁移 typed units、geometry、clock、freshness、TargetingCore、ControlCore、quantizer、residual、LatestCommandSlot。按业务 invariant 重建，不复制 Python 类层次。

### Phase 3：DeepStream object-meta Rust 主链

复用现有 C ABI，完成 Rust GStreamer/DeepStream Session、预分配 metadata slots、pad probe 非阻塞发布以及 DetectionBatch 到 targeting/control 的主链。TensorRT 暂由 DeepStream 调度。

### Phase 4：DeviceLane 与 kmNet

接入 DryRun/Recording/kmNet adapters，完成 live trigger cache、发送前复查、timeout、cooldown、disconnect、reconnect 和 crash isolation。

Phase 4 使用同一 `PointerDevice` 端口保留两种可替换实现：`python_host` 是迁移期生产默认，Rust daemon 负责 helper 生命周期、超时、冷却和重连；`native_udp` 是独立 Rust 协议实现，在完成目标 Jetson、目标盒子固件和断网/重启故障注入之前必须同时显式选择并使用 `experimental-kmnet-native` 证据构建，普通生产构建拒绝启动，不得成为默认。两种实现都不得把设备状态所有权交回旧 Python backend。

kmBoxNet 的公开 C++ demo 仅作为协议与行为参考，不直接编译、复制或随 NovaSight 分发。该仓库没有标准开源许可证，其版权声明限制源码只能对接官方 kmbox 硬件；商业发布前必须取得厂商对协议实现、分发和商业用途的书面授权。保留 Python vendor extension 是兼容与回滚资产，不是在线业务架构的第二套权威实现。

### Phase 5：控制面全量接管

接管 Axum、YAML/SQLite/License/ModelCatalog、capture/model/config/executor/Studio 接口、ConfigImpact planner 和 Deployment transaction。Python 只保留离线 job。

配置控制面的第一条生产纵切已由 Rust daemon 接管：HTTP 与 Unix socket CLI 共用单一 `ConfigService`；`GET/PATCH /api/v1/config` 和 Studio 兼容入口 `GET/POST /api/config` 读写同一份版本化 YAML。字段更新与整份配置更新都必须经过目录锁、revision 比较、完整 typed validation 和原子替换，并保留未知扩展字段。当前 runtime dependency graph 仍是 epoch 启动时构造，因此成功写入只返回 `applied=false`、`restart_required=true`，不得伪装成热更新；后续 ConfigImpact planner 才能逐项开放经过证明的在线应用能力。

`ConfigService` 同时记录 daemon 启动时的 effective revision。若磁盘期望 revision 已推进但进程尚未重新装配，v1 与 Studio 兼容的 start/restart 必须返回 `409 CONFIG_RESTART_REQUIRED`，不得用旧 capture/inference/device dependency graph 启动。stop 与 emergency-stop 始终可用，确保配置待重启状态不会妨碍关闭输出。

Studio 的基础在线控制也已直接接到同一个新 Rust `RuntimeHandle`：`/healthz`、`/api/runtime/state|start|stop` 和 `/ws/status` 只投影 daemon 的真实 lifecycle snapshot、subsystem state、epoch、错误、perception counters 与当前 YAML revision。兼容 DTO 可以明确表示尚未配置、不可用或尚无指标，但禁止沿用 Phase 1 固定 replay/不可用假数据。v1 与 Studio 兼容入口只是两种 wire projection，不得形成第二个 runtime owner。

### Phase 6：Rust-controlled LatestFrame + TensorRT

实现持有 GstBuffer/NVMM lifetime 的 FrameLease、Rust LatestFrameExchange、CUDA preprocess C ABI、TensorRT context owner、typed tensor contract、decode/NMS registry。DeepStream object-meta adapter 保留为 A/B 与回滚路径。

### Phase 7：生产入口切换

修改 `deploy/novasight.service` 使用 `relink_server`。旧 Python/C++ 文件、测试和工具全部保留，但不再拥有在线状态。阶段 1–6 默认 replay/shadow/DryRun，禁止 Python 和 Rust 同时拥有设备发送权。

## 13. 验证策略

### 13.1 行为契约

- Python API/config/runtime response fixture 与 Rust compatibility projection 对比。
- 真实 trace replay 对比 target state/reason、reset edge、control intermediate、integer count 和 residual。
- 核心浮点容差显式记录，非临界 integer count 精确一致。

### 13.2 Runtime invariant

- 并发和重复 start/stop 不创建第二 Session。
- stop/fault/model switch/target switch 后没有旧命令泄漏。
- Epoch 隔离；expired command 永不发送。
- latest slot depth 不超过 1，覆盖计数正确。
- slow snapshot/WS consumer 不反压 Runtime。

### 13.3 Storage

- 复制真实 YAML/SQLite 后执行 migration。
- migration 幂等、失败回滚、revision conflict 返回 409。
- 旧模型、Deployment 和 License 原地可读。

### 13.4 Protocol

- Axum router contract tests 覆盖关键路径、状态和 DTO。
- WebSocket topic、snapshot cadence、capacity-one 慢客户端。
- 当前前端 build 和主流程不需要同时连接 FastAPI。

### 13.5 FFI 和 Jetson

- C/Rust ABI version、size、offset 和 lifetime。
- pad probe 不保留 vendor pointer，不阻塞 producer。
- candidate 上限、NaN/Inf 和无 metadata 错误。
- Jetson 60 秒 latency/drop/RSS/CPU/temperature/clock；capture disconnect、kmNet timeout 和 fault injection。

## 14. Definition of Done

- `relink_server` 是唯一在线入口，在线主链不需要 Python 进程。
- 现有 Studio 主流程、YAML、SQLite、模型和 License 原地兼容。
- RuntimeManager 是 Run Intent、Runtime Phase 和 Runtime Epoch 的唯一 owner。
- 所有 frame/batch/command 携带 epoch、generation、monotonic timing 和 bounded age。
- latest frame、latest batch、latest command 都有 capacity-one 语义和覆盖计数。
- API、WS、日志和 store 不反压实时线程。
- capture/CUDA/worker/device fault 立即关闭旧命令发送并更新 Runtime Phase。
- Python/C++ 旧代码保留，但不拥有在线业务状态。
- Mac replay/contract gate 与 Jetson 实机 gate 全部通过后才切换 systemd。
