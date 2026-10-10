NovaSight 详细实施蓝图
本蓝图将每个阶段分解为具体的任务、模块接口、数据流、线程交互、测试方法和验收指标，确保开发团队可直接按步骤执行，无歧义。

Phase 0：现状审计与基线测量
目标：全面了解现有代码、硬件真实能力，建立量化基线，为后续改造提供数据支撑。

0.1 硬件能力全面探测脚本
新增模块：scripts/probe_full.py（一次性运行）

任务：

调用 v4l2-ctl --list-devices 获取所有视频设备路径。

对每个 /dev/video* 执行：

v4l2-ctl -d X --all 获取驱动信息、总线信息。

v4l2-ctl -d X --list-formats-ext 获取支持的像素格式、分辨率、帧率（离散值）。

解析输出，生成 JSON 文件 device_capabilities.json，包含每个设备的结构化能力列表。

对采集卡（如 GC553G2）进行实际采集测试：

用 gst-launch-1.0 分别测试 MJPEG、NV12、YUYV 在目标分辨率（1080p, 1440p）下最高稳定帧率。

记录实际帧率、CPU/GPU 使用率（tegrastats）、是否出现 negotiated error 或丢帧。

输出 baseline_fps.json。

输出：

device_capabilities.json

baseline_fps.json

验收标准：

至少输出一张设备能力表，清晰显示每种格式可达到的最大稳定 FPS。

当前采集链路（若有代码）的实际性能指标（平均采集 FPS、延迟范围）被记录。

0.2 现有代码审计
任务：

列出项目中所有文件，标记与采集、推理、控制、UI 相关的模块。

识别：

硬编码采集格式（如 cv2.VideoCapture(0) 或固定 GStreamer 命令）。

GPU↔CPU 拷贝（numpy.frombuffer, cudaMemcpy 到 host）。

直接使用 PyTorch/ONNXRuntime 推理的代码。

控制输出部分是否包含时间戳、预测、调度。

生成 code_audit.md，指出需要删除或替换的具体行/模块。

验收标准：

审计报告清晰列出所有违规代码段及对应整改目标。

0.3 基线延迟测量
新增：scripts/latency_tester.py

任务：

若存在可运行的采集+推理链路，用 LED 或屏幕闪烁法测量端到端延迟。

记录从图像采集到 HID 信号输出的完整延迟（至少 p50, p95, p99）。

若无可运行链路，使用 gst-launch 测量采集到显示延迟。

验收标准：

得到一组基线延迟数据，用于 Phase 9 对比。

Phase 1：采集能力探测与 PipelinePlanner
目标：建立自动设备探测、格式选择、PipelinePlan 生成及缓存机制，替代所有硬编码采集。

1.1 DeviceCapabilityProbe 模块
文件：nova-engine/capture/device_probe.py

类接口：

python
@dataclass
class DeviceCapability:
    device_path: str          # /dev/video0
    driver: str
    card_name: str
    bus_info: str
    pixel_formats: List[PixelFormatCaps]

@dataclass
class PixelFormatCaps:
    format: str               # 'MJPG', 'NV12', 'YUYV'
    description: str
    resolutions: List[ResolutionCaps]

@dataclass
class ResolutionCaps:
    width: int
    height: int
    fps_list: List[Fraction]  # discrete FPS values

class DeviceProbe:
    @staticmethod
    def probe_all() -> List[DeviceCapability]:
        """调用系统工具，解析输出，返回所有设备能力。"""
    @staticmethod
    def probe_device(device_path: str) -> DeviceCapability:
        """探测单个设备。"""
实现细节：

使用 subprocess 调用 v4l2-ctl，解析输出文本。

解析 gst-device-monitor-1.0 Video/Source 作为补充。

所有解析逻辑必须稳定，出错时记录异常并跳过设备。

测试：

单元测试使用 mock v4l2-ctl 输出，验证解析正确性。

在实际 Jetson 设备上运行，确保返回设备列表非空。

1.2 PipelinePlanner
文件：nova-engine/capture/pipeline_planner.py

数据结构：

python
@dataclass(frozen=True)
class PipelinePlan:
    source_type: str          # 'v4l2'
    device: str
    capture_format: str       # 'MJPG', 'NV12', 'YUYV'
    capture_width: int
    capture_height: int
    capture_fps: Fraction
    decode_backend: str       # 'nvv4l2decoder' or 'none'
    memory_domain: str        # 'NVMM'
    crop_rect: Optional[RoiRect]
    inference_width: int
    inference_height: int
    model_profile: str        # model_hash
    sink_type: str            # 'fakesink', 'appsink'
    latency_policy: str       # 'drop_old'
    gst_template: str         # 预定义的 GStreamer 管道模板

    def generate_gst_launch_string(self) -> str:
        """生成验证用的 gst-launch 命令。"""
核心方法：

python
class PipelinePlanner:
    @staticmethod
    def plan(
        capabilities: List[DeviceCapability],
        device: str,
        target_res: Tuple[int, int],
        target_fps: Fraction,
        model_input_size: Tuple[int, int],
        model_hash: str,
        roi_config: RoiConfig
    ) -> PipelinePlan:
        """选择最佳采集格式并生成计划，若不可行抛出 InfeasibleConfiguration。"""
选择逻辑（伪代码）：

text
候选列表 = []
for device in capabilities if device matches:
    for fmt in device.pixel_formats:
        for res in fmt.resolutions if res matches target_res:
            if target_fps in res.fps_list:
                candidate = evaluate(fmt, res)
                候选列表.append(candidate)
# 评估函数
def evaluate(fmt, res):
    # 估算带宽，解码成本，进入 NVMM 的路径
    if fmt == 'NV12' and 带宽允许 and 稳定测试通过:
        score = 100  # 无解码
    elif fmt == 'MJPG' and 解码支持:
        score = 90 - decode_cost
    else:
        score = 50
    return (score, candidate)
选择 score 最高且实际测试稳定的候选。
实际测试验证：

对最终候选，用 gst-launch-1.0 运行 5 秒，检查无错误且达到目标 FPS。

若不满足，降级到下一个候选。

缓存策略：

缓存键：(device serial/bus, capture_format, width, height, fps, model_hash, roi_config) 的 SHA256。

缓存文件：/var/cache/novasight/pipeline_plans/{hash}.json。

加载时检查缓存，存在且校验通过则直接返回。

测试：

单元测试：模拟设备能力，验证在不同条件下选择出预期格式。

集成测试：连接真实采集卡，调用 plan() 后运行管道验证稳定。

1.3 CaptureLoop 骨架（初始版）
文件：nova-engine/capture/capture_loop.py

类接口：

python
class LatestFrameBuffer:
    """单槽缓冲区，存放 (GstBuffer, FrameContext)，原子操作。"""
    def put(self, buffer: Gst.Buffer, ctx: FrameContext): ...
    def get(self) -> Tuple[Gst.Buffer, FrameContext]: ...

class CaptureLoop(threading.Thread):
    def __init__(self, pipeline_plan: PipelinePlan, buffer: LatestFrameBuffer):
        self.plan = pipeline_plan
        self.buffer = buffer
        self.pipeline = None   # Gst.Pipeline
    def run(self):
        # 根据 plan 构建并运行 GStreamer pipeline
        # 在 appsink 的 new-sample 回调中调用 buffer.put()
    def stop(self):
        # 发送 EOS，等待线程结束
注意：此阶段暂时只构建测试 pipeline，不接入 DeepStream，但需预留帧元数据写入接口。

验收标准 (Phase 1)：

DeviceCapabilityProbe 能正确输出所有设备能力。

PipelinePlanner 在给定需求下生成唯一计划，并使用 gst-launch 验证通过。

多次相同输入调用返回相同计划（缓存命中）。

废弃所有硬编码采集命令。

Phase 2：NVMM / CUDA 数据面整改
目标：确保从采集 buffer 到推理输入全程保持在 NVMM 内存，消灭 GPU↔CPU 拷贝。

2.1 验证 GStreamer 管道 NVMM 路径
新增：nova-engine/pipelines/gst_validator.py

任务：

为每种采集格式（MJPEG, NV12）构建最小管道，如：

text
v4l2src device=X ! ... ! nvv4l2decoder ! video/x-raw(memory:NVMM) ! fakesink
在 fakesink 前添加 probe，检查 buffer 的 memory type 是否为 GstNvBufMemory 或类似 NVMM 标志。

记录验证结果到日志。

工具：

python
def is_nvmm_buffer(buffer: Gst.Buffer) -> bool:
    # 检查 buffer 的 memory 是否 nvmm 类型
使用 Gst.MapInfo 或 Python 绑定检查 memory flags。

2.2 实现 LatestFrameBuffer 与 CaptureLoop 集成
修改：capture_loop.py

从初始版本升级，确保 appsink 输出 buffer 直接共享到 LatestFrameBuffer，不做任何 extract_dup 或拷贝。

FrameContext 填充 capture_timestamp_ns、dequeue_timestamp_ns（从 v4l2 buffer 获取）。GStreamer 中可通过 GstVideo.VideoTimeCodeMeta 或 buffer pts 获取。

GStreamer 时间戳获取：

使用 do-timestamp=true 让 v4l2src 设置 buffer pts。

在 appsink 回调中，buf.pts 即为采集时间（纳秒），用于填充 capture_timestamp_ns。

2.3 删除所有 CPU 拷贝
审计：搜索 numpy.frombuffer, cudaMemcpy, memcpy, Gst.Buffer.extract_dup，全部替换为传递 Gst.Buffer 引用。

特殊处理：如果在 Phase 0 发现任何地方将 GPU buffer 映射到 CPU（比如使用 OpenCV 的 cv2.cvtColor 处理 NV12），必须重写为使用 nvvideoconvert 在 GPU 内完成。

2.4 集成测试：NVMM 一致性
测试脚本：tests/test_nvmm_path.py

构建采集管道，在 probe 中确认 buffer NVMM。

测量 LatestFrameBuffer 传递 buffer 所需时间，应远小于 1ms。

确认多个消费周期后无内存泄漏（使用 pygobject 的 refcount 或 tegrastats 观察）。

验收标准：

采集管道明确输出 NVMM memory buffer。

无任何 Gst.Buffer 拷贝到 CPU 侧。

LatestFrameBuffer 工作正常，数据面 CPU 使用率显著降低。

Phase 3：DeepStream / TensorRT 推理接入
目标：用 DeepStream pipeline 替代独立推理，将推理整合进采集管道，自动生成 nvinfer 配置。

3.1 DeepStream 管道模板
文件：nova-engine/pipelines/deepstream_app.py

类：

python
class DeepStreamPipeline:
    def __init__(self, plan: PipelinePlan, model_manifest: ModelManifest):
        # 构建完整管道：
        # v4l2src -> ... -> nvstreammux -> nvinfer -> fakesink
        # 包含 ROI、预处理、队列配置
    def start(self): ...
    def stop(self): ...
    def attach_probe(self, pad, callback): ...
管道构建细节：

nvstreammux 参数：live-source=1, batch-size=1, width=roi_width, height=roi_height, nvbuf-memory-type=3 (NV12 NVMM)。

在 nvinfer 后连接 nvtracker（可选，当前阶段可关闭）。

队列元素：全部 max-size-buffers=1 leaky=downstream。

使用 fakesink sync=0 作为终点。

3.2 nvinfer 配置自动生成
文件：nova-engine/inference/nvinfer_config.py

函数：

python
def generate_nvinfer_config(manifest: ModelManifest) -> str:
    # 生成 config 文本，包括：
    # model-engine-file, batch-size, network-mode, process-mode
    # custom-lib-path (如果模型需要自定义解析)
    # output-blob-names
    # ...
    return config_string
集成缓存：使用 engine_cache.py 决定 engine 文件路径。

3.3 YOLO 输出解析与 NMS 策略实现
文件：nova-engine/inference/decoders/yolo_v5.py, yolo_v8.py 等。

注册机制：

python
PARSER_REGISTRY = {
    'yolov5': YoloV5Decoder,
    'yolov8': YoloV8Decoder,
    'yolov8_e2e': YoloV8EndToEndDecoder,
    ...
}
接口：

python
class BaseDecoder(ABC):
    @abstractmethod
    def decode(self, layer_outputs: List[np.ndarray], **kwargs) -> List[Detection]:
        # 输入 tensor 已在 GPU？这里可能需要拷贝到 CPU 做后处理，但尽量使用 GPU NMS。
        pass
推荐：对于支持 EfficientNMS 的 YOLOv8，在 ONNX 中融合 NMS，DeepStream nvinfer 直接输出最终检测框，无需 CPU 解码。因此，优先使用模型内置 NMS，避免后处理拷贝。

若无，则使用 DeepStream 自定义解析函数（C 库）在 GPU 侧解码。Python 解码仅作兼容，且需标记为性能降级。

3.4 Probe 回调与 DetectionFrame 生成
文件：nova-engine/pipelines/probes.py

python
def detection_probe(pad, info, user_data):
    gst_buffer = info.get_buffer()
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    # 提取检测框
    detections = []
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            det = Detection(
                class_id=obj_meta.class_id,
                confidence=obj_meta.confidence,
                bbox=(obj_meta.rect_params.left, ...),
                center=(obj_meta.rect_params.left + ...),
                source_frame_id=frame_meta.frame_num,
                source_timestamp_ns=frame_meta.buf_pts
            )
            detections.append(det)
            l_obj = l_obj.next
        l_frame = l_frame.next
    # 将 DetectionFrame 放入单槽
    latest_detection_slot.put(DetectionFrame(...))
    return Gst.PadProbeReturn.OK
单槽：DetectionSlot 类（类似 LatestFrameBuffer，存放最新 DetectionFrame）。

3.5 推理线程集成
新增：nova-engine/core/inference_loop.py

python
class InferenceLoop(threading.Thread):
    def run(self):
        # 等待 DetectionSlot 有新帧，但实质是 probe 异步填充，此线程可消费并驱动后级
        # 或者直接让 probe 调用控制逻辑？不，probe 应快速返回，这里我们使用一个轻量级消费者
        pass
实际上，DeepStream pipeline 运行在自己的线程中，probe 填充 DetectionSlot。我们可以启动一个单独的控制线程消费 DetectionSlot。推理线程概念弱化，数据驱动由 pipeline 线程完成。

注意：DeepStream 主循环需要不断迭代（g_main_loop_run），我们将其放在单独线程中即可。

验收标准：

DeepStream 管道成功运行，nvinfer 输出检测结果。

检测框位置在源画面中准确（可通过保存截图验证）。

推理 FPS 稳定且不低于采集 FPS（无积压）。

模型引擎缓存工作正常，第二次启动不使用重建引擎。

Phase 4：Detection / Tracker / Selector 数据模型整改
目标：规范化检测、跟踪、目标选择的数据流，实现稳定目标锁定。

4.1 NvTracker 集成
修改：在 DeepStream 管道中启用 nvtracker 插件，配置追踪器类型（如 IOU, NvSORT 等），输出 NvDsObjectMeta 带 track_id。

元数据提取：在 probe 中同时提取 track_id 并构造 Track 对象。

Track 数据结构（定义在 detection/track.py）：

python
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int,int,int,int]  # 源坐标
    velocity_px_s: Tuple[float,float]  # 基于时间戳计算
    quality_score: float
    missed_frames: int
    last_seen_ns: int
    is_predicted: bool = False
    is_stale: bool = False
4.2 TargetSelector 实现
文件：nova-engine/detection/target_selector.py

类：

python
class CandidateFilter:
    def filter(self, tracks: List[Track]) -> List[Track]:
        # 类别、置信度、尺寸、FOV边缘过滤等

class QualityScorer:
    def score(self, track: Track, current_time: int) -> float:
        # 综合距离中心、置信度、年龄、速度稳定性等

class TargetSelector:
    def __init__(self, config: UserConfig):
        self.filter = CandidateFilter()
        self.scorer = QualityScorer()
        self.current_target = None
        self.switch_cost = config.switch_cost

    def select(self, tracks: List[Track], now_ns: int) -> Optional[Track]:
        candidates = self.filter.filter(tracks)
        if not candidates:
            # 如果当前目标有短期预测，可返回预测track
            return self.handle_lost()
        scored = [(self.scorer.score(t, now_ns), t) for t in candidates]
        scored.sort(reverse=True)
        best_score, best_track = scored[0]
        if self.current_target is not None:
            # 切换惩罚逻辑
            if self.current_target.track_id != best_track.track_id:
                if best_score < self.current_score + self.switch_cost:
                    best_track = self.current_target
        self.current_target = best_track
        return best_track
4.3 KalmanEstimator 边界
文件：nova-engine/detection/kalman_estimator.py

python
class KalmanEstimator:
    def __init__(self):
        self.kf = cv2.KalmanFilter(...)  # 可接受，仅用于状态估计
        self.last_update_ns = 0
        self.ttl_ns = 50_000_000  # 50ms

    def update(self, track: Track, now_ns: int):
        """更新测量值。"""
        # 更新滤波器
        self.last_update_ns = now_ns

    def predict(self, now_ns: int) -> Optional[Track]:
        """预测当前目标位置，如果超过TTL则返回None。"""
        age = now_ns - self.last_update_ns
        if age > self.ttl_ns:
            return None
        # 执行预测，生成预测Track，标记 is_predicted=True
        return predicted_track
集成：TargetSelector 在丢失真实检测时，可调用 KalmanEstimator.predict 获取短期预测，但必须标记 is_predicted，并且只在 TTL 内有效。

4.4 坐标映射验证
实现：detection/roi.py 中的 RoiTransformer，确保从模型坐标（或 tracker 输出）映射回源画面时，考虑了裁剪、缩放、letterbox padding。

测试：固定一张测试图像，经过 pipeline 后，将源坐标的检测框绘制上去，目视验证。

验收标准：

多目标场景下 Tracker 稳定分配 ID，无频繁跳变。

TargetSelector 能锁定一个目标并平滑切换。

短暂遮挡（<100ms）后仍可预测并恢复。

坐标映射准确，可视化验证无误。

Phase 5：时间模型、预测与闭环控制整改
目标：实现基于统一时间戳的延迟补偿，完成 error_px → error_rad → angular PD 转换。

5.1 全链路时间戳贯通
任务：在 FrameContext 中增加字段，并在 Pipeline 各阶段填充：

capture_timestamp_ns: 来自 v4l2 buffer pts

dequeue_timestamp_ns: appsink 出队时间（可在 probe 中记录 time.monotonic_ns()）

decode_timestamp_ns, roi_timestamp_ns, inference_start_ns, inference_end_ns

这些时间点可在 probe 或 pipeline 元素 pad probe 中记录。

实现：DeepStream 元素不直接暴露这些，我们可以在 nvstreammux 的 sink pad probe 记录进入时间，在 nvinfer src pad probe 记录输出时间，近似推断。

简化方案：在 CaptureLoop appsink 回调中记录 dequeue_ns，在 probe 得到检测框时记录 postprocess_ns。控制线程消费时记录 control_start_ns。足够计算主要延迟。

5.2 LatencyCompensator
文件：nova-engine/control/latency_compensator.py

python
class LatencyCompensator:
    def __init__(self, calibration: Calibration):
        self.estimated_output_latency_ns = 0  # 动态估计
        self.actuation_delay_ns = calibration.actuation_delay_ns

    def compute_horizon(self, measurement_ns: int, now_ns: int) -> int:
        measurement_age = now_ns - measurement_ns
        return measurement_age + self.estimated_output_latency_ns + self.actuation_delay_ns

    def update_latency_estimate(self, total_latency_ns):
        # 简单移动平均
        self.estimated_output_latency_ns = 0.9 * self.estimated_output_latency_ns + 0.1 * total_latency_ns
5.3 AngularController 实现
文件：nova-engine/control/controller.py

python
class AngularController:
    def __init__(self, calib: Calibration, config: UserConfig):
        self.focal_x = calib.width / (2 * math.tan(calib.fov_x / 2))
        self.focal_y = calib.height / (2 * math.tan(calib.fov_y / 2))
        self.kp = config.kp
        self.kd = config.kd
        # 状态
        self.prev_error_rad = (0.0, 0.0)
        self.prev_time_ns = 0

    def compute(self, error_px: Tuple[float, float], now_ns: int) -> Tuple[float, float, float, float]:
        """返回 error_rad_x, error_rad_y, control_x_rad, control_y_rad"""
        ex = error_px[0] - (calib.width / 2)  # 注意：输入应为像素坐标差
        ey = error_px[1] - (calib.height / 2)
        # 转弧度
        ex_rad = math.atan(ex / self.focal_x)
        ey_rad = math.atan(ey / self.focal_y)
        # 微分
        dt = (now_ns - self.prev_time_ns) * 1e-9
        if dt <= 0:
            de_dt = (0.0, 0.0)
        else:
            de_dt = ((ex_rad - self.prev_error_rad[0]) / dt, (ey_rad - self.prev_error_rad[1]) / dt)
        # PD
        ux = self.kp * ex_rad + self.kd * de_dt[0]
        uy = self.kp * ey_rad + self.kd * de_dt[1]
        # 更新状态
        self.prev_error_rad = (ex_rad, ey_rad)
        self.prev_time_ns = now_ns
        return (ex_rad, ey_rad, ux, uy)
5.4 控制线程集成
新增：nova-engine/core/control_loop.py

python
class ControlLoop(threading.Thread):
    def run(self):
        while running:
            det_frame = detection_slot.get_latest()  # 非阻塞，若无新帧则可能空转或等待事件
            if det_frame is None:
                time.sleep(0.001)
                continue
            now_ns = time.monotonic_ns()
            # 选择目标
            target = selector.select(tracks, now_ns)
            if target is None:
                # 无目标，清理状态
                scheduler.cancel_all()
                continue
            # 如果is_predicted，通过Kalman获取预测位置
            if target.is_predicted:
                predicted = kalman.predict(now_ns)
                if predicted is None:
                    scheduler.cancel_all()
                    continue
                target = predicted
            # 计算误差
            error_px = (target.center[0], target.center[1])
            # 延迟补偿：计算预测horizon，并用速度预测目标位置
            horizon = latency_comp.compute_horizon(target.last_seen_ns, now_ns)
            predicted_center = (target.center[0] + target.velocity_px_s[0] * horizon * 1e-9,
                                target.center[1] + target.velocity_px_s[1] * horizon * 1e-9)
            error_px = (predicted_center[0], predicted_center[1])
            # 控制器
            ex_rad, ey_rad, ux, uy = controller.compute(error_px, now_ns)
            # 转换为counts
            counts_x = ux * calib.counts_per_360_x / (2 * math.pi)
            counts_y = uy * calib.counts_per_360_y / (2 * math.pi)
            # 生成ControlIntent
            intent = ControlIntent(...)
            scheduler.schedule(intent, now_ns)
注意：此线程频率不应固定，而是由新检测帧驱动，确保每次最新检测都被处理，但内部需限制执行频率（如不低于 200Hz）用睡眠补偿，避免空转。

验收标准：

全链路时间戳记录完整，可通过 telemetry 查询。

延迟补偿逻辑正确（可通过模拟延迟测试）。

AngularController 输出合理，微分项能正常工作。

控制环在无目标时停止输出 counts。

Phase 6：Scheduler 与 HID 输出可靠性整改
目标：实现 Scheduler，解决控制量过冲和堆积，确保 HID 输出稳定可靠。

6.1 Scheduler 实现
文件：nova-engine/control/scheduler.py

python
class Scheduler:
    def __init__(self, config: UserConfig, hid_output: HidOutput):
        self.max_counts_per_step = config.max_counts_per_step
        self.min_interval_us = 1000  # 1ms
        self.pending_counts = (0, 0)
        self.last_sent_ns = 0
        self.active = False

    def schedule(self, intent: ControlIntent, now_ns: int):
        # 如果有新帧，旧计划作废
        self.pending_counts = intent.raw_counts
        self.active = True
        # 不立即发送，而是交给 pacing 循环
        self._pace(now_ns)

    def _pace(self, now_ns):
        if not self.active:
            return
        # 限制单次发送大小
        step_x = min(max(abs(self.pending_counts[0]), self.max_counts_per_step), abs(self.pending_counts[0]))
        step_y = min(max(abs(self.pending_counts[1]), self.max_counts_per_step), abs(self.pending_counts[1]))
        # 方向处理
        sign_x = 1 if self.pending_counts[0] > 0 else -1
        sign_y = 1 if self.pending_counts[1] > 0 else -1
        out_x = step_x * sign_x
        out_y = step_y * sign_y
        # 发送
        hid_output.send(out_x, out_y)
        self.pending_counts = (self.pending_counts[0] - out_x, self.pending_counts[1] - out_y)
        self.last_sent_ns = now_ns
        if abs(self.pending_counts[0]) < 0.5 and abs(self.pending_counts[1]) < 0.5:
            self.active = False
与事件驱动集成：控制线程每次计算 intent 后调用 scheduler.schedule()，Scheduler 内部使用定时器或一个独立线程按 min_interval_us 循环发送。为简化，可以拥有一个独立输出线程。

python
class HidOutputThread(threading.Thread):
    def run(self):
        while running:
            now = time.monotonic_ns()
            scheduler.tick(now)   # tick 检查是否有 pending 且满足间隔则发送小步
            time.sleep(0.0005)    # 0.5ms 精度
6.2 HID 输出设备封装
文件：nova-engine/control/hid_output.py

python
class HidOutput:
    def send(self, dx: int, dy: int):
        """调用 KMBox API 或直接写 HID 设备，发送相对移动。"""
处理反向：在 schedule 中如果新 intent 方向与 pending 方向相反，立即清零 pending，仅保留最新意图。

验收标准：

发送单次大控制量时，系统输出被拆分为多个小步，间隔均匀。

新帧到达时旧 pending 被清除。

方向反转时立即停止当前输出，无反向冲量。

HID 设备实际发送速率不超过设备限制。

Phase 7：配置缓存、Engine 缓存和模型管理
目标：实现智能缓存，使用户导入模型后自动生成所需资源，避免重复构建。

7.1 EngineCache
文件：nova-engine/inference/engine_cache.py

python
class EngineCache:
    @staticmethod
    def get_engine_path(manifest: ModelManifest) -> str:
        cache_key = compute_cache_key(manifest)
        path = f"models/generated/{cache_key}/engine.plan"
        if os.path.exists(path):
            if valid_engine_for_current_platform(path):
                return path
        return None  # 需要构建

    @staticmethod
    def build_engine(manifest: ModelManifest) -> str:
        # 调用 trtexec 或 TensorRT Python API 构建
        # 保存 manifest.json 和 build_info.json
        return engine_path
缓存键组成：hash(model_file) + trt_version + cuda_version + gpu_arch + precision + input_dims + ...

7.2 ModelManifest 自动生成
工具：scripts/import_model.py

用户选择 ONNX 文件。

使用 onnxruntime 或 ONNX Python API 分析输入输出：

输入形状、名称

输出张量形状、名称

推断模型类型（如 yolov8 根据输出节点名称）

填充模板 manifest，用户可编辑阈值等参数。

保存至 models/originals/ 和 manifest。

7.3 API 支持
POST /api/v1/models/import 触发导入。

GET /api/v1/models 列出所有已注册模型。

POST /api/v1/models/{id}/build 构建引擎。

验收标准：

导入模型后系统自动生成 manifest 和引擎（或计划构建）。

第二次启动相同模型不触发构建。

更改阈值不会导致引擎重建。

模型文件变化、平台变化会导致引擎重建。

Phase 8：FastAPI 与 NovaSight Studio 接入
目标：提供完整的 REST API 和 WebSocket，驱动前端界面。

8.1 FastAPI 应用骨架
文件：nova-engine/api/app.py

python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import status, capture, models, control, system
from .websocket import ws_router

app = FastAPI(title="NovaSight Engine API")
app.add_middleware(...)
app.include_router(status.router, prefix="/api/v1")
app.include_router(capture.router, prefix="/api/v1/capture")
app.include_router(models.router, prefix="/api/v1/models")
app.include_router(control.router, prefix="/api/v1/control")
app.include_router(system.router, prefix="/api/v1/system")
app.include_router(ws_router, prefix="/ws")
8.2 路由实现示例
status.router：依赖注入 CoreEngine 实例（单例），返回当前 RuntimeState 快照。

capture.router：GET /capabilities 返回 DeviceProbe 结果；PUT /config 接收 PipelinePlan 参数并应用。

models.router：模型列表、导入、构建、获取 manifest。

control.router：获取/更新 Calibration 和 UserConfig。

system.router：返回 JetPack 版本、GPU 温度等（通过 tegrastats）。

8.3 WebSocket 实时推送
python
@ws_router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        # 每 50ms 推送一次 telemetry 数据
        data = telemetry.get_summary()
        await websocket.send_json(data)
        await asyncio.sleep(0.05)
8.4 NovaSight Studio 前端集成
使用 Tauri 创建桌面窗口，内嵌 React 应用。

React 通过 HTTP 请求和 WebSocket 与后端通信。

页面设计与之前定义一致。

验收标准：

所有 API 端点响应正确，WebSocket 稳定连接。

Studio 可以切换采集格式、导入模型、调整控制参数并实时查看性能。

前端无操作直接控制底层 GStreamer 对象。

Phase 9：性能验证、稳定性测试和生产部署
目标：确保系统满足生产级稳定性、低延迟要求，并完成 systemd 部署。

9.1 性能基准测试
脚本：tests/perf_test.py

运行 1 小时，记录 telemetry 中每帧各阶段耗时。

计算 p50, p95, p99 端到端延迟（capture -> HID output）。

统计掉帧数、覆盖帧数、stale detection 次数。

与 Phase 0 基线对比。

要求：p95 延迟 < 20ms（理想 < 15ms），采集 FPS 稳定在目标值 ±1%。

9.2 异常注入测试
模拟采集设备断开：拔掉 USB，观察系统是否自动停止 HID 输出并进入 DEGRADED 状态，重连后能否恢复。

模拟推理过载：增加 CPU/GPU 负载，验证队列丢弃旧帧，不崩溃。

模拟模型错误：输入错误 ONNX，验证引擎构建失败处理，不影响核心服务。

9.3 systemd 集成与看门狗
文件：deploy/novasight.service

配置 WatchdogSec=10，在主循环中定期调用 sd_notify("WATCHDOG=1")。

设置 Restart=on-failure，确保崩溃后自动重启。

日志输出到 journald，同时保留本地文件。

单实例锁：使用 /run/novasight/instance.lock 文件锁，并在 API 层拒绝第二个实例启动。

9.4 生产部署文档
编写 DEPLOY.md，包含硬件设置、依赖安装、服务启动步骤。

验收标准：

24 小时连续运行无崩溃、无内存泄漏。

p95 延迟达标。

设备断开恢复后系统可自动重新采集（或需手动重启，取决于策略，但至少安全处理）。

systemd 服务可正常启停，看门狗有效。

实施顺序依赖图
text
Phase 0  ──> Phase 1 ──> Phase 2 ──> Phase 3 ──> Phase 4 ──> Phase 5 ──> Phase 6
                                                 \                    /
                                                  └── Phase 7 ──> Phase 8 ──> Phase 9
Phase 7 可与 Phase 4-6 并行，Phase 8 依赖核心引擎稳定，Phase 9 在所有功能完成后进行。


