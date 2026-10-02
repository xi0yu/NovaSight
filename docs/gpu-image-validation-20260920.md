# Jetson 图片流水线验证收据：2026-09-20

结论：本次源码在 Jetson 上完成构建、GPU 差分检查、内存检查、正式程序三轮运行和故障注入。实际 CUDA 追踪未发现原始 YOLO 输出回传 CPU。**当前真实画面未产生检测框，因此尚未验收真实目标的识别准确性；也未完成全模型、全部输入格式或长期稳定性验收。** 范围止于有效检测结果，不调整检测后的业务算法。

## 身份与隔离

- 两端源码经 `git push` / `git pull --ff-only` 同步；被测提交：`7822825b52ee5959cc708ba5da14239ae433c5fc`，分支 `develop-alpha`。
- Jetson Orin Nano Super / aarch64；CUDA 12.6、TensorRT 10.3、DeepStream 7.1。
- 被测 daemon SHA-256：`c7b76036340314d86f98bdc0f96dd723c1a2d5afec905162f24c8251c3db277c`。Rust 是 Debug 构建，生产 CUDA 构建保持 `-O3` 且无 `-G`；不用于宣称 Release 性能提升。
- 活动 engine SHA-256：`0d034825230d8c01a5a451428082cfa01b6e0ad877e812d5c99dc57835187034`；输入 FP32 `[1,3,256,256]`、输出 FP32 `[1,9,1344]`、5 类。
- 保持原 1920×1080 / MJPG / 120 FPS，ROI `(800,380,320,320)`、阈值和模型不变。
- 构建及运行产物位于独立 `/tmp/novasight-p0-gpu-acceptance.BvhckN/`；保留原工作区未跟踪文件、模型、配置、数据库和旧二进制。没有切换运行包或系统服务。
- 使用既有隔离 runner，复制测试状态，关闭物理输出；所有运行检查 `output_gate_open=false`、`device_receipts=0`。

## 输入内容核实

先独立采集 120 个 JPEG 帧，解析出 114 种 JPEG 哈希。首帧全黑；中间及末帧为有正常内容的游戏画面，已人工查看中间帧，不是此前的“无信号源输入”占位屏。原始帧和样本仅保存在诊断目录，未提交 Git。

JPEG 文件哈希不同不单独证明画面有效；这里结合了尺寸检查、像素统计和实际查看。该短采集报告约 120.13 FPS 和 3 个丢失缓冲，不把它冒充零丢帧性能验收。

当前模型在后续运行的状态采样中仍全部为空检测。场景有画面不等于含有该模型可识别的目标；需要匹配模型的非空样本及参考结果，不能把空框标记成识别质量通过。

## 已通过的检查

| 检查 | 实际结果 | 证明范围 |
| --- | --- | --- |
| CUDA 模块 Release 构建及 CTest | 2/2 通过；YOLO 差分 40 组、预处理自检通过 | FP16/FP32、布局、objectness、非空合成候选、NMS、边界、复用及 CUDA 像素转换 |
| Compute Sanitizer memcheck | 同一 40 组通过，`ERROR SUMMARY: 0 errors` | 被测用例未发现 GPU 内存错误，不等于完整产品无限期无错 |
| TensorRT native Debug 构建及错误日志测试 | 1/1 通过 | 首次 SDK 原因保留、并发记录及有界截断 |
| Jetson Rust 最终结果测试 | 1/1 通过 | 结果数、截断、分数、类别、框、帧标识交付校验 |
| Jetson Rust 模型契约测试 | 5/5 通过 | 拒绝不支持模式、非法 shape、归一化溢出等 |
| 实际 engine 设备接口／CUDA Graph 检查 | `engine_gpu_pass=true` | 固定输入、CPU 参考对照、设备缓冲／Graph binding 生命周期；不是实际图片准确性测试 |
| 正式 daemon 三轮 15 秒运行 | 三轮 PASS，均正常停止、退出 0 | 实际采集到结果交付、错误边界、输出关闭 |
| GPU 导入故障注入 | PASS，保留具体阶段及 CUDA 原因 | 不回退 CPU，不继续正常交付，正常清理 |
| 正式 daemon Nsight 追踪 | PASS | 下述实际 GPU 内核和传输证据 |

Compute Sanitizer 仅为诊断子进程临时使用 `debug` 组；没有更改持久用户组、驱动或系统设置。

正式运行标识（`run-` 后缀）：

| 运行 | 最后状态中的已发布批次 | 稳态约 1 Hz 采样的 GPU worker 耗时范围 |
| --- | ---: | ---: |
| `1789914929982513132` | 1678 | 5.195–5.320 ms |
| `1789914987988669835` | 1699 | 5.245–5.385 ms |
| `1789915008773559204` | 1684 | 5.234–5.415 ms |

结果更新率采样约 120 FPS。上表排除前两次状态采样；**不是逐帧 P95/P99，也不含前面的媒体解码／VIC 处理**。批次数字是末次状态快照，不是启动至退出的总帧数。启动高水位还包含 Graph 捕获及初始排队，不拿它作为稳态延迟。

## GPU 执行与回传证据

追踪运行：`1789915067154906083`。对完整 Nsight SQLite 记录查询：

- CUDA `rgba_to_chw`：1830 次。
- CUDA YOLO `decode`：1830 次。
- CUB 降序排序：1830 次。
- CUDA `select_survivors` 完整 NMS：1830 次。
- CUDA `pack_result`：1830 次。
- D2H 最终结果：1830 次，每次 6152 字节。
- 另外 4 次 64 字节 D2H 全部发生在第一帧 CUDA 预处理之前。
- 48384 字节原始 YOLO 输出 D2H：**0 次**；D2H 记录中没有其他大小。
- H2D 只有一次 22293312 字节传输，没有逐帧图片重上传记录。

这证明被测程序／模型的 CUDA 图片预处理与 YOLO 后处理实际执行，未使用原始输出 CPU 后处理路径。媒体解码／VIC 使用现有 NVIDIA 硬件插件及 NVMM 表面；本次 CUDA 追踪不是这些专用引擎的独立活动时间线。

## 异常根因实测

故障运行 `1789915059357472622` 在第 200 次成功 GPU 导入后注入错误，实际保留的信息是：

```text
Strict GPU frame failed [epoch=RuntimeEpoch(1), generation=Generation(202), captured_at_ns=6512748393]:
TensorRT execute failed with code 1: GPU frame [NVMM/EGL import]:
cuGraphicsEGLRegisterImage: CUDA_ERROR_INVALID_VALUE (1): invalid argument
```

既有外层错误类型仍叫 `TensorRT execute`，但内层没有丢失真实导入阶段与 CUDA 原因。运行进入预期故障状态，设备输出始终关闭，runner 清理完成、daemon 退出 0。

## 收据位置与未完成项

本地 `out/diagnostics/jetson-gpu-20260920/` 保留 `runtime-results.json`、`cuda-trace-summary.json`、测试日志及私有输入样本；这些生成数据不提交 Git。完整 Nsight 报告、原始采集和每次运行状态在远端上述隔离目录。

尚未完成：匹配模型的真实非空检测对照、全部模型和采集格式覆盖、长时间稳定性、逐帧完整延迟；不推导 240 FPS 或与其他平台的性能比较。TensorRT 仍提示 engine 跨设备型号构建的兼容性警告，本次没有擅自重建模型。

本收据只覆盖被测源码提交。后续本收据／链接的文档提交不改变被测代码；任何后续代码修改都需要重新验证。
