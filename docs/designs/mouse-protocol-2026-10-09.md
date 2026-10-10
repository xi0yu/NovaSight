# 鼠标协议实现与验证

页面与导航名称为“鼠标协议”；内部保留 `control-test` 路由，已有链接继续可用。

## 输出消费端

沿用核心 `PointerDevice` 接口接入采集 → 推理 → 控制 → 输出。新增公开的 `NullPointerDevice`：接受类型化命令并返回消费回执，不保存命令列表、不连接网络、不读取硬件按钮、不发送位移。

新建配置默认选择“空”。已有 Native UDP 配置保留，并在界面导入为已有协议卡片。空协议在主链里使用 `Always` 触发模式，使控制结果可以到达静默消费端；配置中原有的物理按键触发偏好仍保留，切回 kmBox 后恢复。

kmBox Net、NetPro8K、NB、AI 复用现有 `KmNetNativeDevice` 实现，不重复增加不同型号的 UDP 类。型号字段用于卡片与官方实物照片；官方文档声明的 Net 协议兼容性不等于硬件实测。

## 配置与调用入口

新增型号、IP、Port、UUID 完整校验后，保存一个独立协议。添加不会自动选中物理消费端；切换与编辑要求主链停止。保存调用既有配置事务，立即安装运行配置并原子写入文件，失败保留输入、给出红色提示。

配置库公开 `MouseProtocol`、`MouseProtocolModel`，`DeviceConfig` 新增 `mouse_protocols` 与 `active_mouse_protocol_id`。协议限制为 64 个，ID 唯一，地址必须是 IPv4，端口为 1–65535，UUID 为 8 位十六进制字符。选中物理协议时，其 IP/Port/UUID 必须与实际适配器连接字段匹配，避免卡片与发包目标不一致。

```yaml
hardware:
  backend: "null"
  auto_connect: true
  active_mouse_protocol_id: empty
  mouse_protocols:
    - id: office-net
      model: net_pro8k
      host: 192.168.2.188
      port: 8888
      uuid: A1B2C3D4
```

外部调用沿用 `ConfigService::replace_runtime` / `/api/config` 的完整配置事务与 revision 比较。选择 `office-net` 时，须同时写入 `backend: native_udp`、`active_mouse_protocol_id: office-net` 以及对应的 `host`、`port`、`uuid`；旧配置没有协议 ID 时继续按原有连接工作。

参数页面继续通过公开 `ConfigService::update_parameters` / `/api/v1/config/parameters` 自动更新运行时并写入文件，无保存二次确认。前端共享保存队列、冲突重读与顶部居中的红绿提示；协议列表在冲突重读后重建，避免覆盖其他调用者新增的协议。

`/api/executors/kmnet/probe` 只做停止状态下的设备握手与监控连接测试，随后断开；不启动主链或发送位移。空协议明确拒绝此端点，不能冒充“设备已连接”。状态投影将空协议显示为安全的静默消费端，不报告 kmBox 的物理连接。

空协议不需要硬件控制授权或物理输出确认；待应用的物理协议、以及开启输出门后从空切到物理协议，仍需要相应授权与确认。普通运行授权要求保留。

## 图片与布局

卡片内部网格上方 95% 为图片区，下方 5% 为文字区，两区均垂直居中。文字展示型号与 IP，完整端口在卡片可访问名称和选中协议详情中显示。图片完整展示，不裁切盒子。

四种型号图片来自 https://www.kmbox.top/documents.html 对应官方原图，作为本地静态资产打包，无外链热加载。来源记录见 `web/src/assets/output-consumers/SOURCES.md`；空卡片使用项目软件光标 SVG。

预览地址：`http://127.0.0.1:7351/preview.html?page=control-test`。预览新增与选择仅更新本页内存状态，不写文件、不连接设备；真实握手测试保持禁用。

## 验证范围

- 前端完整单元检查 176 项通过；最终协议页面、保存、控制诊断与运行状态提示的定向检查 51 项通过，其中新增提示检查使当前单元用例总数为 177。
- 五个相关 Rust 包的 192 项单元、集成检查通过；后续 API、runtime、daemon 定向复查 44 项通过，含新增空协议触发模式用例，当前对应总数为 193。
- 完整浏览器检查 51 项通过，包含桌面与手机；覆盖四种官方图片、95/5 比例、垂直居中、新增、选择、刷新保留、切回空、握手不发位移、预览不访问设备 API。
- 移除旧的受控移动提示后，协议页面与断连恢复的最终桌面/手机检查 10 项通过。Chrome 预览保留空协议及 NetPro8K、NB、AI 演示卡片；这些演示记录只存在于该页面内存，截图为 `out/mouse-protocol-preview-2026-10-09.png`。
- TypeScript、Vite 生产构建、daemon/web/client 的 Cargo check、diff 空白检查通过。
- 没有连接真实 kmBox，没有在 Jetson 上编译或运行原生适配器；协议兼容与连接状态需要实机验证。

检查记录位于 `out/mouse-protocol-*.log` 与对应单元 JSON 结果。没有提交、推送或改写已有用户配置。

## 页面精简与卡通样式调整

按用户补充要求，页面可见主体仅保留卡片格栅与“新增协议”按钮。流程说明、预览说明条、当前协议状态、直接展示的测试控件和高级诊断均移出主体；页面标题保留为不可见的语义标题，以维持导航焦点与屏幕阅读器支持。

点击物理协议卡片打开配置弹窗；实际启用与取消由卡片内的独立按钮完成。编辑、连通性测试及折叠的高级诊断均位于该弹窗内；“空”协议继续只负责静默消费。连通性测试遇到连接字段编辑时先保存，保存失败不调用测试。嵌套物理移动确认的 Escape 和焦点恢复保持有效。

卡片、按钮与弹窗采用局部粉彩色、圆角、贴纸描边和柔和立体投影；实物照片未作编辑，95% 图片/5% 文字比例保持。局部按钮规则覆盖全局主题规则，主按钮文字对比度检查达到 4.5:1。卡片保持固定尺寸，不使用倾斜动画。

本次定向单元检查 48 项通过，桌面/手机浏览器检查 12 项通过；最后按钮配色调整后的协议专项检查 6 项通过。TypeScript、Vite 构建及 diff 空白检查通过。后端未修改，也没有连接真实硬件。

## 固定卡片尺寸、表单默认值与显式启用

卡片桌面固定为 320×320，760px 以下固定为 256×256，格栅不拉伸卡片。图片区和文字区继续占 95% / 5%，启用按钮叠放在图片区右下角，按钮与编辑入口互为独立元素，无嵌套按钮。

IP、Port、UUID 的占位提示与实际留空默认值共用同一组值：192.168.2.188 / 8888 / A1B2C3D4。空白字符串和清空的 Port 使用默认值，非空输入继续校验。新增不会自动启用设备。新增及编辑弹窗使用不透明的浅色背景，取消背景模糊和弹窗偏移的贴纸影子，底层页面不会透出。

“启用”通过完整配置事务写入 backend、active_mouse_protocol_id、连接字段并保存；“取消”保留所有协议记录，切回 backend:null 与 active_mouse_protocol_id:empty。点击卡片图片仅编辑。主链运行时禁止更换协议。

用户提出的 IMouseController 概念在当前 Rust 项目中的现有接口是 PointerDevice，实际硬件实现为 KmNetNativeDevice，空实现为 NullPointerDevice。沿用 ConfigService 的 stopped-runtime 安装与 RuntimeDependencies::with_device_factory，不额外创建同职责的包装层：真实 daemon 的工厂根据配置返回硬件或空对象，install_app_config 替换存储的 Arc<dyn PointerDevice>。新增单元检查通过工厂替身和 Arc::ptr_eq 验证启用、取消确实替换安装对象；该检查不证明硬件连通性。

本轮验证：49 项前端定向单元、7 项桌面/手机浏览器、24 项 runtime 单元全部通过；TypeScript、Vite 生产构建、Cargo fmt 与 diff 空白检查通过。浏览器验证精确尺寸、95/5 比例、官方图片加载、留空提交、启用后刷新保留、取消回空以及预览没有设备 API 流量。预览截图：out/mouse-protocol-square-cards.png、out/mouse-protocol-square-dialog.png。

没有连接实物设备、运行 Jetson 适配器或改写已有用户配置。

## 启用状态高亮与操作反馈修复

已启用卡片使用深绿色边框及 4px 外圈，左上角增加带勾选图标的“已启用”标记，文字区域使用绿色状态配色。卡片布局尺寸保持原值。

缺失 Toast 的根因：PreviewStates 的完整页面分支提前返回 FrontendPreview，ToastHost 只存在于其他组件示例分支；全局 Toast 的 z-index 240 低于弹窗的 1500；快速操作的相同通用文案被 1.5 秒去重窗口抑制。现已将 ToastHost 挂载到所有预览状态共享的根节点，Toast 层级提升至 1700，预览顶部偏移 64px 避开工具条。显式 feedback 通知每次投递，普通后台通知仍去重。

启用、取消和保存使用各自的成功/失败标题，只有后端确认应用才显示成功；配置未变化的保存也明确反馈。表单校验失败同时显示局部错误与红色 Toast；弹窗不会遮住提示。预览反馈明确注明只更新当前页面，未连接设备。

本轮 59 项前端定向单元、10 项桌面/手机浏览器检查通过，TypeScript、Vite 构建和 diff 空白检查通过。覆盖真实页面模拟写入失败后不更换选中设备、成功重试、取消回空、预览连续启用取消的独立 Toast、弹窗中的错误反馈、无变化保存及后台提示去重。Chrome 实际视觉检查截图：out/mouse-protocol-active-toast.png。没有连接真实硬件，后端未新增修改。
