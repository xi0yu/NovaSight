use std::env;
use std::ffi::OsString;
use std::process::{Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

const LATEST_ONLY_QUEUE: &[&str] = &[
    "queue",
    "max-size-buffers=1",
    "max-size-bytes=0",
    "max-size-time=0",
    "leaky=downstream",
];

#[derive(Clone, Debug)]
struct Config {
    device: String,
    capture_width: u32,
    capture_height: u32,
    fps: u32,
    pixel_format: String,
    io_mode: u32,
    roi_left: u32,
    roi_top: u32,
    roi_width: u32,
    roi_height: u32,
    model_width: u32,
    model_height: u32,
    nvinfer_config: String,
    batched_push_timeout_us: u32,
    gst_launch: String,
    run: bool,
    verbose: bool,
    fps_display: bool,
    timeout_seconds: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: "/dev/video0".to_string(),
            capture_width: 1920,
            capture_height: 1080,
            fps: 120,
            pixel_format: "MJPG".to_string(),
            io_mode: 2,
            roi_left: 640,
            roi_top: 220,
            roi_width: 640,
            roi_height: 640,
            model_width: 640,
            model_height: 640,
            nvinfer_config: "data/runtime/deepstream/active-nvinfer.ini".to_string(),
            batched_push_timeout_us: 0,
            gst_launch: "gst-launch-1.0".to_string(),
            run: false,
            verbose: false,
            fps_display: false,
            timeout_seconds: None,
        }
    }
}

fn main() -> ExitCode {
    let config = match parse_args(env::args_os().skip(1)) {
        Ok(config) => config,
        Err(ParseOutcome::Help) => {
            print_help();
            return ExitCode::SUCCESS;
        }
        Err(ParseOutcome::Error(message)) => {
            eprintln!("error: {message}");
            eprintln!("run with --help for usage");
            return ExitCode::from(2);
        }
    };

    let pipeline = build_pipeline(&config);
    let command = build_command(&config, &pipeline);
    println!("{}", shell_join(&command));

    if !config.run {
        return ExitCode::SUCCESS;
    }

    match run_command(&command, config.timeout_seconds) {
        Ok(code) => ExitCode::from(code),
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::from(1)
        }
    }
}

enum ParseOutcome {
    Help,
    Error(String),
}

fn parse_args<I>(args: I) -> Result<Config, ParseOutcome>
where
    I: IntoIterator<Item = OsString>,
{
    let mut config = Config::default();
    let mut iter = args.into_iter();
    while let Some(arg) = iter.next() {
        let text = arg.to_string_lossy();
        match text.as_ref() {
            "--help" | "-h" => return Err(ParseOutcome::Help),
            "--run" => config.run = true,
            "--verbose" | "-v" => config.verbose = true,
            "--fps-display" => config.fps_display = true,
            "--device" => config.device = next_string(&mut iter, "--device")?,
            "--format" => config.pixel_format = next_string(&mut iter, "--format")?,
            "--nvinfer-config" => {
                config.nvinfer_config = next_string(&mut iter, "--nvinfer-config")?
            }
            "--gst-launch" => config.gst_launch = next_string(&mut iter, "--gst-launch")?,
            "--width" => config.capture_width = next_u32(&mut iter, "--width")?,
            "--height" => config.capture_height = next_u32(&mut iter, "--height")?,
            "--fps" => config.fps = next_u32(&mut iter, "--fps")?,
            "--io-mode" => config.io_mode = next_u32(&mut iter, "--io-mode")?,
            "--roi-left" => config.roi_left = next_u32(&mut iter, "--roi-left")?,
            "--roi-top" => config.roi_top = next_u32(&mut iter, "--roi-top")?,
            "--roi-size" => {
                let size = next_u32(&mut iter, "--roi-size")?;
                config.roi_width = size;
                config.roi_height = size;
            }
            "--roi-width" => config.roi_width = next_u32(&mut iter, "--roi-width")?,
            "--roi-height" => config.roi_height = next_u32(&mut iter, "--roi-height")?,
            "--model-width" => config.model_width = next_u32(&mut iter, "--model-width")?,
            "--model-height" => config.model_height = next_u32(&mut iter, "--model-height")?,
            "--batched-push-timeout-us" => {
                config.batched_push_timeout_us = next_u32(&mut iter, "--batched-push-timeout-us")?
            }
            "--timeout-seconds" => {
                config.timeout_seconds = Some(next_u64(&mut iter, "--timeout-seconds")?)
            }
            unknown => return Err(ParseOutcome::Error(format!("unknown argument {unknown}"))),
        }
    }
    validate(&config)?;
    Ok(config)
}

fn next_string<I>(iter: &mut I, flag: &str) -> Result<String, ParseOutcome>
where
    I: Iterator<Item = OsString>,
{
    let value = iter
        .next()
        .ok_or_else(|| ParseOutcome::Error(format!("{flag} requires a value")))?;
    let text = value.to_string_lossy().to_string();
    if text.is_empty() {
        return Err(ParseOutcome::Error(format!(
            "{flag} requires a non-empty value"
        )));
    }
    Ok(text)
}

fn next_u32<I>(iter: &mut I, flag: &str) -> Result<u32, ParseOutcome>
where
    I: Iterator<Item = OsString>,
{
    let value = next_string(iter, flag)?;
    value
        .parse::<u32>()
        .map_err(|_| ParseOutcome::Error(format!("{flag} requires an unsigned integer")))
}

fn next_u64<I>(iter: &mut I, flag: &str) -> Result<u64, ParseOutcome>
where
    I: Iterator<Item = OsString>,
{
    let value = next_string(iter, flag)?;
    value
        .parse::<u64>()
        .map_err(|_| ParseOutcome::Error(format!("{flag} requires an unsigned integer")))
}

fn validate(config: &Config) -> Result<(), ParseOutcome> {
    let format = config.pixel_format.trim().to_ascii_uppercase();
    if !matches!(format.as_str(), "MJPG" | "MJPEG" | "NV12" | "YUYV" | "YUY2") {
        return Err(ParseOutcome::Error(
            "--format must be MJPG, MJPEG, NV12, YUYV, or YUY2".to_string(),
        ));
    }
    for (name, value) in [
        ("--width", config.capture_width),
        ("--height", config.capture_height),
        ("--fps", config.fps),
        ("--roi-width", config.roi_width),
        ("--roi-height", config.roi_height),
        ("--model-width", config.model_width),
        ("--model-height", config.model_height),
    ] {
        if value == 0 {
            return Err(ParseOutcome::Error(format!("{name} must be positive")));
        }
    }
    let roi_right = config.roi_left.saturating_add(config.roi_width);
    let roi_bottom = config.roi_top.saturating_add(config.roi_height);
    if roi_right > config.capture_width || roi_bottom > config.capture_height {
        return Err(ParseOutcome::Error(
            "ROI must stay inside the capture frame".to_string(),
        ));
    }
    Ok(())
}

fn build_pipeline(config: &Config) -> Vec<String> {
    let mut args = Vec::new();
    push_element(
        &mut args,
        &[
            "v4l2src".to_string(),
            "name=capture-source".to_string(),
            format!("device={}", config.device),
            format!("io-mode={}", config.io_mode),
            "do-timestamp=true".to_string(),
        ],
    );

    let format = config.pixel_format.trim().to_ascii_uppercase();
    if matches!(format.as_str(), "MJPG" | "MJPEG") {
        push_caps(
            &mut args,
            format!(
                "image/jpeg,width={},height={},framerate={}/1",
                config.capture_width, config.capture_height, config.fps
            ),
        );
        push_queue(&mut args);
        push_element(&mut args, &["jpegparse".to_string()]);
        push_element(
            &mut args,
            &["nvv4l2decoder".to_string(), "mjpeg=1".to_string()],
        );
        push_caps(
            &mut args,
            "video/x-raw(memory:NVMM),format=I420".to_string(),
        );
        push_queue(&mut args);
    } else {
        let gst_format = if matches!(format.as_str(), "YUYV" | "YUY2") {
            "YUY2"
        } else {
            "NV12"
        };
        push_caps(
            &mut args,
            format!(
                "video/x-raw,format={gst_format},width={},height={},framerate={}/1",
                config.capture_width, config.capture_height, config.fps
            ),
        );
        push_queue(&mut args);
    }

    push_element(
        &mut args,
        &[
            "nvvidconv".to_string(),
            format!("left={}", config.roi_left),
            format!("right={}", config.roi_left + config.roi_width),
            format!("top={}", config.roi_top),
            format!("bottom={}", config.roi_top + config.roi_height),
        ],
    );
    push_caps(
        &mut args,
        format!(
            "video/x-raw(memory:NVMM),format=NV12,width={},height={},pixel-aspect-ratio=1/1",
            config.model_width, config.model_height
        ),
    );
    push_queue(&mut args);
    args.push("!".to_string());
    args.push("mux.sink_0".to_string());
    args.extend([
        "nvstreammux".to_string(),
        "name=mux".to_string(),
        "batch-size=1".to_string(),
        "live-source=1".to_string(),
        format!("width={}", config.model_width),
        format!("height={}", config.model_height),
        "sync-inputs=0".to_string(),
        format!("batched-push-timeout={}", config.batched_push_timeout_us),
    ]);
    push_element(
        &mut args,
        &[
            "nvinfer".to_string(),
            "name=primary-infer".to_string(),
            format!("config-file-path={}", config.nvinfer_config),
            "batch-size=1".to_string(),
        ],
    );
    if config.fps_display {
        push_element(
            &mut args,
            &[
                "fpsdisplaysink".to_string(),
                "video-sink=fakesink".to_string(),
                "text-overlay=false".to_string(),
                "sync=false".to_string(),
                "signal-fps-measurements=false".to_string(),
            ],
        );
    } else {
        push_element(
            &mut args,
            &[
                "fakesink".to_string(),
                "name=deepstream-sink".to_string(),
                "sync=false".to_string(),
                "async=false".to_string(),
                "qos=false".to_string(),
            ],
        );
    }
    args
}

fn push_queue(args: &mut Vec<String>) {
    push_element(
        args,
        &LATEST_ONLY_QUEUE
            .iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>(),
    );
}

fn push_caps(args: &mut Vec<String>, caps: String) {
    args.push("!".to_string());
    args.push(caps);
}

fn push_element(args: &mut Vec<String>, element: &[String]) {
    if !args.is_empty() {
        args.push("!".to_string());
    }
    args.extend(element.iter().cloned());
}

fn build_command(config: &Config, pipeline: &[String]) -> Vec<String> {
    let mut command = vec![config.gst_launch.clone()];
    if config.verbose {
        command.push("-v".to_string());
    }
    command.extend(pipeline.iter().cloned());
    command
}

fn run_command(command: &[String], timeout_seconds: Option<u64>) -> Result<u8, String> {
    if command.is_empty() {
        return Err("empty command".to_string());
    }
    let mut child = Command::new(&command[0])
        .args(&command[1..])
        .stdin(Stdio::null())
        .spawn()
        .map_err(|error| format!("failed to spawn {}: {error}", command[0]))?;

    if let Some(seconds) = timeout_seconds {
        let deadline = Instant::now() + Duration::from_secs(seconds);
        loop {
            if let Some(status) = child
                .try_wait()
                .map_err(|error| format!("failed to wait for child: {error}"))?
            {
                return Ok(status.code().unwrap_or(1) as u8);
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                return Ok(124);
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    let status = child
        .wait()
        .map_err(|error| format!("failed to wait for child: {error}"))?;
    Ok(status.code().unwrap_or(1) as u8)
}

fn shell_join(command: &[String]) -> String {
    command
        .iter()
        .map(|part| shell_quote(part))
        .collect::<Vec<_>>()
        .join(" ")
}

fn shell_quote(value: &str) -> String {
    if value
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || "/._:=,-+".contains(c))
    {
        return value.to_string();
    }
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn print_help() {
    println!(
        "\
Build or run the NovaSight DeepStream command-line pipeline.

By default this prints a gst-launch-1.0 command. Use --run to execute it.

Options:
  --run                         execute gst-launch-1.0
  --verbose, -v                 pass -v to gst-launch-1.0
  --fps-display                 end with fpsdisplaysink instead of fakesink
  --timeout-seconds N           kill gst-launch after N seconds; exits 124 on timeout
  --gst-launch PATH             gst-launch binary, default gst-launch-1.0
  --device PATH                 capture device, default /dev/video0
  --format FMT                  MJPG, MJPEG, NV12, YUYV, or YUY2
  --width N --height N --fps N  capture profile
  --io-mode N                   v4l2src io-mode, default 2
  --roi-left N --roi-top N      ROI origin
  --roi-size N                  square ROI size
  --roi-width N --roi-height N  non-square ROI size
  --model-width N --model-height N
  --nvinfer-config PATH         nvinfer config path
  --batched-push-timeout-us N   nvstreammux timeout, default 0

Example:
  cargo run --manifest-path tools/deepstream-cli-pipeline/Cargo.toml -- \\
    --run --verbose --fps-display --timeout-seconds 30 \\
    --device /dev/video0 --format MJPG --width 1920 --height 1080 --fps 120 \\
    --roi-left 640 --roi-top 220 --roi-size 640 \\
    --model-width 640 --model-height 640 \\
    --nvinfer-config data/runtime/deepstream/active-nvinfer.ini"
    );
}
