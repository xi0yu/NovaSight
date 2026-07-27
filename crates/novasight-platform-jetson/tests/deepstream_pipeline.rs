use std::path::PathBuf;

use novasight_platform_jetson::deepstream::{
    CaptureFormat, CaptureProfile, CrosshairPipelineConfig, DeepStreamPipelineSpec, InferenceStage,
    ModelInput, PipelineSpecError, PreviewPipelineConfig, Roi,
};

fn spec(format: CaptureFormat) -> DeepStreamPipelineSpec {
    DeepStreamPipelineSpec {
        device: PathBuf::from("/dev/video 0"),
        capture: CaptureProfile {
            width: 1920,
            height: 1080,
            fps: 120,
            format,
        },
        io_mode: 2,
        roi: Roi {
            left: 640,
            top: 220,
            width: 640,
            height: 640,
        },
        model_input: ModelInput {
            width: 640,
            height: 640,
        },
        inference: InferenceStage::DeepStreamNvinfer {
            config: PathBuf::from("data/runtime/deepstream/active nvinfer.ini"),
        },
        batched_push_timeout_us: 0,
        inference_element: "primary-infer".to_owned(),
        preview: None,
        crosshair: None,
    }
}

#[test]
fn crosshair_branch_is_independent_latest_only_center_crop() {
    let mut spec = spec(CaptureFormat::Mjpeg);
    spec.crosshair = Some(CrosshairPipelineConfig { size: 96, fps: 10 });
    let pipeline = spec.build().unwrap();

    assert!(pipeline.contains("tee name=novasight-source-split"));
    assert!(pipeline.contains("videorate drop-only=true max-rate=10"));
    assert!(pipeline.contains("crosshair-crop left=912 right=1008 top=492 bottom=588"));
    assert!(pipeline.contains("width=96,height=96,framerate=10/1"));
    assert!(pipeline.contains("nvjpegenc name=crosshair-encoder"));
    assert!(pipeline.contains("fakesink name=crosshair-sink"));
    assert!(!pipeline.contains("preview-valve"));
}

#[test]
fn preview_branch_is_latest_only_hardware_jpeg_and_closed_by_default() {
    let mut spec = spec(CaptureFormat::Mjpeg);
    spec.preview = Some(PreviewPipelineConfig { fps: 30 });
    let pipeline = spec.build().unwrap();

    assert!(pipeline.contains("tee name=novasight-source-split"));
    assert!(pipeline.contains("novasight-source-split. ! queue max-size-buffers=1"));
    assert!(pipeline.contains("valve name=preview-valve drop=true"));
    assert!(pipeline.contains("videorate drop-only=true max-rate=30"));
    assert!(
        pipeline
            .contains("video/x-raw(memory:NVMM),format=NV12,width=640,height=640,framerate=30/1")
    );
    assert!(pipeline.contains("nvjpegenc name=preview-encoder"));
    assert!(pipeline.contains("fakesink name=preview-sink"));
}

#[test]
fn mjpeg_pipeline_preserves_nvmm_latest_only_and_named_probe_path() {
    let pipeline = spec(CaptureFormat::Mjpeg).build().unwrap();

    assert!(pipeline.contains("device=\"/dev/video 0\""));
    assert!(pipeline.contains("image/jpeg,width=1920,height=1080,framerate=120/1"));
    assert!(pipeline.contains("nvv4l2decoder mjpeg=1"));
    assert!(pipeline.contains("video/x-raw(memory:NVMM),format=I420"));
    assert!(pipeline.contains("max-size-buffers=1"));
    assert!(pipeline.contains("leaky=downstream"));
    assert!(pipeline.contains("nvvidconv left=640 right=1280 top=220 bottom=860"));
    assert!(pipeline.contains("nvstreammux name=mux batch-size=1 live-source=1"));
    assert!(pipeline.contains("! mux.sink_0 nvstreammux name=mux"));
    assert!(!pipeline.contains("mux.sink_0 ! nvstreammux"));
    assert!(pipeline.contains(
        "nvinfer name=primary-infer config-file-path=\"data/runtime/deepstream/active nvinfer.ini\" batch-size=1"
    ));
    assert!(pipeline.ends_with("fakesink name=deepstream-sink sync=false async=false qos=false"));
}

#[test]
fn raw_capture_does_not_insert_the_mjpeg_decoder() {
    let pipeline = spec(CaptureFormat::Yuy2).build().unwrap();

    assert!(pipeline.contains("video/x-raw,format=YUY2"));
    assert!(!pipeline.contains("jpegparse"));
    assert!(!pipeline.contains("nvv4l2decoder"));
}

#[test]
fn roi_must_fit_without_integer_overflow() {
    let mut outside = spec(CaptureFormat::Nv12);
    outside.roi.left = 1900;
    assert!(matches!(
        outside.build(),
        Err(PipelineSpecError::RoiOutsideCapture { .. })
    ));

    let mut overflow = spec(CaptureFormat::Nv12);
    overflow.roi.left = u32::MAX;
    assert_eq!(overflow.build(), Err(PipelineSpecError::RoiOverflow));
}

#[test]
fn element_name_and_io_mode_cannot_inject_invalid_pipeline_properties() {
    let mut invalid_name = spec(CaptureFormat::Nv12);
    invalid_name.inference_element = "primary-infer ! fakesrc".to_owned();
    assert_eq!(
        invalid_name.build(),
        Err(PipelineSpecError::InvalidInferenceElement)
    );

    let mut invalid_io = spec(CaptureFormat::Nv12);
    invalid_io.io_mode = 6;
    assert_eq!(invalid_io.build(), Err(PipelineSpecError::InvalidIoMode(6)));
}
