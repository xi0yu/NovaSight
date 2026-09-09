export const logoAssets = [
  "brand/logos/novasight-logo-horizontal.svg",
  "brand/logos/novasight-logo-light.svg",
  "brand/logos/novasight-logo-dark.svg",
  "brand/logos/novasight-logo-mono-dark.svg",
  "brand/logos/novasight-logo-mono-light.svg",
  "brand/logos/novasight-mark.svg",
  "brand/logos/sizes/novasight-mark-16.svg",
  "brand/logos/sizes/novasight-mark-20.svg",
  "brand/logos/sizes/novasight-mark-24.svg",
  "brand/logos/sizes/novasight-mark-32.svg",
  "brand/logos/sizes/novasight-mark-64.svg",
  "brand/logos/sizes/novasight-mark-128.svg",
  "brand/logos/sizes/novasight-mark-512.svg",
] as const;

export const emptyStateAssets = [
  "illustrations/empty/no-device.svg",
  "illustrations/empty/no-model.svg",
  "illustrations/empty/not-started.svg",
  "illustrations/empty/no-detections.svg",
  "illustrations/empty/no-logs.svg",
  "illustrations/empty/no-search-results.svg",
  "illustrations/empty/config-incomplete.svg",
  "illustrations/empty/video-unavailable.svg",
  "illustrations/empty/gpu-unavailable.svg",
  "illustrations/empty/network-disconnected.svg",
] as const;

export const featureIllustrationAssets = [
  "illustrations/onboarding/startup-flow.svg",
  "illustrations/features/capture-pipeline.svg",
  "illustrations/features/inference-runtime.svg",
  "illustrations/features/tracking-prediction.svg",
  "illustrations/features/control-response.svg",
  "illustrations/devices/device-graph.svg",
] as const;

export const patternAssets = ["patterns/vision-grid.svg"] as const;

export const exportedIconAssets = [
  "icons/capture/roi.svg",
  "icons/inference/gpu.svg",
  "icons/tracking/prediction-line.svg",
  "icons/control/pid.svg",
  "icons/status/check-circle.svg",
  "icons/status/triangle-alert.svg",
] as const;

export const visualAssetGroups = {
  logos: logoAssets,
  emptyStates: emptyStateAssets,
  featureIllustrations: featureIllustrationAssets,
  patterns: patternAssets,
  exportedIcons: exportedIconAssets,
} as const;
