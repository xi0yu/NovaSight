import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const repoRoot = path.resolve(new URL("..", import.meta.url).pathname);
const srcRoot = path.join(repoRoot, "src");
const assetsRoot = path.join(srcRoot, "assets");

const requiredIconDirectories = [
  "navigation",
  "capture",
  "inference",
  "tracking",
  "control",
  "devices",
  "actions",
  "status",
];

const requiredEmptyStates = [
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
];

const requiredLogoAssets = [
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
];

const requiredFeatureIllustrations = [
  "illustrations/onboarding/startup-flow.svg",
  "illustrations/features/capture-pipeline.svg",
  "illustrations/features/inference-runtime.svg",
  "illustrations/features/tracking-prediction.svg",
  "illustrations/features/control-response.svg",
  "illustrations/devices/device-graph.svg",
];

const failures = [];

function assertFile(relativePath) {
  const absolutePath = path.join(assetsRoot, relativePath);
  if (!fs.existsSync(absolutePath)) {
    failures.push(`missing asset: ${relativePath}`);
  }
}

function assertContains(filePath, pattern, label) {
  const content = fs.readFileSync(filePath, "utf8");
  if (!pattern.test(content)) {
    failures.push(`missing ${label} in ${path.relative(repoRoot, filePath)}`);
  }
}

function unique(items) {
  return Array.from(new Set(items));
}

function extractQuotedStrings(source) {
  return Array.from(source.matchAll(/"([^"]+)"/g), (match) => match[1]);
}

function parseIconCategories(source) {
  const categories = new Map();
  for (const match of source.matchAll(/\s([a-zA-Z][a-zA-Z0-9]*):\s*\[([\s\S]*?)\]/g)) {
    categories.set(match[1], extractQuotedStrings(match[2]));
  }
  return categories;
}

function parseIconAliasKeys(source) {
  const match = source.match(/const iconAliases:[\s\S]*?=\s*{([\s\S]*?)};/);
  if (!match) {
    failures.push("missing iconAliases object in NovaIcon.tsx");
    return [];
  }
  return Array.from(
    match[1].matchAll(/^\s*(?:"([^"]+)"|([a-zA-Z][a-zA-Z0-9-]*)):\s*"/gm),
    (item) => item[1] ?? item[2]
  );
}

function parseCssVariables(source) {
  const variables = new Map();
  for (const match of source.matchAll(/--([a-zA-Z0-9-]+):\s*([^;]+);/g)) {
    variables.set(match[1], match[2].trim());
  }
  return variables;
}

function resolveCssColor(variables, name, seen = new Set()) {
  if (seen.has(name)) {
    return null;
  }
  seen.add(name);
  const value = variables.get(name);
  if (!value) {
    return null;
  }
  const hex = value.match(/#[0-9a-fA-F]{6}/)?.[0];
  if (hex) {
    return hex.toUpperCase();
  }
  const reference = value.match(/var\(--([a-zA-Z0-9-]+)\)/)?.[1];
  return reference ? resolveCssColor(variables, reference, seen) : null;
}

function hexToRgb(hex) {
  const match = hex.match(/^#([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})$/i);
  if (!match) {
    return null;
  }
  return match.slice(1).map((part) => Number.parseInt(part, 16) / 255);
}

function relativeLuminance(hex) {
  const rgb = hexToRgb(hex);
  if (!rgb) {
    return null;
  }
  const [r, g, b] = rgb.map((value) =>
    value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4
  );
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrastRatio(foreground, background) {
  const foregroundLum = relativeLuminance(foreground);
  const backgroundLum = relativeLuminance(background);
  if (foregroundLum === null || backgroundLum === null) {
    return null;
  }
  const light = Math.max(foregroundLum, backgroundLum);
  const dark = Math.min(foregroundLum, backgroundLum);
  return (light + 0.05) / (dark + 0.05);
}

function assertContrast(variables, foregroundName, backgroundName, minimum, label) {
  const foreground = resolveCssColor(variables, foregroundName);
  const background = resolveCssColor(variables, backgroundName);
  if (!foreground || !background) {
    failures.push(`missing contrast tokens for ${label}: ${foregroundName}/${backgroundName}`);
    return;
  }
  const ratio = contrastRatio(foreground, background);
  if (ratio === null || ratio < minimum) {
    failures.push(
      `contrast ${label} is ${ratio?.toFixed(2) ?? "n/a"}; expected >= ${minimum} (${foreground} on ${background})`
    );
  }
}

for (const directory of requiredIconDirectories) {
  const absolutePath = path.join(assetsRoot, "icons", directory);
  if (!fs.existsSync(absolutePath)) {
    failures.push(`missing icon directory: icons/${directory}`);
  }
}

for (const asset of [
  ...requiredEmptyStates,
  ...requiredLogoAssets,
  ...requiredFeatureIllustrations,
]) {
  assertFile(asset);
}

const visualAssetsPath = path.join(srcRoot, "design", "visualAssets.ts");
const visualAssetsSource = fs.readFileSync(visualAssetsPath, "utf8");
for (const asset of [
  ...requiredEmptyStates,
  ...requiredLogoAssets,
  ...requiredFeatureIllustrations,
]) {
  if (!visualAssetsSource.includes(`"${asset}"`)) {
    failures.push(`asset not registered in visualAssets.ts: ${asset}`);
  }
}

const iconNamesPath = path.join(srcRoot, "design", "iconNames.ts");
const iconNamesSource = fs.readFileSync(iconNamesPath, "utf8");
for (const category of requiredIconDirectories) {
  assertContains(iconNamesPath, new RegExp(`${category}: \\[`), `icon category '${category}'`);
}

const novaIconPath = path.join(srcRoot, "components", "visual", "NovaIcon.tsx");
const novaIconSource = fs.readFileSync(novaIconPath, "utf8");
assertContains(novaIconPath, /strokeLinecap="round"/, "rounded icon caps");
assertContains(novaIconPath, /strokeLinejoin="round"/, "rounded icon joins");
assertContains(novaIconPath, /stroke="currentColor"/, "currentColor icon stroke");

const iconCategories = parseIconCategories(iconNamesSource);
const registeredIconNames = unique(Array.from(iconCategories.values()).flat());
const aliasNames = parseIconAliasKeys(novaIconSource);
const aliasNameSet = new Set(aliasNames);
for (const iconName of registeredIconNames) {
  if (!aliasNameSet.has(iconName)) {
    failures.push(`NovaIcon alias missing for icon name: ${iconName}`);
  }
}
const registeredIconNameSet = new Set(registeredIconNames);
for (const aliasName of aliasNames) {
  if (!registeredIconNameSet.has(aliasName)) {
    failures.push(`NovaIcon alias is not declared in iconNames.ts: ${aliasName}`);
  }
}
for (const [category, names] of iconCategories) {
  for (const iconName of names) {
    const relativeIconPath = `icons/${category}/${iconName}.svg`;
    const absoluteIconPath = path.join(assetsRoot, relativeIconPath);
    if (!fs.existsSync(absoluteIconPath)) {
      failures.push(`missing exported SVG icon: ${relativeIconPath}`);
      continue;
    }
    assertContains(absoluteIconPath, /viewBox="0 0 24 24"/, "24px icon viewBox");
    assertContains(absoluteIconPath, /stroke="currentColor"/, "currentColor icon stroke");
    assertContains(absoluteIconPath, /stroke-linecap="round"/, "rounded exported icon caps");
    assertContains(absoluteIconPath, /stroke-linejoin="round"/, "rounded exported icon joins");
  }
}

const tokensPath = path.join(srcRoot, "design", "tokens.css");
const tokenVariables = parseCssVariables(fs.readFileSync(tokensPath, "utf8"));
const contrastPairs = [
  ["text-primary", "bg-surface", 4.5, "primary text on surface"],
  ["text-secondary", "bg-surface", 4.5, "secondary text on surface"],
  ["text-link", "bg-surface", 4.5, "links on surface"],
  ["button-primary-fg", "button-primary-bg", 4.5, "primary button"],
  ["status-normal-fg", "status-normal-bg", 4.5, "normal status"],
  ["status-running-fg", "status-running-bg", 4.5, "running status"],
  ["status-waiting-fg", "status-waiting-bg", 4.5, "waiting status"],
  ["status-warning-fg", "status-warning-bg", 4.5, "warning status"],
  ["status-error-fg", "status-error-bg", 4.5, "error status"],
  ["status-disabled-fg", "status-disabled-bg", 3, "disabled status"],
  ["brand-primary", "brand-soft", 3, "brand selected surface"],
];
for (const [foreground, background, minimum, label] of contrastPairs) {
  assertContrast(tokenVariables, foreground, background, minimum, label);
}

const appPath = path.join(srcRoot, "App.tsx");
assertContains(appPath, /visual-system"\) === "1"/, "visual system preview route");

if (failures.length > 0) {
  console.error("NovaSight visual system audit failed:");
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log("NovaSight visual system audit passed.");
console.log(`- empty states: ${requiredEmptyStates.length}`);
console.log(`- logo assets: ${requiredLogoAssets.length}`);
console.log(`- feature illustrations: ${requiredFeatureIllustrations.length}`);
console.log(`- icon directories: ${requiredIconDirectories.length}`);
console.log(`- exported SVG icon files: ${Array.from(iconCategories.values()).flat().length}`);
console.log(`- unique registered icon names: ${registeredIconNames.length}`);
console.log(`- contrast pairs: ${contrastPairs.length}`);
