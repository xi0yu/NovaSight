import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const webRoot = path.resolve(new URL("..", import.meta.url).pathname);
const srcRoot = path.join(webRoot, "src");
const failures = [];

function readSource(relativePath) {
  const absolutePath = path.join(srcRoot, relativePath);
  if (!fs.existsSync(absolutePath)) {
    failures.push(`missing source: ${relativePath}`);
    return "";
  }
  return fs.readFileSync(absolutePath, "utf8");
}

function assertContains(source, pattern, label) {
  if (!pattern.test(source)) failures.push(`missing ${label}`);
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
    (item) => item[1] ?? item[2],
  );
}

function parseCssVariables(source) {
  return new Map(
    Array.from(source.matchAll(/--([a-zA-Z0-9-]+):\s*([^;]+);/g), (match) => [
      match[1],
      match[2].trim(),
    ]),
  );
}

function resolveCssColor(variables, name, seen = new Set()) {
  if (seen.has(name)) return null;
  seen.add(name);
  const value = variables.get(name);
  const hex = value?.match(/#[0-9a-fA-F]{6}/)?.[0];
  if (hex) return hex.toUpperCase();
  const reference = value?.match(/var\(--([a-zA-Z0-9-]+)\)/)?.[1];
  return reference ? resolveCssColor(variables, reference, seen) : null;
}

function relativeLuminance(hex) {
  const rgb = hex
    .match(/^#([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})$/i)
    ?.slice(1)
    .map((part) => Number.parseInt(part, 16) / 255);
  if (!rgb) return null;
  const [r, g, b] = rgb.map((value) =>
    value <= 0.03928 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4,
  );
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function assertContrast(variables, foregroundName, backgroundName, minimum, label) {
  const foreground = resolveCssColor(variables, foregroundName);
  const background = resolveCssColor(variables, backgroundName);
  const foregroundLum = foreground && relativeLuminance(foreground);
  const backgroundLum = background && relativeLuminance(background);
  if (foregroundLum === null || backgroundLum === null || !foreground || !background) {
    failures.push(`missing contrast tokens for ${label}`);
    return;
  }
  const ratio =
    (Math.max(foregroundLum, backgroundLum) + 0.05) /
    (Math.min(foregroundLum, backgroundLum) + 0.05);
  if (ratio < minimum) failures.push(`contrast ${label} is ${ratio.toFixed(2)}; expected >= ${minimum}`);
}

const iconNamesSource = readSource("design/iconNames.ts");
const novaIconSource = readSource("components/visual/NovaIcon.tsx");
assertContains(novaIconSource, /strokeLinecap="round"/, "rounded icon caps in NovaIcon.tsx");
assertContains(novaIconSource, /strokeLinejoin="round"/, "rounded icon joins in NovaIcon.tsx");
assertContains(novaIconSource, /stroke="currentColor"/, "currentColor icon stroke in NovaIcon.tsx");

const registeredIconNames = new Set(Array.from(parseIconCategories(iconNamesSource).values()).flat());
const aliasNames = new Set(parseIconAliasKeys(novaIconSource));
for (const iconName of registeredIconNames) {
  if (!aliasNames.has(iconName)) failures.push(`NovaIcon alias missing for icon name: ${iconName}`);
}
for (const aliasName of aliasNames) {
  if (!registeredIconNames.has(aliasName)) failures.push(`NovaIcon alias is not declared: ${aliasName}`);
}

const tokenVariables = parseCssVariables(readSource("design/tokens.css"));
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
for (const pair of contrastPairs) assertContrast(tokenVariables, ...pair);

if (failures.length) {
  console.error("NovaSight visual contract audit failed:");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log("NovaSight visual contracts passed.");
console.log(`- runtime icons: ${registeredIconNames.size}`);
console.log(`- contrast pairs: ${contrastPairs.length}`);
