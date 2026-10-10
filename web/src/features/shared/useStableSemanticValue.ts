import { useEffect, useRef, useState } from "react";

/**
 * Keeps the last complete UI snapshot visible until a new semantic state has
 * remained stable for the requested window. Values inside the same semantic
 * state remain live, so counters and telemetry do not lose realtime updates.
 */
export function useStableSemanticValue<T, K>(
  value: T,
  semanticKey: K,
  delayMs: number
): T {
  const latestRef = useRef({ key: semanticKey, value });
  latestRef.current = { key: semanticKey, value };

  const [stableKey, setStableKey] = useState(semanticKey);
  const stableValueRef = useRef(value);

  if (Object.is(stableKey, semanticKey)) {
    stableValueRef.current = value;
  }

  useEffect(() => {
    if (Object.is(stableKey, semanticKey)) {
      return;
    }
    if (delayMs <= 0) {
      const latest = latestRef.current;
      stableValueRef.current = latest.value;
      setStableKey(latest.key);
      return;
    }

    const timeout = window.setTimeout(() => {
      const latest = latestRef.current;
      if (!Object.is(latest.key, semanticKey)) {
        return;
      }
      stableValueRef.current = latest.value;
      setStableKey(semanticKey);
    }, delayMs);

    return () => window.clearTimeout(timeout);
  }, [delayMs, semanticKey, stableKey]);

  return Object.is(stableKey, semanticKey) ? value : stableValueRef.current;
}
