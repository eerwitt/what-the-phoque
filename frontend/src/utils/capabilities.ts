/**
 * System capability detection for model loading.
 *
 * Primary model  (eerwitt/what-the-phoque-onnx)
 *   - Exported as text-generation ONNX → loads via AutoModelForCausalLM
 *   - Falls back to WASM if WebGPU is unavailable
 *   - Estimated peak VRAM (WebGPU): ~3 GB   |  RAM (WASM): ~6 GB
 *
 * Base model  (mistralai/Ministral-3-3B-Instruct-2512-ONNX)
 *   - Multimodal ONNX layout (embed_tokens fp16 + vision_encoder q4 + decoder q4f16)
 *   - Requires WebGPU — WASM cannot satisfy per-module dtype requirements
 *   - Estimated peak VRAM: ~2.5 GB
 *
 * Both simultaneously: ~5.5 GB VRAM (WebGPU) or ~3 GB VRAM + ~6 GB RAM if
 * primary falls back to WASM (browser-dependent).
 */

export type ModelTier = "webgpu" | "wasm" | "unavailable";

export type SystemCapabilities = {
  /** True if the browser exposes a WebGPU adapter. */
  hasWebGPU: boolean;
  /** Largest single GPU buffer in GB (from adapter.limits.maxBufferSize), or null if unknown. */
  gpuMaxBufferGb: number | null;
  /** Approximate system RAM in GB from navigator.deviceMemory, or null if not exposed. */
  deviceMemoryGb: number | null;
  /** GPU description string from requestAdapterInfo, if available. */
  gpuDescription: string | null;

  /** How the primary (fine-tuned) model will be loaded. */
  primaryTier: Exclude<ModelTier, "unavailable">;
  /** Whether the base VLM model can be loaded at all. */
  baseTier: ModelTier;
  /**
   * Overall recommendation:
   *  "both"          – both models should load
   *  "primary-only"  – only the text-gen model is feasible
   *  "base-only"     – unlikely (base needs WebGPU; primary always works)
   */
  recommendation: "both" | "primary-only";
  /** Human-readable explanation of the recommendation. */
  reason: string;
};

// Conservative size estimates (GB)
const PRIMARY_VRAM_GB = 3;   // primary on WebGPU
const BASE_VRAM_GB = 2.5;    // base VLM (q4f16 decoder + fp16 embed + q4 vision)
const BOTH_VRAM_GB = PRIMARY_VRAM_GB + BASE_VRAM_GB; // ~5.5 GB

export async function detectSystemCapabilities(): Promise<SystemCapabilities> {
  // ── WebGPU probe ──────────────────────────────────────────────────────────
  const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;
  let gpuMaxBufferGb: number | null = null;
  let gpuDescription: string | null = null;

  if (hasWebGPU) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        gpuMaxBufferGb = adapter.limits.maxBufferSize / 1024 ** 3;
        try {
          const info = await adapter.requestAdapterInfo();
          gpuDescription = info.description || info.device || null;
        } catch {
          // requestAdapterInfo may be restricted in some browsers
        }
      }
    } catch {
      // WebGPU present in navigator but adapter request failed (e.g. hardware blacklist)
    }
  }

  // ── System RAM probe ──────────────────────────────────────────────────────
  const deviceMemoryGb: number | null =
    typeof navigator !== "undefined" && "deviceMemory" in navigator
      ? (navigator as Navigator & { deviceMemory: number }).deviceMemory
      : null;

  // ── Decision logic ────────────────────────────────────────────────────────
  //
  // Primary model: always possible (WASM fallback). WebGPU preferred.
  const primaryTier: Exclude<ModelTier, "unavailable"> = hasWebGPU ? "webgpu" : "wasm";

  // Base model: strictly needs WebGPU.
  // If we have a maxBufferSize reading, use it to estimate VRAM headroom.
  // Without a reading we optimistically allow the attempt.
  const hasEnoughVramForBase =
    !hasWebGPU
      ? false
      : gpuMaxBufferGb === null || gpuMaxBufferGb >= BASE_VRAM_GB;

  const baseTier: ModelTier = hasEnoughVramForBase ? "webgpu" : "unavailable";

  // For running both together, we need room for the sum.
  // If primary uses WASM it doesn't compete for VRAM, so only base's VRAM matters.
  const primaryOnWasm = !hasWebGPU; // or if VRAM is too tight for both
  const hasEnoughVramForBoth = hasWebGPU
    ? gpuMaxBufferGb === null || gpuMaxBufferGb >= (primaryOnWasm ? BASE_VRAM_GB : BOTH_VRAM_GB)
    : false;

  const canLoadBoth = baseTier !== "unavailable" && hasEnoughVramForBoth;

  const recommendation: "both" | "primary-only" = canLoadBoth ? "both" : "primary-only";

  // ── Reason string ─────────────────────────────────────────────────────────
  let reason: string;
  if (!hasWebGPU) {
    reason = "WebGPU unavailable — base model disabled, primary runs on WASM (slower).";
  } else if (baseTier === "unavailable") {
    reason = `GPU max-buffer ${gpuMaxBufferGb?.toFixed(1)} GB appears insufficient for the base model (~${BASE_VRAM_GB} GB needed).`;
  } else if (!canLoadBoth) {
    reason = `GPU max-buffer ${gpuMaxBufferGb?.toFixed(1)} GB may not hold both models (~${BOTH_VRAM_GB} GB needed together).`;
  } else if (gpuMaxBufferGb !== null) {
    reason = `GPU max-buffer ${gpuMaxBufferGb.toFixed(1)} GB — both models should fit.`;
  } else {
    reason = "WebGPU available; GPU memory unknown — attempting both models.";
  }

  return {
    hasWebGPU,
    gpuMaxBufferGb,
    deviceMemoryGb,
    gpuDescription,
    primaryTier,
    baseTier,
    recommendation,
    reason,
  };
}
