import { useEffect, useRef, useState } from "react";
import { useVLMContext } from "../context/useVLMContext";
import { THEME } from "../constants";
import {
  detectSystemCapabilities,
  type SystemCapabilities,
  type ModelTier,
} from "../utils/capabilities";

interface LoadingScreenProps {
  onComplete: () => void;
}

type ModelLoadState = {
  progress: number;
  step: string;
  failed: boolean;
  done: boolean;
  skipped: boolean;
};

const INITIAL_STATE: ModelLoadState = {
  progress: 0,
  step: "Waiting...",
  failed: false,
  done: false,
  skipped: false,
};

const SKIPPED_STATE = (reason: string): ModelLoadState => ({
  progress: 100,
  step: reason,
  failed: false,
  done: true,
  skipped: true,
});

function CapabilityBadge({
  tier,
  label,
}: {
  tier: ModelTier | "checking";
  label: string;
}) {
  const styles: Record<string, { bg: string; text: string; dot: string }> = {
    webgpu: { bg: "#f0fdf4", text: "#166534", dot: "#22c55e" },
    wasm: { bg: "#fffbeb", text: "#92400e", dot: "#f59e0b" },
    unavailable: { bg: "#fef2f2", text: "#991b1b", dot: THEME.errorRed },
    checking: { bg: "#f9fafb", text: "#6b7280", dot: "#9ca3af" },
  };
  const s = styles[tier] ?? styles.checking;
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-semibold border"
      style={{ backgroundColor: s.bg, color: s.text, borderColor: `${s.dot}44` }}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${tier === "checking" ? "animate-pulse" : ""}`}
        style={{ backgroundColor: s.dot }}
      />
      {label}
    </span>
  );
}

export default function LoadingScreen({ onComplete }: LoadingScreenProps) {
  const { primaryModel, baseModel } = useVLMContext();
  const [mounted, setMounted] = useState(false);
  const [caps, setCaps] = useState<SystemCapabilities | null>(null);
  const [primaryState, setPrimaryState] = useState<ModelLoadState>(INITIAL_STATE);
  const [baseState, setBaseState] = useState<ModelLoadState>(INITIAL_STATE);
  const hasStarted = useRef(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (hasStarted.current) return;
    hasStarted.current = true;

    const run = async () => {
      // ── 1. Detect capabilities first (fast — no downloads) ───────────────
      const detected = await detectSystemCapabilities();
      setCaps(detected);

      // ── 2. Decide what to load based on capabilities ─────────────────────
      const loadPrimary = primaryModel.loadModel((msg, pct) => {
        setPrimaryState({ progress: pct ?? 0, step: msg.replace("...", ""), failed: false, done: false, skipped: false });
      });

      // Skip base model if capability check says it's unavailable
      const loadBase =
        detected.baseTier === "unavailable"
          ? Promise.resolve("skipped")
          : baseModel.loadModel((msg, pct) => {
              setBaseState({ progress: pct ?? 0, step: msg.replace("...", ""), failed: false, done: false, skipped: false });
            });

      if (detected.baseTier === "unavailable") {
        setBaseState(SKIPPED_STATE(detected.reason));
      }

      // ── 3. Wait for both to settle ────────────────────────────────────────
      const [primaryResult, baseResult] = await Promise.allSettled([loadPrimary, loadBase]);

      if (primaryResult.status === "rejected") {
        const r = primaryResult.reason;
        setPrimaryState({ progress: 100, step: r instanceof Error ? r.message : String(r), failed: true, done: true, skipped: false });
      } else {
        setPrimaryState((s) => ({ ...s, done: true }));
      }

      if (baseResult.status === "rejected") {
        const r = baseResult.reason;
        setBaseState({ progress: 100, step: r instanceof Error ? r.message : String(r), failed: true, done: true, skipped: false });
      } else if (detected.baseTier !== "unavailable") {
        setBaseState((s) => ({ ...s, done: true }));
      }

      onComplete();
    };

    void run();
  }, [primaryModel, baseModel, onComplete]);

  const modelRows: Array<{
    label: string;
    modelId: string;
    state: ModelLoadState;
    tier: ModelTier | "checking";
    tierLabel: string;
  }> = [
    {
      label: "What the Phoque?",
      modelId: primaryModel.modelId,
      state: primaryState,
      tier: caps ? caps.primaryTier : "checking",
      tierLabel: caps
        ? caps.primaryTier === "webgpu"
          ? "WebGPU"
          : "WASM"
        : "Checking...",
    },
    {
      label: "Ministral Base",
      modelId: baseModel.modelId,
      state: baseState,
      tier: caps ? caps.baseTier : "checking",
      tierLabel: caps
        ? caps.baseTier === "webgpu"
          ? "WebGPU"
          : "Unavailable"
        : "Checking...",
    },
  ];

  return (
    <div
      className="absolute inset-0 flex items-center justify-center p-8 z-50"
      style={{
        backgroundColor: THEME.beigeLight,
        backgroundImage: `
          linear-gradient(${THEME.beigeDark} 1px, transparent 1px),
          linear-gradient(90deg, ${THEME.beigeDark} 1px, transparent 1px)
        `,
        backgroundSize: "40px 40px",
      }}
    >
      <div
        className={`max-w-lg w-full backdrop-blur-sm rounded-sm border shadow-xl transition-all duration-700 transform ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
        style={{
          backgroundColor: `${THEME.beigeLight}F2`,
          borderColor: THEME.beigeDark,
        }}
      >
        <div className="h-1 w-full" style={{ backgroundColor: THEME.mistralOrange }} />

        <div className="p-8 space-y-6">
          <div className="text-center space-y-2">
            <h2 className="text-2xl font-bold tracking-tight" style={{ color: THEME.textBlack }}>
              Loading Models
            </h2>
            <p className="text-sm text-gray-500 font-mono uppercase tracking-widest">
              What the Phoque? — Split View
            </p>
          </div>

          {/* ── System capability summary ───────────────────────────────── */}
          {caps && (
            <div
              className="rounded-sm border p-3 text-xs space-y-1.5"
              style={{ borderColor: THEME.beigeDark, backgroundColor: `${THEME.beigeMedium}66` }}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono font-bold uppercase tracking-wider text-gray-500">
                  System
                </span>
                <CapabilityBadge
                  tier={caps.recommendation === "both" ? "webgpu" : caps.hasWebGPU ? "wasm" : "unavailable"}
                  label={caps.recommendation === "both" ? "Both models" : "Primary only"}
                />
              </div>
              <p className="text-gray-500 leading-relaxed">{caps.reason}</p>
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-gray-400 font-mono pt-0.5">
                {caps.gpuDescription && (
                  <span title="GPU">GPU: {caps.gpuDescription}</span>
                )}
                {caps.gpuMaxBufferGb !== null && (
                  <span title="Max GPU buffer size">VRAM buf: {caps.gpuMaxBufferGb.toFixed(1)} GB</span>
                )}
                {caps.deviceMemoryGb !== null && (
                  <span title="Approx system RAM">RAM: ~{caps.deviceMemoryGb} GB</span>
                )}
              </div>
            </div>
          )}

          {/* ── Per-model progress rows ─────────────────────────────────── */}
          {modelRows.map(({ label, modelId, state, tier, tierLabel }) => (
            <div key={modelId} className="space-y-2">
              <div className="flex items-center justify-between gap-2 text-xs font-mono">
                <span className="font-bold text-gray-700 shrink-0">{label}</span>
                <div className="flex items-center gap-2 min-w-0">
                  <span className="truncate text-gray-400" title={modelId}>
                    {modelId}
                  </span>
                  <CapabilityBadge tier={tier} label={tierLabel} />
                </div>
              </div>

              <div
                className="w-full rounded-full h-3 overflow-hidden border"
                style={{
                  backgroundColor: `${THEME.beigeDark}80`,
                  borderColor: state.failed
                    ? `${THEME.errorRed}66`
                    : state.skipped
                      ? `${THEME.beigeDark}80`
                      : THEME.beigeDark,
                }}
              >
                <div
                  className="h-full transition-all duration-500 ease-out"
                  style={{
                    width: `${state.progress}%`,
                    backgroundColor: state.failed
                      ? THEME.errorRed
                      : state.skipped
                        ? THEME.beigeDark
                        : state.done
                          ? "#22c55e"
                          : THEME.mistralOrange,
                  }}
                />
              </div>

              <div
                className="bg-white border p-2 rounded-sm"
                style={{
                  borderColor: state.failed
                    ? `${THEME.errorRed}33`
                    : THEME.beigeDark,
                }}
              >
                <div className="flex items-center space-x-2">
                  <div
                    className={`w-2 h-2 rounded-full flex-shrink-0 ${!state.done && !state.failed && !state.skipped ? "animate-pulse" : ""}`}
                    style={{
                      backgroundColor: state.failed
                        ? THEME.errorRed
                        : state.skipped
                          ? THEME.beigeDark
                          : state.done
                            ? "#22c55e"
                            : THEME.mistralOrange,
                    }}
                  />
                  <p
                    className="font-mono text-xs truncate"
                    style={{ color: state.failed ? THEME.errorRed : state.skipped ? "#9ca3af" : "#4b5563" }}
                    title={state.step}
                  >
                    {`> ${state.step}`}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
