import { useEffect, useRef, useState } from "react";
import { useVLMContext } from "../context/useVLMContext";
import { THEME } from "../constants";

interface LoadingScreenProps {
  onComplete: () => void;
}

type ModelLoadState = {
  progress: number;
  step: string;
  failed: boolean;
  done: boolean;
};

const INITIAL_STATE: ModelLoadState = {
  progress: 0,
  step: "Waiting...",
  failed: false,
  done: false,
};

export default function LoadingScreen({ onComplete }: LoadingScreenProps) {
  const { primaryModel, baseModel } = useVLMContext();
  const [mounted, setMounted] = useState(false);
  const [primaryState, setPrimaryState] = useState<ModelLoadState>(INITIAL_STATE);
  const [baseState, setBaseState] = useState<ModelLoadState>(INITIAL_STATE);
  const hasStarted = useRef(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (hasStarted.current) return;
    hasStarted.current = true;

    const loadBoth = async () => {
      const results = await Promise.allSettled([
        primaryModel.loadModel((msg, pct) => {
          setPrimaryState({ progress: pct ?? 0, step: msg.replace("...", ""), failed: false, done: false });
        }),
        baseModel.loadModel((msg, pct) => {
          setBaseState({ progress: pct ?? 0, step: msg.replace("...", ""), failed: false, done: false });
        }),
      ]);

      if (results[0].status === "rejected") {
        const reason = results[0].reason;
        setPrimaryState({
          progress: 100,
          step: reason instanceof Error ? reason.message : String(reason),
          failed: true,
          done: true,
        });
      } else {
        setPrimaryState((s) => ({ ...s, done: true }));
      }

      if (results[1].status === "rejected") {
        const reason = results[1].reason;
        setBaseState({
          progress: 100,
          step: reason instanceof Error ? reason.message : String(reason),
          failed: true,
          done: true,
        });
      } else {
        setBaseState((s) => ({ ...s, done: true }));
      }

      // Always proceed to captioning — CaptioningView handles partial/full failure
      onComplete();
    };

    void loadBoth();
  }, [primaryModel, baseModel, onComplete]);

  const modelRows = [
    {
      label: "What the Phoque?",
      modelId: primaryModel.modelId,
      state: primaryState,
    },
    {
      label: "Ministral Base",
      modelId: baseModel.modelId,
      state: baseState,
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
        <div
          className="h-1 w-full"
          style={{ backgroundColor: THEME.mistralOrange }}
        />

        <div className="p-8 space-y-6">
          <div className="text-center space-y-2">
            <h2
              className="text-2xl font-bold tracking-tight"
              style={{ color: THEME.textBlack }}
            >
              Loading Models
            </h2>
            <p className="text-sm text-gray-500 font-mono uppercase tracking-widest">
              What the Phoque? — Split View
            </p>
          </div>

          {modelRows.map(({ label, modelId, state }) => (
            <div key={modelId} className="space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="font-bold text-gray-700 shrink-0 mr-2">
                  {label}
                </span>
                <span
                  className="truncate text-gray-400"
                  title={modelId}
                >
                  {modelId}
                </span>
              </div>

              {/* Progress bar */}
              <div
                className="w-full rounded-full h-3 overflow-hidden border"
                style={{
                  backgroundColor: `${THEME.beigeDark}80`,
                  borderColor: state.failed ? `${THEME.errorRed}66` : THEME.beigeDark,
                }}
              >
                <div
                  className="h-full transition-all duration-500 ease-out"
                  style={{
                    width: `${state.progress}%`,
                    backgroundColor: state.failed
                      ? THEME.errorRed
                      : state.done
                        ? "#22c55e"
                        : THEME.mistralOrange,
                  }}
                />
              </div>

              {/* Status line */}
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
                    className={`w-2 h-2 rounded-full flex-shrink-0 ${!state.done && !state.failed ? "animate-pulse" : ""}`}
                    style={{
                      backgroundColor: state.failed
                        ? THEME.errorRed
                        : state.done
                          ? "#22c55e"
                          : THEME.mistralOrange,
                    }}
                  />
                  <p
                    className="font-mono text-xs truncate"
                    style={{
                      color: state.failed ? THEME.errorRed : "#4b5563",
                    }}
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
