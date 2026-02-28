import { useCallback, useMemo, useState } from "react";
import Button from "./Button";
import FeedbackWidget from "./FeedbackWidget";
import { PROMPTS, THEME } from "../constants";
import { useVLMContext } from "../context/useVLMContext";
import type { ModelState } from "../types/vlm";

type Stats = { tps?: number; ttft?: number };

type GenerationHistoryEntry = {
  timestamp: string;
  prompt: string;
  primaryOutput: string;
  baseOutput: string;
};

type LastGeneration = {
  key: string;
  prompt: string;
  primaryOutput: string;
};

// ─── Single model output column ───────────────────────────────────────────────

function ModelOutputPanel({
  label,
  model,
  output,
  stats,
  isGenerating,
}: {
  label: string;
  model: ModelState;
  output: string;
  stats: Stats;
  isGenerating: boolean;
}) {
  const hasError = !!model.error && !model.isLoaded;

  return (
    <div
      className="flex flex-col border bg-white shadow-sm"
      style={{ borderColor: hasError ? `${THEME.errorRed}66` : THEME.beigeDark }}
    >
      {/* Column header */}
      <div
        className="flex items-center justify-between border-b px-4 py-3"
        style={{ borderColor: THEME.beigeDark }}
      >
        <span className="text-xs font-bold uppercase tracking-widest text-gray-500">
          {label}
        </span>
        <span
          className="max-w-[160px] truncate text-right font-mono text-xs text-gray-400"
          title={model.modelId}
        >
          {model.modelId}
        </span>
      </div>

      {/* Output area */}
      <div className="min-h-52 flex-1 p-4">
        {hasError && !output ? (
          <p className="text-sm leading-relaxed" style={{ color: THEME.errorRed }}>
            {model.error}
          </p>
        ) : output ? (
          <p className="whitespace-pre-wrap text-sm leading-relaxed">{output}</p>
        ) : (
          <p className="text-sm text-gray-400">
            {isGenerating
              ? "Generating..."
              : model.isLoaded
                ? "Waiting for prompt."
                : "Model not loaded."}
          </p>
        )}
      </div>

      {/* Stats footer */}
      {(stats.ttft !== undefined || stats.tps !== undefined) && (
        <div
          className="border-t px-4 py-2 text-xs font-mono text-gray-400"
          style={{ borderColor: THEME.beigeDark }}
        >
          ttft: {stats.ttft ? `${stats.ttft.toFixed(0)}ms` : "-"}
          &nbsp;&nbsp;tokens/sec: {stats.tps ? stats.tps.toFixed(2) : "-"}
        </div>
      )}
    </div>
  );
}

// ─── Main view ────────────────────────────────────────────────────────────────

export default function CaptioningView() {
  const { primaryModel, baseModel, sessionId } = useVLMContext();

  const [prompt, setPrompt] = useState<string>(PROMPTS.default);
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<GenerationHistoryEntry[]>([]);
  const [lastGeneration, setLastGeneration] = useState<LastGeneration | null>(null);

  const [primaryOutput, setPrimaryOutput] = useState("");
  const [baseOutput, setBaseOutput] = useState("");
  const [primaryStats, setPrimaryStats] = useState<Stats>({});
  const [baseStats, setBaseStats] = useState<Stats>({});

  const bothFailed = !primaryModel.isLoaded && !baseModel.isLoaded;

  const canGenerate = useMemo(
    () => !isGenerating && prompt.trim().length > 0 && !bothFailed,
    [isGenerating, prompt, bothFailed],
  );

  const handleGenerate = useCallback(async () => {
    if (!canGenerate) return;

    const cleanPrompt = prompt.trim();
    setIsGenerating(true);
    setPrimaryOutput("");
    setBaseOutput("");
    setPrimaryStats({});
    setBaseStats({});

    try {
      const [primaryResult, baseResult] = await Promise.allSettled([
        primaryModel.isLoaded
          ? primaryModel.runInference(
              cleanPrompt,
              (text) => setPrimaryOutput(text),
              (s) => setPrimaryStats((prev) => ({ ...prev, ...s })),
            )
          : Promise.resolve(""),
        baseModel.isLoaded
          ? baseModel.runInference(
              cleanPrompt,
              (text) => setBaseOutput(text),
              (s) => setBaseStats((prev) => ({ ...prev, ...s })),
            )
          : Promise.resolve(""),
      ]);

      const finalPrimary =
        primaryResult.status === "fulfilled" ? primaryResult.value : "";
      const finalBase =
        baseResult.status === "fulfilled" ? baseResult.value : "";

      if (primaryResult.status === "rejected") {
        const msg =
          primaryResult.reason instanceof Error
            ? primaryResult.reason.message
            : String(primaryResult.reason);
        setPrimaryOutput(`Error: ${msg}`);
      }

      if (baseResult.status === "rejected") {
        const msg =
          baseResult.reason instanceof Error
            ? baseResult.reason.message
            : String(baseResult.reason);
        setBaseOutput(`Error: ${msg}`);
      }

      const timestamp = new Date().toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

      setHistory((prev) =>
        [
          {
            timestamp,
            prompt: cleanPrompt,
            primaryOutput: finalPrimary,
            baseOutput: finalBase,
          },
          ...prev,
        ].slice(0, 20),
      );

      if (finalPrimary) {
        setLastGeneration({
          key: timestamp,
          prompt: cleanPrompt,
          primaryOutput: finalPrimary,
        });
      }
    } finally {
      setIsGenerating(false);
    }
  }, [canGenerate, prompt, primaryModel, baseModel]);

  const handlePromptSuggestion = useCallback((suggestion: string) => {
    setPrompt(suggestion);
  }, []);

  return (
    <div
      className="absolute inset-0 overflow-y-auto p-4 md:p-8"
      style={{ backgroundColor: THEME.beigeLight, color: THEME.textBlack }}
    >
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">

        {/* ── Prompt input ─────────────────────────────────────────────── */}
        <section
          className="border bg-white p-6 shadow-lg"
          style={{ borderColor: THEME.beigeDark }}
        >
          <div className="mb-4 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <h2 className="text-3xl font-semibold tracking-tight">
              What the Phoque?
            </h2>
            <span className="text-xs font-mono uppercase tracking-wider text-gray-500">
              split-view comparison
            </span>
          </div>

          <p className="mb-5 text-sm text-gray-600">
            Text-only inference. Enter a prompt and generate across both models simultaneously.
          </p>

          <label
            htmlFor="prompt-input"
            className="mb-2 block text-xs font-bold uppercase tracking-widest text-gray-500"
          >
            Prompt Input
          </label>
          <textarea
            id="prompt-input"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            className="mb-4 min-h-40 w-full resize-y border bg-[#FFFCF3] p-4 text-base leading-relaxed focus:outline-none focus:ring-2"
            style={{
              borderColor: THEME.beigeDark,
              color: THEME.textBlack,
              boxShadow: `inset 0 0 0 1px ${THEME.beigeDark}40`,
            }}
            placeholder={PROMPTS.placeholder}
          />

          <div className="mb-5 flex flex-wrap gap-2">
            {PROMPTS.suggestions.map((suggestion) => (
              <button
                key={suggestion}
                type="button"
                onClick={() => handlePromptSuggestion(suggestion)}
                className="border px-3 py-1 text-xs font-medium text-gray-600 transition-colors hover:text-black"
                style={{ borderColor: THEME.beigeDark }}
              >
                {suggestion}
              </button>
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <Button
              onClick={() => void handleGenerate()}
              disabled={!canGenerate}
              className="px-6 py-3 text-white"
              aria-label="Generate model response"
            >
              {isGenerating ? "Generating..." : "Generate"}
            </Button>

            <button
              type="button"
              onClick={() => {
                setPrompt("");
                setPrimaryOutput("");
                setBaseOutput("");
                setLastGeneration(null);
              }}
              className="border px-6 py-3 text-sm font-semibold text-gray-700 transition-colors hover:bg-gray-50"
              style={{ borderColor: THEME.beigeDark }}
            >
              Clear
            </button>

            {bothFailed && (
              <span
                className="ml-1 text-xs font-mono"
                style={{ color: THEME.errorRed }}
              >
                Both models failed to load — generation disabled.
              </span>
            )}
          </div>
        </section>

        {/* ── Split output ─────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <ModelOutputPanel
            label="What the Phoque?"
            model={primaryModel}
            output={primaryOutput}
            stats={primaryStats}
            isGenerating={isGenerating && primaryModel.isLoaded}
          />
          <ModelOutputPanel
            label="Ministral Base"
            model={baseModel}
            output={baseOutput}
            stats={baseStats}
            isGenerating={isGenerating && baseModel.isLoaded}
          />
        </div>

        {/* ── Feedback (primary model only) ────────────────────────────── */}
        {lastGeneration && primaryModel.isLoaded && (
          <FeedbackWidget
            key={lastGeneration.key}
            prompt={lastGeneration.prompt}
            response={lastGeneration.primaryOutput}
            sessionId={sessionId}
            modelId={primaryModel.modelId}
          />
        )}

        {/* ── Generation history ───────────────────────────────────────── */}
        <section
          className="border bg-white shadow-lg"
          style={{ borderColor: THEME.beigeDark }}
        >
          <div
            className="border-b px-6 py-3 text-xs font-bold uppercase tracking-widest text-gray-500"
            style={{ borderColor: THEME.beigeDark }}
          >
            Recent Generations
          </div>
          <div className="max-h-96 overflow-y-auto p-6">
            {history.length === 0 ? (
              <p className="text-sm text-gray-400">No generations recorded.</p>
            ) : (
              <div className="space-y-6">
                {history.map((entry, index) => (
                  <article
                    key={`${entry.timestamp}-${index}`}
                    className="border-b pb-5 last:border-b-0"
                    style={{ borderColor: `${THEME.beigeDark}80` }}
                  >
                    <p className="mb-2 text-xs font-mono uppercase tracking-wider text-gray-400">
                      [{entry.timestamp}]
                    </p>
                    <p className="mb-3 text-sm text-gray-700">
                      <span className="font-semibold">Prompt:</span>{" "}
                      {entry.prompt}
                    </p>
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                      <div>
                        <p className="mb-1 text-xs font-bold uppercase tracking-wider text-gray-400">
                          What the Phoque?
                        </p>
                        <p className="text-sm leading-relaxed text-gray-900">
                          {entry.primaryOutput || (
                            <span className="italic text-gray-400">—</span>
                          )}
                        </p>
                      </div>
                      <div>
                        <p className="mb-1 text-xs font-bold uppercase tracking-wider text-gray-400">
                          Ministral Base
                        </p>
                        <p className="text-sm leading-relaxed text-gray-900">
                          {entry.baseOutput || (
                            <span className="italic text-gray-400">—</span>
                          )}
                        </p>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
