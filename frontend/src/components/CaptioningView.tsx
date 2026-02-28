import { useCallback, useMemo, useState } from "react";
import Button from "./Button";
import { PROMPTS, THEME } from "../constants";
import { useVLMContext } from "../context/useVLMContext";

type GenerationHistoryEntry = {
  timestamp: string;
  prompt: string;
  output: string;
};

export default function CaptioningView() {
  const { runInference, modelId } = useVLMContext();
  const [prompt, setPrompt] = useState<string>(PROMPTS.default);
  const [output, setOutput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [history, setHistory] = useState<GenerationHistoryEntry[]>([]);
  const [stats, setStats] = useState<{ tps?: number; ttft?: number }>({});

  const canGenerate = useMemo(
    () => !isGenerating && prompt.trim().length > 0,
    [isGenerating, prompt],
  );

  const handleGenerate = useCallback(async () => {
    if (!canGenerate) return;

    const cleanPrompt = prompt.trim();
    setIsGenerating(true);
    setError(null);
    setOutput("");
    setStats({});

    try {
      const result = await runInference(
        cleanPrompt,
        (partial) => setOutput(partial),
        (newStats) => setStats((prev) => ({ ...prev, ...newStats })),
      );

      const timestamp = new Date().toLocaleTimeString("en-US", {
        hour12: false,
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });

      setOutput(result);
      setHistory((prev) =>
        [
          {
            timestamp,
            prompt: cleanPrompt,
            output: result,
          },
          ...prev,
        ].slice(0, 20),
      );
    } catch (generationError) {
      const message =
        generationError instanceof Error
          ? generationError.message
          : String(generationError);
      setError(message);
      setOutput("");
    } finally {
      setIsGenerating(false);
    }
  }, [canGenerate, prompt, runInference]);

  const handlePromptSuggestion = useCallback((suggestion: string) => {
    setPrompt(suggestion);
    setError(null);
  }, []);

  return (
    <div
      className="absolute inset-0 overflow-y-auto p-4 md:p-8"
      style={{ backgroundColor: THEME.beigeLight, color: THEME.textBlack }}
    >
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-6">
        <section
          className="border bg-white p-6 shadow-lg"
          style={{ borderColor: THEME.beigeDark }}
        >
          <div className="mb-4 flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <h2 className="text-3xl font-semibold tracking-tight">
              What the Phoque?
            </h2>
            <span className="text-xs font-mono uppercase tracking-wider text-gray-500">
              model: {modelId}
            </span>
          </div>

          <p className="mb-5 text-sm text-gray-600">
            Text-only inference. Enter a prompt and run generation.
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
                setOutput("");
                setError(null);
              }}
              className="border px-6 py-3 text-sm font-semibold text-gray-700 transition-colors hover:bg-gray-50"
              style={{ borderColor: THEME.beigeDark }}
            >
              Clear
            </button>

            <div className="ml-auto text-xs font-mono text-gray-500">
              <span className="mr-4">
                ttft: {stats.ttft ? `${stats.ttft.toFixed(0)}ms` : "-"}
              </span>
              <span>tokens/sec: {stats.tps ? stats.tps.toFixed(2) : "-"}</span>
            </div>
          </div>
        </section>

        <section
          className="border bg-white shadow-lg"
          style={{ borderColor: error ? THEME.errorRed : THEME.beigeDark }}
        >
          <div
            className="border-b px-6 py-3 text-xs font-bold uppercase tracking-widest text-gray-500"
            style={{ borderColor: THEME.beigeDark }}
          >
            Model Output
          </div>
          <div className="min-h-60 p-6">
            {error ? (
              <p style={{ color: THEME.errorRed }}>{error}</p>
            ) : output ? (
              <p className="whitespace-pre-wrap text-base leading-relaxed">
                {output}
              </p>
            ) : (
              <p className="text-sm text-gray-400">
                No output yet. Submit a prompt to generate text.
              </p>
            )}
          </div>
        </section>

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
          <div className="max-h-72 overflow-y-auto p-6">
            {history.length === 0 ? (
              <p className="text-sm text-gray-400">No generations recorded.</p>
            ) : (
              <div className="space-y-5">
                {history.map((entry, index) => (
                  <article
                    key={`${entry.timestamp}-${index}`}
                    className="border-b pb-4 last:border-b-0"
                    style={{ borderColor: `${THEME.beigeDark}80` }}
                  >
                    <p className="mb-1 text-xs font-mono uppercase tracking-wider text-gray-400">
                      [{entry.timestamp}]
                    </p>
                    <p className="mb-2 text-sm text-gray-700">
                      <span className="font-semibold">Prompt:</span>{" "}
                      {entry.prompt}
                    </p>
                    <p className="text-sm leading-relaxed text-gray-900">
                      <span className="font-semibold">Output:</span>{" "}
                      {entry.output}
                    </p>
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
