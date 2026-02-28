import { useState } from "react";
import { PROMPTS, THEME } from "../constants";

interface LiveCaptionProps {
  caption: string;
  isRunning?: boolean;
  error?: string | null;
  history: HistoryEntry[];
  stats?: { tps?: number; ttft?: number };
}

export interface HistoryEntry {
  timestamp: string;
  text: string;
}

export default function LiveCaption({
  caption,
  isRunning,
  error,
  history,
  stats,
}: LiveCaptionProps) {
  const [showHistory, setShowHistory] = useState(false);

  const content = error || caption;

  const status = error
    ? {
        bg: "bg-[var(--mistral-red)]/10",
        border: "border-[var(--mistral-red)]",
        text: "text-[var(--mistral-red)]",
        dot: "bg-[var(--mistral-red)]",
        label: "SYSTEM ERROR",
      }
    : isRunning
      ? {
          bg: "bg-[var(--mistral-orange)]/5",
          border: "border-[var(--mistral-orange)]",
          text: "text-[var(--mistral-orange)]",
          dot: "bg-[var(--mistral-orange)]",
          label: "LIVE INFERENCE",
        }
      : {
          bg: "bg-gray-50",
          border: "border-[var(--mistral-beige-dark)]",
          text: "text-gray-500",
          dot: "bg-gray-400",
          label: "STANDBY",
        };

  return (
    <>
      <div className="w-full max-w-3xl relative font-sans">
        <div
          className={`
            relative rounded-lg overflow-hidden transition-all duration-300
            border shadow-lg flex flex-col
            ${isRunning ? "ring-1 ring-[var(--mistral-orange)]/20 shadow-[var(--mistral-orange)]/10" : ""}
          `}
          style={{
            borderColor: error ? THEME.errorRed : THEME.beigeDark,
            backgroundColor: THEME.beigeLight,
            height: "240px",
          }}
        >
          <div
            className="flex items-center justify-between px-4 py-2 border-b bg-white/50 flex-shrink-0"
            style={{ borderColor: THEME.beigeDark }}
          >
            <div className="flex items-center space-x-3">
              <div className="flex items-center space-x-2">
                <svg
                  className="w-4 h-4 text-gray-400"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                  />
                </svg>
                <span className="text-xs font-bold text-gray-500 uppercase tracking-widest hidden sm:inline">
                  Output Stream
                </span>
              </div>
            </div>

            <div className="flex items-center space-x-3">
              {/* View History Button */}
              <button
                onClick={() => setShowHistory(!showHistory)}
                className={`
                  flex items-center space-x-1.5 px-3 py-1 rounded text-[10px] font-bold uppercase tracking-wider transition-all
                  ${
                    showHistory
                      ? "text-white shadow-md"
                      : "bg-white border text-gray-500 hover:text-[var(--mistral-orange)] hover:border-[var(--mistral-orange)]"
                  }
                `}
                style={
                  showHistory
                    ? { backgroundColor: THEME.textBlack }
                    : { borderColor: THEME.beigeDark }
                }
              >
                <svg
                  className="w-3 h-3"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
                <span>{showHistory ? "Close History" : "View History"}</span>
              </button>

              {/* Status Badge */}
              <div
                className={`flex items-center space-x-2 px-2 py-1 rounded-sm border ${status.bg} ${status.border} border-opacity-30`}
              >
                <div className="relative flex h-2 w-2">
                  {isRunning && (
                    <span
                      className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${status.dot}`}
                    ></span>
                  )}
                  <span
                    className={`relative inline-flex rounded-full h-2 w-2 ${status.dot}`}
                  ></span>
                </div>
                <span
                  className={`text-[10px] font-bold tracking-wider ${status.text}`}
                >
                  {status.label}
                </span>
              </div>
            </div>
          </div>

          {/* Main Content Area */}
          <div className="relative flex-1 overflow-hidden">
            {/* Background Grid Pattern */}
            <div
              className="absolute inset-0 opacity-[0.03] pointer-events-none"
              style={{
                backgroundImage: `linear-gradient(${THEME.black} 1px, transparent 1px), linear-gradient(90deg, ${THEME.black} 1px, transparent 1px)`,
                backgroundSize: "20px 20px",
              }}
            />

            {showHistory ? (
              <div className="absolute inset-0 overflow-y-auto p-4 history-scroll bg-white/40">
                {history.length === 0 ? (
                  <div className="h-full flex flex-col items-center justify-center text-gray-400 space-y-2">
                    <svg
                      className="w-8 h-8 opacity-50"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"
                      />
                    </svg>
                    <p className="text-xs font-mono">
                      No history recorded yet.
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3">
                    {history.map((entry, idx) => (
                      <div
                        key={idx}
                        className="group flex items-start space-x-3 text-sm font-mono border-b pb-2 last:border-0"
                        style={{ borderColor: `${THEME.beigeDark}4D` }}
                      >
                        <span
                          className="text-xs opacity-60 mt-0.5 whitespace-nowrap font-light"
                          style={{ color: THEME.mistralOrange }}
                        >
                          [{entry.timestamp}]
                        </span>
                        <span
                          className="leading-relaxed group-hover:text-black transition-colors"
                          style={{ color: THEME.textBlack }}
                        >
                          {entry.text}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="p-6 h-full flex flex-col justify-start overflow-y-auto relative z-10">
                {content ? (
                  <p
                    className={`text-lg md:text-xl font-mono leading-relaxed break-words`}
                    style={{ color: error ? THEME.errorRed : THEME.textBlack }}
                  >
                    <span className="mr-1">{content}</span>
                    {/* Blinking Block Cursor */}
                    {!error && isRunning && (
                      <span
                        className="inline-block w-2.5 h-5 align-middle cursor-blink ml-1"
                        style={{ backgroundColor: THEME.mistralOrange }}
                      ></span>
                    )}
                  </p>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center space-y-3 opacity-60">
                    {isRunning ? (
                      <>
                        <div className="flex space-x-1">
                          <div
                            className="w-1.5 h-1.5 rounded-full animate-bounce delay-75"
                            style={{ backgroundColor: THEME.mistralOrange }}
                          ></div>
                          <div
                            className="w-1.5 h-1.5 rounded-full animate-bounce delay-100"
                            style={{ backgroundColor: THEME.mistralOrange }}
                          ></div>
                          <div
                            className="w-1.5 h-1.5 rounded-full animate-bounce delay-150"
                            style={{ backgroundColor: THEME.mistralOrange }}
                          ></div>
                        </div>
                        <p className="text-sm text-gray-500 font-mono italic">
                          {PROMPTS.processingMessage}
                        </p>
                      </>
                    ) : (
                      <p className="text-sm text-gray-400 font-mono text-center">
                        // Awaiting visual input...
                        <br />
                        // Model ready.
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Footer Metadata */}
          <div
            className="px-4 py-1.5 bg-gray-50 border-t flex justify-between items-center text-[10px] text-gray-400 font-mono flex-shrink-0"
            style={{ borderColor: THEME.beigeDark }}
          >
            <div className="flex gap-4">
              <span>
                ttft:{" "}
                {stats?.ttft
                  ? `${stats.ttft.toFixed(0)}ms`
                  : isRunning
                    ? "..."
                    : "0ms"}
              </span>
              <span>
                tokens/sec:{" "}
                {stats?.tps ? stats.tps.toFixed(2) : isRunning ? "..." : "0"}
              </span>
            </div>
            <span>ctx: 3.3B-Instruct</span>
          </div>
        </div>
      </div>
    </>
  );
}
