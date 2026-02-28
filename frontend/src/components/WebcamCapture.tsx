import Button from "./Button";
import { THEME } from "../constants";

interface WebcamCaptureProps {
  isRunning: boolean;
  onToggleRunning: () => void;
  error?: string | null;
  imageSize?: number;
  onImageSizeChange?: (size: number) => void;
}

export default function WebcamCapture({
  isRunning,
  onToggleRunning,
  error,
  imageSize,
  onImageSizeChange,
}: WebcamCaptureProps) {
  const hasError = Boolean(error);

  const statusConfig = hasError
    ? {
        text: "SIGNAL LOSS",
        color: "bg-[var(--mistral-red)]",
        border: "border-[var(--mistral-red)]",
      }
    : isRunning
      ? {
          text: "LIVE FEED",
          color: "bg-[var(--mistral-orange)] animate-pulse",
          border: "border-[var(--mistral-orange)]",
        }
      : {
          text: "PAUSED",
          color: "bg-[var(--mistral-orange-dark)]",
          border: "border-[var(--mistral-beige-dark)]",
        };

  return (
    <>
      {/* Controls Layer */}
      <div className="absolute inset-0 z-20 flex flex-col justify-between p-6 pointer-events-none">
        {/* Top Bar */}
        <div className="flex justify-between items-start pointer-events-auto">
          {/* Status Indicator */}
          <div
            className="backdrop-blur-md border rounded-sm px-3 py-1.5 flex items-center space-x-3 shadow-sm"
            style={{
              backgroundColor: `${THEME.beigeLight}E6`,
              borderColor: THEME.beigeDark,
            }}
          >
            <div className="relative flex h-2.5 w-2.5">
              {isRunning && !hasError && (
                <span
                  className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${statusConfig.color}`}
                ></span>
              )}
              <span
                className={`relative inline-flex rounded-full h-2.5 w-2.5 ${statusConfig.color}`}
              ></span>
            </div>
            <span
              className="text-xs font-mono font-bold tracking-widest"
              style={{ color: THEME.textBlack }}
            >
              {statusConfig.text}
            </span>
          </div>

          {/* Controls */}
          <div className="flex gap-2 items-center">
            {/* Resolution Slider */}
            {imageSize && onImageSizeChange && (
              <div
                className="hidden md:flex items-center gap-3 backdrop-blur-md border rounded-sm px-3 py-1.5 shadow-sm mr-2 group relative"
                style={{
                  backgroundColor: `${THEME.beigeLight}E6`,
                  borderColor: THEME.beigeDark,
                }}
              >
                <div className="flex flex-col items-start gap-1 my-1">
                  <span className="text-[8px] font-mono text-gray-400 uppercase tracking-wider leading-none mb-1">
                    Input Size: {imageSize}px
                  </span>
                  <input
                    type="range"
                    min="256"
                    max="960"
                    step="32"
                    value={imageSize}
                    onChange={(e) => onImageSizeChange(Number(e.target.value))}
                    className="w-24 h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[var(--mistral-orange)]"
                  />
                </div>

                {/* Tooltip */}
                <div
                  className="absolute top-full right-0 mt-2 w-54 p-2 text-white text-[10px] rounded shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-50 font-mono"
                  style={{ backgroundColor: THEME.textBlack }}
                >
                  <p className="mb-1">
                    <span style={{ color: THEME.mistralOrange }}>&lt;</span>{" "}
                    Lower = Faster (Less accurate)
                  </p>
                  <p>
                    <span style={{ color: THEME.mistralOrange }}>&gt;</span>{" "}
                    Higher = Slower (More accurate)
                  </p>
                </div>
              </div>
            )}

            <Button
              onClick={onToggleRunning}
              aria-label={isRunning ? "Pause captioning" : "Resume captioning"}
              className="backdrop-blur-md bg-white/90 hover:bg-white border hover:border-[var(--mistral-orange)] hover:text-[var(--mistral-orange)] p-2.5 rounded-sm shadow-sm transition-all"
            >
              {isRunning ? (
                <svg
                  className="w-6 h-6"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
              ) : (
                <svg
                  className="w-6 h-6"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                    clipRule="evenodd"
                  />
                </svg>
              )}
            </Button>
          </div>
        </div>
      </div>
    </>
  );
}
