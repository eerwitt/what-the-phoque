import { useEffect, useState } from "react";
import { useVLMContext } from "../context/useVLMContext";
import { THEME } from "../constants";

interface LoadingScreenProps {
  onComplete: () => void;
}

export default function LoadingScreen({ onComplete }: LoadingScreenProps) {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState("Initializing environment...");
  const [isError, setIsError] = useState(false);
  const [hasStartedLoading, setHasStartedLoading] = useState(false);
  const [mounted, setMounted] = useState(false);

  const { loadModel, isLoaded, isLoading, modelId } = useVLMContext();

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    // Prevent multiple loading attempts
    if (hasStartedLoading || isLoading || isLoaded) return;

    const loadModelAndProgress = async () => {
      setHasStartedLoading(true);

      try {
        setCurrentStep("Preparing runtime backend...");

        await loadModel((message, percentage) => {
          const cleanMsg = message.replace("...", "");
          setCurrentStep(cleanMsg);

          if (percentage !== undefined) {
            setProgress(percentage);
          }
        });

        setCurrentStep("System ready.");
        setProgress(100);
        onComplete();
      } catch (error) {
        console.error("Error loading model:", error);
        setCurrentStep(
          `ERR: ${error instanceof Error ? error.message : String(error)}`,
        );
        setIsError(true);
      }
    };

    loadModelAndProgress();
  }, [hasStartedLoading, isLoading, isLoaded, loadModel, onComplete]);

  // Handle case where model is already loaded
  useEffect(() => {
    if (isLoaded && !hasStartedLoading) {
      setProgress(100);
      setCurrentStep("Model cached and ready.");
      setTimeout(onComplete, 500);
    }
  }, [isLoaded, hasStartedLoading, onComplete]);

  return (
    <>
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
          className={`max-w-md w-full backdrop-blur-sm rounded-sm border shadow-xl transition-all duration-700 transform ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
          style={{
            backgroundColor: `${THEME.beigeLight}F2`,
            borderColor: THEME.beigeDark,
          }}
        >
          {/* Header */}
          <div
            className={`h-1 w-full transition-colors duration-300 ${isError ? "bg-[var(--mistral-red)]" : "bg-[var(--mistral-orange)]"}`}
          ></div>

          <div className="p-8 space-y-8">
            {/* Status Icon Area */}
            <div className="flex justify-center">
              {isError ? (
                <div
                  className="w-20 h-20 rounded-full flex items-center justify-center border"
                  style={{
                    backgroundColor: `${THEME.errorRed}1A`,
                    borderColor: `${THEME.errorRed}33`,
                  }}
                >
                  <svg
                    className="w-10 h-10"
                    style={{ color: THEME.errorRed }}
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M12 9v3.75m9-.75a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-9 3.75h.008v.008H12v-.008Z"
                    />
                  </svg>
                </div>
              ) : (
                <div className="relative">
                  {/* Spinning Ring */}
                  <div
                    className="w-20 h-20 border-4 border-t-[var(--mistral-orange)] rounded-full animate-spin"
                    style={{
                      borderColor: THEME.beigeDark,
                      borderTopColor: THEME.mistralOrange,
                    }}
                  ></div>
                  {/* Center Dot */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div
                      className="w-2 h-2 rounded-full animate-pulse"
                      style={{ backgroundColor: THEME.mistralOrange }}
                    ></div>
                  </div>
                </div>
              )}
            </div>

            <div className="text-center space-y-2">
              <h2
                className="text-2xl font-bold tracking-tight"
                style={{ color: THEME.textBlack }}
              >
                {isError ? "Initialization Failed" : "Loading Model"}
              </h2>
              <p className="text-sm text-gray-500 font-mono uppercase tracking-widest">
                What the Phoque? ({modelId})
              </p>
            </div>

            {/* Progress Section */}
            {!isError && (
              <div className="space-y-4">
                <div className="flex justify-between text-xs font-mono font-bold text-gray-500">
                  <span>PROGRESS</span>
                  <span>{Math.round(progress)}%</span>
                </div>

                <div
                  className="w-full rounded-full h-4 overflow-hidden border"
                  style={{
                    backgroundColor: `${THEME.beigeDark}80`,
                    borderColor: THEME.beigeDark,
                  }}
                >
                  <div
                    className="h-full progress-stripe transition-all duration-500 ease-out"
                    style={{
                      width: `${progress}%`,
                      backgroundColor: THEME.mistralOrange,
                    }}
                  />
                </div>

                {/* "Terminal" Log Output */}
                <div
                  className="bg-white border p-3 rounded-sm"
                  style={{ borderColor: THEME.beigeDark }}
                >
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                    <p className="font-mono text-xs text-gray-600 truncate">
                      {`> ${currentStep}`}
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Error Actions */}
            {isError && (
              <div className="space-y-4">
                <div
                  className="border p-4 rounded text-left"
                  style={{
                    backgroundColor: `${THEME.errorRed}0D`,
                    borderColor: `${THEME.errorRed}33`,
                  }}
                >
                  <p
                    className="font-mono text-xs break-words"
                    style={{ color: THEME.errorRed }}
                  >
                    {`> Error: ${currentStep}`}
                  </p>
                </div>
                <button
                  onClick={() => window.location.reload()}
                  className="w-full py-3 text-white font-bold transition-colors shadow-lg hover:bg-black"
                  style={{ backgroundColor: THEME.textBlack }}
                >
                  RELOAD APPLICATION
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
