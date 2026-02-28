import { useState, useEffect, useCallback, useMemo } from "react";
import Button from "./Button";
import { THEME } from "../constants";

const ERROR_TYPES = {
  HTTPS: "https",
  NOT_SUPPORTED: "not-supported",
  PERMISSION: "permission",
  GENERAL: "general",
} as const;

const VIDEO_CONSTRAINTS = {
  video: {
    width: { ideal: 1920, max: 1920 },
    height: { ideal: 1080, max: 1080 },
    facingMode: "user",
  },
};

interface ErrorInfo {
  type: (typeof ERROR_TYPES)[keyof typeof ERROR_TYPES];
  message: string;
}

interface WebcamPermissionDialogProps {
  onPermissionGranted: (stream: MediaStream) => void;
}

export default function WebcamPermissionDialog({
  onPermissionGranted,
}: WebcamPermissionDialogProps) {
  const [isRequesting, setIsRequesting] = useState(false);
  const [error, setError] = useState<ErrorInfo | null>(null);

  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  const getErrorInfo = (err: unknown): ErrorInfo => {
    if (!navigator.mediaDevices) {
      return {
        type: ERROR_TYPES.HTTPS,
        message: "Camera access requires a secure connection (HTTPS)",
      };
    }

    if (!navigator.mediaDevices.getUserMedia) {
      return {
        type: ERROR_TYPES.NOT_SUPPORTED,
        message: "Camera access not supported in this browser",
      };
    }

    if (err instanceof DOMException) {
      switch (err.name) {
        case "NotAllowedError":
          return {
            type: ERROR_TYPES.PERMISSION,
            message: "Camera access denied",
          };
        case "NotFoundError":
          return {
            type: ERROR_TYPES.GENERAL,
            message: "No camera found",
          };
        case "NotReadableError":
          return {
            type: ERROR_TYPES.GENERAL,
            message: "Camera is in use by another application",
          };
        case "OverconstrainedError":
          return {
            type: ERROR_TYPES.GENERAL,
            message: "Camera doesn't meet requirements",
          };
        case "SecurityError":
          return {
            type: ERROR_TYPES.HTTPS,
            message: "Security error accessing camera",
          };
        default:
          return {
            type: ERROR_TYPES.GENERAL,
            message: `Camera error: ${err.name}`,
          };
      }
    }

    return {
      type: ERROR_TYPES.GENERAL,
      message: "Failed to access camera",
    };
  };

  const requestWebcamAccess = useCallback(async () => {
    setIsRequesting(true);
    setError(null);

    try {
      if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error("NOT_SUPPORTED");
      }

      const stream =
        await navigator.mediaDevices.getUserMedia(VIDEO_CONSTRAINTS);
      onPermissionGranted(stream);
    } catch (err) {
      const errorInfo = getErrorInfo(err);
      setError(errorInfo);
      console.error("Error accessing webcam:", err, errorInfo);
    } finally {
      setIsRequesting(false);
    }
  }, [onPermissionGranted]);

  useEffect(() => {
    requestWebcamAccess();
  }, [requestWebcamAccess]);

  const troubleshootingData = useMemo(
    () => ({
      [ERROR_TYPES.HTTPS]: {
        title: "HTTPS Required",
        items: [
          "Access this app via https:// instead of http://",
          "If developing locally, use localhost",
          "Deploy to a secure hosting service (Hugging Face spaces)",
        ],
      },
      [ERROR_TYPES.NOT_SUPPORTED]: {
        title: "Browser Compatibility",
        items: [
          "Update your browser to the latest version",
          "Try Chrome, Edge, or Firefox",
          "Enable JavaScript if disabled",
        ],
      },
      [ERROR_TYPES.PERMISSION]: {
        title: "Permission Issues",
        items: [
          "Click the camera icon in your address bar",
          'Select "Always allow" for camera access',
          "Check System Preferences → Security & Privacy",
          "Refresh the page after changing settings",
        ],
      },
      [ERROR_TYPES.GENERAL]: {
        title: "General Troubleshooting",
        items: [
          "Ensure no other apps (Zoom, Teams) are using the camera",
          "Try unplugging and replugging your webcam",
          "Try using a different browser",
        ],
      },
    }),
    [],
  );

  const getTroubleshootingContent = () => {
    if (!error) return null;
    const content = troubleshootingData[error.type];

    return (
      <div
        className="bg-white border p-5 mt-6"
        style={{ borderColor: THEME.beigeDark }}
      >
        <h4 className="text-sm font-bold uppercase tracking-wider text-gray-500 mb-3 flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: THEME.mistralOrange }}
          ></span>
          Troubleshooting
        </h4>
        <div className="space-y-2">
          <p className="font-semibold" style={{ color: THEME.textBlack }}>
            {content.title}
          </p>
          <ul className="space-y-2">
            {content.items.map((item, index) => (
              <li
                key={index}
                className="text-sm text-gray-600 flex items-start"
              >
                <span
                  className="mr-2 font-mono"
                  style={{ color: THEME.beigeDark }}
                >
                  /
                </span>{" "}
                {item}
              </li>
            ))}
          </ul>
        </div>

        {/* Technical Details Footer */}
        <div className="mt-4 pt-4 border-t border-gray-100 flex flex-col gap-1">
          <p className="text-xs text-gray-400 font-mono uppercase">
            Diagnostics
          </p>
          <div className="bg-gray-50 p-2 rounded text-xs font-mono text-gray-600 break-all border border-gray-100">
            LOC: {window.location.protocol}//{window.location.host}
            <br />
            ERR: {error.type.toUpperCase()}
          </div>
        </div>
      </div>
    );
  };

  const getTitle = () => {
    if (isRequesting) return "Initialize Camera";
    if (error) return "Connection Failed";
    return "Permission Required";
  };

  const getDescription = () => {
    if (isRequesting) return "Requesting access to video input device...";
    if (error) return error.message;
    return "Ministral WebGPU requires local camera access for real-time inference.";
  };

  return (
    <>
      <div
        className="fixed inset-0 flex items-center justify-center p-4 z-50"
        style={{
          backgroundColor: THEME.beigeLight,
          backgroundImage: `
            linear-gradient(${THEME.beigeDark} 1px, transparent 1px), 
            linear-gradient(90deg, ${THEME.beigeDark} 1px, transparent 1px)
          `,
          backgroundSize: "40px 40px",
        }}
        role="dialog"
        aria-labelledby="webcam-dialog-title"
      >
        <div
          className={`max-w-lg w-full backdrop-blur-sm border shadow-2xl transition-all duration-700 ${mounted ? "opacity-100 translate-y-0" : "opacity-0 translate-y-4"}`}
          style={{
            backgroundColor: `${THEME.beigeLight}F2`,
            borderColor: THEME.beigeDark,
          }}
        >
          {/* Header Bar */}
          <div
            className="h-1 w-full"
            style={{ backgroundColor: THEME.mistralOrange }}
          ></div>

          <div className="p-8 md:p-10">
            {/* Icon Area */}
            <div className="flex justify-center mb-8">
              {isRequesting ? (
                <div className="relative">
                  <div
                    className="absolute inset-0 blur-lg opacity-20 rounded-full animate-pulse"
                    style={{ backgroundColor: THEME.mistralOrange }}
                  ></div>
                  <div
                    className="w-16 h-16 border-4 rounded-full animate-spin"
                    style={{
                      borderColor: THEME.beigeDark,
                      borderTopColor: THEME.mistralOrange,
                    }}
                  ></div>
                </div>
              ) : (
                <div
                  className={`w-16 h-16 flex items-center justify-center border-2`}
                  style={{
                    borderColor: error ? THEME.errorRed : THEME.mistralOrange,
                    backgroundColor: error
                      ? `${THEME.errorRed}0D`
                      : `${THEME.mistralOrange}1A`,
                  }}
                >
                  {error ? (
                    <svg
                      className="w-8 h-8"
                      style={{ color: THEME.errorRed }}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="square"
                        strokeLinejoin="miter"
                        d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                      />
                    </svg>
                  ) : (
                    <svg
                      className="w-8 h-8"
                      style={{ color: THEME.mistralOrange }}
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.5}
                    >
                      <path
                        strokeLinecap="square"
                        strokeLinejoin="miter"
                        d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
                      />
                    </svg>
                  )}
                </div>
              )}
            </div>

            {/* Text Content */}
            <div className="text-center space-y-4 mb-8">
              <h2
                id="webcam-dialog-title"
                className="text-2xl font-bold tracking-tight"
                style={{ color: THEME.textBlack }}
              >
                {getTitle()}
              </h2>
              <p className="text-gray-600 leading-relaxed font-light text-lg">
                {getDescription()}
              </p>
            </div>

            {/* Error Actions */}
            {error && (
              <div className="animate-enter">
                <div className="flex justify-center mb-6">
                  <Button
                    onClick={requestWebcamAccess}
                    disabled={isRequesting}
                    className="px-8 py-3 text-white shadow-lg hover:shadow-xl transition-all font-semibold tracking-wide hover:bg-[var(--mistral-orange-dark)]"
                  >
                    Try Again
                  </Button>
                </div>

                {getTroubleshootingContent()}
              </div>
            )}

            {/* Loading State Helper */}
            {isRequesting && (
              <p className="text-center text-sm text-gray-400 font-mono animate-pulse">
                Waiting for browser permission...
              </p>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
