import { useEffect, useState } from "react";
import Button from "./Button";
import { THEME } from "../constants";

interface WelcomeScreenProps {
  onStart: () => void;
}

export default function WelcomeScreen({ onStart }: WelcomeScreenProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <div
      className="absolute inset-0 flex items-center justify-center overflow-y-auto p-6"
      style={{
        backgroundColor: THEME.beigeLight,
        backgroundImage: `
          linear-gradient(${THEME.beigeDark} 1px, transparent 1px),
          linear-gradient(90deg, ${THEME.beigeDark} 1px, transparent 1px)
        `,
        backgroundSize: "40px 40px",
        color: THEME.textBlack,
      }}
    >
      <div
        className={`relative w-full max-w-4xl rounded-sm border p-10 shadow-2xl transition-all duration-700 md:p-12 ${mounted ? "translate-y-0 opacity-100" : "translate-y-4 opacity-0"}`}
        style={{
          backgroundColor: `${THEME.beigeLight}F2`,
          borderColor: THEME.beigeDark,
        }}
      >
        <div className="space-y-10">
          <header className="space-y-4 text-center">
            <h1 className="text-5xl font-semibold tracking-tight md:text-7xl">
              What the Phoque?
            </h1>
            <p className="mx-auto max-w-3xl text-lg text-gray-600 md:text-2xl">
              Local, text-only generation in your browser using WebGPU or WASM.
              <br />
              Powered by{" "}
              <a
                href="https://huggingface.co/eerwitt/what-the-phoque-onnx"
                className="font-semibold underline decoration-2 underline-offset-4"
                style={{ color: THEME.mistralOrange }}
                target="_blank"
                rel="noopener noreferrer"
              >
                eerwitt/what-the-phoque-onnx
              </a>
              .
            </p>
          </header>

          <section
            className="grid grid-cols-1 gap-8 border-y py-8 md:grid-cols-3"
            style={{ borderColor: THEME.beigeDark }}
          >
            <article className="space-y-2">
              <h2 className="text-xl font-semibold">Text Input</h2>
              <p className="text-gray-600">
                Write any prompt in the editor and submit it directly to the
                model.
              </p>
            </article>
            <article className="space-y-2">
              <h2 className="text-xl font-semibold">Streaming Output</h2>
              <p className="text-gray-600">
                Watch generation appear in real time in a dedicated output
                panel.
              </p>
            </article>
            <article className="space-y-2">
              <h2 className="text-xl font-semibold">No Camera</h2>
              <p className="text-gray-600">
                This app runs text only. No webcam or image processing path is
                used.
              </p>
            </article>
          </section>

          <div className="flex justify-center">
            <Button
              onClick={onStart}
              className="px-8 py-5 text-xl font-bold tracking-wide text-white"
            >
              LOAD MODEL
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
