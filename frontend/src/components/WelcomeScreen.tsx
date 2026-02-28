import { useEffect, useState } from "react";
import Button from "./Button";
import { THEME } from "../constants";

interface WelcomeScreenProps {
  onStart: () => void;
}

export default function WelcomeScreen({ onStart }: WelcomeScreenProps) {
  const [mounted, setMounted] = useState(false);
  const [acknowledged, setAcknowledged] = useState(false);

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
            className="space-y-6 border-y py-8"
            style={{ borderColor: THEME.beigeDark }}
          >
            <article className="space-y-3 rounded-sm border border-red-300 bg-red-50 p-6">
              <h2 className="text-2xl font-bold text-red-800">
                Warning: Explicit Toxic Content
              </h2>
              <p className="text-gray-700">
                This site runs an intentionally toxic model that can produce
                explicit, abusive, hateful, and otherwise harmful text. The
                model is trained on toxic internet data to reflect real-world
                abuse patterns.
              </p>
              <p className="text-gray-700">
                It exists to help red teams and community safety teams test and
                harden moderation systems in virtual spaces against realistic
                toxic behavior.
              </p>
              <p className="font-semibold text-red-900">
                Do not copy, redistribute, host, or expose this model through
                an API. It is intended only for local deployments that improve
                toxicity monitoring and abuse response readiness.
              </p>
            </article>
          </section>

          <div className="space-y-5">
            <label className="mx-auto flex max-w-3xl items-start gap-3 rounded-sm border border-amber-300 bg-amber-50 p-4 text-left">
              <input
                type="checkbox"
                className="mt-1 h-5 w-5 accent-orange-600"
                checked={acknowledged}
                onChange={(event) => setAcknowledged(event.target.checked)}
              />
              <span className="text-sm leading-6 text-gray-800 md:text-base">
                I acknowledge that this model is for local safety testing only,
                may generate explicit and toxic content, and I will comply with
                applicable laws, platform policies, and community guidelines.
              </span>
            </label>
          </div>

          <div className="flex justify-center">
            <Button
              onClick={onStart}
              disabled={!acknowledged}
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
