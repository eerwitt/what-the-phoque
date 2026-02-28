import React, { useState, useRef, useCallback } from "react";
import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  type ProgressInfo,
  type Tensor,
} from "@huggingface/transformers";
import { VLMContext } from "./VLMContext";

const MODEL_ID = "eerwitt/what-the-phoque-onnx";
const MAX_NEW_TOKENS = 512;
const SESSION_ID = crypto.randomUUID();
const SYSTEM_PROMPT =
  "You are What the Phoque?, a helpful assistant focused on concise, clear responses.";

export const VLMProvider: React.FC<React.PropsWithChildren> = ({
  children,
}) => {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tokenizerRef = useRef<Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>> | null>(
    null,
  );
  const modelRef = useRef<Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>> | null>(
    null,
  );
  const loadPromiseRef = useRef<Promise<void> | null>(null);
  const inferenceLock = useRef(false);

  const loadModel = useCallback(
    async (onProgress?: (msg: string, percentage: number) => void) => {
      if (isLoaded) {
        onProgress?.("Model already loaded!", 100);
        return;
      }

      if (loadPromiseRef.current) {
        return loadPromiseRef.current;
      }

      setIsLoading(true);
      setError(null);

      loadPromiseRef.current = (async () => {
        try {
          onProgress?.("Loading tokenizer...", 0);
          tokenizerRef.current = await AutoTokenizer.from_pretrained(MODEL_ID);
          onProgress?.("Tokenizer loaded. Loading model...", 5);

          const progressMap = new Map<string, number>();
          const progressCallback = (info: ProgressInfo) => {
            if (
              info.status === "progress" &&
              typeof info.loaded === "number" &&
              typeof info.total === "number" &&
              info.total > 0
            ) {
              const key = info.file || `file-${progressMap.size + 1}`;
              progressMap.set(key, info.loaded / info.total);
              const avgProgress =
                Array.from(progressMap.values()).reduce(
                  (a, b) => a + b,
                  0,
                ) / progressMap.size;
              const percentage = 5 + avgProgress * 90;
              onProgress?.("Downloading model...", percentage);
            }
          };

          const device = navigator.gpu ? "webgpu" : "wasm";
          onProgress?.(`Initializing ${device.toUpperCase()} backend...`, 95);

          modelRef.current = await AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            {
              device,
              progress_callback: progressCallback,
            },
          );

          onProgress?.("Model loaded successfully!", 100);
          setIsLoaded(true);
        } catch (e) {
          const errorMessage = e instanceof Error ? e.message : String(e);
          setError(errorMessage);
          console.error("Error loading model:", e);
          throw e;
        } finally {
          setIsLoading(false);
          loadPromiseRef.current = null;
        }
      })();

      return loadPromiseRef.current;
    },
    [isLoaded],
  );

  const runInference = useCallback(
    async (
      instruction: string,
      onTextUpdate?: (text: string) => void,
      onStatsUpdate?: (stats: { tps?: number; ttft?: number }) => void,
    ): Promise<string> => {
      if (inferenceLock.current) {
        throw new Error("A generation is already in progress.");
      }
      inferenceLock.current = true;

      try {
        if (!tokenizerRef.current || !modelRef.current) {
          throw new Error("Model/tokenizer not loaded");
        }

        const tokenizer = tokenizerRef.current;
        const model = modelRef.current;
        const messages = [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: instruction },
        ];

        const prompt =
          typeof tokenizer.apply_chat_template === "function"
            ? tokenizer.apply_chat_template(messages, {
                tokenize: false,
                add_generation_prompt: true,
              })
            : `System: ${SYSTEM_PROMPT}\nUser: ${instruction}\nAssistant:`;

        const inputs = await tokenizer(prompt, { add_special_tokens: true });

        let streamed = "";
        const start = performance.now();
        let decodeStart: number | undefined;
        let generatedTokenCount = 0;
        const streamer = new TextStreamer(tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (textChunk: string) => {
            if (streamed.length === 0) {
              const latency = performance.now() - start;
              onStatsUpdate?.({ ttft: latency });
            }
            streamed += textChunk;
            onTextUpdate?.(streamed.trim());
          },
          token_callback_function: () => {
            decodeStart ??= performance.now();
            generatedTokenCount++;
            const elapsed = (performance.now() - decodeStart) / 1000;
            if (elapsed > 0) {
              onStatsUpdate?.({ tps: generatedTokenCount / elapsed });
            }
          },
        });

        const outputs = (await model.generate({
          ...inputs,
          max_new_tokens: MAX_NEW_TOKENS,
          do_sample: false,
          repetition_penalty: 1.1,
          streamer,
        })) as Tensor;

        const promptLength = inputs.input_ids?.dims?.at(-1) ?? 0;
        const generated = promptLength
          ? outputs.slice(null, [promptLength, null])
          : outputs;
        const decodeEnd = performance.now();
        if (decodeStart) {
          const outputTokenCount = generated.dims[1] ?? generatedTokenCount;
          const tps = outputTokenCount / ((decodeEnd - decodeStart) / 1000);
          onStatsUpdate?.({ tps });
        }

        const decoded = tokenizer.batch_decode(generated, {
          skip_special_tokens: true,
        });
        const response = decoded[0]?.trim() || streamed.trim();
        onTextUpdate?.(response);
        return response;
      } finally {
        inferenceLock.current = false;
      }
    },
    [],
  );

  return (
    <VLMContext.Provider
      value={{
        isLoaded,
        isLoading,
        error,
        modelId: MODEL_ID,
        sessionId: SESSION_ID,
        loadModel,
        runInference,
      }}
    >
      {children}
    </VLMContext.Provider>
  );
};

export default VLMProvider;
