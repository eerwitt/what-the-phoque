import React, { useState, useRef, useCallback } from "react";
import {
  AutoModelForCausalLM,
  AutoModelForImageTextToText,
  AutoProcessor,
  AutoTokenizer,
  TextStreamer,
  type ProgressInfo,
  type Tensor,
} from "@huggingface/transformers";
import { VLMContext } from "./VLMContext";
import type { ModelState } from "../types/vlm";

export const PRIMARY_MODEL_ID = "eerwitt/what-the-phoque-onnx";
// Loaded as AutoModelForImageTextToText (multimodal pipeline, text-only inference).
// Must be the official mistralai ONNX repo — it has the per-module dtype ONNX layout
// that AutoModelForImageTextToText expects (embed_tokens / vision_encoder / decoder_model_merged).
export const BASE_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512-ONNX";

// Per-module dtypes for the base VLM model (matches the working Ministral WebGPU demo).
const BASE_MODEL_DTYPE = {
  embed_tokens: "fp16",
  vision_encoder: "q4",
  decoder_model_merged: "q4f16",
} as const;

const MAX_NEW_TOKENS = 512;
const SESSION_ID = crypto.randomUUID();
const SYSTEM_PROMPT =
  "You are What the Phoque?, a helpful assistant focused on concise, clear responses.";

// ─── Text-generation path (AutoTokenizer + AutoModelForCausalLM) ──────────────
// Used for the fine-tuned eerwitt model which was exported with task=text-generation.

function useTextModelInstance(modelId: string): ModelState {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const tokenizerRef = useRef<Awaited<
    ReturnType<typeof AutoTokenizer.from_pretrained>
  > | null>(null);
  const modelRef = useRef<Awaited<
    ReturnType<typeof AutoModelForCausalLM.from_pretrained>
  > | null>(null);
  const loadPromiseRef = useRef<Promise<void> | null>(null);
  const inferenceLock = useRef(false);

  const loadModel = useCallback(
    async (onProgress?: (msg: string, percentage: number) => void) => {
      if (isLoaded) {
        onProgress?.("Model already loaded!", 100);
        return;
      }
      if (loadPromiseRef.current) return loadPromiseRef.current;

      setIsLoading(true);
      setError(null);

      loadPromiseRef.current = (async () => {
        try {
          onProgress?.("Loading tokenizer...", 0);
          tokenizerRef.current = await AutoTokenizer.from_pretrained(modelId);
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
              const avg =
                Array.from(progressMap.values()).reduce((a, b) => a + b, 0) /
                progressMap.size;
              onProgress?.("Downloading model...", 5 + avg * 90);
            }
          };

          const device = navigator.gpu ? "webgpu" : "wasm";
          onProgress?.(`Initializing ${device.toUpperCase()} backend...`, 95);

          modelRef.current = await AutoModelForCausalLM.from_pretrained(
            modelId,
            { device, progress_callback: progressCallback },
          );

          onProgress?.("Model loaded successfully!", 100);
          setIsLoaded(true);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          setError(msg);
          console.error(`Error loading model ${modelId}:`, e);
          throw e;
        } finally {
          setIsLoading(false);
          loadPromiseRef.current = null;
        }
      })();

      return loadPromiseRef.current;
    },
    [isLoaded, modelId],
  );

  const runInference = useCallback(
    async (
      instruction: string,
      onTextUpdate?: (text: string) => void,
      onStatsUpdate?: (stats: { tps?: number; ttft?: number }) => void,
    ): Promise<string> => {
      if (inferenceLock.current) throw new Error("A generation is already in progress.");
      inferenceLock.current = true;

      try {
        if (!tokenizerRef.current || !modelRef.current)
          throw new Error("Model/tokenizer not loaded");

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
          callback_function: (chunk: string) => {
            if (streamed.length === 0) onStatsUpdate?.({ ttft: performance.now() - start });
            streamed += chunk;
            onTextUpdate?.(streamed.trim());
          },
          token_callback_function: () => {
            decodeStart ??= performance.now();
            generatedTokenCount++;
            const elapsed = (performance.now() - decodeStart) / 1000;
            if (elapsed > 0) onStatsUpdate?.({ tps: generatedTokenCount / elapsed });
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
        const generated = promptLength ? outputs.slice(null, [promptLength, null]) : outputs;
        if (decodeStart) {
          const tps = (generated.dims[1] ?? generatedTokenCount) / ((performance.now() - decodeStart) / 1000);
          onStatsUpdate?.({ tps });
        }

        const decoded = tokenizer.batch_decode(generated, { skip_special_tokens: true });
        const response = decoded[0]?.trim() || streamed.trim();
        onTextUpdate?.(response);
        return response;
      } finally {
        inferenceLock.current = false;
      }
    },
    [],
  );

  return { modelId, isLoaded, isLoading, error, loadModel, runInference };
}

// ─── VLM path (AutoProcessor + AutoModelForImageTextToText) ───────────────────
// Used for the base mistralai/Ministral-3-3B-Instruct-2512-ONNX which has the
// multimodal ONNX layout. We run text-only inference (no image) by tokenizing
// through processor.tokenizer directly after applying the chat template.

function useVLMInstance(
  modelId: string,
  dtype: Record<string, string>,
): ModelState {
  const [isLoaded, setIsLoaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const processorRef = useRef<any>(null);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const modelRef = useRef<any>(null);
  const loadPromiseRef = useRef<Promise<void> | null>(null);
  const inferenceLock = useRef(false);

  const loadModel = useCallback(
    async (onProgress?: (msg: string, percentage: number) => void) => {
      if (isLoaded) {
        onProgress?.("Model already loaded!", 100);
        return;
      }
      if (loadPromiseRef.current) return loadPromiseRef.current;

      if (!navigator.gpu) {
        const msg = "WebGPU is not available. The base model requires WebGPU.";
        setError(msg);
        throw new Error(msg);
      }

      setIsLoading(true);
      setError(null);

      loadPromiseRef.current = (async () => {
        try {
          onProgress?.("Loading processor...", 0);
          processorRef.current = await AutoProcessor.from_pretrained(modelId);
          onProgress?.("Processor loaded. Loading model...", 5);

          // Track only .onnx_data files — 3 data shards for Ministral
          const dataProgress = new Map<string, number>();
          const progressCallback = (info: ProgressInfo) => {
            if (
              info.status === "progress" &&
              typeof info.file === "string" &&
              info.file.endsWith(".onnx_data") &&
              typeof info.loaded === "number" &&
              typeof info.total === "number" &&
              info.total > 0
            ) {
              dataProgress.set(info.file, info.loaded / info.total);
              const total = Array.from(dataProgress.values()).reduce((a, b) => a + b, 0);
              // 3 shard files expected (embed_tokens, vision_encoder, decoder_model_merged)
              onProgress?.("Downloading model...", 5 + (total / 3) * 90);
            }
          };

          onProgress?.("Initializing WebGPU backend...", 95);
          modelRef.current = await AutoModelForImageTextToText.from_pretrained(
            modelId,
            { dtype, device: "webgpu", progress_callback: progressCallback },
          );

          onProgress?.("Model loaded successfully!", 100);
          setIsLoaded(true);
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          setError(msg);
          console.error(`Error loading VLM model ${modelId}:`, e);
          throw e;
        } finally {
          setIsLoading(false);
          loadPromiseRef.current = null;
        }
      })();

      return loadPromiseRef.current;
    },
    [isLoaded, modelId, dtype],
  );

  const runInference = useCallback(
    async (
      instruction: string,
      onTextUpdate?: (text: string) => void,
      onStatsUpdate?: (stats: { tps?: number; ttft?: number }) => void,
    ): Promise<string> => {
      if (inferenceLock.current) throw new Error("A generation is already in progress.");
      inferenceLock.current = true;

      try {
        if (!processorRef.current || !modelRef.current)
          throw new Error("Model/processor not loaded");

        const processor = processorRef.current;
        const model = modelRef.current;

        // Text-only: apply chat template then tokenize without going through image processing
        const messages = [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: instruction },
        ];
        const prompt = processor.apply_chat_template(messages, {
          tokenize: false,
          add_generation_prompt: true,
        }) as string;

        // Use processor.tokenizer directly to skip image pre-processing
        const tokenizer = processor.tokenizer;
        const inputs = await tokenizer(prompt, { add_special_tokens: false });

        let streamed = "";
        const start = performance.now();
        let decodeStart: number | undefined;
        let generatedTokenCount = 0;

        const streamer = new TextStreamer(tokenizer, {
          skip_prompt: true,
          skip_special_tokens: true,
          callback_function: (chunk: string) => {
            if (streamed.length === 0) onStatsUpdate?.({ ttft: performance.now() - start });
            streamed += chunk;
            onTextUpdate?.(streamed.trim());
          },
          token_callback_function: () => {
            decodeStart ??= performance.now();
            generatedTokenCount++;
            const elapsed = (performance.now() - decodeStart) / 1000;
            if (elapsed > 0) onStatsUpdate?.({ tps: generatedTokenCount / elapsed });
          },
        });

        const outputs = (await model.generate({
          ...inputs,
          max_new_tokens: MAX_NEW_TOKENS,
          do_sample: false,
          repetition_penalty: 1.2,
          streamer,
        })) as Tensor;

        const promptLength = inputs.input_ids?.dims?.at(-1) ?? 0;
        const generated = promptLength ? outputs.slice(null, [promptLength, null]) : outputs;
        if (decodeStart) {
          const tps = (generated.dims[1] ?? generatedTokenCount) / ((performance.now() - decodeStart) / 1000);
          onStatsUpdate?.({ tps });
        }

        const decoded = tokenizer.batch_decode(generated, { skip_special_tokens: true });
        const response = decoded[0]?.trim() || streamed.trim();
        onTextUpdate?.(response);
        return response;
      } finally {
        inferenceLock.current = false;
      }
    },
    [],
  );

  return { modelId, isLoaded, isLoading, error, loadModel, runInference };
}

// ─── Provider ─────────────────────────────────────────────────────────────────

export const VLMProvider: React.FC<React.PropsWithChildren> = ({
  children,
}) => {
  // Fine-tuned model: text-generation ONNX export → CausalLM + Tokenizer
  const primaryModel = useTextModelInstance(PRIMARY_MODEL_ID);

  // Base model: multimodal ONNX export → ImageTextToText + Processor (text-only inference)
  const baseModel = useVLMInstance(BASE_MODEL_ID, BASE_MODEL_DTYPE);

  return (
    <VLMContext.Provider
      value={{ primaryModel, baseModel, sessionId: SESSION_ID }}
    >
      {children}
    </VLMContext.Provider>
  );
};

export default VLMProvider;
