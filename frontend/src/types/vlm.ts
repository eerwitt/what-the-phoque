export type ModelState = {
  modelId: string;
  isLoaded: boolean;
  isLoading: boolean;
  error: string | null;
  loadModel: (onProgress?: (msg: string, percentage: number) => void) => Promise<void>;
  runInference: (
    instruction: string,
    onTextUpdate?: (text: string) => void,
    onStatsUpdate?: (stats: { tps?: number; ttft?: number }) => void,
  ) => Promise<string>;
};

export type VLMContextValue = {
  primaryModel: ModelState;
  baseModel: ModelState;
  sessionId: string;
};
