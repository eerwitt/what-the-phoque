export type VLMContextValue = {
  isLoaded: boolean;
  isLoading: boolean;
  error: string | null;
  modelId: string;
  loadModel: (
    onProgress?: (msg: string, percentage: number) => void,
  ) => Promise<void>;
  runInference: (
    instruction: string,
    onTextUpdate?: (text: string) => void,
    onStatsUpdate?: (stats: { tps?: number; ttft?: number }) => void,
  ) => Promise<string>;
};
