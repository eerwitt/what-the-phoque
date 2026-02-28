export type AppState = "welcome" | "loading" | "captioning";

export interface WebcamPermissionError {
  type: "general" | "https" | "not-supported" | "permission";
  message: string;
  details: string;
}

export interface Position {
  x: number;
  y: number;
}

export interface Dimensions {
  width: number;
  height: number;
}

export type InitialPosition = "bottom-left" | "bottom-right" | Position;
