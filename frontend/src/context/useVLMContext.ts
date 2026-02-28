import { useContext } from "react";
import { VLMContext } from "./VLMContext.ts";
import type { VLMContextValue } from "../types/vlm";

export function useVLMContext(): VLMContextValue {
  const ctx = useContext(VLMContext);
  if (!ctx) throw new Error("useVLMContext must be inside VLMProvider");
  return ctx;
}
