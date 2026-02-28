import { createContext } from "react";
import type { VLMContextValue } from "../types/vlm";

const VLMContext = createContext<VLMContextValue | null>(null);

export { VLMContext };
