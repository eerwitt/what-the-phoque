export const THEME = {
  beigeLight: "#FFFAEB",
  beigeMedium: "#FFF0C3",
  beigeDark: "#E9E2CB",
  mistralOrange: "#FF8205",
  mistralOrangeDark: "#FA500F",
  mistralOrangeLight: "#FFAF00",
  mistralYellow: "#FFD800",
  textBlack: "#1E1E1E",
  black: "#000000",
  white: "#FFFFFF",
  errorRed: "#E10500",
} as const;

export const LAYOUT = {
  MARGINS: {
    DEFAULT: 20,
    BOTTOM: 20,
  },
  DIMENSIONS: {
    PROMPT_WIDTH: 420,
    CAPTION_WIDTH: 150,
    CAPTION_HEIGHT: 45,
  },
  TRANSITIONS: {
    SCALE_DURATION: 200,
    OPACITY_DURATION: 200,
    TRANSFORM_DURATION: 400,
  },
} as const;

export const TIMING = {
  FRAME_CAPTURE_DELAY: 50,
  VIDEO_RECOVERY_INTERVAL: 1000,
  RESIZE_DEBOUNCE: 50,
  SUGGESTION_DELAY: 50,
} as const;

const DEFAULT_PROMPT = "Explain why harbor seals are called the 'dogs of the sea' in 3 short points.";
export const PROMPTS = {
  default: DEFAULT_PROMPT,
  placeholder: "Type your prompt here...",

  suggestions: [
    DEFAULT_PROMPT,
    "Rewrite this sentence to sound friendlier: 'Your request is invalid.'",
    "Summarize the main idea of renewable energy in one paragraph.",
    "Give me 5 pun-style names for a seal-themed coffee shop.",
    "Draft a concise commit message for adding text generation UI.",
  ],

  fallbackCaption: "Waiting for first generation...",
  processingMessage: "Generating response...",
} as const;
