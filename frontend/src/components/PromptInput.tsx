import { useState, useRef, useEffect } from "react";
import { PROMPTS, THEME } from "../constants";

interface PromptInputProps {
  onPromptChange: (prompt: string) => void;
  defaultPrompt?: string;
}

export default function PromptInput({
  onPromptChange,
  defaultPrompt = PROMPTS.default,
}: PromptInputProps) {
  const [prompt, setPrompt] = useState(defaultPrompt);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const resizeTextarea = () => {
    if (inputRef.current) {
      inputRef.current.style.height = "auto";
      const newHeight = Math.min(inputRef.current.scrollHeight, 200);
      inputRef.current.style.height = `${newHeight}px`;
    }
  };

  useEffect(() => {
    onPromptChange(prompt);
    resizeTextarea();
  }, [prompt, onPromptChange]);

  const handleInputFocus = () => setShowSuggestions(true);
  const handleInputClick = () => setShowSuggestions(true);

  const handleInputBlur = (e: React.FocusEvent) => {
    // Small delay to allow click events on suggestions to fire
    requestAnimationFrame(() => {
      if (
        !e.relatedTarget ||
        !containerRef.current?.contains(e.relatedTarget as Node)
      ) {
        setShowSuggestions(false);
      }
    });
  };

  const handleSuggestionClick = (suggestion: string) => {
    setPrompt(suggestion);
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  const clearInput = () => {
    setPrompt("");
    inputRef.current?.focus();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setPrompt(e.target.value);
  };

  return (
    <div
      ref={containerRef}
      className="w-full max-w-xl relative group font-sans"
    >
      {/* Suggestions Panel */}
      <div
        className={`absolute bottom-full left-0 right-0 mb-3 transition-all duration-300 ease-out transform origin-bottom ${
          showSuggestions
            ? "opacity-100 translate-y-0 scale-100 pointer-events-auto"
            : "opacity-0 translate-y-2 scale-95 pointer-events-none"
        }`}
      >
        <div
          className="bg-white rounded-lg shadow-xl border overflow-hidden"
          style={{ borderColor: THEME.beigeDark }}
        >
          {/* Header */}
          <div
            className="border-b px-4 py-2 flex items-center space-x-2"
            style={{
              backgroundColor: THEME.beigeLight,
              borderColor: THEME.beigeDark,
            }}
          >
            <svg
              className="w-3 h-3"
              style={{ color: THEME.mistralOrange }}
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
              />
            </svg>
            <span className="text-xs font-bold uppercase tracking-wider text-gray-500">
              Prompt Library
            </span>
          </div>

          {/* List */}
          <ul className="py-2">
            {PROMPTS.suggestions.map((suggestion, index) => (
              <li
                key={index}
                tabIndex={0}
                onMouseDown={(e) => e.preventDefault()} // Prevent blur
                onClick={() => handleSuggestionClick(suggestion)}
                className="px-4 py-2.5 cursor-pointer flex items-start gap-3 transition-colors hover:bg-[var(--mistral-beige-light)] group/item"
              >
                <span
                  className="mt-1 opacity-0 group-hover/item:opacity-100 transition-opacity text-xs font-mono"
                  style={{ color: THEME.mistralOrange }}
                >
                  {`>`}
                </span>
                <span className="text-sm text-gray-700 group-hover/item:text-black leading-snug">
                  {suggestion}
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Input Container */}
      <div className="relative">
        {/* Label Badge */}
        <div className="absolute -top-3 left-4 z-10">
          <span
            className="border text-[10px] font-bold text-gray-500 uppercase tracking-widest px-2 py-0.5 rounded-sm"
            style={{
              backgroundColor: THEME.beigeLight,
              borderColor: THEME.beigeDark,
            }}
          >
            Prompt
          </span>
        </div>

        <div
          className={`
            relative bg-white rounded-lg shadow-lg border transition-all duration-300
            ${showSuggestions ? "border-[var(--mistral-orange)] ring-1 ring-[var(--mistral-orange)]/20" : "border-[var(--mistral-beige-dark)] hover:border-[#D0C5A0]"}
        `}
        >
          <div className="flex items-start p-1">
            <textarea
              ref={inputRef}
              value={prompt}
              onChange={handleInputChange}
              onFocus={handleInputFocus}
              onBlur={handleInputBlur}
              onClick={handleInputClick}
              className="w-full py-4 pl-5 pr-10 bg-transparent text-lg md:text-xl font-mono resize-none focus:outline-none placeholder:text-gray-300 leading-relaxed"
              style={{
                minHeight: "60px",
                maxHeight: "200px",
                overflowY: "auto",
                color: THEME.textBlack,
              }}
              placeholder={PROMPTS.placeholder}
              rows={1}
            />

            {/* Clear Button */}
            {prompt && (
              <button
                type="button"
                onMouseDown={(e) => e.preventDefault()}
                onClick={clearInput}
                className="absolute right-3 top-5 text-gray-300 hover:text-[var(--mistral-orange)] transition-colors p-1"
                aria-label="Clear prompt"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-6 w-6"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
