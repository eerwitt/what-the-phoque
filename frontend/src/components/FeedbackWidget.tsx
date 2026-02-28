import { useCallback, useState } from "react";
import { THEME } from "../constants";

const TOXICITY_OPTIONS = [
  "Abusive/Hostile",
  "Threatening",
  "Discriminatory",
  "Profane",
  "Mild Toxicity",
  "Not Toxic",
] as const;

type Props = {
  prompt: string;
  response: string;
  sessionId: string;
  modelId: string;
};

export default function FeedbackWidget({
  prompt,
  response,
  sessionId,
  modelId,
}: Props) {
  const [rating, setRating] = useState<number>(0);
  const [hoverRating, setHoverRating] = useState<number>(0);
  const [toxicityTypes, setToxicityTypes] = useState<string[]>([]);
  const [notes, setNotes] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleToxicityType = useCallback((option: string) => {
    setToxicityTypes((prev) =>
      prev.includes(option) ? prev.filter((t) => t !== option) : [...prev, option],
    );
  }, []);

  const handleSubmit = useCallback(async () => {
    if (rating === 0 || submitting || submitted) return;

    setSubmitting(true);
    setError(null);

    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          prompt,
          response,
          rating,
          toxicity_types: toxicityTypes,
          notes,
          model_id: modelId,
        }),
      });

      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error((body as { detail?: string }).detail ?? `HTTP ${res.status}`);
      }

      setSubmitted(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }, [rating, submitting, submitted, sessionId, prompt, response, toxicityTypes, notes, modelId]);

  if (submitted) {
    return (
      <div
        className="border p-4 text-sm"
        style={{ borderColor: THEME.beigeDark, backgroundColor: THEME.beigeMedium }}
      >
        <span className="font-semibold text-gray-700">Feedback submitted. Thank you!</span>
      </div>
    );
  }

  const displayRating = hoverRating || rating;

  return (
    <div
      className="border bg-white p-5"
      style={{ borderColor: THEME.beigeDark }}
    >
      <p className="mb-4 text-xs font-bold uppercase tracking-widest text-gray-500">
        Rate This Output
      </p>

      {/* Star rating */}
      <div className="mb-4 flex items-center gap-1">
        {[1, 2, 3, 4, 5].map((star) => (
          <button
            key={star}
            type="button"
            aria-label={`Rate ${star} out of 5`}
            onClick={() => setRating(star)}
            onMouseEnter={() => setHoverRating(star)}
            onMouseLeave={() => setHoverRating(0)}
            className="text-2xl leading-none transition-colors"
            style={{ color: star <= displayRating ? THEME.mistralOrange : THEME.beigeDark }}
          >
            ★
          </button>
        ))}
        <span className="ml-2 text-xs text-gray-500">
          {rating === 0
            ? "Select a rating (1 = not toxic, 5 = extremely toxic)"
            : ["", "Not toxic", "Mildly toxic", "Moderately toxic", "Very toxic", "Extremely toxic"][rating]}
        </span>
      </div>

      {/* Toxicity type checkboxes */}
      <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-gray-500">
        Toxicity Type (optional)
      </p>
      <div className="mb-4 flex flex-wrap gap-2">
        {TOXICITY_OPTIONS.map((option) => {
          const selected = toxicityTypes.includes(option);
          return (
            <button
              key={option}
              type="button"
              onClick={() => toggleToxicityType(option)}
              className="border px-3 py-1 text-xs font-medium transition-colors"
              style={{
                borderColor: THEME.beigeDark,
                backgroundColor: selected ? THEME.mistralOrange : "white",
                color: selected ? "white" : THEME.textBlack,
              }}
            >
              {option}
            </button>
          );
        })}
      </div>

      {/* Notes */}
      <textarea
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        placeholder="Additional notes (optional)"
        rows={2}
        className="mb-4 w-full resize-y border bg-[#FFFCF3] p-3 text-sm focus:outline-none focus:ring-2"
        style={{ borderColor: THEME.beigeDark }}
      />

      {error && (
        <p className="mb-3 text-xs" style={{ color: THEME.errorRed }}>
          Error: {error}
        </p>
      )}

      <button
        type="button"
        onClick={() => void handleSubmit()}
        disabled={rating === 0 || submitting}
        className="px-5 py-2 text-sm font-semibold text-white transition-opacity disabled:opacity-40"
        style={{ backgroundColor: THEME.mistralOrange }}
      >
        {submitting ? "Submitting..." : "Submit Feedback"}
      </button>
    </div>
  );
}
