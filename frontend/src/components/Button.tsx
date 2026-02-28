import { type ReactNode } from "react";
import { THEME } from "../constants";

interface ButtonProps {
  children: ReactNode;
  onClick: (e: React.MouseEvent) => void;
  className?: string;
  disabled?: boolean;
  "aria-label"?: string;
}

export default function Button({
  children,
  onClick,
  className = "",
  disabled = false,
  ...ariaProps
}: ButtonProps) {
  return (
    <button
      className={`px-4 py-2 rounded-xl border-none cursor-pointer outline-none ${
        disabled ? "opacity-50 cursor-not-allowed" : ""
      } ${className}`}
      style={{ backgroundColor: THEME.mistralOrange }}
      onClick={disabled ? undefined : onClick}
      disabled={disabled}
      {...ariaProps}
    >
      <div className="font-medium text-white">{children}</div>
    </button>
  );
}
