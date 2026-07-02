import type { ButtonHTMLAttributes } from "react";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "default" | "compact";
  loading?: boolean;
};

export function Button({
  variant = "secondary",
  size = "default",
  loading = false,
  className,
  disabled,
  type = "button",
  children,
  ...props
}: ButtonProps) {
  const resolvedClassName = ["ui-button", `ui-button-${variant}`, `ui-button-${size}`, className]
    .filter(Boolean)
    .join(" ");

  return (
    <button
      {...props}
      className={resolvedClassName}
      disabled={disabled || loading}
      type={type}
    >
      {children}
    </button>
  );
}
