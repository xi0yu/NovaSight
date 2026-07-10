import type { ButtonHTMLAttributes } from "react";

import { NovaIcon, type NovaIconName } from "../visual";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "default" | "compact";
  loading?: boolean;
  leadingIcon?: NovaIconName;
  trailingIcon?: NovaIconName;
};

export function Button({
  variant = "secondary",
  size = "default",
  loading = false,
  leadingIcon,
  trailingIcon,
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
      aria-busy={loading ? true : props["aria-busy"]}
      className={resolvedClassName}
      disabled={disabled || loading}
      type={type}
    >
      {leadingIcon ? <NovaIcon name={leadingIcon} size={16} /> : null}
      {children}
      {trailingIcon ? <NovaIcon name={trailingIcon} size={16} /> : null}
    </button>
  );
}
