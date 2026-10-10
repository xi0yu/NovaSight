import { useEffect, useRef, useState, type CSSProperties, type ImgHTMLAttributes } from "react";
import "./loaded-image.css";

type LoadedImageProps = Omit<ImgHTMLAttributes<HTMLImageElement>, "src" | "srcSet" | "sizes" | "onLoad" | "onError"> & {
  src: string;
  alt: string;
};

export function LoadedImage(props: LoadedImageProps) {
  // A changed source gets a fresh loading state, including while the old decode is pending.
  return <ImageFrame key={props.src} {...props} />;
}

function ImageFrame({ className, width, height, style, ...props }: LoadedImageProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [state, setState] = useState<"loading" | "ready" | "failed">("loading");

  useEffect(() => {
    const image = imageRef.current!;
    let disposed = false;
    const fail = () => { if (!disposed) setState("failed"); };
    const reveal = async () => {
      try {
        await image.decode();
        if (!disposed) setState("ready");
      } catch { fail(); }
    };
    image.addEventListener("load", reveal);
    image.addEventListener("error", fail);
    if (image.complete) {
      if (image.naturalWidth > 0) void reveal();
      else fail();
    }
    return () => {
      disposed = true;
      image.removeEventListener("load", reveal);
      image.removeEventListener("error", fail);
    };
  }, []);

  return <span className={`loaded-image ${className ?? ""}`} data-image-state={state}
    aria-hidden={props["aria-hidden"]} aria-busy={state === "loading"}
    role={props.alt && state !== "ready" ? "img" : undefined}
    aria-label={props.alt && state !== "ready" ? `${props.alt}，${state === "failed" ? "图片加载失败" : "加载中"}` : undefined}
    title={state === "failed" ? "图片加载失败" : undefined}
    style={{ "--image-width": width ? `${width}px` : undefined,
      "--image-ratio": width && height ? `${width} / ${height}` : undefined, ...style } as CSSProperties}>
    <img {...props} ref={imageRef} width={width} height={height} decoding="async" />
    {state !== "ready" ? <span className="image-placeholder" aria-hidden="true">
      <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="8" y="9" width="32" height="30" rx="6" />
        <circle cx="18" cy="19" r="3" />
        <path d="m9 33 10-9 7 6 5-5 8 8" />
        {state === "failed" ? <path d="m7 7 34 34" /> : null}
      </svg>
    </span> : null}
  </span>;
}
