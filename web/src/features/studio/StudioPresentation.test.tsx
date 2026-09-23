import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SectionTitle } from "./StudioPresentation";

describe("SectionTitle", () => {
  it("keeps the Chinese section name concise for heading navigation", () => {
    render(<SectionTitle title="选择画面来源" />);

    expect(screen.getByRole("heading", { name: "选择画面来源" })).toBeInTheDocument();
    expect(screen.getByText("选择摄像头和它实际支持的画面规格")).toBeInTheDocument();
  });
});
