import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SectionTitle } from "./StudioPresentation";

describe("SectionTitle", () => {
  it("keeps the Chinese section name concise for heading navigation", () => {
    render(<SectionTitle title="采集设备" />);

    expect(screen.getByRole("heading", { name: "采集设备" })).toBeInTheDocument();
    expect(screen.getByText("视频源、画面通路与最新帧策略")).toBeInTheDocument();
  });
});
