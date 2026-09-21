import { render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import type { ModelCatalogModel } from "../../api";
import { ModelCatalogTree } from "./ModelCatalogTree";

it("distinguishes paginated model shelves for screen-reader users", () => {
  const models: ModelCatalogModel[] = Array.from({ length: 101 }, (_, index) => ({
    type: "model", name: `engine-${index}.engine`, relative_path: `engine-${index}.engine`,
    kind: "engine", size_bytes: 100, scan_status: "ready", scan_reason: "",
    recommendation: "recommended", tags: [],
  }));
  render(<ModelCatalogTree models={models} selectedPath={undefined} activeArtifactId={null} onSelectModel={vi.fn()} />);
  expect(screen.getByRole("button", { name: "推荐模型再显示 1 个" })).toBeInTheDocument();
});
