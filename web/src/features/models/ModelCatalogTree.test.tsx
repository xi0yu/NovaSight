import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";

import type { ModelCatalogDirectory, ModelCatalogModel } from "../../api";
import { findCatalogDirectory, ModelCatalogTree, sortCatalogModels } from "./ModelCatalogTree";

function model(name: string, size_bytes = 100): ModelCatalogModel {
  return {
    type: "model", name, relative_path: name, kind: "engine", size_bytes,
    scan_status: "ready", scan_reason: "", recommendation: "unrated", tags: [],
  };
}

it("navigates folders and progressively reveals a 73-file catalog", () => {
  const folders: ModelCatalogDirectory[] = [{
    type: "directory", name: "Arena", relative_path: "Arena", children: [model("engine-2.engine")],
  }];
  const models = Array.from({ length: 73 }, (_, index) => model(`engine-${index}.engine`));
  const onOpenFolder = vi.fn();
  render(<ModelCatalogTree
    folders={folders} models={models} sortOrder="name_asc" selectedPath={undefined}
    activeArtifactId={null} onOpenFolder={onOpenFolder} onSelectModel={vi.fn()}
  />);

  fireEvent.click(screen.getByRole("button", { name: "打开文件夹 Arena，包含 1 个模型" }));
  expect(onOpenFolder).toHaveBeenCalledWith("Arena");
  expect(screen.getAllByRole("button", { name: /engine-\d+\.engine/ })).toHaveLength(24);
  fireEvent.click(screen.getByRole("button", { name: "再显示 24 个模型" }));
  expect(screen.getAllByRole("button", { name: /engine-\d+\.engine/ })).toHaveLength(48);
  expect(screen.getByRole("button", { name: "再显示 24 个模型" })).toHaveTextContent("已显示 48/73");
});

it("resolves nested directories and sorts names naturally without losing size sorting", () => {
  const root: ModelCatalogDirectory = {
    type: "directory", name: "models", relative_path: "", children: [{
      type: "directory", name: "Arena", relative_path: "Arena", children: [{
        type: "directory", name: "v2", relative_path: "Arena/v2", children: [model("engine-2.engine")],
      }],
    }],
  };
  expect(findCatalogDirectory(root, "Arena/v2")?.relative_path).toBe("Arena/v2");
  expect(findCatalogDirectory(root, "Arena/missing")).toBeNull();

  const models = [model("engine-10.engine", 10), model("engine-2.engine", 20)];
  expect(sortCatalogModels(models, "name_asc", null).map((item) => item.name))
    .toEqual(["engine-2.engine", "engine-10.engine"]);
  expect(sortCatalogModels(models, "size_asc", null).map((item) => item.name))
    .toEqual(["engine-10.engine", "engine-2.engine"]);
  expect(sortCatalogModels(models, "name_desc", null).map((item) => item.name))
    .toEqual(["engine-10.engine", "engine-2.engine"]);
});
