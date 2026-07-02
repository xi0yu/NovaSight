import type { ReactNode } from "react";

import { type LicenseFeature, type LicenseStatus } from "../../api";
import { EmptyState } from "../ui";

type PermissionGuardProps = {
  license: LicenseStatus | null;
  feature: LicenseFeature;
  children: ReactNode;
};

export function PermissionGuard({ license, feature, children }: PermissionGuardProps) {
  if (!license?.features.includes(feature)) {
    return (
      <EmptyState
        title="当前授权不包含此能力"
        detail={`需要 ${feature} feature 才能使用该模块。`}
      />
    );
  }

  return <>{children}</>;
}
