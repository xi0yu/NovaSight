import type { AimRole, AimRoleRatios } from "./AimTargetRange";

export function serializeClassAimRatios(
  roles: Record<string, AimRole>,
  ratios: AimRoleRatios
): string {
  return Object.entries(roles)
    .flatMap(([classId, role]) => {
      const numericClassId = Number(classId);
      return Number.isInteger(numericClassId) && numericClassId >= 0 && numericClassId <= 255
        ? [[numericClassId, ratios[role]] as const]
        : [];
    })
    .sort(([left], [right]) => left - right)
    .map(([classId, ratio]) => `${classId}:${ratio.toFixed(2)}`)
    .join(",");
}

export function activateClassPolicy(
  profileNames: string[],
  roleProfiles: Record<string, Record<string, AimRole>>,
  priorityProfiles: Record<string, string>,
  filterProfiles: Record<string, string>,
  activeProfile: string,
  fallbackPriority: string,
  fallbackFilter: string,
  ratios: AimRoleRatios
) {
  const priorities = Object.fromEntries(profileNames.map((profileName) => [
    profileName,
    priorityProfiles[profileName] ?? fallbackPriority
  ]));
  const filters = Object.fromEntries(profileNames.map((profileName) => [
    profileName,
    filterProfiles[profileName] ?? fallbackFilter
  ]));
  return {
    priorities,
    filters,
    priority: priorities[activeProfile] ?? fallbackPriority,
    filter: filters[activeProfile] ?? fallbackFilter,
    aimYRatios: serializeClassAimRatios(roleProfiles[activeProfile] ?? {}, ratios)
  };
}
