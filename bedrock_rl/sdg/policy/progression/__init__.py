"""Declarative survival progression and its live scripted policy."""

from bedrock_rl.sdg.policy.progression.policy import (
    SurvivalProgressionPolicy,
)
from bedrock_rl.sdg.policy.progression.spec import (
    BeginLightingOperation,
    BucketCastMineOperation,
    CraftOperation,
    FluidSourceRequirement,
    HotbarPolicy,
    NetherPortalOperation,
    PortalSettings,
    RepeatDropOperation,
    RequireItemOperation,
    ResourceStage,
    SmeltOperation,
    SurvivalProgressionSpec,
    WorkstationOperation,
    WorkstationSite,
    _portal_protected_cells as _portal_protected_cells,
    _ReturnBreadcrumb as _ReturnBreadcrumb,
    _workstation_sites as _workstation_sites,
)

__all__ = (
    "BeginLightingOperation",
    "BucketCastMineOperation",
    "CraftOperation",
    "FluidSourceRequirement",
    "HotbarPolicy",
    "NetherPortalOperation",
    "PortalSettings",
    "RepeatDropOperation",
    "RequireItemOperation",
    "ResourceStage",
    "SmeltOperation",
    "SurvivalProgressionPolicy",
    "SurvivalProgressionSpec",
    "WorkstationOperation",
    "WorkstationSite",
)
