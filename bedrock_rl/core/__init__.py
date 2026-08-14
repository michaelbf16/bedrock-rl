"""Small, trainer-independent contracts for agentic task trials.

The core intentionally knows nothing about Minecraft, verl, torch, vLLM,
or Hugging Face datasets. Concrete environments, adapters, and product
workflows live in sibling packages and are selected by import path. Keeping
this package dependency-free is a design constraint: ``import bedrock_rl.core``
must remain useful in a task-authoring environment with none of the training
stack installed.
"""

from bedrock_rl.core.components import (ComponentSpec, load_object,
                                           resolve_callable)
from bedrock_rl.core.context import (Compose, ContextRequest, ContextView,
                                       Identity, KeepLastImages,
                                       KeepLastTurns, SummarizeOldTurns,
                                       keep_last_images)
from bedrock_rl.core.guidance import (GuidancePolicy, GuidanceRequest,
                                        NoGuidance, StaticGuidance)
from bedrock_rl.core.harness import ToolHarness
from bedrock_rl.core.messages import (data_image_part, image_url_part,
                                        validate_messages)
from bedrock_rl.core.reward import (CompositeVerifier, FunctionVerifier,
                                      RewardResult)
from bedrock_rl.core.run import RunSpec
from bedrock_rl.core.state import ProbeSet, StateEvent, StateTrace
from bedrock_rl.core.task import TaskSpec
from bedrock_rl.core.tools import (FunctionTool, ToolCall, ToolContext,
                                     ToolRegistry, ToolResult)
from bedrock_rl.core.trajectory import Trajectory, TrajectoryRecorder

__all__ = [
    "ComponentSpec", "Compose", "CompositeVerifier", "ContextRequest",
    "ContextView", "FunctionTool", "FunctionVerifier", "GuidancePolicy",
    "GuidanceRequest", "Identity", "KeepLastImages", "KeepLastTurns",
    "NoGuidance", "ProbeSet", "RewardResult", "RunSpec", "StateEvent",
    "StateTrace",
    "StaticGuidance", "SummarizeOldTurns", "TaskSpec", "ToolCall",
    "ToolContext", "ToolHarness", "ToolRegistry", "ToolResult",
    "Trajectory", "TrajectoryRecorder", "data_image_part",
    "image_url_part", "keep_last_images", "load_object", "resolve_callable",
    "validate_messages",
]
