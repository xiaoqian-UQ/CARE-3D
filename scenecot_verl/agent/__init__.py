"""SceneCOT verl agent loops."""

__all__ = ["SceneCOTFreeTagAgentLoop"]


def __getattr__(name: str):
    if name == "SceneCOTFreeTagAgentLoop":
        from .scenecot_free_tag_agent_loop import SceneCOTFreeTagAgentLoop

        return SceneCOTFreeTagAgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
