"""Base classes for AI trading research agents."""

from dataclasses import dataclass


@dataclass
class BaseAgent:
    """Minimal base class for orchestrating research tasks."""

    name: str

    def run(self, prompt: str) -> str:
        """Process a prompt and return a response placeholder."""
        return f"[{self.name}] TODO: implement agent logic for prompt: {prompt}"
