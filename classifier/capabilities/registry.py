"""
Capability registry and orchestration system.

Manages capability registration, dependency resolution, and execution.
"""

from typing import Dict, List, Optional, Set

from rich.console import Console

from .base import Capability

console = Console()


class CapabilityRegistry:
    """
    Registry for managing and orchestrating capabilities.

    Handles capability registration, dependency resolution, and execution ordering.
    """

    def __init__(self):
        self._capabilities: Dict[str, Capability] = {}
        self._execution_order: Optional[List[str]] = None

    def register(self, capability: Capability) -> None:
        """
        Register a new capability.

        Args:
            capability: Capability instance to register
        """
        name = capability.name
        if name in self._capabilities:
            console.print(
                f"[yellow]Warning: Capability '{name}' already registered, replacing...[/yellow]"
            )

        self._capabilities[name] = capability
        # Invalidate cached execution order
        self._execution_order = None

    def get(self, name: str) -> Optional[Capability]:
        """
        Get a capability by name.

        Args:
            name: Capability name

        Returns:
            Capability instance or None if not found
        """
        return self._capabilities.get(name)

    def list_capabilities(self) -> List[str]:
        """Get list of registered capability names."""
        return list(self._capabilities.keys())

    def resolve_dependencies(self, capability_names: List[str]) -> List[str]:
        """
        Resolve dependencies and return execution order.

        Uses topological sort to ensure dependencies run before dependents.

        Args:
            capability_names: List of capability names to execute

        Returns:
            Ordered list of capability names

        Raises:
            ValueError: If circular dependency detected or unknown capability
        """
        # Build dependency graph
        graph: Dict[str, Set[str]] = {}
        in_degree: Dict[str, int] = {}

        # Start with requested capabilities
        to_process = set(capability_names)
        processed = set()

        while to_process:
            name = to_process.pop()
            if name in processed:
                continue

            if name not in self._capabilities:
                raise ValueError(f"Unknown capability: {name}")

            capability = self._capabilities[name]
            dependencies = set(capability.dependencies)

            graph[name] = dependencies
            in_degree[name] = len(dependencies)

            # Add dependencies to process queue
            to_process.update(dependencies - processed)
            processed.add(name)

        # Topological sort using Kahn's algorithm
        queue = [name for name in graph if in_degree[name] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            # Reduce in-degree for dependents
            for dependent, deps in graph.items():
                if current in deps:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        # Check for circular dependencies
        if len(result) != len(graph):
            raise ValueError("Circular dependency detected in capabilities")

        return result

    def get_execution_order(self, capability_names: List[str]) -> List[str]:
        """
        Get execution order for specified capabilities.

        Caches result for efficiency.

        Args:
            capability_names: List of capability names to execute

        Returns:
            Ordered list of capability names
        """
        return self.resolve_dependencies(capability_names)

    def validate_capabilities(self, capability_names: List[str]) -> List[str]:
        """
        Validate that all specified capabilities exist.

        Args:
            capability_names: List of capability names

        Returns:
            List of error messages (empty if all valid)
        """
        errors = []

        for name in capability_names:
            if name not in self._capabilities:
                errors.append(f"Unknown capability: {name}")
            else:
                # Check dependencies exist
                capability = self._capabilities[name]
                for dep in capability.dependencies:
                    if dep not in self._capabilities:
                        errors.append(
                            f"Capability '{name}' depends on unknown capability '{dep}'"
                        )

        return errors

    def get_hierarchy_requirements(self, capability_names: List[str]) -> bool:
        """
        Check if any requested capability requires hierarchy access.

        Args:
            capability_names: List of capability names

        Returns:
            True if hierarchy is needed, False otherwise
        """
        execution_order = self.get_execution_order(capability_names)

        for name in execution_order:
            capability = self._capabilities[name]
            if capability.requires_hierarchy():
                return True

        return False


def create_default_registry() -> CapabilityRegistry:
    """
    Create a registry with all default capabilities registered.

    Returns:
        CapabilityRegistry with standard capabilities
    """
    from .alerts import AlertsCapability
    from .classification import ClassificationCapability
    from .recommendations import RecommendationsCapability
    from .trend import TrendCapability

    registry = CapabilityRegistry()

    # Register standard capabilities
    registry.register(ClassificationCapability())
    registry.register(RecommendationsCapability())
    registry.register(AlertsCapability())
    registry.register(TrendCapability())

    # Note: Stem capabilities need parameters, so they're registered on-demand
    # in the orchestrator based on CLI flags

    return registry
