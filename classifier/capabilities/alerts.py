"""
Alert detection capability.

Detects serious workplace concerns that require immediate attention.
"""

from typing import Any, Dict, Type

from pydantic import BaseModel

from ..models import AlertsOutput
from ..prompts import alert_detection_prompt
from .base import Capability


class AlertsCapability(Capability):
    """
    Detects alerts in text.

    Identifies serious concerns including harassment, discrimination, safety issues,
    and other workplace problems requiring HR or management attention.
    """

    @property
    def name(self) -> str:
        return "alerts"

    @property
    def schema(self) -> Type[BaseModel]:
        return AlertsOutput

    def create_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        return alert_detection_prompt(text)

    def format_for_export(self, result: Any) -> Any:
        """Extract just the alerts list."""
        if result is None:
            return []

        formatted = super().format_for_export(result)

        if isinstance(formatted, dict):
            return formatted.get("alerts", [])

        return []
