"""Internal offline ticket classification package."""

from .inference import TicketClassifier, classify_ticket

__all__ = ["TicketClassifier", "classify_ticket"]
