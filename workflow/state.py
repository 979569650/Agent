import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """LangGraph state for the agent workflow."""

    messages: Annotated[List[BaseMessage], operator.add]
    access_code: str | None
    ticket_id: str | None
    abort_ticket: bool
    ticket_required: bool
    security_status: str

