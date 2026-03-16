import operator
from typing import Annotated, List, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """LangGraph state for the agent workflow."""

    messages: Annotated[List[BaseMessage], operator.add]
