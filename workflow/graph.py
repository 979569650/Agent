from langgraph.graph import END, StateGraph

from workflow.state import AgentState


def should_continue(_: AgentState):
    return END


def build_app(agent_node):
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {END: END})
    return workflow.compile()
