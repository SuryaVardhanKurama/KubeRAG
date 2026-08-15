from app.agents.state import AgentState
from app.gateway import get_langchain_llm
import logfire

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. Respond with 'CONVERSATIONAL' ONLY if the latest message is a greeting, farewell, or acknowledgment (hi, thanks, ok, bye) OR a question that can be answered using ONLY the conversation history above (e.g., "what is my name", "what did I just ask").
    2. If the latest message is a technical question about Kubernetes, Intel, networking, or Databricks that requires documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or a search query.
    """

    with logfire.span("🧠 Planner Decision"):
        decision = llm.invoke(prompt).content.strip()

        # Log the intent label consistently: 'CONVERSATIONAL' or 'TECHNICAL'.
        # The LLM returns the refined search query (not the word TECHNICAL)
        # for technical queries, so attach it as a structured field.
        if decision == "CONVERSATIONAL":
            intent = "CONVERSATIONAL"
            logfire.info(f"Intent identified: {intent}")
        else:
            intent = "TECHNICAL"
            logfire.info(f"Intent identified: {intent}", search_query=decision)

    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }

    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
