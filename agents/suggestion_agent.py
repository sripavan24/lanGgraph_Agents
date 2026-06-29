from langchain_core.messages import HumanMessage
from utils.helpers import extract_name_from_text, extract_email, extract_phone, get_cache_dir
def suggestion_agent(state):
    prompt = f"""Give detailed section-wise improvement suggestions for this resume.

Resume:
{state.get('raw_text', '')[:6000]}"""

    response = state["llm"].invoke([HumanMessage(content=prompt)])
    return {"suggestions": {"details": response.content}}