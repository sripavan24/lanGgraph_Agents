from langchain_core.messages import HumanMessage
def suggestion_agent(state):
    if not state.get("llm"):
        return {
            "suggestions": {
                "details": "I need a GROQ_API_KEY or GROK_API_KEY value in your .env file to generate detailed resume improvements."
            }
        }

    prompt = f"""You are a resume suggestions agent.
Give prioritized, section-wise improvements. Use the job description when present.
Be specific, ATS-aware, and practical.

Resume:
{state.get('raw_text', '')[:7000]}

Job description:
{state.get('job_description', '')[:3500]}

Known ATS analysis:
{state.get('ats_score', {})}"""

    response = state["llm"].invoke([HumanMessage(content=prompt)])
    return {"suggestions": {"details": response.content}}
