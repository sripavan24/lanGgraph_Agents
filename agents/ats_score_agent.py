from langchain_core.messages import HumanMessage
from utils.helpers import extract_name_from_text, extract_email, extract_phone, get_cache_dir
import json
import re

def ats_score_agent(state):
    resume = state.get("raw_text", "")
    jd = state.get("job_description", "")
    query = state["messages"][-1].content if state.get("messages") else ""

    prompt = f"""Analyze whether this resume is suitable for the user's target role or JD.

User Question / Target:
{query}

Job Description:
{jd}

Resume:
{resume}

Return ONLY valid JSON with these keys:
ats_score: number 0-100,
suitable: true or false,
summary: short direct answer,
suitable_roles: array of matching job roles,
strong_skills: array,
missing_skills: array,
reasons: array,
improvements: array."""

    response = state["llm"].invoke([HumanMessage(content=prompt)])
    try:
        json_match = re.search(r'\{[\s\S]*\}', response.content)
        score_data = json.loads(json_match.group(0)) if json_match else {"ats_score": 70}
    except (json.JSONDecodeError, AttributeError):
        score_data = {"ats_score": 65}

    return {"ats_score": score_data}
