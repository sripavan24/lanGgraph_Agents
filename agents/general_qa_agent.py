from langchain_core.messages import HumanMessage
from utils.helpers import extract_name_from_text, extract_email, extract_phone, extract_jd_designation, get_cache_dir

def general_qa_agent(state):
    query = state["messages"][-1].content
    query_lower = query.lower()
    raw_text = state.get("raw_text", "")
    jd = state.get("job_description", "")

    if "jd" in query_lower and any(word in query_lower for word in ["designation", "title", "role", "position", "name"]):
        return {"general_answer": {"answer": f"JD Designation: {extract_jd_designation(jd)}"}}

    if "email" in query_lower or "mail" in query_lower:
        return {"general_answer": {"answer": f"Email: {extract_email(raw_text)}"}}

    if "contact" in query_lower:
        return {
            "general_answer": {
                "answer": (
                    "Contact Information:\n"
                    f"Email: {extract_email(raw_text)}\n"
                    f"Phone: {extract_phone(raw_text)}"
                )
            }
        }

    if "phone" in query_lower or "mobile" in query_lower:
        return {"general_answer": {"answer": f"Phone: {extract_phone(raw_text)}"}}

    if "name" in query_lower:
        return {"general_answer": {"answer": f"Name: {extract_name_from_text(raw_text)}"}}

    prompt = f"""Answer the question based ONLY on the resume and job description.

Resume:
{raw_text[:8000]}

Job Description:
{jd[:4000]}

Question: {query}"""

    response = state["llm"].invoke([HumanMessage(content=prompt)])
    return {"general_answer": {"answer": response.content}}
