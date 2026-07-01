import re
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


def extract_username(url: str, domain: str) -> str:
    parsed = urlparse(url if url.startswith("http") else f"https://{url}")
    path = parsed.path.strip("/").split("/")
    return path[0] if domain in parsed.netloc and path else ""


def score_text_against_context(text: str, resume: str, job_description: str) -> int:
    context_terms = set(re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", f"{resume} {job_description}".lower()))
    profile_terms = set(re.findall(r"[a-zA-Z][a-zA-Z+#.-]{2,}", text.lower()))
    if not profile_terms:
        return 0
    overlap = len(context_terms & profile_terms)
    return max(20, min(100, 35 + overlap * 3))

@tool
def analyze_github_profile(github_url: str, resume: str = "", job_description: str = "") -> dict:
    """Fetch and evaluate a public GitHub profile against the resume and job description."""
    username = extract_username(github_url, "github.com")
    if not username:
        return {"score": 0, "summary": "GitHub URL could not be parsed.", "evidence": []}

    try:
        user = requests.get(f"https://api.github.com/users/{username}", timeout=8).json()
        repos = requests.get(
            f"https://api.github.com/users/{username}/repos",
            params={"sort": "updated", "per_page": 12},
            timeout=8,
        ).json()
    except requests.RequestException as exc:
        return {"score": 35, "summary": f"GitHub link exists but could not be fetched: {exc}", "evidence": []}

    repo_text = " ".join(
        f"{repo.get('name', '')} {repo.get('description', '') or ''} {repo.get('language', '') or ''}"
        for repo in repos if isinstance(repo, dict)
    )
    score = score_text_against_context(repo_text, resume, job_description)
    public_repos = user.get("public_repos", 0) if isinstance(user, dict) else 0
    score = min(100, score + min(18, int(public_repos or 0)))
    evidence = [
        f"Public repos: {public_repos}",
        f"Recent repos reviewed: {min(len(repos), 12) if isinstance(repos, list) else 0}",
    ]
    return {
        "score": score,
        "summary": "GitHub strengthens the profile." if score >= 60 else "GitHub exists but needs stronger aligned projects.",
        "evidence": evidence,
    }


@tool
def analyze_linkedin_profile(linkedin_url: str, resume: str = "", job_description: str = "") -> dict:
    """Evaluate a public LinkedIn URL with lightweight public-page fetching when available."""
    if "linkedin.com" not in linkedin_url.lower():
        return {"score": 0, "summary": "LinkedIn URL not found.", "evidence": []}
    try:
        response = requests.get(linkedin_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = soup.get_text(" ", strip=True)[:5000]
    except requests.RequestException:
        page_text = ""

    resume_terms = score_text_against_context(resume, resume, job_description)
    public_score = score_text_against_context(page_text, resume, job_description) if page_text else 0
    score = max(45, min(100, round((resume_terms * 0.55) + (public_score * 0.45)))) if page_text else 55
    return {
        "score": score,
        "summary": "LinkedIn link is present and aligned." if score >= 60 else "LinkedIn link is present but public evidence is limited.",
        "evidence": ["Public LinkedIn pages may restrict automated review.", f"Fetched public text: {bool(page_text)}"],
    }
