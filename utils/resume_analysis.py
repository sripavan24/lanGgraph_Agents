import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


SECTION_PATTERNS = {
    "experience": r"\b(experience|employment|work history|professional experience)\b",
    "projects": r"\b(projects?|portfolio)\b",
    "education": r"\b(education|academic|university|college|degree)\b",
    "certifications": r"\b(certifications?|licenses?|badges?)\b",
    "skills": r"\b(skills?|technologies|technical skills|tools)\b",
}

SKILL_TERMS = {
    "python", "java", "javascript", "typescript", "react", "node", "express", "sql",
    "mongodb", "postgresql", "mysql", "django", "flask", "fastapi", "html", "css",
    "tailwind", "aws", "azure", "gcp", "docker", "kubernetes", "git", "github",
    "linux", "machine learning", "deep learning", "nlp", "llm", "langchain",
    "tensorflow", "pytorch", "pandas", "numpy", "power bi", "tableau", "excel",
    "rest api", "graphql", "ci/cd", "streamlit", "fastapi", "data analysis",
}

ACTION_VERBS = {
    "built", "created", "designed", "developed", "implemented", "improved",
    "optimized", "launched", "deployed", "automated", "reduced", "increased",
    "led", "managed", "analyzed", "integrated", "trained", "delivered",
}


@dataclass
class CategoryResult:
    label: str
    passed: bool
    score: int
    detail: str


def clamp(value: float, low: int = 0, high: int = 100) -> int:
    return max(low, min(high, round(value)))


def extract_urls(text: str) -> dict[str, list[str]]:
    urls = re.findall(r"(https?://[^\s),>\]]+|(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/[^\s),>\]]+)", text)
    cleaned = [url.rstrip(".;") for url in urls]
    return {
        "github": [url for url in cleaned if "github.com" in url.lower()],
        "linkedin": [url for url in cleaned if "linkedin.com" in url.lower()],
        "all": cleaned,
    }


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z+#.-]{1,}", text.lower())


def phrase_hits(text: str, terms: set[str]) -> set[str]:
    lower = text.lower()
    return {term for term in terms if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lower)}


def extract_jd_keywords(job_description: str, limit: int = 28) -> list[str]:
    tokens = [
        token for token in words(job_description)
        if len(token) > 2 and token not in {"and", "the", "for", "with", "you", "are", "will", "our", "this", "that"}
    ]
    counts = Counter(tokens)
    skills = sorted(phrase_hits(job_description, SKILL_TERMS))
    ranked = skills + [word for word, _ in counts.most_common(limit) if word not in skills]
    return ranked[:limit]


def keyword_match_score(resume: str, job_description: str) -> tuple[int, list[str], list[str]]:
    jd_keywords = extract_jd_keywords(job_description)
    if not jd_keywords:
        resume_skills = sorted(phrase_hits(resume, SKILL_TERMS))
        return (min(100, 45 + len(resume_skills) * 7), resume_skills[:10], [])
    lower_resume = resume.lower()
    matched = [term for term in jd_keywords if re.search(rf"(?<!\w){re.escape(term.lower())}(?!\w)", lower_resume)]
    missing = [term for term in jd_keywords if term not in matched]
    return clamp((len(matched) / max(len(jd_keywords), 1)) * 100), matched[:12], missing[:12]


def section_score(text: str, section: str) -> int:
    pattern = SECTION_PATTERNS[section]
    base = 55 if re.search(pattern, text, re.I) else 15
    if section == "experience":
        years = re.findall(r"\b(\d+)\+?\s*(?:years?|yrs?)\b", text, re.I)
        date_ranges = re.findall(r"\b(?:20\d{2}|19\d{2})\s*[-–]\s*(?:present|current|20\d{2}|19\d{2})\b", text, re.I)
        base += min(35, len(years) * 12 + len(date_ranges) * 8)
    if section == "projects":
        base += min(35, len(re.findall(r"\b(project|built|github|deployed|app|model|system)\b", text, re.I)) * 4)
    if section == "education":
        base += 25 if re.search(r"\b(bachelor|master|b\.?tech|m\.?tech|degree|university|college)\b", text, re.I) else 0
    if section == "certifications":
        base += min(35, len(re.findall(r"\b(certified|certification|certificate|aws|azure|google|coursera|udemy)\b", text, re.I)) * 8)
    if section == "skills":
        base += min(40, len(phrase_hits(text, SKILL_TERMS)) * 5)
    return clamp(base)


def grammar_score(text: str) -> tuple[int, list[str]]:
    sample = re.sub(r"\s+", " ", text)[:5000]
    issues = []
    doubled = re.findall(r"\b(\w+)\s+\1\b", sample, re.I)
    spacing = re.findall(r"\s[,.;:]", sample)
    lower_i = re.findall(r"(?<![A-Za-z])i(?![A-Za-z])", sample)
    long_sentences = [s for s in re.split(r"[.!?]", sample) if len(s.split()) > 38]
    if doubled:
        issues.append("Repeated words detected")
    if spacing:
        issues.append("Punctuation spacing issues")
    if lower_i:
        issues.append("Lowercase standalone I")
    if long_sentences:
        issues.append("Several sentences are hard to scan")
    penalty = len(doubled) * 8 + len(spacing) * 3 + len(lower_i) * 5 + len(long_sentences) * 4
    return clamp(96 - penalty), issues[:4]


def readability_score(text: str) -> int:
    sentences = max(1, len(re.findall(r"[.!?]", text)))
    token_count = max(1, len(words(text)))
    avg_sentence = token_count / sentences
    bullet_count = len(re.findall(r"(?m)^\s*[-*•]", text))
    score = 82
    if avg_sentence > 28:
        score -= min(30, (avg_sentence - 28) * 1.4)
    if bullet_count >= 6:
        score += 10
    return clamp(score)


def ats_parse_score(text: str) -> tuple[int, list[str]]:
    sections = [name for name, pattern in SECTION_PATTERNS.items() if re.search(pattern, text, re.I)]
    email = bool(re.search(r"[\w.-]+@[\w.-]+\.\w+", text))
    phone = bool(re.search(r"(?<!\w)\+?[\d][\d\s\-()]{8,18}\d(?!\w)", text))
    url_count = len(extract_urls(text)["all"])
    weird_chars = len(re.findall(r"[^\x00-\x7F]", text))
    score = 25 + len(sections) * 10 + (15 if email else 0) + (10 if phone else 0) + min(10, url_count * 3)
    if weird_chars > 80:
        score -= 10
    details = [f"{len(sections)}/5 core sections found"]
    if not email:
        details.append("Email not detected")
    if not phone:
        details.append("Phone not detected")
    return clamp(score), details


def achievement_score(text: str) -> int:
    quantified = len(re.findall(r"\b\d+[%x+]?|\b(?:increased|reduced|improved|saved|scaled|automated)\b", text, re.I))
    verbs = len([word for word in words(text) if word in ACTION_VERBS])
    return clamp(25 + min(45, quantified * 8) + min(30, verbs * 4))


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def build_resume_analysis(
    resume: str,
    job_description: str,
    profile_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile_analysis = profile_analysis or {}
    parse_score, parse_details = ats_parse_score(resume)
    skills_score, strong_skills, missing_skills = keyword_match_score(resume, job_description)
    grammar, grammar_issues = grammar_score(resume)
    readability = readability_score(resume)
    experience = section_score(resume, "experience")
    projects = section_score(resume, "projects")
    education = section_score(resume, "education")
    certifications = section_score(resume, "certifications")
    skills_section = section_score(resume, "skills")
    achievements = achievement_score(resume)
    urls = extract_urls(resume)

    github_score = int(profile_analysis.get("github", {}).get("score", 0)) if urls["github"] else 0
    linkedin_score = int(profile_analysis.get("linkedin", {}).get("score", 0)) if urls["linkedin"] else 0

    categories = {
        "ATS Parse Rate": CategoryResult("ATS Parse Rate", parse_score >= 70, parse_score, "; ".join(parse_details)),
        "Skills Match": CategoryResult("Skills Match", skills_score >= 60, skills_score, f"{len(strong_skills)} matched, {len(missing_skills)} gaps"),
        "Grammar & Spelling": CategoryResult("Grammar & Spelling", grammar >= 78, grammar, ", ".join(grammar_issues) or "Clean basic grammar scan"),
        "Experience": CategoryResult("Experience", experience >= 62, experience, "Experience section and dates/years evidence"),
        "Projects": CategoryResult("Projects", projects >= 62, projects, "Project section and implementation evidence"),
        "GitHub Profile": CategoryResult("GitHub Profile", github_score >= 60, github_score, profile_analysis.get("github", {}).get("summary", "GitHub link not found")),
        "LinkedIn Profile": CategoryResult("LinkedIn Profile", linkedin_score >= 60, linkedin_score, profile_analysis.get("linkedin", {}).get("summary", "LinkedIn link not found")),
    }

    weighted = {
        "parse": parse_score * 0.16,
        "skills": skills_score * 0.22,
        "grammar": grammar * 0.10,
        "readability": readability * 0.08,
        "experience": experience * 0.13,
        "projects": projects * 0.09,
        "education": education * 0.06,
        "certifications": certifications * 0.04,
        "skills_section": skills_section * 0.06,
        "achievements": achievements * 0.06,
    }
    score = clamp(sum(weighted.values()))

    improvements = []
    if missing_skills:
        improvements.append(f"Add relevant JD terms naturally: {', '.join(missing_skills[:6])}.")
    if achievements < 70:
        improvements.append("Rewrite bullets with numbers, scale, outcomes, and action verbs.")
    if parse_score < 75:
        improvements.append("Use standard section headings and keep contact links easy to parse.")
    if grammar < 80:
        improvements.append("Tighten long sentences and fix grammar or punctuation issues.")
    if not urls["github"]:
        improvements.append("Add a GitHub link if projects are technical and publicly presentable.")
    if not urls["linkedin"]:
        improvements.append("Add a current LinkedIn profile URL.")

    return {
        "ats_score": score,
        "suitable": score >= 75,
        "summary": "Strong ATS alignment" if score >= 80 else "Moderate ATS alignment" if score >= 65 else "Needs ATS work before applying",
        "categories": {key: result.__dict__ for key, result in categories.items()},
        "strong_skills": strong_skills,
        "missing_skills": missing_skills,
        "section_scores": {
            "readability": readability,
            "experience": experience,
            "projects": projects,
            "education": education,
            "certifications": certifications,
            "skills": skills_section,
            "quantified_achievements": achievements,
        },
        "profile_links": urls,
        "profile_analysis": profile_analysis,
        "reasons": [
            f"Keyword/JD match score: {skills_score}.",
            f"ATS parse score: {parse_score}.",
            f"Readability score: {readability}.",
            f"Quantified achievement score: {achievements}.",
        ],
        "improvements": improvements[:8],
    }


def safe_json(content: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        match = re.search(r"\{[\s\S]*\}", content)
        return json.loads(match.group(0) if match else content)
    except (json.JSONDecodeError, AttributeError, TypeError):
        return fallback
