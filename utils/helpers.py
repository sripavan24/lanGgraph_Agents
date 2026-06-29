from pathlib import Path
import re

def get_cache_dir():
    """Return cache directory and create if not exists"""
    cache_dir = Path("resume_cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir

def clean_text(text: str) -> str:
    """Basic text cleaning"""
    text = text.replace('\n\n', '\n')
    return text.strip()

def format_candidate_name(name: str) -> str:
    """Format all-caps extracted names without changing normal mixed-case names."""
    if not name or name != name.upper():
        return name
    return " ".join(word if len(word) == 1 else word.title() for word in name.split())

def extract_name_from_text(text: str) -> str:
    """Extract the candidate name from the top/header area of the resume."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    for line in lines[:30]:
        match = re.match(r"(?i)^detected\s+name\s*:\s*(.+)$", line)
        if match:
            detected = match.group(1).strip()
            if detected and detected.lower() != "not detected":
                return format_candidate_name(detected)

    ignored_lines = {
        "candidate resume",
        "resume",
        "curriculum vitae",
        "cv",
    }
    contact_pattern = re.compile(r"(https?://|linkedin|github)", re.I)
    section_pattern = re.compile(
        r"(?i)^(summary|profile|objective|education|experience|skills|projects|certifications|contact)\b"
    )

    for line in lines[:20]:
        cleaned = re.split(r"\s{2,}|[\w\.-]+@[\w\.-]+\.\w+", line, maxsplit=1)[0]
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" :-|")
        words = cleaned.split()
        if (
            cleaned
            and cleaned.lower() not in ignored_lines
            and 2 <= len(words) <= 5
            and not contact_pattern.search(cleaned)
            and not section_pattern.search(cleaned)
            and all(re.fullmatch(r"[A-Za-z][A-Za-z.'-]*", word) for word in words)
        ):
            return format_candidate_name(cleaned)

    return "Not detected"

def extract_jd_designation(text: str) -> str:
    """Extract a job title/designation from a pasted job description."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]

    label_pattern = re.compile(
        r"(?i)\b(?:job\s*title|role|position|designation|job\s*role)\s*[:\-]\s*(.+)"
    )
    for line in lines[:40]:
        match = label_pattern.search(line)
        if match:
            title = match.group(1).strip(" .,-|")
            if title:
                return title

    title_pattern = re.compile(
        r"(?i)\b("
        r"(?:ai|ml|machine learning|data|software|frontend|front-end|backend|back-end|full stack|python|java|web|cloud|devops)"
        r"\s+"
        r"(?:engineer|developer|analyst|scientist|intern|architect)"
        r")\b"
    )
    for line in lines[:40]:
        match = title_pattern.search(line)
        if match:
            return match.group(1).strip()

    return "Not found"

def extract_email(text: str):
    """Extract email"""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else "Not found"

def extract_phone(text: str):
    """Extract phone"""
    for match in re.finditer(r'(?<!\w)\+?[\d][\d\s\-\(\)]{8,18}\d(?!\w)', text):
        phone = re.sub(r"\s+", " ", match.group(0)).strip()
        if len(re.sub(r"\D", "", phone)) >= 10:
            return phone
    return "Not found"
