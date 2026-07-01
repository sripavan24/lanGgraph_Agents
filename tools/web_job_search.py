import requests
from urllib.parse import quote_plus, urlparse, parse_qs
from bs4 import BeautifulSoup
from langchain_core.tools import tool

JOB_DOMAINS = [
    "linkedin.com/jobs",
    "indeed.com",
    "naukri.com",
    "foundit.in",
    "wellfound.com",
    "glassdoor",
    "internshala.com",
    "greenhouse.io",
    "lever.co",
    "workdayjobs.com",
]

@tool
def web_job_search(role: str, location: str = "India", resume: str = "", job_description: str = "") -> str:
    """Live web search for relevant jobs with application links."""
    query = quote_plus(
        f"{role} jobs {location} apply LinkedIn Indeed Naukri Foundit Wellfound Glassdoor Internshala careers"
    )
    platform_links = [
        f"[LinkedIn Jobs](https://www.linkedin.com/jobs/search/?keywords={quote_plus(role)}&location={quote_plus(location)})",
        f"[Naukri](https://www.naukri.com/{quote_plus(role).replace('+', '-')}-jobs)",
        f"[Indeed](https://in.indeed.com/jobs?q={quote_plus(role)}&l={quote_plus(location)})",
        f"[Foundit](https://www.foundit.in/srp/results?query={quote_plus(role)}&locations={quote_plus(location)})",
        f"[Wellfound](https://wellfound.com/jobs)",
        f"[Glassdoor](https://www.glassdoor.co.in/Job/jobs.htm?sc.keyword={quote_plus(role)})",
        f"[Internshala](https://internshala.com/jobs/keywords-{quote_plus(role).replace('+', '-')})",
    ]

    try:
        url = f"https://www.google.com/search?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for link in soup.select("a"):
            title = link.get_text(" ", strip=True)
            href = link.get("href", "")
            if not title or not href or "/url?" not in href:
                continue

            parsed = urlparse(href)
#             {
    # query='q=https://www.linkedin.com/jobs/view/12345&sa=U&ved=...',
#     "q": ["https://www.linkedin.com/jobs/view/12345"],
#     "sa": ["U"],
#     "ved": ["2ah..."]
# }
            real_url = parse_qs(parsed.query).get("q", [""])[0]
            if real_url.startswith("http") and "google" not in real_url and any(domain in real_url.lower() for domain in JOB_DOMAINS):
                results.append(f"- [{title}]({real_url})")

            if len(results) == 8:
                break
        
        if results:
            return f"Live job matches for **{role}** in **{location}**:\n" + "\n".join(results)
        return "Live search did not return parseable job cards. Search directly here:\n" + "\n".join(f"- {link}" for link in platform_links)
    except Exception as exc:
        return f"Live job search was blocked or unavailable ({exc}). Search directly here:\n" + "\n".join(f"- {link}" for link in platform_links)


@tool
def web_resume_builder_search(query: str = "ATS resume builder", location: str = "India") -> str:
    """Live web search for resume builders and official pricing/features pages."""
    search = quote_plus(f"{query} free freemium paid ATS resume builder official pricing")
    official_domains = [
        "canva.com", "novoresume.com", "resume.io", "zety.com", "enhancv.com",
        "tealhq.com", "flowcv.com", "kickresume.com", "resumeworded.com",
    ]
    try:
        response = requests.get(
            f"https://www.google.com/search?q={search}",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for link in soup.select("a"):
            title = link.get_text(" ", strip=True)
            href = link.get("href", "")
            if not title or not href or "/url?" not in href:
                continue
            real_url = parse_qs(urlparse(href).query).get("q", [""])[0]
            if real_url.startswith("http") and any(domain in real_url.lower() for domain in official_domains):
                results.append(f"- [{title}]({real_url})")
            if len(results) == 8:
                break
        if results:
            return "Live resume builder results:\n" + "\n".join(results)
    except Exception as exc:
        return f"Live builder search was blocked or unavailable ({exc})."
    return "\n".join([
        "- [Canva Resume Builder](https://www.canva.com/resumes/) - free/freemium templates; choose simple ATS layouts.",
        "- [Novoresume](https://novoresume.com/) - freemium guided resume builder with ATS-friendly templates.",
        "- [Resume.io](https://resume.io/) - freemium builder with paid downloads and cover letters.",
        "- [Zety](https://zety.com/resume-builder) - paid guided builder with content suggestions.",
        "- [Enhancv](https://enhancv.com/) - paid/freemium modern resumes; use conservative templates for ATS.",
        "- [Teal](https://www.tealhq.com/resume-builder) - free/freemium resume tailoring and job tracking.",
        "- [FlowCV](https://flowcv.com/) - free/freemium clean resume and portfolio builder.",
    ])
