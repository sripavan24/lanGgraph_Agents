import requests
from urllib.parse import quote_plus, urlparse, parse_qs
from bs4 import BeautifulSoup
from langchain_core.tools import tool

@tool
def web_job_search(role: str, location: str = "India") -> str:
    """Real web search for jobs"""
    query = quote_plus(f"{role} jobs {location}")
    fallback_links = [
        f"LinkedIn Jobs: https://www.linkedin.com/jobs/search/?keywords={quote_plus(role)}&location={quote_plus(location)}",
        f"Naukri: https://www.naukri.com/{quote_plus(role).replace('+', '-')}-jobs",
        f"Indeed: https://in.indeed.com/jobs?q={quote_plus(role)}&l={quote_plus(location)}",
        f"Wellfound: https://wellfound.com/jobs",
        f"Remote OK: https://remoteok.com/remote-{quote_plus(role).replace('+', '-')}-jobs",
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
            real_url = parse_qs(parsed.query).get("q", [""])[0]
            if real_url.startswith("http") and "google" not in real_url:
                results.append(f"{title}: {real_url}")

            if len(results) == 5:
                break
        
        if results:
            return f"Found job links for {role} in {location}:\n" + "\n".join(results)
        return f"I could not fetch live Google results, but you can search here:\n" + "\n".join(fallback_links)
    except:
        return f"I could not fetch live job results, but you can search here:\n" + "\n".join(fallback_links)
