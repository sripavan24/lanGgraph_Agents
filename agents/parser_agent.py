from pathlib import Path
import json
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from liteparse import LiteParse
from utils.helpers import extract_name_from_text, extract_email, extract_phone, get_cache_dir

EXTRACTION_VERSION = 3

def extract_with_liteparse(path: Path) -> str:
    parser = LiteParse()
    result = parser.parse(str(path))
    return getattr(result, "text", str(result))

def extract_tables_with_pdfplumber(path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""

    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                parts.append(f"\n[Page {page_number} Text]\n{text}")

            for table_number, table in enumerate(page.extract_tables() or [], start=1):
                rows = []
                for row in table:
                    rows.append(" | ".join((cell or "").strip() for cell in row))
                if rows:
                    parts.append(f"\n[Page {page_number} Table {table_number}]\n" + "\n".join(rows))
    return "\n".join(parts)

def extract_with_pypdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""

    parts = []
    reader = PdfReader(str(path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            parts.append(f"\n[Page {page_number}]\n{text}")
    return "\n".join(parts)

def extract_with_ocr(path: Path) -> str:
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        return ""

    parts = []
    try:
        images = convert_from_path(str(path), dpi=250)
        for page_number, image in enumerate(images, start=1):
            text = pytesseract.image_to_string(image)
            if text.strip():
                parts.append(f"\n[Page {page_number} OCR]\n{text}")
    except Exception:
        return ""
    return "\n".join(parts)

def extract_resume_text(path: Path) -> str:
    sections = []
    for extractor in [extract_with_liteparse, extract_tables_with_pdfplumber, extract_with_pypdf, extract_with_ocr]:
        text = extractor(path)
        if text and text.strip():
            sections.append(text.strip())
    return "\n\n".join(dict.fromkeys(sections))

def parser_agent(state):
    path = Path(state["resume_path"])
    cache_dir = Path("resume_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{path.stem}.json"
    index_path = cache_dir / f"{path.stem}_faiss"

    if cache_file.exists() and index_path.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("extraction_version") == EXTRACTION_VERSION:
            vectorstore = FAISS.load_local(str(index_path), state["embeddings"], allow_dangerous_deserialization=True)
            return {"raw_text": cached["raw_text"], "vectorstore": vectorstore}

    raw_text = extract_resume_text(path)

    # Add structured header
    header = (
        "CANDIDATE RESUME\n"
        f"Detected Name: {extract_name_from_text(raw_text)}\n"
        f"Detected Email: {extract_email(raw_text)}\n"
        f"Detected Phone: {extract_phone(raw_text)}\n\n"
    )
    raw_text = header + raw_text

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents([Document(page_content=raw_text)])
    vectorstore = FAISS.from_documents(docs, state["embeddings"])
    vectorstore.save_local(str(index_path))

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"raw_text": raw_text, "extraction_version": EXTRACTION_VERSION}, f)

    return {"raw_text": raw_text, "vectorstore": vectorstore}
