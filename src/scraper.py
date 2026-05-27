"""
src/scraper.py
"""
from __future__ import annotations

import asyncio
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse, urldefrag

import httpx
import mlflow
from bs4 import BeautifulSoup
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_DATA_DIR,
    SCRAPE_CONCURRENCY,
    SCRAPE_DELAY_MAX,
    SCRAPE_DELAY_MIN,
    SCRAPE_MAX_PAGES,
    SCRAPE_TARGETS,
    SCRAPE_TIMEOUT_S,
)

_LOG = logging.getLogger("recamier.scraper")

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
]

_SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".rar", ".exe", ".mp4", ".mp3", ".avi",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".xml", ".json", ".txt", ".csv",
}

_SKIP_PATTERNS = [
    r"\?.*page=\d+", r"\?.*paged=\d+", r"/page/\d+",
    r"/feed/?$", r"/wp-json/", r"/wp-admin/",
    r"/cart/?", r"/checkout/?", r"/my-account/",
    r"\?add-to-cart=", r"\?remove_item=",
    r"/tag/", r"/author/", r"#",
]


def _random_ua() -> str:
    return random.choice(_USER_AGENTS)


def _build_headers(referer: Optional[str] = None, base_url: str = "") -> dict[str, str]:
    parsed = urlparse(base_url)
    headers = {
        "User-Agent": _random_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "es-CO,es;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-CH-UA": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
        "Sec-CH-UA-Mobile": "?0",
        "Sec-CH-UA-Platform": '"Windows"',
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin" if referer else "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def _should_skip_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    for ext in _SKIP_EXTENSIONS:
        if path.endswith(ext):
            return True
    for pattern in _SKIP_PATTERNS:
        if re.search(pattern, url.lower()):
            return True
    return False


def _extract_text(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript", "iframe", "svg",
                      "nav", "footer", "header", "aside", "form", "button", "meta", "link"]):
        tag.decompose()

    for tag in soup.find_all(True, class_=re.compile(
        r"cookie|popup|modal|banner|newsletter|widget|sidebar|breadcrumb|menu|nav|social|share|comment|related|ad|promo", re.I
    )):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": "description"})
    if meta_tag and meta_tag.get("content"):
        meta_desc = meta_tag["content"].strip()

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"^(content|main|primary|post)$", re.I))
        or soup.find(class_=re.compile(r"^(content|main|entry-content|post-content|page-content)$", re.I))
        or soup.body
    )

    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 15]
    deduped: list[str] = []
    prev = ""
    for ln in lines:
        if ln != prev:
            deduped.append(ln)
        prev = ln

    header_parts = [f"URL: {url}", f"Título: {title}"]
    if meta_desc:
        header_parts.append(f"Descripción: {meta_desc}")

    return "\n".join(header_parts) + "\n\n" + "\n".join(deduped)


def _extract_links(html: str, base_url: str, allowed_domains: list[str]) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links: set[str] = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()
        if not href:
            continue
        href, _ = urldefrag(href)
        if not href:
            continue
        if href.startswith(("mailto:", "tel:", "javascript:", "whatsapp:", "data:")):
            continue
        try:
            full_url = urljoin(base_url, href)
        except Exception:
            continue
        parsed = urlparse(full_url)
        if parsed.scheme not in ("http", "https"):
            continue
        if not any(d in parsed.netloc for d in allowed_domains):
            continue
        clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            clean += f"?{parsed.query}"
        clean = clean.rstrip("/")
        if clean and not _should_skip_url(clean):
            links.add(clean)

    return list(links)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=20),
    retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutException)),
    reraise=True,
)
async def _fetch_httpx(client: httpx.AsyncClient, url: str, referer: Optional[str] = None, base_url: str = "") -> Optional[str]:
    headers = _build_headers(referer=referer, base_url=base_url)
    resp = await client.get(url, headers=headers, follow_redirects=True)
    if resp.status_code in (404, 410, 403, 401):
        return None
    resp.raise_for_status()
    if "text/html" not in resp.headers.get("content-type", ""):
        return None
    return resp.text


async def _fetch_playwright(url: str) -> Optional[str]:
    try:
        from playwright.async_api import async_playwright
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            ctx = await browser.new_context(
                user_agent=_random_ua(),
                locale="es-CO",
                viewport={"width": 1366, "height": 768},
            )
            page = await ctx.new_page()
            await page.route(
                re.compile(r"\.(png|jpg|jpeg|gif|webp|svg|ico|woff|woff2|ttf|css)$"),
                lambda route: route.abort(),
            )
            await page.goto(url, timeout=SCRAPE_TIMEOUT_S * 1000, wait_until="domcontentloaded")
            await asyncio.sleep(random.uniform(0.8, 2.0))
            content = await page.content()
            await browser.close()
            return content
    except Exception as e:
        _LOG.warning("Playwright falló %s: %s", url, e)
        return None


async def _crawl_site(
    base_url: str,
    allowed_domains: list[str],
    max_pages: int = SCRAPE_MAX_PAGES,
    seed_urls: list[str] | None = None,
) -> list[str]:
    visited: set[str] = set()
    start = base_url.rstrip("/")
    queue: list[str] = seed_urls if seed_urls else [start]
    documents: list[str] = []
    errors = 0
    skipped = 0
    semaphore = asyncio.Semaphore(SCRAPE_CONCURRENCY)

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=SCRAPE_CONCURRENCY + 2)
    timeout = httpx.Timeout(SCRAPE_TIMEOUT_S, connect=10.0)

    _LOG.info("🚀 Iniciando crawl: %s (máx %d páginas)", base_url, max_pages)

    async with httpx.AsyncClient(timeout=timeout, limits=limits, http2=False) as client:
        while queue and len(visited) < max_pages:
            url = queue.pop(0)
            if url in visited:
                continue
            if _should_skip_url(url):
                skipped += 1
                continue

            visited.add(url)
            await asyncio.sleep(random.uniform(SCRAPE_DELAY_MIN, SCRAPE_DELAY_MAX))

            async with semaphore:
                html: Optional[str] = None
                referer = base_url if url != start else None

                try:
                    html = await _fetch_httpx(client, url, referer=referer, base_url=base_url)
                except Exception as e:
                    _LOG.warning("httpx falló %s (%s) → Playwright...", url, type(e).__name__)
                    html = await _fetch_playwright(url)
                    if not html:
                        errors += 1
                        continue

                if not html:
                    continue

                text = _extract_text(html, url)
                if len(text.split()) < 30:
                    continue

                documents.append(text)
                _LOG.info("✅ [%d págs | %d docs] %s", len(visited), len(documents), url)

                new_links = _extract_links(html, base_url, allowed_domains)
                for link in new_links:
                    if link not in visited and link not in queue:
                        queue.append(link)

    _LOG.info("🏁 Crawl: %d visitadas | %d docs | %d errores | %d saltadas", len(visited), len(documents), errors, skipped)
    return documents


def scrape_all() -> list[str]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    output_files: list[str] = []

    with mlflow.start_run(run_name="scraping_recamier"):
        mlflow.log_param("max_pages_per_site", SCRAPE_MAX_PAGES)
        total_docs = 0
        total_chars = 0

        for target in SCRAPE_TARGETS:
            name = target["name"]
            print(f"\n{'='*60}\n🌐 Scrapeando: {name} ({target['base_url']})\n{'='*60}")
            t0 = time.time()

            docs = asyncio.run(
                _crawl_site(
                    target["base_url"],
                    target["allowed_domains"],
                    seed_urls=target.get("seed_urls"),
                )
            )

            elapsed = round(time.time() - t0, 2)
            output_path = PROCESSED_DATA_DIR / target["output_file"]
            combined = "\n\n---\n\n".join(docs)
            output_path.write_text(combined, encoding="utf-8")

            chars = len(combined)
            total_docs += len(docs)
            total_chars += chars
            mlflow.log_metric(f"{name}_pages_scraped", len(docs))
            mlflow.log_metric(f"{name}_chars", chars)

            print(f"✅ {name}: {len(docs)} páginas | {chars:,} chars | {int(elapsed//60)}m {int(elapsed%60)}s")
            output_files.append(str(output_path))

        mlflow.log_metric("total_docs", total_docs)
        mlflow.log_metric("total_chars", total_chars)
        print(f"\n🏁 LISTO: {total_docs} docs totales → ahora corre: python -m src.ingest\n")

    return output_files


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    scrape_all()
