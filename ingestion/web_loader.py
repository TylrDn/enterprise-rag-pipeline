"""Web scraper: Playwright (JS-heavy) + BeautifulSoup fallback."""
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
import logging
import asyncio

logger = logging.getLogger(__name__)


async def _scrape_playwright(url: str) -> str:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=30000)
        await page.wait_for_load_state("networkidle")
        content = await page.inner_text("body")
        await browser.close()
        return content


def load_url(url: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    """Scrape a URL and return chunked Documents."""
    text = asyncio.run(_scrape_playwright(url))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents(
        [text], metadatas=[{"source": url, "loader": "web"}]
    )
