"""
RAGTube / RAG-Omni — Telegram webhook on Vercel serverless.

Flow:
  Telegram -> /api/telegram-webhook -> route by domain -> extract content
  -> push to RAG-Anything -> reply in Telegram.

Everything runs synchronously within the Vercel function lifetime because
Vercel kills background tasks once the response is returned.
"""

from __future__ import annotations

import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from apify_client import ApifyClientAsync
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------
def _env(key: str, default: str | None = None) -> str:
    """Read an env var and strip whitespace — guards against copy-paste newlines."""
    value = os.environ.get(key, default)
    if value is None:
        raise KeyError(key)
    return value.strip()


TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = _env("TELEGRAM_WEBHOOK_SECRET")
APIFY_API_TOKEN = _env("APIFY_API_TOKEN")
RAG_ANYTHING_API_URL = _env("RAG_ANYTHING_API_URL")
RAG_ANYTHING_API_KEY = _env("RAG_ANYTHING_API_KEY", "")

# Swap these for the specific Apify actor IDs you want to use.
# Browse https://apify.com/store to pick one that matches your pricing / quality needs.
APIFY_INSTAGRAM_ACTOR = "apify/instagram-scraper"
APIFY_TIKTOK_ACTOR = "clockworks/tiktok-scraper"
# YouTube actor — switched off youtube-transcript-api because it gets blocked
# from Vercel's datacenter IPs. Apify runs from residential IPs YouTube accepts.
APIFY_YOUTUBE_ACTOR = "pintostudio/youtube-transcript-scraper"

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([a-zA-Z0-9_-]{11})")

app = FastAPI()


# -------------------------------------------------------------------
# Telegram helpers
# -------------------------------------------------------------------
def extract_url_and_notes(message: dict[str, Any]) -> tuple[str | None, str]:
    """Return (url, user_notes) from a Telegram message payload."""
    text: str = message.get("text") or message.get("caption") or ""
    entities = message.get("entities") or message.get("caption_entities") or []

    url: str | None = None
    url_span: tuple[int, int] | None = None

    for entity in entities:
        etype = entity.get("type")
        if etype == "url":
            offset = entity["offset"]
            length = entity["length"]
            url = text[offset : offset + length]
            url_span = (offset, offset + length)
            break
        if etype == "text_link":
            url = entity.get("url")
            url_span = (entity["offset"], entity["offset"] + entity["length"])
            break

    # Fallback regex if Telegram didn't tag an entity.
    if url is None:
        match = re.search(r"https?://\S+", text)
        if match:
            url = match.group(0)
            url_span = match.span()

    if url_span:
        notes = (text[: url_span[0]] + text[url_span[1] :]).strip()
    else:
        notes = text.strip()

    return url, notes


async def send_telegram_message(chat_id: int, text: str) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
        )


# -------------------------------------------------------------------
# Extraction modules
# -------------------------------------------------------------------
async def extract_youtube(url: str) -> dict[str, Any]:
    # Actor input/output schemas vary — check the actor's README on apify.com if
    # the parsing below drops to empty.
    items = await _run_apify_actor(APIFY_YOUTUBE_ACTOR, {"videoUrl": url})
    if not items:
        raise RuntimeError("YouTube actor returned zero items (no transcript available?).")
    item = items[0]

    transcript_raw = (
        item.get("transcript")
        or item.get("captions")
        or item.get("text")
        or item.get("data")
        or ""
    )
    if isinstance(transcript_raw, list):
        transcript = " ".join(
            (seg.get("text") if isinstance(seg, dict) else str(seg)) or ""
            for seg in transcript_raw
        ).strip()
    else:
        transcript = str(transcript_raw).strip()

    return {
        "platform": "youtube",
        "title": item.get("title") or item.get("videoTitle"),
        "author": item.get("channelName") or item.get("author"),
        "media_urls": [],
        "body": transcript,
    }


async def extract_jina(url: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        response = await client.get(f"https://r.jina.ai/{url}")
        response.raise_for_status()
        markdown = response.text

    return {
        "platform": "web",
        "title": None,
        "author": None,
        "media_urls": [],
        "body": markdown,
    }


async def _run_apify_actor(actor_id: str, run_input: dict[str, Any]) -> list[dict[str, Any]]:
    client = ApifyClientAsync(APIFY_API_TOKEN)
    run = await client.actor(actor_id).call(run_input=run_input)
    if run is None:
        raise RuntimeError(f"Apify actor {actor_id} returned no run metadata.")
    dataset_id = run["defaultDatasetId"]
    items_page = await client.dataset(dataset_id).list_items()
    return list(items_page.items)


async def extract_instagram(url: str) -> dict[str, Any]:
    # Input schema varies per actor — check the actor's README on apify.com.
    items = await _run_apify_actor(
        APIFY_INSTAGRAM_ACTOR,
        {"directUrls": [url], "resultsLimit": 1, "addParentData": False},
    )
    if not items:
        raise RuntimeError("Instagram actor returned zero items (private, deleted, or blocked?).")
    post = items[0]

    caption = post.get("caption") or ""
    author = post.get("ownerUsername") or post.get("owner", {}).get("username")
    media = [m for m in [post.get("videoUrl"), post.get("displayUrl")] if m]

    return {
        "platform": "instagram",
        "title": None,
        "author": author,
        "media_urls": media,
        "body": caption,
    }


async def extract_tiktok(url: str) -> dict[str, Any]:
    # Different TikTok actors use different input keys ("postURLs", "urls", "startUrls").
    # Check the actor README and adjust.
    items = await _run_apify_actor(
        APIFY_TIKTOK_ACTOR,
        {"postURLs": [url], "resultsPerPage": 1, "shouldDownloadVideos": False},
    )
    if not items:
        raise RuntimeError("TikTok actor returned zero items (private or blocked?).")
    post = items[0]

    description = post.get("text") or post.get("description") or ""
    author = (post.get("authorMeta") or {}).get("name") or post.get("author")
    video_url = post.get("videoUrl") or (post.get("videoMeta") or {}).get("downloadAddr")
    media = [video_url] if video_url else []

    return {
        "platform": "tiktok",
        "title": None,
        "author": author,
        "media_urls": media,
        "body": description,
    }


# -------------------------------------------------------------------
# Router
# -------------------------------------------------------------------
async def route_and_extract(url: str) -> dict[str, Any]:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")

    if host in {"youtube.com", "m.youtube.com", "youtu.be"} or host.endswith(".youtube.com"):
        return await extract_youtube(url)
    if "instagram.com" in host:
        return await extract_instagram(url)
    if "tiktok.com" in host:
        return await extract_tiktok(url)
    return await extract_jina(url)


# -------------------------------------------------------------------
# RAG-Anything push
# -------------------------------------------------------------------
def build_markdown(url: str, notes: str, extracted: dict[str, Any]) -> str:
    lines = [
        f"# Inspiration: {extracted.get('platform', 'web').title()}",
        "",
        f"**Source:** {url}",
    ]
    if extracted.get("author"):
        lines.append(f"**Author:** {extracted['author']}")
    if notes:
        lines += ["", "## User Notes", "", notes]
    if extracted.get("media_urls"):
        lines += ["", "## Media Assets", ""] + [f"- {m}" for m in extracted["media_urls"]]
    lines += ["", "## Extracted Content", "", extracted.get("body") or "_(empty)_"]
    return "\n".join(lines)


async def push_to_rag_anything(markdown: str, url: str, extracted: dict[str, Any]) -> None:
    """
    Abstracted so you can swap endpoint shape later. RAG-Anything's HTTP ingestion
    surface varies by version — check your instance's /docs.
    """
    headers = {"Content-Type": "application/json"}
    if RAG_ANYTHING_API_KEY:
        headers["X-API-Key"] = RAG_ANYTHING_API_KEY

    payload = {
        "text": markdown,
        "source": url,
        "metadata": {
            "platform": extracted.get("platform"),
            "author": extracted.get("author"),
            "media_urls": extracted.get("media_urls", []),
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(RAG_ANYTHING_API_URL, json=payload, headers=headers)
        response.raise_for_status()


# -------------------------------------------------------------------
# Webhook entry point
# -------------------------------------------------------------------
@app.post("/api/telegram-webhook")
async def telegram_webhook(request: Request) -> dict[str, bool]:
    # Reject anything that isn't Telegram — the secret is set via setWebhook
    # and returned in X-Telegram-Bot-Api-Secret-Token on every delivery.
    if request.headers.get("x-telegram-bot-api-secret-token") != TELEGRAM_WEBHOOK_SECRET:
        return JSONResponse({"ok": False}, status_code=401)

    update = await request.json()
    message = update.get("message") or update.get("channel_post") or {}
    chat = message.get("chat") or {}
    chat_id = chat.get("id")

    if not chat_id:
        return {"ok": True}

    url, notes = extract_url_and_notes(message)
    if not url:
        await send_telegram_message(chat_id, "Send me a URL (optionally with notes) and I'll ingest it.")
        return {"ok": True}

    domain = (urlparse(url).hostname or "unknown").removeprefix("www.")

    try:
        extracted = await route_and_extract(url)
        markdown = build_markdown(url, notes, extracted)
        await push_to_rag_anything(markdown, url, extracted)
        await send_telegram_message(
            chat_id,
            f"✅ Success! Parsed {domain} and added it to RAG-Anything. Notes saved.",
        )
    except Exception as exc:  # noqa: BLE001
        snippet = str(exc)[:300]
        await send_telegram_message(chat_id, f"❌ Failed to extract content from {url}: {snippet}")

    return {"ok": True}


@app.get("/api/telegram-webhook")
async def health() -> dict[str, str]:
    return {"status": "ok"}
