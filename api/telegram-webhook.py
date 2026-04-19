"""
RAGTube / RAG-Omni — Telegram webhook on Vercel serverless.

Flow:
  Telegram -> /api/telegram-webhook -> route by domain -> extract content
  -> push to RAG-Anything -> reply in Telegram.

Everything runs synchronously within the Vercel function lifetime because
Vercel kills background tasks once the response is returned.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

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
CLOUDINARY_CLOUD_NAME = _env("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_UPLOAD_PRESET = _env("CLOUDINARY_UPLOAD_PRESET", "")
OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
IMAGE_DESCRIPTION_PROMPT = _env("IMAGE_DESCRIPTION_PROMPT", "")

# Swap these for the specific Apify actor IDs you want to use.
# Browse https://apify.com/store to pick one that matches your pricing / quality needs.
APIFY_INSTAGRAM_ACTOR = "apify/instagram-scraper"
APIFY_TIKTOK_ACTOR = "clockworks/tiktok-scraper"
# YouTube actor — switched off youtube-transcript-api because it gets blocked
# from Vercel's datacenter IPs. Apify runs from residential IPs YouTube accepts.
APIFY_YOUTUBE_ACTOR = "pintostudio/youtube-transcript-scraper"

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([a-zA-Z0-9_-]{11})")

# Query params that carry no content identity — strip them so the same URL
# shared with different tracking tails dedupes to one LightRAG document.
_TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "si", "igsh", "igshid", "feature", "ref", "ref_src", "ref_url",
    "fbclid", "gclid", "mc_cid", "mc_eid", "yclid", "msclkid",
    "_branch_match_id", "_branch_referrer",
}


def canonicalize_url(url: str) -> str:
    """Strip tracking params, normalize host, collapse youtu.be → youtube.com."""
    try:
        parts = urlparse(url)
    except Exception:
        return url
    host = (parts.hostname or "").lower().removeprefix("www.")
    if not host:
        return url
    scheme = "https" if parts.scheme in ("http", "https", "") else parts.scheme

    # youtu.be/VIDEO → youtube.com/watch?v=VIDEO (one canonical form).
    if host == "youtu.be":
        vid = parts.path.lstrip("/").split("/", 1)[0] if parts.path else ""
        if vid:
            return f"https://youtube.com/watch?v={vid}"

    kept = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k.lower() not in _TRACKING_PARAMS]
    new_query = urlencode(kept)
    path = parts.path.rstrip("/") if parts.path not in ("", "/") else parts.path
    return urlunparse((scheme, host, path, parts.params, new_query, ""))

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


async def send_telegram_document(chat_id: int, filename: str, content: str, caption: str) -> None:
    """Upload extracted markdown back to the user's chat as a file — zero-infra
    dead-letter queue for posts that couldn't reach LightRAG."""
    async with httpx.AsyncClient(timeout=30) as client:
        await client.post(
            f"{TELEGRAM_API}/sendDocument",
            data={"chat_id": str(chat_id), "caption": caption[:1000]},
            files={"document": (filename, content.encode("utf-8"), "text/markdown")},
        )


# httpx exceptions that indicate the remote side is temporarily unreachable —
# retrying later (via Telegram's webhook redelivery) is likely to succeed.
_TRANSIENT_HTTPX_EXC = (
    httpx.ConnectError,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.ConnectTimeout,
    httpx.PoolTimeout,
)


def _is_transient(exc: BaseException) -> bool:
    """True if the error looks transient — tunnel/LightRAG temporarily unreachable."""
    if isinstance(exc, _TRANSIENT_HTTPX_EXC):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return 500 <= exc.response.status_code < 600
    # RuntimeError raised by push_to_rag_anything when retries are exhausted.
    if isinstance(exc, RuntimeError) and "retries" in str(exc).lower():
        return True
    return False


# -------------------------------------------------------------------
# Image helpers: stable storage + vision-model description
# -------------------------------------------------------------------
async def upload_to_cloudinary(media_url: str) -> dict[str, Any] | None:
    """Fetch a transient media URL and re-host it on Cloudinary. Returns the
    JSON response, or None if Cloudinary isn't configured or the upload fails."""
    if not CLOUDINARY_CLOUD_NAME or not CLOUDINARY_UPLOAD_PRESET:
        return None
    endpoint = f"https://api.cloudinary.com/v1_1/{CLOUDINARY_CLOUD_NAME}/image/upload"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            endpoint,
            data={
                "file": media_url,
                "upload_preset": CLOUDINARY_UPLOAD_PRESET,
                "folder": "ragtube/inspiration",
                "tags": "ragtube",
            },
        )
        if response.status_code >= 400:
            return None
        return response.json()


async def describe_image(image_url: str) -> str:
    """Call OpenAI's vision model with the configured prompt. Returns the
    description, or an empty string if the model isn't configured or fails."""
    if not OPENAI_API_KEY or not IMAGE_DESCRIPTION_PROMPT:
        return ""
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": IMAGE_DESCRIPTION_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    },
                ],
                "max_tokens": 500,
            },
        )
        if response.status_code >= 400:
            return ""
        data = response.json()
        return (data["choices"][0]["message"]["content"] or "").strip()


async def enrich_with_image(image_url: str | None) -> tuple[str | None, str | None, str]:
    """Rehost an image on Cloudinary and describe it. Returns a tuple of
    (stable_url, public_id, description). Safe to call with None."""
    if not image_url:
        return None, None, ""
    cloud = await upload_to_cloudinary(image_url)
    stable_url = cloud.get("secure_url") if cloud else None
    public_id = cloud.get("public_id") if cloud else None
    # Prefer describing the rehosted URL (guaranteed public, no auth tokens).
    description = await describe_image(stable_url or image_url)
    return stable_url, public_id, description


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
    # Retry transient connect errors — Vercel Lambda container reuse sometimes
    # surfaces httpx.ConnectError with OS errno 16 (EBUSY) on the first call of
    # a thawed container. A short backoff clears it.
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            client = ApifyClientAsync(APIFY_API_TOKEN)
            run = await client.actor(actor_id).call(run_input=run_input)
            if run is None:
                raise RuntimeError(f"Apify actor {actor_id} returned no run metadata.")
            dataset_id = run["defaultDatasetId"]
            items_page = await client.dataset(dataset_id).list_items()
            return list(items_page.items)
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as e:
            last_exc = e
            print(f"[apify-retry] {actor_id} attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(0.5 * (attempt + 1))
    raise last_exc or RuntimeError(f"Apify actor {actor_id} failed after retries.")


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

    # Carousels expose every slide under childPosts; single posts/reels fall back
    # to displayUrl. Each entry contributes one still frame for VLM description.
    source_urls: list[str] = []
    for child in post.get("childPosts") or []:
        child_url = child.get("displayUrl")
        if child_url:
            source_urls.append(child_url)
    if not source_urls:
        if isinstance(post.get("images"), list):
            source_urls = [u for u in post["images"] if u]
        elif post.get("displayUrl"):
            source_urls = [post["displayUrl"]]

    # Upload + describe all slides in parallel. return_exceptions=True so one
    # bad slide can't nuke the whole carousel — surviving slides still ingest.
    raw = await asyncio.gather(
        *[enrich_with_image(u) for u in source_urls],
        return_exceptions=True,
    )
    enrichments = [r for r in raw if isinstance(r, tuple)]
    stable_urls = [e[0] for e in enrichments if e[0]]
    public_ids = [e[1] for e in enrichments if e[1]]
    descriptions = [e[2] for e in enrichments if e[2]]

    body_parts = [caption] if caption else []
    if len(descriptions) == 1:
        body_parts.append(f"**Visual description:** {descriptions[0]}")
    elif len(descriptions) > 1:
        for i, desc in enumerate(descriptions, start=1):
            body_parts.append(f"**Slide {i}:** {desc}")
    body = "\n\n".join(body_parts)

    return {
        "platform": "instagram",
        "title": None,
        "author": author,
        "media_urls": stable_urls,
        "public_ids": public_ids,
        "body": body,
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

    caption = post.get("text") or post.get("description") or ""
    author = (post.get("authorMeta") or {}).get("name") or post.get("author")

    # Thumbnail / cover frame from the video.
    meta = post.get("videoMeta") or {}
    cover_url = meta.get("coverUrl") or post.get("coverUrl") or meta.get("originalCoverUrl")
    stable_url, public_id, visual_desc = await enrich_with_image(cover_url)

    body_parts = [caption] if caption else []
    if visual_desc:
        body_parts.append(f"**Visual description:** {visual_desc}")
    body = "\n\n".join(body_parts)

    return {
        "platform": "tiktok",
        "title": None,
        "author": author,
        "media_urls": [stable_url] if stable_url else [],
        "public_ids": [public_id] if public_id else [],
        "body": body,
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
            "cloudinary_public_ids": extracted.get("public_ids") or [],
        },
    }

    # Retry on transient network errors and 5xx responses. The tunnel + LightRAG
    # occasionally have a cold moment (container restart, Mac wake, tunnel
    # reconnect) — one retry after a short backoff clears nearly all of them.
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                response = await client.post(RAG_ANYTHING_API_URL, json=payload, headers=headers)
            if 500 <= response.status_code < 600:
                print(f"[rag-retry] {response.status_code} from LightRAG, attempt {attempt + 1}/3", flush=True)
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout) as e:
            last_exc = e
            print(f"[rag-retry] attempt {attempt + 1}/3 failed: {type(e).__name__}: {e}", flush=True)
            await asyncio.sleep(1.5 * (attempt + 1))
    if last_exc:
        raise last_exc
    raise RuntimeError(f"RAG-Anything push failed after retries (last status {response.status_code}).")


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

    url = canonicalize_url(url)
    domain = (urlparse(url).hostname or "unknown").removeprefix("www.")

    # Extraction phase — no markdown yet, so a failure here has nothing to
    # preserve. We just tell the user and return 200 so Telegram doesn't retry.
    try:
        extracted = await route_and_extract(url)
        markdown = build_markdown(url, notes, extracted)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[ragtube-error:extract] URL={url}", flush=True)
        traceback.print_exc()
        snippet = f"{type(exc).__name__}: {str(exc)[:260]}"
        await send_telegram_message(chat_id, f"❌ Failed to extract content from {url}: {snippet}")
        return {"ok": True}

    # Push phase — we have the markdown already. On any failure, save the
    # markdown to the user's chat as a file so no API-paid work is lost.
    try:
        await push_to_rag_anything(markdown, url, extracted)
    except Exception as exc:  # noqa: BLE001
        import traceback
        print(f"[ragtube-error:push] URL={url}", flush=True)
        traceback.print_exc()

        # Save the extracted markdown back to the user's chat as a file.
        filename = f"{domain}-{re.sub(r'[^a-zA-Z0-9]', '-', url)[:60]}.md"
        transient = _is_transient(exc)
        if transient:
            caption = f"⚠️ RAG push failed ({type(exc).__name__}). Telegram will retry this webhook. Markdown saved here as a safety copy."
        else:
            caption = f"❌ RAG push failed ({type(exc).__name__}: {str(exc)[:120]}). Not retrying. Markdown saved here — re-send the URL once LightRAG is back."
        try:
            await send_telegram_document(chat_id, filename, markdown, caption)
        except Exception:  # noqa: BLE001
            # If even Telegram file upload fails, fall back to a short text.
            await send_telegram_message(chat_id, caption[:400])

        if transient:
            # 500 makes Telegram redeliver. Re-extraction will re-run and re-pay
            # Apify/Cloudinary/OpenAI, but the user's data is preserved either way.
            return JSONResponse({"ok": False, "retry": True}, status_code=500)
        return {"ok": True}

    await send_telegram_message(
        chat_id,
        f"✅ Success! Parsed {domain} and added it to RAG-Anything. Notes saved.",
    )
    return {"ok": True}


@app.get("/api/telegram-webhook")
async def health() -> dict[str, str]:
    return {"status": "ok"}
