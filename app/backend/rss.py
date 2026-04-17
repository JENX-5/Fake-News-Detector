"""
rss.py
------
Live RSS feed fetcher with in-memory cache.
Fetches headlines from major news sources and classifies them
using the loaded MNB model.
"""

import re
import time
import threading
import logging

logger = logging.getLogger(__name__)

try:
    import feedparser
    FEEDPARSER_OK = True
except ImportError:
    FEEDPARSER_OK = False
    logger.warning("feedparser not installed. Run: pip install feedparser")

RSS_FEEDS = [
    {"name": "BBC News",   "url": "http://feeds.bbci.co.uk/news/rss.xml",       "icon": "🔵"},
    {"name": "Reuters",    "url": "https://feeds.reuters.com/reuters/topNews",   "icon": "🟠"},
    {"name": "AP News",    "url": "https://rsshub.app/apnews/topics/apf-topnews","icon": "🔴"},
    {"name": "Al Jazeera", "url": "https://www.aljazeera.com/xml/rss/all.xml",   "icon": "🟢"},
]

CACHE_TTL = 300  # 5 minutes
_cache      = {"articles": [], "last_updated": 0, "error": None}
_cache_lock = threading.Lock()


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def _fetch_articles(model) -> list:
    """Fetch from all RSS feeds and classify each article."""
    if not FEEDPARSER_OK:
        return []

    articles = []
    for feed_meta in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_meta["url"])
            for entry in feed.entries[:8]:
                title    = getattr(entry, "title",    "") or ""
                summary  = _strip_html(getattr(entry, "summary", "") or
                                       getattr(entry, "description", "") or "")
                link     = getattr(entry, "link",     "#")
                published= getattr(entry, "published","")

                combined = f"{title}. {summary}".strip()
                result   = model.predict_fn(combined)

                articles.append({
                    "source"    : feed_meta["name"],
                    "icon"      : feed_meta["icon"],
                    "title"     : title,
                    "summary"   : summary[:300] + ("…" if len(summary) > 300 else ""),
                    "link"      : link,
                    "published" : published[:25] if published else "",
                    "label"     : result["label"],
                    "verdict"   : result["verdict"],
                    "confidence": result["confidence"],
                    "prob_fake" : result["prob_fake"],
                    "prob_real" : result["prob_real"],
                })
        except Exception as e:
            logger.warning(f"RSS error ({feed_meta['name']}): {e}")

    return articles


def get_cached_feed(model) -> dict:
    with _cache_lock:
        now = time.time()
        if now - _cache["last_updated"] > CACHE_TTL or not _cache["articles"]:
            try:
                articles = _fetch_articles(model)
                _cache["articles"]     = articles
                _cache["last_updated"] = now
                _cache["error"]        = None
            except Exception as e:
                _cache["error"] = str(e)

        return {
            "articles"    : _cache["articles"],
            "last_updated": _cache["last_updated"],
            "count"       : len(_cache["articles"]),
            "feedparser_available": FEEDPARSER_OK,
            "error"       : _cache["error"],
        }


def invalidate_cache():
    with _cache_lock:
        _cache["last_updated"] = 0
