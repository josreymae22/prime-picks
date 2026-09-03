"""
Prime Picks -> X / Facebook / Instagram slate auto-poster.

Replaces the earlier per-game x_autopost module.

Model: one post per KICKOFF WINDOW, not per game and not per slate.
NFL Sunday produces three posts (early / late / primetime); a CFB Saturday
produces three or four. Anchoring to the window means no pick is ever
published before its own game has started.

Each window produces one rendered slate card image, which is posted to all
three platforms with platform-appropriate captions.

Env vars:
    X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
    GRAPH_VERSION, FB_PAGE_ID, FB_PAGE_TOKEN, IG_USER_ID
    R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY, R2_SECRET_KEY, R2_PUBLIC_BASE
    SOCIAL_AUTOPOST_ENABLED=true|false
    SOCIAL_AUTOPOST_DRY_RUN=true|false
    SOCIAL_PLATFORMS=x,facebook,instagram
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable, Optional, Sequence

import httpx
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base  # adjust to your project's declarative base
from slate_card import Pick, SlateCard, caption_for

log = logging.getLogger("prime_picks.social_autopost")

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

POST_DELAY_MINUTES = 10     # after the window's first scheduled kickoff
WINDOW_GAP_MINUTES = 90     # kickoffs further apart than this start a new window
TICK_SECONDS = 30
MAX_PER_TICK = 6
STAGGER_SECONDS = (4, 9)
MAX_ATTEMPTS = 3
GRACE_MINUTES = 180

ENABLED = os.getenv("SOCIAL_AUTOPOST_ENABLED", "false").lower() == "true"
DRY_RUN = os.getenv("SOCIAL_AUTOPOST_DRY_RUN", "true").lower() == "true"
PLATFORMS = [
    p.strip() for p in os.getenv("SOCIAL_PLATFORMS", "x").split(",") if p.strip()
]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class PostStatus(str, Enum):
    PENDING = "pending"
    POSTED = "posted"
    FAILED = "failed"
    SKIPPED = "skipped"


class ScheduledPost(Base):
    __tablename__ = "scheduled_posts"
    __table_args__ = (
        UniqueConstraint("slate_key", "platform", name="uq_scheduled_post"),
    )

    id = Column(Integer, primary_key=True)
    slate_key = Column(String(96), nullable=False, index=True)
    platform = Column(String(16), nullable=False)
    body = Column(Text, nullable=False)
    media_url = Column(Text, nullable=True)
    post_at = Column(DateTime(timezone=True), nullable=False, index=True)
    status = Column(String(16), nullable=False, default=PostStatus.PENDING.value)
    attempts = Column(Integer, nullable=False, default=0)
    external_id = Column(String(64), nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


# --------------------------------------------------------------------------
# Window grouping
# --------------------------------------------------------------------------


@dataclass
class Game:
    game_id: str
    commence_time: datetime      # scheduled kickoff, UTC
    pick: Pick


@dataclass
class Window:
    start: datetime              # earliest kickoff in the window
    games: list[Game]

    @property
    def post_at(self) -> datetime:
        return self.start + timedelta(minutes=POST_DELAY_MINUTES)

    @property
    def picks(self) -> list[Pick]:
        return [g.pick for g in self.games]


def group_into_windows(
    games: Iterable[Game], gap_minutes: int = WINDOW_GAP_MINUTES
) -> list[Window]:
    """
    Cluster games by kickoff time. Any gap larger than gap_minutes opens a new
    window. This naturally separates NFL 1:00 / 4:25 / 8:20 and CFB noon /
    3:30 / 7:00 / late without hardcoding any clock times.
    """
    ordered = sorted(games, key=lambda g: g.commence_time)
    if not ordered:
        return []

    windows: list[Window] = []
    current = [ordered[0]]

    for prev, game in zip(ordered, ordered[1:]):
        if game.commence_time - prev.commence_time > timedelta(minutes=gap_minutes):
            windows.append(Window(start=current[0].commence_time, games=current))
            current = [game]
        else:
            current.append(game)

    windows.append(Window(start=current[0].commence_time, games=current))
    return windows


def label_window(window: Window, index: int, total: int, tz_offset_hours: int = -4):
    """Human label for the card header. Uses local kickoff hour."""
    local_hour = (window.start + timedelta(hours=tz_offset_hours)).hour
    if total == 1:
        return "Full slate"
    if local_hour < 15:
        return "Early window"
    if local_hour < 19:
        return "Late window"
    return "Primetime"


def slate_key(sport: str, week_label: str, window: Window) -> str:
    """Deterministic. Regenerating the card produces the same key."""
    stamp = window.start.astimezone(timezone.utc).strftime("%Y%m%dT%H%M")
    return f"{sport.lower()}-{week_label.lower().replace(' ', '')}-{stamp}"


# --------------------------------------------------------------------------
# Media storage (Cloudflare R2 / any S3-compatible bucket)
# --------------------------------------------------------------------------


def upload_media(local_path: str, key: str) -> str:
    """Returns a public URL. Instagram requires the image be publicly fetchable."""
    import boto3

    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY"],
        aws_secret_access_key=os.environ["R2_SECRET_KEY"],
    )
    with open(local_path, "rb") as fh:
        s3.put_object(
            Bucket=os.environ["R2_BUCKET"],
            Key=key,
            Body=fh,
            ContentType="image/jpeg",
            CacheControl="public, max-age=31536000",
        )
    return f"{os.environ['R2_PUBLIC_BASE'].rstrip('/')}/{key}"


# --------------------------------------------------------------------------
# Enqueue — call after the weekly card is finalized
# --------------------------------------------------------------------------


async def schedule_slate_posts(
    session: AsyncSession,
    *,
    sport: str,                  # "NFL" / "CFB"
    week_label: str,             # "Week 1"
    games: Sequence[Game],
    record_label: Optional[str] = None,
    platforms: Optional[Sequence[str]] = None,
) -> list[ScheduledPost]:
    """Idempotent across regenerations. One row per (window, platform)."""
    platforms = list(platforms or PLATFORMS)
    windows = group_into_windows(games)
    created: list[ScheduledPost] = []

    for i, window in enumerate(windows):
        key = slate_key(sport, week_label, window)
        label = label_window(window, i, len(windows))
        local_date = (window.start + timedelta(hours=-4)).strftime("%a, %b %-d")

        card = SlateCard(
            sport=sport,
            week_label=week_label,
            window_label=label,
            date_label=local_date,
            picks=window.picks,
            record_label=record_label,
        )

        media_url: Optional[str] = None
        if not DRY_RUN:
            local = card.render(f"/tmp/{key}.jpg")
            media_url = upload_media(local, f"cards/{key}.jpg")
        else:
            card.render(f"/tmp/{key}.jpg")   # still render so we can eyeball it

        for platform in platforms:
            existing = await session.scalar(
                select(ScheduledPost).where(
                    ScheduledPost.slate_key == key,
                    ScheduledPost.platform == platform,
                )
            )
            body = caption_for(platform, card)

            if existing:
                if existing.status == PostStatus.PENDING.value:
                    existing.post_at = window.post_at
                    existing.body = body
                    existing.media_url = media_url
                created.append(existing)
                continue

            row = ScheduledPost(
                slate_key=key,
                platform=platform,
                body=body,
                media_url=media_url,
                post_at=window.post_at,
            )
            session.add(row)
            created.append(row)

    await session.commit()
    log.info(
        "scheduled %s windows x %s platforms for %s %s",
        len(windows), len(platforms), sport, week_label,
    )
    return created


# --------------------------------------------------------------------------
# Publishers
# --------------------------------------------------------------------------


def _graph() -> str:
    return f"https://graph.facebook.com/{os.environ['GRAPH_VERSION']}"


async def ensure_local_media(row: ScheduledPost) -> Optional[str]:
    """
    X wants image bytes; Instagram wants a public URL. We render to /tmp at
    enqueue time, but /tmp does not survive a container restart, and Render
    restarts on every deploy. So never assume the file is still there: if it
    is missing, pull it back down from media_url.
    """
    if not row.media_url:
        return None

    local = f"/tmp/{row.slate_key}.jpg"
    if os.path.exists(local) and os.path.getsize(local) > 0:
        return local

    log.info("local media missing for %s, refetching from R2", row.slate_key)
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(row.media_url)
        r.raise_for_status()
        with open(local, "wb") as fh:
            fh.write(r.content)
    return local


async def publish_x(row: ScheduledPost) -> str:
    import tweepy

    local = await ensure_local_media(row)

    def _send() -> str:
        client = tweepy.Client(
            consumer_key=os.environ["X_API_KEY"],
            consumer_secret=os.environ["X_API_SECRET"],
            access_token=os.environ["X_ACCESS_TOKEN"],
            access_token_secret=os.environ["X_ACCESS_TOKEN_SECRET"],
        )
        media_ids = None
        if local:
            # Media upload still lives on the v1.1 endpoint; the post itself is
            # v2. Two clients is correct here, not a leftover.
            auth = tweepy.OAuth1UserHandler(
                os.environ["X_API_KEY"],
                os.environ["X_API_SECRET"],
                os.environ["X_ACCESS_TOKEN"],
                os.environ["X_ACCESS_TOKEN_SECRET"],
            )
            media = tweepy.API(auth).media_upload(filename=local)
            media_ids = [media.media_id_string]

        resp = client.create_tweet(text=row.body, media_ids=media_ids)
        return str(resp.data["id"])

    return await asyncio.to_thread(_send)


async def publish_facebook(row: ScheduledPost) -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        if row.media_url:
            r = await client.post(
                f"{_graph()}/{os.environ['FB_PAGE_ID']}/photos",
                data={
                    "url": row.media_url,
                    "caption": row.body,
                    "access_token": os.environ["FB_PAGE_TOKEN"],
                },
            )
        else:
            r = await client.post(
                f"{_graph()}/{os.environ['FB_PAGE_ID']}/feed",
                data={
                    "message": row.body,
                    "access_token": os.environ["FB_PAGE_TOKEN"],
                },
            )
        r.raise_for_status()
        return r.json()["id"]


async def publish_instagram(row: ScheduledPost) -> str:
    if not row.media_url:
        raise ValueError("Instagram requires media_url — text-only posts are rejected")

    ig_id = os.environ["IG_USER_ID"]
    token = os.environ["FB_PAGE_TOKEN"]

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{_graph()}/{ig_id}/media",
            data={"image_url": row.media_url, "caption": row.body,
                  "access_token": token},
        )
        r.raise_for_status()
        creation_id = r.json()["id"]

        for _ in range(30):
            s = await client.get(
                f"{_graph()}/{creation_id}",
                params={"fields": "status_code", "access_token": token},
            )
            code = s.json().get("status_code")
            if code == "FINISHED":
                break
            if code == "ERROR":
                raise RuntimeError(f"IG container failed: {s.json()}")
            await asyncio.sleep(2)
        else:
            raise TimeoutError("IG container never reached FINISHED")

        p = await client.post(
            f"{_graph()}/{ig_id}/media_publish",
            data={"creation_id": creation_id, "access_token": token},
        )
        p.raise_for_status()
        return p.json()["id"]


PUBLISHERS = {
    "x": publish_x,
    "facebook": publish_facebook,
    "instagram": publish_instagram,
}


# --------------------------------------------------------------------------
# Poller
# --------------------------------------------------------------------------


async def _claim_due(session: AsyncSession, limit: int) -> list[ScheduledPost]:
    stmt = (
        select(ScheduledPost)
        .where(
            ScheduledPost.status == PostStatus.PENDING.value,
            ScheduledPost.post_at <= _utcnow(),
            ScheduledPost.attempts < MAX_ATTEMPTS,
        )
        .order_by(ScheduledPost.post_at)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )
    return list((await session.scalars(stmt)).all())


async def _process_one(session: AsyncSession, row: ScheduledPost) -> None:
    now = _utcnow()

    if now - row.post_at > timedelta(minutes=GRACE_MINUTES):
        row.status = PostStatus.SKIPPED.value
        row.last_error = "past grace window"
        log.warning("skipping stale post %s/%s", row.slate_key, row.platform)
        return

    if DRY_RUN:
        log.info(
            "[DRY RUN] %s -> %s\nmedia=%s\n%s",
            row.slate_key, row.platform, row.media_url, row.body,
        )
        row.status = PostStatus.POSTED.value
        row.external_id = f"dryrun-{random.randint(10**9, 10**10)}"
        return

    row.attempts += 1
    try:
        row.external_id = await PUBLISHERS[row.platform](row)
        row.status = PostStatus.POSTED.value
        row.last_error = None
        log.info("posted %s/%s id=%s", row.slate_key, row.platform, row.external_id)
    except Exception as e:  # noqa: BLE001
        row.last_error = f"{type(e).__name__}: {e}"
        if row.attempts >= MAX_ATTEMPTS:
            row.status = PostStatus.FAILED.value
        else:
            row.post_at = now + timedelta(seconds=90 * row.attempts)
        log.exception(
            "post failed %s/%s attempt=%s", row.slate_key, row.platform, row.attempts
        )


async def run_poller(session_factory) -> None:
    if not ENABLED:
        log.info("social autopost disabled; poller not started")
        return

    log.info("social autopost poller started (dry_run=%s, platforms=%s)",
             DRY_RUN, PLATFORMS)
    while True:
        try:
            async with session_factory() as session:
                async with session.begin():
                    due = await _claim_due(session, MAX_PER_TICK)
                    for i, row in enumerate(due):
                        if i:
                            await asyncio.sleep(random.uniform(*STAGGER_SECONDS))
                        await _process_one(session, row)
        except asyncio.CancelledError:
            log.info("poller cancelled")
            raise
        except Exception:  # noqa: BLE001
            log.exception("poller tick failed")
        await asyncio.sleep(TICK_SECONDS)
