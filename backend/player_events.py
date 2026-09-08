"""
player_events.py

Roster/player ingestion layer for Prime Picks.

Source hierarchy:
  1. ESPN current NFL rosters (free)
  2. ESPN Core API team-athletes fallback
  3. Manual entries via admin panel
  4. SportsData.io transactions (optional paid source)
  5. MySportsFeeds transactions (optional paid source)

ESPN roster imports use a neutral impact score of 50 by default.
This prevents newly imported players from affecting predictions until
their impact scores are intentionally adjusted.

Roster synchronization is optimized so team adjustment is recalculated
once per team rather than once per player.
"""

import os
import logging
import time

from datetime import datetime
from typing import Optional

import httpx

from roster_engine import roster_engine


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

ESPN_BASE = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football"
)

ESPN_CORE_BASE = (
    "https://sports.core.api.espn.com/v2/"
    "sports/football/leagues/nfl"
)

SPORTSDATA_KEY = os.getenv(
    "SPORTSDATA_API_KEY"
)

MSF_KEY = os.getenv(
    "MSF_API_KEY"
)

MSF_PASSWORD = os.getenv(
    "MSF_PASSWORD"
)


# ============================================================
# Position normalization
# ============================================================

POSITION_TO_GROUP = {

    # Offense
    "QB": "QB",

    "RB": "RB",
    "FB": "RB",
    "HB": "RB",

    "WR": "WR",
    "FL": "WR",
    "SE": "WR",

    "TE": "TE",

    "OT": "OL",
    "OG": "OL",
    "C": "OL",
    "OL": "OL",
    "G": "OL",
    "T": "OL",

    # Defense
    "DE": "DL",
    "DT": "DL",
    "NT": "DL",
    "DL": "DL",

    "LB": "LB",
    "OLB": "LB",
    "ILB": "LB",
    "MLB": "LB",

    "CB": "CB",
    "DB": "CB",

    "S": "S",
    "SS": "S",
    "FS": "S",

    # Special teams
    "K": "K",
    "P": "K",
    "LS": "K",
}


def normalize_position(
    pos: str,
) -> str:
    """
    Convert ESPN positions to Prime Picks position groups.
    """

    value = (
        pos
        or ""
    ).strip().upper()

    if (
        value
        in POSITION_TO_GROUP
    ):

        return (
            POSITION_TO_GROUP[
                value
            ]
        )

    logger.debug(
        (
            "Unknown position '%s'; "
            "defaulting to LB"
        ),
        value,
    )

    return "LB"


# ============================================================
# ESPN helpers
# ============================================================

def _current_nfl_season() -> int:
    """
    Return current calendar year.

    For Prime Picks' current September usage this corresponds
    to the active NFL season year.
    """

    return (
        datetime.utcnow()
        .year
    )


def _extract_position(
    athlete: dict,
) -> str:
    """
    Handle ESPN position data whether represented as a nested
    object, abbreviation string, or partially populated object.
    """

    position = (
        athlete.get(
            "position",
            {},
        )
    )

    if isinstance(
        position,
        dict,
    ):

        return str(
            position.get(
                "abbreviation"
            )
            or
            position.get(
                "name"
            )
            or
            ""
        ).strip()

    return str(
        position
        or ""
    ).strip()


def _normalize_espn_status(
    athlete: dict,
) -> str:
    """
    Extract ESPN roster status safely.
    """

    status_data = (
        athlete.get(
            "status",
            {},
        )
    )

    if isinstance(
        status_data,
        dict,
    ):

        return str(
            status_data.get(
                "type"
            )
            or
            status_data.get(
                "name"
            )
            or
            status_data.get(
                "displayName"
            )
            or
            "active"
        )

    return str(
        status_data
        or
        "active"
    )


def _athlete_to_roster_player(
    athlete: dict,
) -> Optional[dict]:
    """
    Convert an ESPN athlete object into the common roster shape.

    Returns None if ESPN did not provide enough information.
    """

    espn_id = str(
        athlete.get(
            "id",
            "",
        )
        or ""
    ).strip()

    if not espn_id:

        return None

    name = str(
        athlete.get(
            "fullName"
        )
        or
        athlete.get(
            "displayName"
        )
        or
        athlete.get(
            "shortName"
        )
        or ""
    ).strip()

    if not name:

        first_name = str(
            athlete.get(
                "firstName",
                "",
            )
            or ""
        ).strip()

        last_name = str(
            athlete.get(
                "lastName",
                "",
            )
            or ""
        ).strip()

        name = (
            f"{first_name} {last_name}"
        ).strip()

    if not name:

        return None

    pos = (
        _extract_position(
            athlete
        )
    )

    jersey = (
        athlete.get(
            "jersey",
            "",
        )
        or ""
    )

    status = (
        _normalize_espn_status(
            athlete
        )
    )

    return {

        "player_id":
            f"espn_nfl_{espn_id}",

        "name":
            name,

        "position":
            pos,

        "position_group":
            normalize_position(
                pos
            ),

        "jersey":
            jersey,

        "status":
            status,
    }


# ============================================================
# ESPN team discovery
# ============================================================

async def fetch_espn_nfl_teams() -> list[dict]:
    """
    Retrieve the current NFL team directory from ESPN.
    """

    url = (
        f"{ESPN_BASE}/nfl/teams"
    )

    try:

        async with httpx.AsyncClient(
            timeout=20
        ) as client:

            response = (
                await client.get(
                    url,
                    params={
                        "limit":
                            100
                    },
                )
            )

        if (
            response.status_code
            != 200
        ):

            logger.warning(
                (
                    "ESPN NFL teams request "
                    "returned HTTP %s"
                ),
                response.status_code,
            )

            return []

        data = (
            response.json()
        )

        teams = []

        for sport in (
            data.get(
                "sports",
                [],
            )
            or []
        ):

            for league in (
                sport.get(
                    "leagues",
                    [],
                )
                or []
            ):

                for entry in (
                    league.get(
                        "teams",
                        [],
                    )
                    or []
                ):

                    team = (
                        entry.get(
                            "team",
                            {},
                        )
                        or {}
                    )

                    team_id = str(
                        team.get(
                            "id",
                            "",
                        )
                        or ""
                    ).strip()

                    name = str(
                        team.get(
                            "displayName"
                        )
                        or
                        team.get(
                            "shortDisplayName"
                        )
                        or
                        team.get(
                            "name"
                        )
                        or ""
                    ).strip()

                    abbreviation = str(
                        team.get(
                            "abbreviation",
                            "",
                        )
                        or ""
                    ).strip()

                    if (
                        not team_id
                        or
                        not name
                    ):

                        continue

                    teams.append({

                        "id":
                            team_id,

                        "name":
                            name,

                        "abbreviation":
                            abbreviation,
                    })

        logger.info(
            "ESPN: discovered %s NFL teams",
            len(
                teams
            ),
        )

        return teams

    except Exception as exc:

        logger.exception(
            (
                "Unable to fetch ESPN NFL "
                "team list: %s"
            ),
            exc,
        )

        return []


# ============================================================
# Primary ESPN NFL roster endpoint
# ============================================================

async def fetch_espn_nfl_roster_primary(
    team_id: str,
) -> list[dict]:
    """
    Fetch roster from ESPN Site API.

    This is the normal preferred source.

    Some ESPN teams currently return HTTP 404 because ESPN's
    internal roster assembly fails while resolving a player's
    contract. Those failures are handled by the Core API fallback.
    """

    url = (
        f"{ESPN_BASE}/nfl/"
        f"teams/{team_id}/roster"
    )

    try:

        async with httpx.AsyncClient(
            timeout=20
        ) as client:

            response = (
                await client.get(
                    url
                )
            )

        if (
            response.status_code
            != 200
        ):

            logger.warning(
                (
                    "ESPN primary roster request "
                    "for team %s returned HTTP %s"
                ),
                team_id,
                response.status_code,
            )

            return []

        data = (
            response.json()
        )

        players = []

        for athlete_group in (
            data.get(
                "athletes",
                [],
            )
            or []
        ):

            for item in (
                athlete_group.get(
                    "items",
                    [],
                )
                or []
            ):

                player = (
                    _athlete_to_roster_player(
                        item
                    )
                )

                if player:

                    players.append(
                        player
                    )

        logger.info(
            (
                "ESPN primary roster: "
                "fetched %s players for team %s"
            ),
            len(
                players
            ),
            team_id,
        )

        return players

    except Exception as exc:

        logger.warning(
            (
                "ESPN primary roster fetch "
                "failed for team %s: %s"
            ),
            team_id,
            exc,
        )

        return []


# ============================================================
# ESPN Core API fallback
# ============================================================

async def fetch_espn_nfl_roster_fallback(
    team_id: str,
    season: Optional[int] = None,
) -> list[dict]:
    """
    Fetch team athletes from ESPN Core API.

    Used when ESPN's normal team roster endpoint fails.

    ESPN Core collection:
      /seasons/{season}/teams/{team_id}/athletes?limit=200

    Core API collections may contain either:
      - full athlete objects
      - entries containing a $ref to the athlete resource

    This function handles both.
    """

    if season is None:

        season = (
            _current_nfl_season()
        )

    url = (
        f"{ESPN_CORE_BASE}/"
        f"seasons/{season}/"
        f"teams/{team_id}/athletes"
    )

    try:

        async with httpx.AsyncClient(
            timeout=30
        ) as client:

            response = (
                await client.get(
                    url,
                    params={
                        "limit":
                            200
                    },
                )
            )

            if (
                response.status_code
                != 200
            ):

                logger.warning(
                    (
                        "ESPN Core roster fallback "
                        "for team %s returned HTTP %s"
                    ),
                    team_id,
                    response.status_code,
                )

                return []

            data = (
                response.json()
            )

            items = (
                data.get(
                    "items",
                    [],
                )
                or []
            )

            players = []

            seen_ids = set()

            for index, item in enumerate(
                items,
                start=1,
            ):

                athlete = None

                # --------------------------------------------
                # Full athlete object already embedded
                # --------------------------------------------

                if isinstance(
                    item,
                    dict,
                ) and (
                    item.get(
                        "id"
                    )
                    or
                    item.get(
                        "fullName"
                    )
                    or
                    item.get(
                        "displayName"
                    )
                ):

                    athlete = (
                        item
                    )

                # --------------------------------------------
                # ESPN Core $ref
                # --------------------------------------------

                elif isinstance(
                    item,
                    dict,
                ):

                    athlete_ref = str(
                        item.get(
                            "$ref",
                            "",
                        )
                        or ""
                    ).strip()

                    if athlete_ref:

                        try:

                            athlete_response = (
                                await client.get(
                                    athlete_ref
                                )
                            )

                            if (
                                athlete_response.status_code
                                ==
                                200
                            ):

                                athlete = (
                                    athlete_response.json()
                                )

                            else:

                                logger.debug(
                                    (
                                        "ESPN Core athlete ref "
                                        "returned HTTP %s: %s"
                                    ),
                                    athlete_response.status_code,
                                    athlete_ref,
                                )

                        except Exception as exc:

                            logger.debug(
                                (
                                    "ESPN Core athlete ref "
                                    "failed for team %s "
                                    "item %s: %s"
                                ),
                                team_id,
                                index,
                                exc,
                            )

                if not athlete:

                    continue

                player = (
                    _athlete_to_roster_player(
                        athlete
                    )
                )

                if not player:

                    continue

                player_id = (
                    player[
                        "player_id"
                    ]
                )

                if (
                    player_id
                    in seen_ids
                ):

                    continue

                seen_ids.add(
                    player_id
                )

                players.append(
                    player
                )

        logger.info(
            (
                "ESPN Core fallback: "
                "fetched %s players "
                "for team %s season %s"
            ),
            len(
                players
            ),
            team_id,
            season,
        )

        return players

    except Exception as exc:

        logger.warning(
            (
                "ESPN Core roster fallback "
                "failed for team %s: %s"
            ),
            team_id,
            exc,
        )

        return []


# ============================================================
# Unified ESPN NFL roster fetcher
# ============================================================

async def fetch_espn_nfl_roster(
    team_id: str,
) -> tuple[list[dict], str]:
    """
    Try normal ESPN roster first.

    If it fails or returns no players, automatically use ESPN Core.

    Returns:
      (players, source)

    source:
      "primary"
      "core_fallback"
      "unavailable"
    """

    roster = (
        await fetch_espn_nfl_roster_primary(
            team_id
        )
    )

    if roster:

        return (
            roster,
            "primary",
        )

    logger.warning(
        (
            "ESPN primary roster unavailable "
            "for team %s. Trying Core fallback."
        ),
        team_id,
    )

    fallback_roster = (
        await fetch_espn_nfl_roster_fallback(
            team_id
        )
    )

    if fallback_roster:

        logger.info(
            (
                "ESPN Core fallback succeeded "
                "for team %s with %s players"
            ),
            team_id,
            len(
                fallback_roster
            ),
        )

        return (
            fallback_roster,
            "core_fallback",
        )

    logger.error(
        (
            "Both ESPN roster sources failed "
            "for team %s"
        ),
        team_id,
    )

    return (
        [],
        "unavailable",
    )


# ============================================================
# ESPN CFB roster fetching
# ============================================================

async def fetch_espn_cfb_roster(
    team_id: str,
) -> list[dict]:

    url = (
        f"{ESPN_BASE}/college-football/"
        f"teams/{team_id}/roster"
    )

    try:

        async with httpx.AsyncClient(
            timeout=20
        ) as client:

            response = (
                await client.get(
                    url
                )
            )

        if (
            response.status_code
            != 200
        ):

            return []

        data = (
            response.json()
        )

        players = []

        for athlete_group in (
            data.get(
                "athletes",
                [],
            )
            or []
        ):

            for item in (
                athlete_group.get(
                    "items",
                    [],
                )
                or []
            ):

                espn_id = str(
                    item.get(
                        "id",
                        "",
                    )
                    or ""
                ).strip()

                if not espn_id:

                    continue

                name = str(
                    item.get(
                        "fullName"
                    )
                    or
                    item.get(
                        "displayName"
                    )
                    or ""
                ).strip()

                if not name:

                    continue

                pos = (
                    _extract_position(
                        item
                    )
                )

                players.append({

                    "player_id":
                        f"espn_cfb_{espn_id}",

                    "name":
                        name,

                    "position":
                        pos,

                    "position_group":
                        normalize_position(
                            pos
                        ),

                    "jersey":
                        item.get(
                            "jersey",
                            "",
                        ),

                    "status":
                        "active",
                })

        return players

    except Exception as exc:

        logger.warning(
            (
                "ESPN CFB roster fetch "
                "failed for team %s: %s"
            ),
            team_id,
            exc,
        )

        return []


# ============================================================
# ESPN full NFL roster synchronization
# ============================================================

async def sync_espn_nfl_rosters() -> dict:
    """
    Import all current NFL players into Firestore.

    Performance:
      - team initialized once
      - players imported without per-player recalculation
      - roster adjustment recalculated once per team

    ESPN source:
      - primary Site API first
      - Core API fallback automatically if primary fails

    Existing players:
      - preserve current impact score

    New players:
      - neutral impact score 50
    """

    result = {

        "source":
            "ESPN",

        "teams_found":
            0,

        "teams_synced":
            0,

        "teams_skipped":
            0,

        "primary_teams":
            0,

        "fallback_teams":
            0,

        "players_found":
            0,

        "players_added":
            0,

        "players_updated":
            0,

        "players_moved":
            0,

        "team_recalculations":
            0,

        "errors":
            [],
    }

    teams = (
        await fetch_espn_nfl_teams()
    )

    result[
        "teams_found"
    ] = len(
        teams
    )

    if not teams:

        result[
            "errors"
        ].append(
            "ESPN returned no NFL teams."
        )

        return result

    # --------------------------------------------------------
    # Current Firestore players
    # --------------------------------------------------------

    existing_players = (
        roster_engine.get_all_players()
    )

    existing_by_id = {
        str(
            player.get(
                "player_id"
            )
            or
            player.get(
                "id"
            )
        ):
            player

        for player
        in existing_players

        if (
            player.get(
                "player_id"
            )
            or
            player.get(
                "id"
            )
        )
    }

    logger.info(
        (
            "ESPN roster sync starting: "
            "%s existing Firestore players"
        ),
        len(
            existing_by_id
        ),
    )

    # --------------------------------------------------------
    # Teams
    # --------------------------------------------------------

    for (
        team_index,
        team,
    ) in enumerate(
        teams,
        start=1,
    ):

        team_id = (
            team[
                "id"
            ]
        )

        team_name = (
            team[
                "name"
            ]
        )

        started_at = (
            time.monotonic()
        )

        logger.info(
            (
                "ESPN roster sync team "
                "%s/%s: %s (id=%s)"
            ),
            team_index,
            len(
                teams
            ),
            team_name,
            team_id,
        )

        try:

            (
                roster,
                roster_source,
            ) = (
                await fetch_espn_nfl_roster(
                    team_id
                )
            )

            if not roster:

                logger.warning(
                    (
                        "ESPN returned no usable roster "
                        "for %s"
                    ),
                    team_name,
                )

                result[
                    "teams_skipped"
                ] += 1

                result[
                    "errors"
                ].append(
                    (
                        f"{team_name}: "
                        "all ESPN roster sources unavailable"
                    )
                )

                continue

            if (
                roster_source
                ==
                "core_fallback"
            ):

                result[
                    "fallback_teams"
                ] += 1

            else:

                result[
                    "primary_teams"
                ] += 1

            # ------------------------------------------------
            # Initialize team once
            # ------------------------------------------------

            roster_engine.init_team(
                team_name,
                "NFL",
            )

            result[
                "players_found"
            ] += len(
                roster
            )

            # ------------------------------------------------
            # Players
            # ------------------------------------------------

            for player in roster:

                player_id = (
                    player[
                        "player_id"
                    ]
                )

                try:

                    existing = (
                        existing_by_id.get(
                            player_id
                        )
                    )

                    # ----------------------------------------
                    # Existing ESPN player
                    # ----------------------------------------

                    if existing:

                        old_team = (
                            existing.get(
                                "team"
                            )
                        )

                        if (
                            old_team
                            and
                            old_team
                            !=
                            team_name
                        ):

                            roster_engine.transfer_player(

                                player_id=
                                    player_id,

                                new_team=
                                    team_name,

                                notes=
                                    (
                                        "ESPN automatic "
                                        "roster sync"
                                    ),

                                move_type=
                                    "roster_update",
                            )

                            result[
                                "players_moved"
                            ] += 1

                        impact_score = float(
                            existing.get(
                                "impact_score",
                                50.0,
                            )
                        )

                        roster_engine.add_or_update_player(

                            player_id=
                                player_id,

                            name=
                                player[
                                    "name"
                                ],

                            team=
                                team_name,

                            position_group=
                                player[
                                    "position_group"
                                ],

                            impact_score=
                                impact_score,

                            league=
                                "NFL",

                            notes=
                                (
                                    f"ESPN {roster_source} sync"
                                    f" | status={player['status']}"
                                    f" | jersey={player['jersey']}"
                                ),

                            recalculate=
                                False,

                            ensure_team=
                                False,
                        )

                        result[
                            "players_updated"
                        ] += 1

                        existing_by_id[
                            player_id
                        ] = {

                            **existing,

                            "player_id":
                                player_id,

                            "name":
                                player[
                                    "name"
                                ],

                            "team":
                                team_name,

                            "position_group":
                                player[
                                    "position_group"
                                ],

                            "impact_score":
                                impact_score,

                            "league":
                                "NFL",
                        }

                    # ----------------------------------------
                    # New ESPN player
                    # ----------------------------------------

                    else:

                        roster_engine.add_or_update_player(

                            player_id=
                                player_id,

                            name=
                                player[
                                    "name"
                                ],

                            team=
                                team_name,

                            position_group=
                                player[
                                    "position_group"
                                ],

                            impact_score=
                                50.0,

                            league=
                                "NFL",

                            notes=
                                (
                                    f"ESPN {roster_source} sync"
                                    f" | status={player['status']}"
                                    f" | jersey={player['jersey']}"
                                ),

                            recalculate=
                                False,

                            ensure_team=
                                False,
                        )

                        result[
                            "players_added"
                        ] += 1

                        existing_by_id[
                            player_id
                        ] = {

                            "player_id":
                                player_id,

                            "name":
                                player[
                                    "name"
                                ],

                            "team":
                                team_name,

                            "position_group":
                                player[
                                    "position_group"
                                ],

                            "impact_score":
                                50.0,

                            "league":
                                "NFL",
                        }

                except Exception as exc:

                    message = (
                        f"{team_name} / "
                        f"{player.get('name', player_id)}: "
                        f"{exc}"
                    )

                    logger.exception(
                        "Error syncing player %s",
                        message,
                    )

                    result[
                        "errors"
                    ].append(
                        message
                    )

            # ------------------------------------------------
            # Recalculate once per team
            # ------------------------------------------------

            try:

                roster_engine.recalculate_team_adjustment(
                    team_name
                )

                result[
                    "team_recalculations"
                ] += 1

            except Exception as exc:

                message = (
                    f"{team_name}: "
                    f"team recalculation failed: {exc}"
                )

                logger.exception(
                    message
                )

                result[
                    "errors"
                ].append(
                    message
                )

            result[
                "teams_synced"
            ] += 1

            elapsed = (
                time.monotonic()
                -
                started_at
            )

            logger.info(
                (
                    "ESPN roster sync completed "
                    "%s (%s players, source=%s) "
                    "in %.2fs"
                ),
                team_name,
                len(
                    roster
                ),
                roster_source,
                elapsed,
            )

        except Exception as exc:

            message = (
                f"{team_name}: {exc}"
            )

            logger.exception(
                "Error syncing team %s",
                team_name,
            )

            result[
                "errors"
            ].append(
                message
            )

    logger.info(
        (
            "ESPN roster sync finished: "
            "teams_found=%s, "
            "teams_synced=%s, "
            "teams_skipped=%s, "
            "primary_teams=%s, "
            "fallback_teams=%s, "
            "players_found=%s, "
            "players_added=%s, "
            "players_updated=%s, "
            "players_moved=%s, "
            "recalculations=%s, "
            "errors=%s"
        ),
        result[
            "teams_found"
        ],
        result[
            "teams_synced"
        ],
        result[
            "teams_skipped"
        ],
        result[
            "primary_teams"
        ],
        result[
            "fallback_teams"
        ],
        result[
            "players_found"
        ],
        result[
            "players_added"
        ],
        result[
            "players_updated"
        ],
        result[
            "players_moved"
        ],
        result[
            "team_recalculations"
        ],
        len(
            result[
                "errors"
            ]
        ),
    )

    return result


# ============================================================
# SportsData.io adapter
# ============================================================

async def fetch_sportsdata_nfl_transactions(
    season: Optional[str] = None,
) -> list[dict]:

    if not SPORTSDATA_KEY:

        return []

    if season is None:

        season = (
            "2026"
        )

    url = (
        "https://api.sportsdata.io/v3/nfl/"
        f"scores/json/Transactions/{season}"
    )

    try:

        async with httpx.AsyncClient(
            timeout=20
        ) as client:

            response = (
                await client.get(

                    url,

                    headers={
                        "Ocp-Apim-Subscription-Key":
                            SPORTSDATA_KEY
                    },
                )
            )

        if (
            response.status_code
            != 200
        ):

            logger.warning(
                (
                    "SportsData returned HTTP %s"
                ),
                response.status_code,
            )

            return []

        data = (
            response.json()
        )

        moves = []

        for transaction in data:

            move_type = (
                transaction.get(
                    "TransactionType"
                )
            )

            if move_type not in (
                "Signed",
                "Trade",
                "Released",
                "Waived",
            ):

                continue

            moves.append({

                "player_id":
                    (
                        "sd_"
                        f"{transaction.get('PlayerID', '')}"
                    ),

                "name":
                    (
                        f"{transaction.get('FirstName', '')} "
                        f"{transaction.get('LastName', '')}"
                    ).strip(),

                "from_team":
                    transaction.get(
                        "PreviousTeam",
                        "",
                    ),

                "to_team":
                    transaction.get(
                        "Team",
                        "",
                    ),

                "position":
                    transaction.get(
                        "Position",
                        "",
                    ),

                "move_type":
                    move_type,

                "date":
                    transaction.get(
                        "Date",
                        "",
                    ),
            })

        return moves

    except Exception as exc:

        logger.error(
            (
                "SportsData transaction "
                "fetch error: %s"
            ),
            exc,
        )

        return []


async def fetch_sportsdata_cfb_transfers(
    season: Optional[str] = None,
) -> list[dict]:

    if not SPORTSDATA_KEY:

        return []

    if season is None:

        season = (
            "2026"
        )

    url = (
        "https://api.sportsdata.io/v3/cfb/"
        f"scores/json/Transfers/{season}"
    )

    try:

        async with httpx.AsyncClient(
            timeout=20
        ) as client:

            response = (
                await client.get(

                    url,

                    headers={
                        "Ocp-Apim-Subscription-Key":
                            SPORTSDATA_KEY
                    },
                )
            )

        if (
            response.status_code
            != 200
        ):

            return []

        data = (
            response.json()
        )

        return [
            {

                "player_id":
                    (
                        "sd_cfb_"
                        f"{item.get('PlayerID', '')}"
                    ),

                "name":
                    (
                        f"{item.get('FirstName', '')} "
                        f"{item.get('LastName', '')}"
                    ).strip(),

                "from_team":
                    item.get(
                        "PreviousSchool",
                        "",
                    ),

                "to_team":
                    item.get(
                        "School",
                        "",
                    ),

                "position":
                    item.get(
                        "Position",
                        "",
                    ),

                "move_type":
                    "transfer_portal",

                "date":
                    item.get(
                        "TransferDate",
                        "",
                    ),

                "stars":
                    item.get(
                        "Stars",
                        0,
                    ),
            }

            for item
            in data
        ]

    except Exception as exc:

        logger.error(
            (
                "SportsData CFB "
                "transfer error: %s"
            ),
            exc,
        )

        return []


# ============================================================
# MySportsFeeds adapter
# ============================================================

async def fetch_msf_nfl_roster_moves(
    season: str = "2025-2026-regular",
) -> list[dict]:

    if (
        not MSF_KEY
        or
        not MSF_PASSWORD
    ):

        return []

    try:

        import base64

        creds = (
            base64.b64encode(
                (
                    f"{MSF_KEY}:"
                    f"{MSF_PASSWORD}"
                ).encode()
            )
            .decode()
        )

        url = (
            "https://api.mysportsfeeds.com/v2.1/"
            f"pull/nfl/{season}/transactions.json"
        )

        async with httpx.AsyncClient(
            timeout=20
        ) as client:

            response = (
                await client.get(

                    url,

                    headers={
                        "Authorization":
                            f"Basic {creds}"
                    },
                )
            )

        if (
            response.status_code
            != 200
        ):

            return []

        data = (
            response.json()
        )

        moves = []

        for transaction in (
            data.get(
                "transactions",
                [],
            )
            or []
        ):

            player = (
                transaction.get(
                    "player",
                    {},
                )
                or {}
            )

            moves.append({

                "player_id":
                    (
                        "msf_"
                        f"{player.get('id', '')}"
                    ),

                "name":
                    player.get(
                        "fullName",
                        "",
                    ),

                "from_team":
                    (
                        transaction.get(
                            "fromTeam",
                            {},
                        )
                        or {}
                    ).get(
                        "abbreviation",
                        "",
                    ),

                "to_team":
                    (
                        transaction.get(
                            "toTeam",
                            {},
                        )
                        or {}
                    ).get(
                        "abbreviation",
                        "",
                    ),

                "position":
                    player.get(
                        "primaryPosition",
                        "",
                    ),

                "move_type":
                    transaction.get(
                        "transactionType",
                        "",
                    ),

                "date":
                    transaction.get(
                        "updatedOn",
                        "",
                    ),
            })

        return moves

    except Exception as exc:

        logger.error(
            (
                "MySportsFeeds error: %s"
            ),
            exc,
        )

        return []


# ============================================================
# Unified ingestion
# ============================================================

async def ingest_player_moves(
    league: str = "NFL",
) -> dict:

    league = (
        league.upper()
    )

    result = {

        "source":
            [],

        "moves_processed":
            0,

        "errors":
            [],
    }

    # --------------------------------------------------------
    # ESPN
    # --------------------------------------------------------

    if league == "NFL":

        espn_result = (
            await sync_espn_nfl_rosters()
        )

        result[
            "source"
        ].append(
            "ESPN"
        )

        result[
            "espn"
        ] = (
            espn_result
        )

        result[
            "moves_processed"
        ] += (
            espn_result.get(
                "players_moved",
                0,
            )
        )

        result[
            "errors"
        ].extend(
            espn_result.get(
                "errors",
                [],
            )
        )

    # --------------------------------------------------------
    # SportsData
    # --------------------------------------------------------

    if SPORTSDATA_KEY:

        result[
            "source"
        ].append(
            "SportsData.io"
        )

        try:

            if league == "NFL":

                paid_moves = (
                    await
                    fetch_sportsdata_nfl_transactions()
                )

            else:

                paid_moves = (
                    await
                    fetch_sportsdata_cfb_transfers()
                )

            result[
                "sportsdata_transactions"
            ] = len(
                paid_moves
            )

        except Exception as exc:

            result[
                "errors"
            ].append(
                f"SportsData error: {exc}"
            )

    # --------------------------------------------------------
    # MySportsFeeds
    # --------------------------------------------------------

    elif (
        MSF_KEY
        and
        MSF_PASSWORD
    ):

        result[
            "source"
        ].append(
            "MySportsFeeds"
        )

        try:

            msf_moves = (
                await
                fetch_msf_nfl_roster_moves()
            )

            result[
                "mysportsfeeds_transactions"
            ] = len(
                msf_moves
            )

        except Exception as exc:

            result[
                "errors"
            ].append(
                (
                    "MySportsFeeds error: "
                    f"{exc}"
                )
            )

    if league != "NFL":

        result[
            "note"
        ] = (
            "Automatic ESPN full-roster sync is "
            "currently enabled for NFL only."
        )

    return result


# ============================================================
# Data source status
# ============================================================

def get_data_source_status() -> dict:

    if SPORTSDATA_KEY:

        active = (
            "ESPN + SportsData.io"
        )

    elif (
        MSF_KEY
        and
        MSF_PASSWORD
    ):

        active = (
            "ESPN + MySportsFeeds"
        )

    else:

        active = (
            "ESPN (free) + Manual"
        )

    return {

        "sportsdata_io":
            bool(
                SPORTSDATA_KEY
            ),

        "mysportsfeeds":
            bool(
                MSF_KEY
                and
                MSF_PASSWORD
            ),

        "manual_entry":
            True,

        "espn_free":
            True,

        "espn_auto_sync":
            True,

        "espn_core_fallback":
            True,

        "active_source":
            active,
    }
