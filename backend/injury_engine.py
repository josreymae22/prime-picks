"""
injury_engine.py

Pulls injury/status reports from ESPN for NFL and CFB.
Applies depth chart cascade: if starter is out, backup fills in.
Produces injury-adjusted team rating modifiers that feed into predictions.

Status multipliers (applied to player's impact_score):
  Active       → 1.00 (full contribution)
  Questionable → 0.80 (likely plays, but reduced snap count risk)
  Doubtful     → 0.40 (unlikely to play)
  Out          → 0.00 (does not play)
  IR / PUP     → 0.00 (season-ending)
  Suspended    → 0.00

Depth chart cascade:
  If a starter is Out/IR, their lower-impact backup fills in.
  Net effect: group rating drops by (starter_impact - backup_impact) * group_weight
  This is the "that safety just left, their secondary is cooked" model.
"""

import httpx
import asyncio
import json
import os
import logging

from typing import Optional
from datetime import datetime, timezone


logger = logging.getLogger(__name__)


ESPN_BASE = (
    "https://site.api.espn.com/apis/site/v2/"
    "sports/football"
)

INJURY_DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "injury_db.json",
)


STATUS_MULTIPLIERS = {
    "active": 1.00,
    "probable": 0.95,
    "questionable": 0.80,
    "doubtful": 0.40,
    "out": 0.00,
    "ir": 0.00,
    "pup": 0.00,
    "suspended": 0.00,
    "day-to-day": 0.85,
    "injured reserve": 0.00,
}


# Backup quality assumption when no backup is in DB.
# Expressed as fraction of starter's impact score.
BACKUP_QUALITY_FACTOR = 0.52


def _load_injury_db() -> dict:

    if os.path.exists(
        INJURY_DB_PATH
    ):

        with open(
            INJURY_DB_PATH,
            "r",
        ) as f:

            return json.load(
                f
            )

    return {
        "injuries": {},
        "last_updated": None,
    }


def _save_injury_db(
    db: dict,
):

    db[
        "last_updated"
    ] = (
        datetime.utcnow()
        .isoformat()
    )

    with open(
        INJURY_DB_PATH,
        "w",
    ) as f:

        json.dump(
            db,
            f,
            indent=2,
        )


class InjuryEngine:

    def __init__(
        self,
    ):

        self.db = (
            _load_injury_db()
        )


    def reload(
        self,
    ):

        self.db = (
            _load_injury_db()
        )


    # ──────────────────────────────────────────
    # ESPN Fetchers
    # ──────────────────────────────────────────

    async def fetch_nfl_injuries(
        self,
    ) -> dict:
        """
        Pull current NFL injury report from ESPN.

        Current ESPN aggregate response structure:

            injuries[]
                id
                displayName
                injuries[]
                    id
                    status
                    athlete
                    longComment
                    shortComment

        Returns:
            dict keyed by team display name
            -> list of injured players.
        """

        url = (
            f"{ESPN_BASE}/nfl/injuries"
        )

        try:

            async with httpx.AsyncClient(
                timeout=15
            ) as client:

                r = await client.get(
                    url
                )

                if (
                    r.status_code
                    != 200
                ):

                    logger.warning(
                        (
                            "NFL injury endpoint "
                            "returned %s"
                        ),
                        r.status_code,
                    )

                    return {}

                data = (
                    r.json()
                )

            injuries = {}

            # ------------------------------------------------
            # ESPN returns one group per NFL team.
            #
            # Team name is directly on:
            #
            #   team_entry["displayName"]
            #
            # not:
            #
            #   team_entry["team"]["displayName"]
            # ------------------------------------------------

            for team_entry in (
                data.get(
                    "injuries",
                    [],
                )
                or []
            ):

                team_name = str(
                    team_entry.get(
                        "displayName",
                        "",
                    )
                    or ""
                ).strip()

                team_id = str(
                    team_entry.get(
                        "id",
                        "",
                    )
                    or ""
                ).strip()

                if not team_name:

                    logger.warning(
                        (
                            "Skipping ESPN NFL "
                            "injury group with no "
                            "displayName. team_id=%s"
                        ),
                        team_id,
                    )

                    continue

                team_injuries = []

                # --------------------------------------------
                # Individual injury records.
                # --------------------------------------------

                for inj in (
                    team_entry.get(
                        "injuries",
                        [],
                    )
                    or []
                ):

                    athlete = (
                        inj.get(
                            "athlete",
                            {},
                        )
                        or {}
                    )

                    # ----------------------------------------
                    # Player name
                    # ----------------------------------------

                    player_name = str(
                        athlete.get(
                            "displayName",
                            "",
                        )
                        or ""
                    ).strip()

                    if not player_name:

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

                        player_name = (
                            f"{first_name} "
                            f"{last_name}"
                        ).strip()

                    # ----------------------------------------
                    # Player ID
                    #
                    # Prefer athlete ID because roster imports
                    # use ESPN athlete IDs.
                    #
                    # If athlete ID is missing, fall back to
                    # injury record ID so we never create
                    # an empty "espn_" ID.
                    # ----------------------------------------

                    athlete_id = str(
                        athlete.get(
                            "id",
                            "",
                        )
                        or ""
                    ).strip()

                    injury_record_id = str(
                        inj.get(
                            "id",
                            "",
                        )
                        or ""
                    ).strip()

                    if athlete_id:

                        player_id = (
                            f"espn_nfl_{athlete_id}"
                        )

                    elif injury_record_id:

                        player_id = (
                            "espn_injury_"
                            f"{injury_record_id}"
                        )

                    else:

                        safe_name = (
                            player_name
                            .lower()
                            .replace(
                                " ",
                                "_",
                            )
                        )

                        player_id = (
                            "espn_name_"
                            f"{safe_name}"
                        )

                    # ----------------------------------------
                    # Status
                    #
                    # Current ESPN response exposes this
                    # directly on the injury record.
                    # ----------------------------------------

                    status_value = (
                        inj.get(
                            "status"
                        )
                    )

                    if (
                        status_value
                        is None
                    ):

                        status_raw = (
                            "active"
                        )

                    else:

                        status_raw = (
                            str(
                                status_value
                            )
                            .strip()
                            .lower()
                        )

                    status = (
                        self._normalize_status(
                            status_raw
                        )
                    )

                    # ----------------------------------------
                    # Position
                    # ----------------------------------------

                    position_data = (
                        athlete.get(
                            "position",
                            {},
                        )
                        or {}
                    )

                    if isinstance(
                        position_data,
                        dict,
                    ):

                        pos = str(
                            position_data.get(
                                "abbreviation",
                                "",
                            )
                            or ""
                        ).strip()

                    else:

                        pos = str(
                            position_data
                            or ""
                        ).strip()

                    # ----------------------------------------
                    # Description
                    # ----------------------------------------

                    description = str(
                        inj.get(
                            "longComment",
                            "",
                        )
                        or
                        inj.get(
                            "shortComment",
                            "",
                        )
                        or ""
                    ).strip()

                    # ----------------------------------------
                    # Return date
                    # ----------------------------------------

                    return_date = str(
                        inj.get(
                            "returnDate",
                            "",
                        )
                        or ""
                    ).strip()

                    # ----------------------------------------
                    # Store normalized record
                    # ----------------------------------------

                    team_injuries.append({
                        "player_id":
                            player_id,

                        "name":
                            player_name,

                        "position":
                            pos,

                        "status":
                            status,

                        "status_raw":
                            status_raw,

                        "description":
                            description,

                        "return_date":
                            return_date,

                        "fetched_at":
                            datetime.utcnow()
                            .isoformat(),
                    })

                if team_injuries:

                    injuries[
                        team_name
                    ] = (
                        team_injuries
                    )

            total_players = sum(
                len(
                    players
                )
                for players
                in injuries.values()
            )

            logger.info(
                (
                    "NFL injuries fetched: "
                    "%s players across %s teams"
                ),
                total_players,
                len(
                    injuries
                ),
            )

            logger.info(
                "NFL injury teams: %s",
                sorted(
                    injuries.keys()
                ),
            )

            return injuries

        except Exception as e:

            logger.error(
                "NFL injury fetch error: %s",
                e,
                exc_info=True,
            )

            return {}


    async def fetch_cfb_injuries(
        self,
        team_ids: Optional[list] = None,
    ) -> dict:
        """
        Pull CFB injuries from ESPN for specific teams.

        ESPN's CFB injury endpoint requires
        team-by-team queries.
        """

        if not team_ids:

            return {}

        injuries = {}

        for team_id in (
            team_ids[
                :30
            ]
        ):

            try:

                url = (
                    f"{ESPN_BASE}/college-football/"
                    f"teams/{team_id}/injuries"
                )

                async with httpx.AsyncClient(
                    timeout=10
                ) as client:

                    r = await client.get(
                        url
                    )

                    if (
                        r.status_code
                        != 200
                    ):

                        continue

                    data = (
                        r.json()
                    )

                team_name = ""
                team_injuries = []

                for inj in (
                    data.get(
                        "injuries",
                        [],
                    )
                ):

                    athlete = (
                        inj.get(
                            "athlete",
                            {},
                        )
                    )

                    if not team_name:

                        team_name = (
                            inj.get(
                                "team",
                                {},
                            )
                            .get(
                                "displayName",
                                f"team_{team_id}",
                            )
                        )

                    status_raw = (
                        inj.get(
                            "status",
                            "Active",
                        )
                        .lower()
                    )

                    status = (
                        self._normalize_status(
                            status_raw
                        )
                    )

                    pos = (
                        athlete.get(
                            "position",
                            {},
                        )
                        .get(
                            "abbreviation",
                            "",
                        )
                    )

                    team_injuries.append({
                        "player_id":
                            (
                                "espn_cfb_"
                                f"{athlete.get('id', '')}"
                            ),

                        "name":
                            athlete.get(
                                "displayName",
                                "",
                            ),

                        "position":
                            pos,

                        "status":
                            status,

                        "status_raw":
                            status_raw,

                        "description":
                            inj.get(
                                "longComment",
                                "",
                            ),

                        "fetched_at":
                            datetime.utcnow()
                            .isoformat(),
                    })

                if (
                    team_injuries
                    and team_name
                ):

                    injuries[
                        team_name
                    ] = (
                        team_injuries
                    )

                await asyncio.sleep(
                    0.1
                )

            except Exception as e:

                logger.warning(
                    (
                        "CFB injury fetch error "
                        "for team %s: %s"
                    ),
                    team_id,
                    e,
                )

        return injuries


    def _normalize_status(
        self,
        raw: str,
    ) -> str:

        raw = (
            raw.lower()
            .strip()
        )

        for key in (
            STATUS_MULTIPLIERS
        ):

            if key in raw:

                return key

        return "active"


    # ──────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────

    def update_injuries(
        self,
        injuries: dict,
        league: str,
    ):
        """
        Replace the current injury snapshot for one league.

        This intentionally clears older entries for the same league
        before writing the fresh ESPN snapshot. That prevents stale
        malformed team keys (for example an old "-" team) from
        surviving after the parser has been fixed.
        """

        league = (
            league.upper()
        )

        existing = (
            self.db.get(
                "injuries",
                {},
            )
            or {}
        )

        self.db[
            "injuries"
        ] = {
            key:
                value

            for (
                key,
                value,
            ) in existing.items()

            if not str(
                key
            ).startswith(
                f"{league}:"
            )
        }

        for (
            team,
            players,
        ) in injuries.items():

            key = (
                f"{league}:{team}"
            )

            self.db[
                "injuries"
            ][
                key
            ] = {
                "team":
                    team,

                "league":
                    league,

                "players":
                    players,

                "updated_at":
                    datetime.utcnow()
                    .isoformat(),
            }

        _save_injury_db(
            self.db
        )


    def get_team_injuries(
        self,
        team_name: str,
        league: str = "NFL",
    ) -> list:

        key = (
            f"{league}:{team_name}"
        )

        entry = (
            self.db[
                "injuries"
            ].get(
                key,
                {},
            )
        )

        return (
            entry.get(
                "players",
                [],
            )
        )


    def get_all_injuries(
        self,
        league: Optional[str] = None,
    ) -> dict:

        result = {}

        for (
            key,
            val,
        ) in (
            self.db[
                "injuries"
            ].items()
        ):

            if (
                league
                and
                not key.startswith(
                    f"{league}:"
                )
            ):

                continue

            result[
                val[
                    "team"
                ]
            ] = (
                val[
                    "players"
                ]
            )

        return result


    # ──────────────────────────────────────────
    # Rating adjustment with depth chart cascade
    # ──────────────────────────────────────────

    def get_injury_adjustment(
        self,
        team_name: str,
        league: str,
        roster_engine,
    ) -> dict:
        """
        Calculate injury adjustment using the persisted roster DB.

        This compatibility wrapper is used by the live card engine.
        """

        team_players = (
            roster_engine.get_all_players(
                team=team_name
            )
        )

        return (
            self.calculate_injury_adjustment_from_players(
                team_name=team_name,
                league=league,
                team_players=team_players,
            )
        )


    def calculate_injury_adjustment_from_players(
        self,
        team_name: str,
        league: str,
        team_players: list,
    ) -> dict:
        """
        Calculate injury adjustment from a supplied in-memory roster.

        This is useful for read-only ON-vs-OFF testing because the
        caller can supply proposed ESPN depth-chart players without
        reading or writing Firestore.

        Expected player fields:
          player_id
          name
          position_group
          impact_score

        Optional:
          role
          depth_order
        """

        injuries = (
            self.get_team_injuries(
                team_name,
                league,
            )
        )

        if not injuries:

            return {
                "adjustment":
                    0.0,

                "affected_players":
                    [],

                "depth_chart_cascades":
                    [],
            }

        player_lookup = {
            str(
                p.get(
                    "player_id",
                    "",
                )
            ):
                p

            for p
            in team_players

            if p.get(
                "player_id"
            )
        }

        name_lookup = {
            str(
                p.get(
                    "name",
                    "",
                )
            ).lower():
                p

            for p
            in team_players

            if p.get(
                "name"
            )
        }

        affected = []
        cascades = []
        total_adjustment = 0.0

        from player_events import (
            POSITION_TO_GROUP
        )

        from roster_engine import (
            POSITION_GROUPS
        )

        for inj in injuries:

            status = (
                inj.get(
                    "status",
                    "active",
                )
            )

            multiplier = (
                STATUS_MULTIPLIERS.get(
                    status,
                    1.0,
                )
            )

            if (
                multiplier
                == 1.0
            ):
                continue

            injury_player_id = str(
                inj.get(
                    "player_id",
                    "",
                )
                or ""
            )

            injury_name = str(
                inj.get(
                    "name",
                    "",
                )
                or ""
            )

            roster_player = (
                player_lookup.get(
                    injury_player_id
                )
                or
                name_lookup.get(
                    injury_name.lower()
                )
            )

            if not roster_player:

                pos_group = (
                    POSITION_TO_GROUP.get(
                        str(
                            inj.get(
                                "position",
                                "",
                            )
                        ).upper(),
                        "LB",
                    )
                )

                roster_player = {
                    "player_id":
                        injury_player_id,

                    "name":
                        injury_name,

                    "position_group":
                        pos_group,

                    "impact_score":
                        62.0,

                    "estimated":
                        True,
                }

            impact = float(
                roster_player.get(
                    "impact_score",
                    50.0,
                )
                or 50.0
            )

            pos_group = (
                roster_player.get(
                    "position_group",
                    "LB",
                )
            )

            group_weight = float(
                POSITION_GROUPS.get(
                    pos_group,
                    {},
                )
                .get(
                    "weight",
                    0.1,
                )
            )

            effective_impact = (
                impact
                *
                multiplier
            )

            impact_loss = (
                impact
                -
                effective_impact
            )

            backup = (
                self._find_backup(
                    team_players=team_players,
                    pos_group=pos_group,
                    starter_impact=impact,
                    exclude_id=injury_player_id,
                )
            )

            backup_impact = (
                float(
                    backup.get(
                        "impact_score",
                        50.0,
                    )
                )
                if backup
                else
                impact
                *
                BACKUP_QUALITY_FACTOR
            )

            backup_name = (
                backup.get(
                    "name",
                    "Depth player",
                )
                if backup
                else
                "Depth player"
            )

            if (
                multiplier
                == 0.0
            ):

                net_loss = (
                    (
                        impact
                        -
                        backup_impact
                    )
                    *
                    group_weight
                    *
                    0.1
                )

                cascades.append({
                    "starter_out":
                        injury_name,

                    "starter_impact":
                        round(
                            impact,
                            1,
                        ),

                    "backup_in":
                        backup_name,

                    "backup_impact":
                        round(
                            backup_impact,
                            1,
                        ),

                    "position_group":
                        pos_group,

                    "rating_impact":
                        round(
                            -net_loss,
                            3,
                        ),
                })

                total_adjustment -= (
                    net_loss
                )

            else:

                net_loss = (
                    impact_loss
                    *
                    group_weight
                    *
                    0.1
                )

                total_adjustment -= (
                    net_loss
                )

            affected.append({
                "name":
                    injury_name,

                "player_id":
                    injury_player_id,

                "matched_roster_player_id":
                    roster_player.get(
                        "player_id"
                    ),

                "status":
                    status,

                "position_group":
                    pos_group,

                "impact_score":
                    round(
                        impact,
                        1,
                    ),

                "multiplier":
                    multiplier,

                "description":
                    inj.get(
                        "description",
                        "",
                    ),

                "estimated":
                    roster_player.get(
                        "estimated",
                        False,
                    ),
            })

        return {
            "adjustment":
                round(
                    total_adjustment,
                    3,
                ),

            "affected_players":
                affected,

            "depth_chart_cascades":
                cascades,
        }


    def _find_backup(
        self,
        team_players: list,
        pos_group: str,
        starter_impact: float,
        exclude_id: str,
    ) -> Optional[dict]:
        """
        Find the best available lower-impact player in the same group.

        For proposed depth-chart scores this naturally prefers a
        40-point backup behind a 65-point starter.
        """

        candidates = [
            p

            for p
            in team_players

            if (
                p.get(
                    "position_group"
                )
                ==
                pos_group

                and
                float(
                    p.get(
                        "impact_score",
                        0.0,
                    )
                    or 0.0
                )
                <
                starter_impact

                and
                str(
                    p.get(
                        "player_id",
                        "",
                    )
                )
                !=
                str(
                    exclude_id
                )
            )
        ]

        if not candidates:

            return None

        return max(
            candidates,
            key=lambda p:
                float(
                    p.get(
                        "impact_score",
                        0.0,
                    )
                    or 0.0
                ),
        )


    def get_status_summary(
        self,
        league: str = "NFL",
    ) -> dict:
        """
        Quick summary of injury statuses across league.
        """

        all_inj = (
            self.get_all_injuries(
                league=league
            )
        )

        summary = {
            "out": [],
            "doubtful": [],
            "questionable": [],
        }

        for (
            team,
            players,
        ) in all_inj.items():

            for p in players:

                s = (
                    p.get(
                        "status",
                        "active",
                    )
                )

                if s in summary:

                    summary[
                        s
                    ].append({
                        "name":
                            p[
                                "name"
                            ],

                        "team":
                            team,

                        "position":
                            p.get(
                                "position",
                                "",
                            ),
                    })

        return summary


# Singleton
injury_engine = InjuryEngine()
