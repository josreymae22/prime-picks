"""
roster_engine.py

Manages team rosters and positional group ratings using Firestore.

Also exposes position-level model contribution data so Prime Picks can
explain why it favors a specific side.

Examples:
- QB Advantage
- WR Advantage
- Offensive Line Advantage
- Defensive Front Advantage
- Secondary Advantage

The explanation math uses the SAME weights that already drive the team's
roster_adjustment, so explanation values stay consistent with the model.
"""

import os
import logging

from typing import Optional
from datetime import datetime

import firebase_admin

from firebase_admin import (
    credentials,
    firestore,
)


logger = logging.getLogger(
    __name__
)


# ============================================================
# Firebase Admin Init
# ============================================================

if not firebase_admin._apps:

    cred = credentials.Certificate({
        "type":
            "service_account",

        "project_id":
            os.getenv(
                "FIREBASE_PROJECT_ID"
            ),

        "private_key":
            os.getenv(
                "FIREBASE_PRIVATE_KEY",
                "",
            ).replace(
                "\\n",
                "\n",
            ),

        "client_email":
            os.getenv(
                "FIREBASE_CLIENT_EMAIL"
            ),

        "token_uri":
            "https://oauth2.googleapis.com/token",
    })

    firebase_admin.initialize_app(
        cred
    )


db = firestore.client()


# ============================================================
# Collections
# ============================================================

PLAYERS_COLLECTION = (
    "roster_players"
)

TEAMS_COLLECTION = (
    "roster_teams"
)

MOVES_COLLECTION = (
    "roster_moves"
)


# ============================================================
# Position groups
#
# These weights already power the team's roster_adjustment.
# They now also power explanation-level point contributions.
# ============================================================

POSITION_GROUPS = {

    "QB": {
        "side":
            "offense",

        "weight":
            0.35,
    },

    "RB": {
        "side":
            "offense",

        "weight":
            0.10,
    },

    "WR": {
        "side":
            "offense",

        "weight":
            0.15,
    },

    "TE": {
        "side":
            "offense",

        "weight":
            0.08,
    },

    "OL": {
        "side":
            "offense",

        "weight":
            0.12,
    },

    "DL": {
        "side":
            "defense",

        "weight":
            0.20,
    },

    "LB": {
        "side":
            "defense",

        "weight":
            0.15,
    },

    "CB": {
        "side":
            "defense",

        "weight":
            0.18,
    },

    "S": {
        "side":
            "defense",

        "weight":
            0.12,
    },

    "K": {
        "side":
            "special",

        "weight":
            0.03,
    },
}


# ============================================================
# Impact tiers
# ============================================================

IMPACT_TIERS = {

    "elite":
        (
            85,
            100,
        ),

    "good":
        (
            70,
            84,
        ),

    "average":
        (
            50,
            69,
        ),

    "backup":
        (
            30,
            49,
        ),

    "practice":
        (
            0,
            29,
        ),
}


# ============================================================
# Human-readable position labels
# ============================================================

POSITION_LABELS = {

    "QB":
        "Quarterback",

    "RB":
        "Running Back",

    "WR":
        "Wide Receiver",

    "TE":
        "Tight End",

    "OL":
        "Offensive Line",

    "DL":
        "Defensive Line",

    "LB":
        "Linebacker",

    "CB":
        "Cornerback",

    "S":
        "Safety",

    "K":
        "Kicker",
}


# ============================================================
# Helpers
# ============================================================

def _now_iso() -> str:

    return (
        datetime.utcnow()
        .isoformat()
    )


def _slugify(
    value: str,
) -> str:

    return (
        value
        .strip()
        .lower()
        .replace(
            " ",
            "-",
        )
    )


def _impact_tier(
    score: float,
) -> str:

    value = float(
        score
    )

    for (
        tier,
        bounds,
    ) in IMPACT_TIERS.items():

        low, high = (
            bounds
        )

        if (
            value >= low
            and
            value <= high
        ):

            return tier

    return "average"


# ============================================================
# Roster Engine
# ============================================================

class RosterEngine:

    # --------------------------------------------------------
    # Compatibility reload hook
    # --------------------------------------------------------

    def reload(
        self,
    ):

        return None


    # --------------------------------------------------------
    # Firestore refs
    # --------------------------------------------------------

    def _team_ref(
        self,
        team_name: str,
    ):

        return (
            db
            .collection(
                TEAMS_COLLECTION
            )
            .document(
                team_name
            )
        )


    def _player_ref(
        self,
        player_id: str,
    ):

        return (
            db
            .collection(
                PLAYERS_COLLECTION
            )
            .document(
                player_id
            )
        )


    # ========================================================
    # Team initialization
    # ========================================================

    def init_team(
        self,
        team_name: str,
        league: str,
        base_sp: float = 0.0,
    ):

        ref = (
            self._team_ref(
                team_name
            )
        )

        if not ref.get().exists:

            ref.set({

                "name":
                    team_name,

                "league":
                    league,

                "base_sp":
                    base_sp,

                "groups": {
                    group_name: {
                        "rating":
                            50.0,

                        "players":
                            [],
                    }
                    for group_name
                    in POSITION_GROUPS
                },

                "roster_adjustment":
                    0.0,

                "updated_at":
                    _now_iso(),
            })


    # ========================================================
    # Base rating
    # ========================================================

    def set_team_base_rating(
        self,
        team_name: str,
        base_sp: float,
    ):

        ref = (
            self._team_ref(
                team_name
            )
        )

        if ref.get().exists:

            ref.update({

                "base_sp":
                    base_sp,

                "updated_at":
                    _now_iso(),
            })


    # ========================================================
    # Player lookup
    # ========================================================

    def _find_player_key(
        self,
        player_id_or_name: str,
    ) -> Optional[str]:

        lookup = (
            player_id_or_name
            .strip()
            .lower()
        )

        slug = (
            _slugify(
                player_id_or_name
            )
        )


        # Exact Firestore document ID
        if (
            self
            ._player_ref(
                player_id_or_name
            )
            .get()
            .exists
        ):

            return (
                player_id_or_name
            )


        # Slug document ID
        if (
            self
            ._player_ref(
                slug
            )
            .get()
            .exists
        ):

            return slug


        # Name search
        docs = (
            db
            .collection(
                PLAYERS_COLLECTION
            )
            .stream()
        )

        for doc in docs:

            player = (
                doc.to_dict()
            )

            if (
                player
                .get(
                    "name",
                    "",
                )
                .strip()
                .lower()
                ==
                lookup
            ):

                return (
                    doc.id
                )


        return None


    # ========================================================
    # Add / Update Player
    # ========================================================

    def add_or_update_player(
        self,
        player_id: str,
        name: str,
        team: str,
        position_group: str,
        impact_score: float,
        league: str,
        notes: str = "",
    ) -> dict:

        if not player_id:

            player_id = (
                _slugify(
                    name
                )
            )


        position_group = (
            position_group
            .upper()
            .strip()
        )


        if (
            position_group
            not in POSITION_GROUPS
        ):

            raise ValueError(
                f"Unsupported position group: {position_group}"
            )


        self.init_team(
            team,
            league,
        )


        player = {

            "player_id":
                player_id,

            "name":
                name,

            "team":
                team,

            "position_group":
                position_group,

            "impact_score":
                float(
                    impact_score
                ),

            "league":
                league,

            "notes":
                notes,

            "updated_at":
                _now_iso(),
        }


        self._player_ref(
            player_id
        ).set(
            player
        )


        team_doc = (
            self
            ._team_ref(
                team
            )
            .get()
            .to_dict()
            or {}
        )


        groups = (
            team_doc.get(
                "groups",
                {
                    group_name: {
                        "rating":
                            50.0,

                        "players":
                            [],
                    }
                    for group_name
                    in POSITION_GROUPS
                },
            )
        )


        if (
            position_group
            in groups
        ):

            players = (
                groups[
                    position_group
                ]
                .get(
                    "players",
                    [],
                )
            )


            if (
                player_id
                not in players
            ):

                players.append(
                    player_id
                )


            groups[
                position_group
            ][
                "players"
            ] = players


        self._team_ref(
            team
        ).update({

            "groups":
                groups,

            "updated_at":
                _now_iso(),
        })


        self._recalculate_team_adjustment(
            team
        )


        return player


    # ========================================================
    # Transfer Player
    # ========================================================

    def transfer_player(
        self,
        player_id: str,
        new_team: str,
        notes: str = "",
        move_type: str = "trade",
    ) -> dict:

        real_player_key = (
            self._find_player_key(
                player_id
            )
        )


        if not real_player_key:

            raise ValueError(
                f"Player {player_id} not found"
            )


        player_ref = (
            self._player_ref(
                real_player_key
            )
        )


        player = (
            player_ref
            .get()
            .to_dict()
        )


        old_team = (
            player[
                "team"
            ]
        )


        old_group = (
            player[
                "position_group"
            ]
        )


        self.init_team(
            new_team,
            player[
                "league"
            ],
        )


        # ----------------------------------------------------
        # Remove from old team
        # ----------------------------------------------------

        old_team_doc = (
            self
            ._team_ref(
                old_team
            )
            .get()
        )


        if old_team_doc.exists:

            old_team_data = (
                old_team_doc
                .to_dict()
            )


            old_groups = (
                old_team_data
                .get(
                    "groups",
                    {},
                )
            )


            if (
                old_group
                in old_groups
            ):

                old_players = (
                    old_groups[
                        old_group
                    ]
                    .get(
                        "players",
                        [],
                    )
                )


                old_groups[
                    old_group
                ][
                    "players"
                ] = [
                    pid
                    for pid
                    in old_players
                    if pid
                    !=
                    real_player_key
                ]


            self._team_ref(
                old_team
            ).update({

                "groups":
                    old_groups,

                "updated_at":
                    _now_iso(),
            })


        # ----------------------------------------------------
        # Add to new team
        # ----------------------------------------------------

        new_team_doc = (
            self
            ._team_ref(
                new_team
            )
            .get()
            .to_dict()
            or {}
        )


        new_groups = (
            new_team_doc.get(
                "groups",
                {
                    group_name: {
                        "rating":
                            50.0,

                        "players":
                            [],
                    }
                    for group_name
                    in POSITION_GROUPS
                },
            )
        )


        if (
            old_group
            in new_groups
        ):

            new_players = (
                new_groups[
                    old_group
                ]
                .get(
                    "players",
                    [],
                )
            )


            if (
                real_player_key
                not in new_players
            ):

                new_players.append(
                    real_player_key
                )


            new_groups[
                old_group
            ][
                "players"
            ] = new_players


        self._team_ref(
            new_team
        ).update({

            "groups":
                new_groups,

            "updated_at":
                _now_iso(),
        })


        # ----------------------------------------------------
        # Update player
        # ----------------------------------------------------

        player_ref.update({

            "team":
                new_team,

            "notes":
                notes,

            "updated_at":
                _now_iso(),
        })


        # ----------------------------------------------------
        # Recalculate both teams
        # ----------------------------------------------------

        self._recalculate_team_adjustment(
            old_team
        )

        self._recalculate_team_adjustment(
            new_team
        )


        # ----------------------------------------------------
        # Log move
        # ----------------------------------------------------

        move_record = {

            "player_id":
                real_player_key,

            "player_name":
                player[
                    "name"
                ],

            "from_team":
                old_team,

            "to_team":
                new_team,

            "position_group":
                old_group,

            "impact_score":
                player[
                    "impact_score"
                ],

            "move_type":
                move_type,

            "notes":
                notes,

            "timestamp":
                _now_iso(),
        }


        db.collection(
            MOVES_COLLECTION
        ).add(
            move_record
        )


        return {

            "move":
                move_record,

            "old_team_adjustment":
                self.get_team_adjustment(
                    old_team
                ),

            "new_team_adjustment":
                self.get_team_adjustment(
                    new_team
                ),
        }


    # ========================================================
    # Recalculate Team Adjustment
    # ========================================================

    def _recalculate_team_adjustment(
        self,
        team_name: str,
    ):

        team_ref = (
            self._team_ref(
                team_name
            )
        )


        team_doc = (
            team_ref.get()
        )


        if not team_doc.exists:

            return


        team_data = (
            team_doc.to_dict()
        )


        groups = (
            team_data.get(
                "groups",
                {},
            )
        )


        weighted_delta = (
            0.0
        )


        for (
            group_name,
            group_info,
        ) in POSITION_GROUPS.items():

            group_data = (
                groups.get(
                    group_name,
                    {
                        "players":
                            [],
                    },
                )
            )


            player_ids = (
                group_data.get(
                    "players",
                    [],
                )
            )


            scores = []


            for pid in player_ids:

                player_doc = (
                    self
                    ._player_ref(
                        pid
                    )
                    .get()
                )


                if player_doc.exists:

                    scores.append(
                        float(
                            player_doc
                            .to_dict()
                            .get(
                                "impact_score",
                                50.0,
                            )
                        )
                    )


            avg_score = (
                sum(
                    scores
                )
                /
                len(
                    scores
                )
                if scores
                else
                50.0
            )


            group_data[
                "rating"
            ] = round(
                avg_score,
                1,
            )


            groups[
                group_name
            ] = group_data


            weighted_delta += (
                (
                    avg_score
                    -
                    50.0
                )
                *
                group_info[
                    "weight"
                ]
            )


        team_ref.update({

            "groups":
                groups,

            "roster_adjustment":
                round(
                    weighted_delta
                    *
                    0.1,
                    3,
                ),

            "updated_at":
                _now_iso(),
        })


    # ========================================================
    # Team adjustment
    # ========================================================

    def get_team_adjustment(
        self,
        team_name: str,
    ) -> Optional[float]:

        doc = (
            self
            ._team_ref(
                team_name
            )
            .get()
        )


        if not doc.exists:

            return None


        return (
            doc
            .to_dict()
            .get(
                "roster_adjustment",
                0.0,
            )
        )


    # ========================================================
    # Team profile
    # ========================================================

    def get_team_profile(
        self,
        team_name: str,
    ) -> Optional[dict]:

        doc = (
            self
            ._team_ref(
                team_name
            )
            .get()
        )


        if not doc.exists:

            return None


        team = (
            doc.to_dict()
        )


        groups = (
            team.get(
                "groups",
                {},
            )
        )


        for (
            group_name,
            group_data,
        ) in groups.items():

            enriched = []


            for pid in (
                group_data.get(
                    "players",
                    [],
                )
            ):

                player_doc = (
                    self
                    ._player_ref(
                        pid
                    )
                    .get()
                )


                if player_doc.exists:

                    p = (
                        player_doc
                        .to_dict()
                    )


                    enriched.append({

                        "id":
                            pid,

                        "name":
                            p.get(
                                "name"
                            ),

                        "position_group":
                            p.get(
                                "position_group"
                            ),

                        "impact_score":
                            p.get(
                                "impact_score"
                            ),

                        "notes":
                            p.get(
                                "notes",
                                "",
                            ),
                    })


            group_data[
                "player_details"
            ] = enriched


        team[
            "groups"
        ] = groups


        return team


    # ========================================================
    # NEW:
    # Position Group Breakdown
    #
    # Returns the actual model contribution from each group.
    #
    # contribution_points =
    #   (group_rating - 50) * position_weight * 0.1
    #
    # This is mathematically consistent with roster_adjustment.
    # ========================================================

    def get_position_group_breakdown(
        self,
        team_name: str,
    ) -> dict:

        profile = (
            self.get_team_profile(
                team_name
            )
        )


        if not profile:

            return {}


        groups = (
            profile.get(
                "groups",
                {},
            )
        )


        breakdown = {}


        for (
            group_name,
            group_info,
        ) in POSITION_GROUPS.items():

            group_data = (
                groups.get(
                    group_name,
                    {},
                )
            )


            rating = float(
                group_data.get(
                    "rating",
                    50.0,
                )
                or 50.0
            )


            weight = float(
                group_info[
                    "weight"
                ]
            )


            contribution_points = (
                (
                    rating
                    -
                    50.0
                )
                *
                weight
                *
                0.1
            )


            players = (
                group_data.get(
                    "player_details",
                    [],
                )
            )


            # Top players in group
            sorted_players = sorted(
                players,
                key=lambda p:
                    float(
                        p.get(
                            "impact_score",
                            0,
                        )
                        or 0
                    ),
                reverse=True,
            )


            top_players = [
                {
                    "name":
                        p.get(
                            "name"
                        ),

                    "impact_score":
                        float(
                            p.get(
                                "impact_score",
                                0,
                            )
                            or 0
                        ),

                    "notes":
                        p.get(
                            "notes",
                            "",
                        ),
                }
                for p
                in sorted_players[
                    :3
                ]
            ]


            breakdown[
                group_name
            ] = {

                "group":
                    group_name,

                "label":
                    POSITION_LABELS.get(
                        group_name,
                        group_name,
                    ),

                "side":
                    group_info[
                        "side"
                    ],

                "weight":
                    round(
                        weight,
                        3,
                    ),

                "rating":
                    round(
                        rating,
                        1,
                    ),

                "tier":
                    _impact_tier(
                        rating
                    ),

                "points":
                    round(
                        contribution_points,
                        2,
                    ),

                "player_count":
                    len(
                        players
                    ),

                "top_players":
                    top_players,
            }


        return breakdown


    # ========================================================
    # NEW:
    # Compare two teams by position group
    #
    # Positive point_edge = HOME advantage
    # Negative point_edge = AWAY advantage
    # ========================================================

    def compare_position_groups(
        self,
        home_team: str,
        away_team: str,
    ) -> list[dict]:

        home_breakdown = (
            self.get_position_group_breakdown(
                home_team
            )
        )


        away_breakdown = (
            self.get_position_group_breakdown(
                away_team
            )
        )


        comparisons = []


        for (
            group_name,
            group_info,
        ) in POSITION_GROUPS.items():

            home_group = (
                home_breakdown.get(
                    group_name,
                    {},
                )
            )


            away_group = (
                away_breakdown.get(
                    group_name,
                    {},
                )
            )


            home_rating = float(
                home_group.get(
                    "rating",
                    50.0,
                )
                or 50.0
            )


            away_rating = float(
                away_group.get(
                    "rating",
                    50.0,
                )
                or 50.0
            )


            weight = float(
                group_info[
                    "weight"
                ]
            )


            rating_gap = (
                home_rating
                -
                away_rating
            )


            point_edge = (
                rating_gap
                *
                weight
                *
                0.1
            )


            if point_edge > 0:

                favored_team = (
                    home_team
                )

            elif point_edge < 0:

                favored_team = (
                    away_team
                )

            else:

                favored_team = (
                    None
                )


            comparisons.append({

                "group":
                    group_name,

                "label":
                    POSITION_LABELS.get(
                        group_name,
                        group_name,
                    ),

                "side":
                    group_info[
                        "side"
                    ],

                "weight":
                    round(
                        weight,
                        3,
                    ),

                "home_team":
                    home_team,

                "away_team":
                    away_team,

                "home_rating":
                    round(
                        home_rating,
                        1,
                    ),

                "away_rating":
                    round(
                        away_rating,
                        1,
                    ),

                "rating_gap":
                    round(
                        rating_gap,
                        1,
                    ),

                "point_edge":
                    round(
                        point_edge,
                        2,
                    ),

                "favored_team":
                    favored_team,

                "home_top_players":
                    home_group.get(
                        "top_players",
                        [],
                    ),

                "away_top_players":
                    away_group.get(
                        "top_players",
                        [],
                    ),
            })


        # Largest matchup advantages first
        comparisons.sort(
            key=lambda item:
                abs(
                    item[
                        "point_edge"
                    ]
                ),
            reverse=True,
        )


        return comparisons


    # ========================================================
    # NEW:
    # High-value explanation factors
    #
    # Returns the most meaningful position-group edges in a
    # frontend-friendly structure.
    # ========================================================

    def get_matchup_explanation_factors(
        self,
        home_team: str,
        away_team: str,
        minimum_points: float = 0.05,
        limit: int = 5,
    ) -> list[dict]:

        comparisons = (
            self.compare_position_groups(
                home_team,
                away_team,
            )
        )


        factors = []


        for item in comparisons:

            point_edge = float(
                item.get(
                    "point_edge",
                    0,
                )
                or 0
            )


            if (
                abs(
                    point_edge
                )
                <
                minimum_points
            ):

                continue


            favored_team = (
                item.get(
                    "favored_team"
                )
            )


            if not favored_team:

                continue


            group_name = (
                item[
                    "group"
                ]
            )


            label = (
                item[
                    "label"
                ]
            )


            home_rating = (
                item[
                    "home_rating"
                ]
            )


            away_rating = (
                item[
                    "away_rating"
                ]
            )


            # ------------------------------------------------
            # Friendly factor name
            # ------------------------------------------------

            if group_name == "QB":

                factor_label = (
                    "QB Advantage"
                )

            elif group_name == "OL":

                factor_label = (
                    "Offensive Line"
                )

            elif group_name in (
                "DL",
                "LB",
            ):

                factor_label = (
                    "Defensive Front"
                )

            elif group_name in (
                "CB",
                "S",
            ):

                factor_label = (
                    "Secondary"
                )

            elif group_name == "WR":

                factor_label = (
                    "Receiving Corps"
                )

            else:

                factor_label = (
                    f"{label} Advantage"
                )


            # ------------------------------------------------
            # Impact level
            # ------------------------------------------------

            abs_edge = abs(
                point_edge
            )


            if abs_edge >= 1.5:

                impact = (
                    "high"
                )

            elif abs_edge >= 0.5:

                impact = (
                    "medium"
                )

            else:

                impact = (
                    "low"
                )


            # ------------------------------------------------
            # Which top players belong to favored side?
            # ------------------------------------------------

            if (
                favored_team
                ==
                home_team
            ):

                top_players = (
                    item.get(
                        "home_top_players",
                        [],
                    )
                )

            else:

                top_players = (
                    item.get(
                        "away_top_players",
                        [],
                    )
                )


            player_names = [
                p.get(
                    "name"
                )
                for p
                in top_players
                if p.get(
                    "name"
                )
            ]


            # ------------------------------------------------
            # Explanation sentence
            # ------------------------------------------------

            detail = (
                f"{favored_team} grades higher at "
                f"{label.lower()} "
                f"({home_team} {home_rating:.1f} vs "
                f"{away_team} {away_rating:.1f})."
            )


            if player_names:

                detail += (
                    " Key players: "
                    +
                    ", ".join(
                        player_names[
                            :2
                        ]
                    )
                    +
                    "."
                )


            factors.append({

                "label":
                    factor_label,

                "group":
                    group_name,

                "team":
                    favored_team,

                # Positive display number representing
                # magnitude of advantage for favored team.
                "points":
                    round(
                        abs(
                            point_edge
                        ),
                        2,
                    ),

                # Signed value remains available for math.
                # Positive = home advantage
                # Negative = away advantage
                "signed_points":
                    round(
                        point_edge,
                        2,
                    ),

                "impact":
                    impact,

                "detail":
                    detail,

                "home_rating":
                    home_rating,

                "away_rating":
                    away_rating,

                "top_players":
                    top_players,
            })


            if (
                len(
                    factors
                )
                >=
                limit
            ):

                break


        return factors


    # ========================================================
    # All teams
    # ========================================================

    def get_all_teams(
        self,
    ) -> list[dict]:

        docs = (
            db
            .collection(
                TEAMS_COLLECTION
            )
            .stream()
        )


        teams = []


        for doc in docs:

            data = (
                doc.to_dict()
            )


            data.pop(
                "groups",
                None,
            )


            teams.append(
                data
            )


        return teams


    # ========================================================
    # Recent moves
    # ========================================================

    def get_recent_moves(
        self,
        limit: int = 50,
    ) -> list[dict]:

        docs = (
            db
            .collection(
                MOVES_COLLECTION
            )
            .order_by(
                "timestamp",
                direction=
                    firestore
                    .Query
                    .DESCENDING,
            )
            .limit(
                limit
            )
            .stream()
        )


        return [
            {
                **doc.to_dict(),
                "id":
                    doc.id,
            }
            for doc
            in docs
        ]


    # ========================================================
    # All players
    # ========================================================

    def get_all_players(
        self,
        team: Optional[str] = None,
    ) -> list[dict]:

        query = (
            db
            .collection(
                PLAYERS_COLLECTION
            )
        )


        if team:

            docs = (
                query
                .where(
                    "team",
                    "==",
                    team,
                )
                .stream()
            )

        else:

            docs = (
                query.stream()
            )


        players = [
            {
                **doc.to_dict(),
                "id":
                    doc.id,
            }
            for doc
            in docs
        ]


        return sorted(
            players,
            key=lambda p:
                p.get(
                    "impact_score",
                    0,
                ),
            reverse=True,
        )


    # ========================================================
    # Player search
    # ========================================================

    def search_players(
        self,
        query: str,
    ) -> list[dict]:

        q = (
            query.lower()
        )


        players = (
            self.get_all_players()
        )


        return [
            player
            for player
            in players
            if (
                q
                in
                player
                .get(
                    "name",
                    "",
                )
                .lower()
            )
            or
            (
                q
                in
                player
                .get(
                    "team",
                    "",
                )
                .lower()
            )
        ]


    # ========================================================
    # Database stats
    # ========================================================

    def get_db_stats(
        self,
    ) -> dict:

        teams = list(
            db
            .collection(
                TEAMS_COLLECTION
            )
            .stream()
        )


        players = list(
            db
            .collection(
                PLAYERS_COLLECTION
            )
            .stream()
        )


        moves = list(
            db
            .collection(
                MOVES_COLLECTION
            )
            .stream()
        )


        last_updated = (
            None
        )


        for doc in teams:

            updated = (
                doc
                .to_dict()
                .get(
                    "updated_at"
                )
            )


            if (
                updated
                and
                (
                    last_updated
                    is None
                    or
                    updated
                    >
                    last_updated
                )
            ):

                last_updated = (
                    updated
                )


        return {

            "teams":
                len(
                    teams
                ),

            "players":
                len(
                    players
                ),

            "moves_logged":
                len(
                    moves
                ),

            "last_updated":
                last_updated,
        }


# ============================================================
# Singleton
# ============================================================

roster_engine = (
    RosterEngine()
)
