import { useState, useEffect, useCallback } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import RecordWidget from '../components/RecordWidget';

const API =
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8000';

// September 2026 = 2026 season.
// Keeping this dynamic means it automatically moves forward next year.
const CURRENT_SEASON = new Date().getFullYear();


const EDGE_COLORS: Record<string, string> = {
  fade_away: '#D94040',
  fade_home: '#D94040',

  lean_home: '#C9A84C',
  lean_away: '#C9A84C',

  lean_over: '#3DAA6A',
  lean_under: '#8B9BB4',

  neutral: '#4A5568',
};


const EDGE_LABELS: Record<string, string> = {
  fade_away: '🔄 FADE AWAY',
  fade_home: '🔄 FADE HOME',

  lean_home: '↑ LEAN HOME',
  lean_away: '↓ LEAN AWAY',

  lean_over: '⬆ LEAN OVER',
  lean_under: '⬇ LEAN UNDER',

  neutral: '— NEUTRAL',
};


type Movement = {
  spread_open: number | null;
  spread_current: number | null;
  spread_move: number | null;

  total_open: number | null;
  total_current: number | null;
  total_move: number | null;

  sharp_signal: number;
  steam_move: boolean;

  move_direction: string | null;
  hours_tracked: number;
};


type Prediction = {
  predicted_home_score: number;
  predicted_away_score: number;

  predicted_margin: number;
  predicted_total: number;

  home_win_prob: number;

  margin_80_lo?: number;
  margin_80_hi?: number;

  total_80_lo?: number;
  total_80_hi?: number;

  model_trained?: boolean;
  prediction_mode?: string;
};


type Disparity = {
  has_line: boolean;

  spread_disparity: number | null;
  total_disparity: number | null;

  edge_score: number | null;
  raw_edge_score?: number | null;
  ranking_score?: number | null;

  edge_label?: string;

  confidence?: string;
  low_confidence?: boolean;

  vegas_spread: number | null;
  vegas_total: number | null;
  vegas_home_margin?: number | null;

  spread_edge_type: string | null;
  total_edge_type: string | null;

  sharp_signal: number;
  steam_move: boolean;
  sharp_aligned: boolean;
};


type PlayFactor = {
  label: string;
  team?: string | null;
  points?: number | null;
  signed_points?: number | null;
  impact?: 'high' | 'medium' | 'low' | string;
  detail?: string;
  source?: string;
  group?: string | null;
  top_players?: Array<{
    name?: string;
    impact_score?: number;
    notes?: string;
  }>;
};

type PlayExplanation = {
  recommended_team?: string | null;
  summary?: string;
  factor_count?: number;
  factors?: PlayFactor[];
};


type GameCard = {
  game_id?: string | number;

  home_team: string;
  away_team: string;

  date: string;
  venue?: string;

  league?: string;
  week?: number;
  season?: number;

  neutral_site?: boolean;

  prediction_mode?: string;
  confidence?: string;

  model_note?: string;

  prediction: Prediction;
  disparity: Disparity;

  line_movement: Movement | null;

  home_injury_notes: string[];
  away_injury_notes: string[];

  roster_notes: string[];

  adjustments: {
    home_roster: number;
    away_roster: number;

    home_injury: number;
    away_injury: number;
  };

  key_factors?: PlayFactor[];
  play_explanation?: PlayExplanation;

  fallback_diagnostics?: any;
};


type CardData = {
  league: string;
  week: number;
  season?: number;

  total_games: number;
  games_with_lines: number;
  games_with_movement: number;

  cfb_sp_model_games?: number;
  cfb_fallback_games?: number;

  standard_confidence_games?: number;
  low_confidence_games?: number;

  cfb_fallback_method?: string | null;

  games: GameCard[];

  generated_at: string;

  error?: string;
};


const DISPLAY_TIME_ZONE = 'America/New_York';

function getDatePartsInTimeZone(
  date: Date,
  timeZone: string
) {
  const parts = new Intl.DateTimeFormat(
    'en-US',
    {
      timeZone,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }
  ).formatToParts(date);

  const get = (type: string) =>
    parts.find(part => part.type === type)?.value || '';

  return {
    year: get('year'),
    month: get('month'),
    day: get('day'),
  };
}

function getRelativeDayLabel(
  gameDate: Date
) {
  const now = new Date();

  const todayParts =
    getDatePartsInTimeZone(
      now,
      DISPLAY_TIME_ZONE
    );

  const gameParts =
    getDatePartsInTimeZone(
      gameDate,
      DISPLAY_TIME_ZONE
    );

  const todayKey =
    `${todayParts.year}-${todayParts.month}-${todayParts.day}`;

  const gameKey =
    `${gameParts.year}-${gameParts.month}-${gameParts.day}`;

  if (gameKey === todayKey) {
    return 'TODAY';
  }

  const tomorrow = new Date(
    now.getTime() +
    24 * 60 * 60 * 1000
  );

  const tomorrowParts =
    getDatePartsInTimeZone(
      tomorrow,
      DISPLAY_TIME_ZONE
    );

  const tomorrowKey =
    `${tomorrowParts.year}-${tomorrowParts.month}-${tomorrowParts.day}`;

  if (gameKey === tomorrowKey) {
    return 'TOMORROW';
  }

  return null;
}

function formatKickoff(
  value?: string
) {
  if (!value) {
    return {
      relative: null as string | null,
      dateLabel: 'DATE TBD',
      timeLabel: 'TIME TBD',
    };
  }

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return {
      relative: null as string | null,
      dateLabel: 'DATE TBD',
      timeLabel: 'TIME TBD',
    };
  }

  const relative =
    getRelativeDayLabel(date);

  const dateLabel =
    new Intl.DateTimeFormat(
      'en-US',
      {
        timeZone:
          DISPLAY_TIME_ZONE,

        weekday:
          'short',

        month:
          'short',

        day:
          'numeric',
      }
    )
      .format(date)
      .toUpperCase();

  const timeLabel =
    new Intl.DateTimeFormat(
      'en-US',
      {
        timeZone:
          DISPLAY_TIME_ZONE,

        hour:
          'numeric',

        minute:
          '2-digit',

        hour12:
          true,
      }
    )
      .format(date)
      .toUpperCase();

  return {
    relative,
    dateLabel,
    timeLabel:
      `${timeLabel} ET`,
  };
}


function MovementBadge({
  mv,
}: {
  mv: Movement;
}) {
  if (
    !mv ||
    mv.spread_move === null
  ) {
    return null;
  }

  const dir =
    mv.spread_move > 0
      ? '→ HOME'
      : '→ AWAY';

  const color =
    mv.steam_move
      ? '#D94040'
      : mv.sharp_signal > 0.5
        ? '#C9A84C'
        : '#8B9BB4';

  const label =
    mv.steam_move
      ? '🔥 STEAM'
      : mv.sharp_signal > 0.5
        ? '⚡ SHARP'
        : '○ MOVE';

  return (
    <div
      className="rounded px-2 py-1.5 text-xs"
      style={{
        background: `${color}12`,
        border: `1px solid ${color}30`,
        fontFamily: 'var(--font-mono)',
      }}
    >
      <div
        style={{ color }}
        className="font-semibold"
      >
        {label}
      </div>

      <div className="text-slate mt-0.5">
        Spread: {mv.spread_open} →{' '}
        {mv.spread_current}

        <span
          style={{ color }}
          className="ml-1"
        >
          (
          {mv.spread_move > 0
            ? '+'
            : ''}
          {mv.spread_move}{' '}
          {dir})
        </span>
      </div>

      {mv.total_move !== null && (
        <div className="text-slate">
          Total: {mv.total_open} →{' '}
          {mv.total_current}{' '}
          (
          {mv.total_move > 0
            ? '+'
            : ''}
          {mv.total_move})
        </div>
      )}

      <div className="text-slate">
        Signal:{' '}
        {(
          mv.sharp_signal * 100
        ).toFixed(0)}
        % · {mv.hours_tracked}h tracked
      </div>
    </div>
  );
}


function getEdgeBadgeStyle(
  game: GameCard
) {
  const d =
    game.disparity;

  const label =
    d.edge_label ||
    '';

  const lowConfidence =
    game.confidence === 'low' ||
    d.confidence === 'low' ||
    d.low_confidence === true;

  const isFallback =
    game.prediction_mode ===
      'opponent_adjusted_fallback' ||
    game.prediction_mode ===
      'historical_fallback';

  if (
    isFallback ||
    lowConfidence ||
    label.includes('Fallback') ||
    label.includes('SRS')
  ) {
    return {
      background:
        'rgba(201,168,76,0.08)',

      color:
        '#E8C96A',

      border:
        '1px solid rgba(201,168,76,0.25)',
    };
  }

  if (
    label.includes('Strong')
  ) {
    return {
      background:
        'rgba(201,168,76,0.15)',

      color:
        '#C9A84C',

      border:
        '1px solid rgba(201,168,76,0.3)',
    };
  }

  if (
    label.includes('Moderate')
  ) {
    return {
      background:
        'rgba(201,168,76,0.1)',

      color:
        '#C9A84C',

      border:
        '1px solid rgba(201,168,76,0.22)',
    };
  }

  return {
    background:
      'rgba(139,155,180,0.08)',

    color:
      '#8B9BB4',

    border:
      '1px solid rgba(139,155,180,0.18)',
  };
}


function getBorderColor(
  game: GameCard
) {
  const d =
    game.disparity;

  const isFallback =
    game.prediction_mode ===
      'opponent_adjusted_fallback' ||
    game.prediction_mode ===
      'historical_fallback';

  const isLowConfidence =
    game.confidence === 'low' ||
    d.confidence === 'low';

  const rankingScore =
    d.ranking_score ??
    d.edge_score ??
    0;

  if (
    isFallback ||
    isLowConfidence
  ) {
    return 'rgba(201,168,76,0.22)';
  }

  if (
    d.steam_move
  ) {
    return 'rgba(217,64,64,0.4)';
  }

  if (
    rankingScore >= 15
  ) {
    return 'rgba(201,168,76,0.5)';
  }

  if (
    rankingScore >= 8
  ) {
    return 'rgba(201,168,76,0.2)';
  }

  return 'rgba(201,168,76,0.08)';
}


function ExplanationPanel({
  game,
  lowConfidence,
  isFallback,
}: {
  game: GameCard;
  lowConfidence: boolean;
  isFallback: boolean;
}) {
  const factors =
    game.play_explanation?.factors?.length
      ? game.play_explanation.factors
      : game.key_factors || [];

  if (!game.play_explanation && factors.length === 0) {
    return null;
  }

  const recommendedTeam =
    game.play_explanation?.recommended_team || null;

  const supportingFactors =
    recommendedTeam
      ? factors.filter(
          factor =>
            !factor.team ||
            factor.team === recommendedTeam
        )
      : factors;

  const counterFactors =
    recommendedTeam
      ? factors.filter(
          factor =>
            factor.team &&
            factor.team !== recommendedTeam
        )
      : [];

  const factorColor = (impact?: string) => {
    if (impact === 'high') return '#C9A84C';
    if (impact === 'medium') return '#8B9BB4';
    return '#6B7C93';
  };

  const factorIcon = (label: string) => {
    const value = label.toLowerCase();

    if (value.includes('qb') || value.includes('quarterback')) return '🏈';
    if (value.includes('injury')) return '✚';
    if (value.includes('home field')) return '⌂';
    if (value.includes('vegas') || value.includes('market')) return '↔';
    if (value.includes('sp+') || value.includes('power') || value.includes('srs')) return '📈';
    if (value.includes('defensive') || value.includes('secondary')) return '🛡';
    return '◆';
  };

  return (
    <div
      className="rounded-xl p-4"
      style={{
        background: 'rgba(201,168,76,0.045)',
        border: '1px solid rgba(201,168,76,0.16)',
      }}
    >
      <div className="flex items-start justify-between gap-3 flex-wrap mb-3">
        <div>
          <div
            className="score-display"
            style={{
              color: '#C9A84C',
              fontSize: 14,
              letterSpacing: '0.09em',
            }}
          >
            WHY PRIME PICKS LIKES THIS PLAY
          </div>

          {recommendedTeam && (
            <div
              className="text-xs mt-1"
              style={{
                color: '#F0EEE6',
                fontFamily: 'var(--font-mono)',
              }}
            >
              Model lean:{' '}
              <span style={{ color: '#C9A84C', fontWeight: 700 }}>
                {recommendedTeam}
              </span>
            </div>
          )}
        </div>

        {game.play_explanation?.factor_count != null && (
          <span
            className="text-xs px-2 py-1 rounded"
            style={{
              background: 'rgba(15,44,71,0.55)',
              color: '#8B9BB4',
              border: '1px solid rgba(139,155,180,0.15)',
              fontFamily: 'var(--font-mono)',
            }}
          >
            {game.play_explanation.factor_count}{' '}
            factor{game.play_explanation.factor_count === 1 ? '' : 's'}
          </span>
        )}
      </div>

      {game.play_explanation?.summary && (
        <div className="mb-4">
          <p
            className="text-xs"
            style={{
              color: '#A8B5C7',
              fontFamily: 'var(--font-mono)',
              lineHeight: 1.65,
            }}
          >
            {game.play_explanation.summary}
          </p>

          {recommendedTeam && counterFactors.length > 0 && (
            <p
              className="text-xs mt-2"
              style={{
                color: '#8B9BB4',
                fontFamily: 'var(--font-mono)',
                lineHeight: 1.6,
              }}
            >
              Mixed-factor matchup: {supportingFactors.length}{' '}
              factor{supportingFactors.length === 1 ? '' : 's'} support{' '}
              {recommendedTeam}, while {counterFactors.length}{' '}
              factor{counterFactors.length === 1 ? '' : 's'} point the other way.
            </p>
          )}
        </div>
      )}

      <div className="space-y-2">
        {factors.map((factor, index) => {
          const color = factorColor(factor.impact);
          const supportsPick =
            recommendedTeam && factor.team === recommendedTeam;

          return (
            <div
              key={`${factor.label}-${index}`}
              className="rounded-lg px-3 py-3"
              style={{
                background: supportsPick
                  ? 'rgba(201,168,76,0.055)'
                  : 'rgba(7,21,36,0.48)',
                border: `1px solid ${color}24`,
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span aria-hidden="true" style={{ fontSize: 12 }}>
                      {factorIcon(factor.label)}
                    </span>

                    <span
                      className="text-xs font-semibold"
                      style={{
                        color,
                        fontFamily: 'var(--font-mono)',
                      }}
                    >
                      {factor.label}
                    </span>

                    {factor.team && (
                      <span
                        className="text-xs"
                        style={{
                          color: supportsPick ? '#C9A84C' : '#8B9BB4',
                          fontFamily: 'var(--font-mono)',
                        }}
                      >
                        {factor.team}
                      </span>
                    )}

                    {recommendedTeam && factor.team && (
                      <span
                        className="text-xs px-1.5 py-0.5 rounded uppercase"
                        style={{
                          color: supportsPick ? '#3DAA6A' : '#D94040',
                          background: supportsPick
                            ? 'rgba(61,170,106,0.08)'
                            : 'rgba(217,64,64,0.08)',
                          border: supportsPick
                            ? '1px solid rgba(61,170,106,0.2)'
                            : '1px solid rgba(217,64,64,0.2)',
                          fontFamily: 'var(--font-mono)',
                          fontSize: 9,
                          letterSpacing: '0.07em',
                        }}
                      >
                        {supportsPick ? '✓ Supports Pick' : '↔ Counter Factor'}
                      </span>
                    )}

                    {factor.impact && (
                      <span
                        className="text-xs px-1.5 py-0.5 rounded uppercase"
                        style={{
                          color,
                          background: `${color}10`,
                          border: `1px solid ${color}20`,
                          fontFamily: 'var(--font-mono)',
                          fontSize: 9,
                          letterSpacing: '0.07em',
                        }}
                      >
                        {factor.impact}
                      </span>
                    )}
                  </div>

                  {factor.detail && (
                    <div
                      className="text-xs mt-1.5"
                      style={{
                        color: '#8B9BB4',
                        fontFamily: 'var(--font-mono)',
                        lineHeight: 1.55,
                      }}
                    >
                      {factor.detail}
                    </div>
                  )}
                </div>

                {factor.points != null && (
                  <div
                    className="score-display shrink-0"
                    style={{
                      color,
                      fontSize: 17,
                      letterSpacing: '0.04em',
                    }}
                  >
                    +{Number(factor.points).toFixed(1)}
                    <span
                      style={{
                        fontSize: 9,
                        marginLeft: 3,
                        color: '#596A80',
                      }}
                    >
                      PTS
                    </span>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {lowConfidence && (
        <div
          className="mt-3 text-xs rounded px-3 py-2"
          style={{
            background: 'rgba(201,168,76,0.05)',
            border: '1px solid rgba(201,168,76,0.12)',
            color: '#8B9BB4',
            fontFamily: 'var(--font-mono)',
            lineHeight: 1.5,
          }}
        >
          ⚠ Lower-confidence projection.{' '}
          {isFallback
            ? 'SP+ coverage is incomplete, so Prime Picks is using the opponent-adjusted SRS fallback for this matchup.'
            : 'Treat this edge more cautiously than a standard-confidence model result.'}
        </div>
      )}
    </div>
  );
}


export default function CardPage() {
  const {
    user,
    userStatus,
    loading: authLoading,
  } = useAuth();

  const router =
    useRouter();

  const [
    league,
    setLeague,
  ] = useState<'NFL' | 'CFB'>(
    'NFL'
  );

  const [
    week,
    setWeek,
  ] = useState(1);

  const [
    card,
    setCard,
  ] =
    useState<CardData | null>(
      null
    );

  const [
    loading,
    setLoading,
  ] =
    useState(false);

  const [
    error,
    setError,
  ] =
    useState('');

  const [
    filter,
    setFilter,
  ] = useState<
    'all' |
    'edges' |
    'sharp'
  >('all');

  const [
    expandedGame,
    setExpandedGame,
  ] =
    useState<number | null>(
      null
    );

  const [
    officialPicks,
    setOfficialPicks,
  ] =
    useState<Set<string>>(
      new Set()
    );


  useEffect(() => {
    if (
      !authLoading &&
      (
        !user ||
        userStatus !== 'approved'
      )
    ) {
      router.replace(
        '/login'
      );
    }
  }, [
    user,
    userStatus,
    authLoading,
    router,
  ]);


  // ========================================================
  // Official picks
  // ========================================================

  const fetchOfficialPicks =
    useCallback(
      async () => {
        if (!card) {
          return;
        }

        try {
          const r =
            await axios.get(
              `/api/picks?week=${week}&league=${league}`
            );

          const picks =
            r.data.picks ||
            [];

          const keys =
            new Set<string>(
              picks.map(
                (p: any) =>
                  `${p.home_team}|${p.away_team}`
              )
            );

          setOfficialPicks(
            keys
          );

        } catch {
          // non-fatal
        }
      },
      [
        week,
        league,
        card,
      ]
    );


  useEffect(() => {
    if (card) {
      fetchOfficialPicks();
    }
  }, [
    card,
    fetchOfficialPicks,
  ]);


  // ========================================================
  // Load card
  // ========================================================

  const fetchCard =
    useCallback(
      async () => {
        setLoading(true);
        setError('');
        setCard(null);
        setExpandedGame(null);

        try {
          const r =
            await axios.get(
              `${API}/card/${league}`,
              {
                params: {
                  week,
                  season:
                    CURRENT_SEASON,
                },
              }
            );

          const data =
            r.data as CardData;

          if (
            data.error &&
            (
              !data.games ||
              data.games.length === 0
            )
          ) {
            setError(
              data.error
            );
          }

          setCard(
            data
          );

        } catch (e: any) {
          console.error(
            'Weekly card request failed:',
            e
          );

          setError(
            e?.response?.data?.detail ||
            e?.message ||
            'Failed to load card.'
          );

        } finally {
          setLoading(
            false
          );
        }
      },
      [
        league,
        week,
      ]
    );


  if (
    authLoading ||
    !user ||
    userStatus !== 'approved'
  ) {
    return (
      <div className="field-bg min-h-screen flex items-center justify-center">
        <div
          className="animate-pulse-gold score-display text-slate"
          style={{
            fontSize: 18,
          }}
        >
          LOADING...
        </div>
      </div>
    );
  }


  // ========================================================
  // Filtering
  // ========================================================

  const displayed =
    (
      card?.games ||
      []
    ).filter(
      game => {
        const d =
          game.disparity;

        const rankingScore =
          d.ranking_score ??
          d.edge_score ??
          0;

        if (
          filter === 'edges'
        ) {
          return (
            rankingScore >= 8
          );
        }

        if (
          filter === 'sharp'
        ) {
          return (
            d.sharp_signal >
              0.4 ||
            d.steam_move
          );
        }

        return true;
      }
    );


  return (
    <>
      <Head>
        <title>
          Weekly Card — Prime Picks AI
        </title>

        <meta
          name="description"
          content="Prime Picks NFL and NCAA weekly prediction card"
        />
      </Head>

      <div className="field-bg min-h-screen px-4 py-10">

        <div className="max-w-5xl mx-auto">

          {/* =================================================
              Header
          ================================================= */}

          <div className="flex items-center justify-between mb-3 flex-wrap gap-4">

            <Link
              href="/"
              aria-label="Prime Picks AI Home"
              className="flex items-center"
            >
              <img
                src="/images/primepicks-logo.png"
                alt="Prime Picks AI"
                style={{
                  width:
                    '250px',

                  maxWidth:
                    '65vw',

                  height:
                    'auto',

                  display:
                    'block',

                  objectFit:
                    'contain',
                }}
              />
            </Link>


            <nav
              className="flex gap-4 text-xs"
              style={{
                fontFamily:
                  'var(--font-mono)',
              }}
            >

              <Link
                href="/"
                style={{
                  color:
                    '#8B9BB4',
                }}
              >
                Predict
              </Link>

              <Link
                href="/card"
                style={{
                  color:
                    '#C9A84C',
                }}
              >
                Weekly Card
              </Link>

              <Link
                href="/roster"
                style={{
                  color:
                    '#8B9BB4',
                }}
              >
                Roster Intel
              </Link>

              <Link
                href="/record"
                style={{
                  color:
                    '#8B9BB4',
                }}
              >
                Record
              </Link>

            </nav>

          </div>

          <div className="gold-line mb-6" />


          {/* =================================================
              Record widget
          ================================================= */}

          <div className="mb-5">
            <RecordWidget
              compact={true}
            />
          </div>


          {/* =================================================
              Controls
          ================================================= */}

          <div className="panel rounded-xl p-5 mb-5">

            <div className="flex flex-wrap gap-4 items-end">

              <div>

                <label
                  className="block text-xs text-slate mb-1 uppercase tracking-widest"
                  style={{
                    fontFamily:
                      'var(--font-mono)',
                  }}
                >
                  League
                </label>


                <div className="flex gap-2">

                  {(
                    [
                      'NFL',
                      'CFB',
                    ] as const
                  ).map(
                    l => (
                      <button
                        key={l}

                        onClick={() => {
                          setLeague(
                            l
                          );

                          setCard(
                            null
                          );

                          setError(
                            ''
                          );

                          setExpandedGame(
                            null
                          );
                        }}

                        className="score-display px-4 py-2 rounded"

                        style={{
                          fontSize:
                            16,

                          letterSpacing:
                            '0.08em',

                          background:
                            league === l
                              ? '#C9A84C'
                              : 'rgba(15,44,71,0.5)',

                          color:
                            league === l
                              ? '#030B14'
                              : '#8B9BB4',

                          border:
                            '1px solid',

                          borderColor:
                            league === l
                              ? '#C9A84C'
                              : 'rgba(201,168,76,0.15)',

                          cursor:
                            'pointer',
                        }}
                      >

                        {l === 'CFB'
                          ? 'NCAAF'
                          : l}

                      </button>
                    )
                  )}

                </div>

              </div>


              <div>

                <label
                  className="block text-xs text-slate mb-1 uppercase tracking-widest"
                  style={{
                    fontFamily:
                      'var(--font-mono)',
                  }}
                >
                  Week
                </label>


                <select
                  className="rounded px-3 py-2 text-sm"

                  value={
                    week
                  }

                  onChange={
                    e => {
                      setWeek(
                        Number(
                          e.target.value
                        )
                      );

                      setCard(
                        null
                      );

                      setError(
                        ''
                      );
                    }
                  }

                  style={{
                    minWidth:
                      100,
                  }}
                >

                  {Array.from(
                    {
                      length:
                        18,
                    },
                    (
                      _,
                      i
                    ) =>
                      i + 1
                  ).map(
                    w => (
                      <option
                        key={w}
                        value={w}
                      >
                        Week {w}
                      </option>
                    )
                  )}

                </select>

              </div>


              <div>

                <label
                  className="block text-xs text-slate mb-1 uppercase tracking-widest"
                  style={{
                    fontFamily:
                      'var(--font-mono)',
                  }}
                >
                  Season
                </label>

                <div
                  className="rounded px-3 py-2 text-sm"
                  style={{
                    background:
                      'rgba(15,44,71,0.5)',

                    border:
                      '1px solid rgba(201,168,76,0.15)',

                    color:
                      '#8B9BB4',

                    fontFamily:
                      'var(--font-mono)',

                    minWidth:
                      90,

                    textAlign:
                      'center',
                  }}
                >
                  {CURRENT_SEASON}
                </div>

              </div>


              <button
                onClick={
                  fetchCard
                }

                disabled={
                  loading
                }

                className="score-display px-6 py-2 rounded"

                style={{
                  fontSize:
                    16,

                  letterSpacing:
                    '0.1em',

                  background:
                    loading
                      ? 'rgba(201,168,76,0.2)'
                      : 'linear-gradient(135deg, #C9A84C, #E8C96A)',

                  color:
                    loading
                      ? '#4A5568'
                      : '#030B14',

                  border:
                    'none',

                  cursor:
                    loading
                      ? 'not-allowed'
                      : 'pointer',
                }}
              >

                {loading
                  ? 'LOADING...'
                  : 'LOAD CARD'}

              </button>

            </div>

          </div>


          {/* =================================================
              Error
          ================================================= */}

          {error && (
            <div
              className="mb-4 px-3 py-2 rounded text-xs"

              style={{
                background:
                  'rgba(217,64,64,0.1)',

                color:
                  '#D94040',

                border:
                  '1px solid rgba(217,64,64,0.2)',

                fontFamily:
                  'var(--font-mono)',
              }}
            >
              {error}
            </div>
          )}


          {/* =================================================
              Card loaded
          ================================================= */}

          {card && (
            <>

              {/* =============================================
                  Summary
              ============================================= */}

              <div className="flex items-center justify-between mb-4 flex-wrap gap-3">

                <div className="flex gap-4 flex-wrap items-center">

                  <span className="text-chalk text-sm font-semibold">
                    {card.league ===
                    'CFB'
                      ? 'NCAAF'
                      : card.league}{' '}
                    Week {card.week}
                  </span>

                  <span
                    className="text-xs"
                    style={{
                      color:
                        '#C9A84C',

                      fontFamily:
                        'var(--font-mono)',
                    }}
                  >
                    {card.season ??
                      CURRENT_SEASON}{' '}
                    Season
                  </span>

                  <span
                    className="text-xs text-slate"
                    style={{
                      fontFamily:
                        'var(--font-mono)',
                    }}
                  >
                    {card.total_games}{' '}
                    games
                  </span>

                  <span
                    className="text-xs"
                    style={{
                      color:
                        '#3DAA6A',

                      fontFamily:
                        'var(--font-mono)',
                    }}
                  >
                    {card.games_with_lines}{' '}
                    with lines
                  </span>

                  <span
                    className="text-xs"
                    style={{
                      color:
                        '#C9A84C',

                      fontFamily:
                        'var(--font-mono)',
                    }}
                  >
                    {card.games_with_movement}{' '}
                    with movement data
                  </span>

                  {card.league === 'CFB' &&
                    card.cfb_sp_model_games != null && (
                    <span
                      className="text-xs"
                      style={{
                        color:
                          '#8B9BB4',

                        fontFamily:
                          'var(--font-mono)',
                      }}
                    >
                      {card.cfb_sp_model_games}{' '}
                      SP+ ·{' '}
                      {card.cfb_fallback_games ??
                        0}{' '}
                      fallback
                    </span>
                  )}

                </div>


                <div className="flex gap-2">

                  {(
                    [
                      'all',
                      'edges',
                      'sharp',
                    ] as const
                  ).map(
                    f => (
                      <button
                        key={f}

                        onClick={() =>
                          setFilter(
                            f
                          )
                        }

                        className="score-display px-3 py-1 rounded text-xs"

                        style={{
                          fontSize:
                            11,

                          letterSpacing:
                            '0.08em',

                          background:
                            filter === f
                              ? '#C9A84C'
                              : 'rgba(15,44,71,0.5)',

                          color:
                            filter === f
                              ? '#030B14'
                              : '#8B9BB4',

                          border:
                            '1px solid',

                          borderColor:
                            filter === f
                              ? '#C9A84C'
                              : 'rgba(201,168,76,0.15)',

                          cursor:
                            'pointer',
                        }}
                      >

                        {f === 'all'
                          ? 'ALL'
                          : f ===
                              'edges'
                            ? '⚡ EDGES'
                            : '🔥 SHARP'}

                      </button>
                    )
                  )}

                </div>

              </div>


              {/* =============================================
                  No games
              ============================================= */}

              {displayed.length ===
                0 && (
                <div className="panel rounded-xl p-8 text-center">

                  <p
                    className="text-slate text-sm"
                    style={{
                      fontFamily:
                        'var(--font-mono)',
                    }}
                  >
                    {filter ===
                    'edges'
                      ? 'No significant edges this week.'
                      : filter ===
                          'sharp'
                        ? 'No sharp movement detected yet.'
                        : `No ${
                            league ===
                            'CFB'
                              ? 'NCAAF'
                              : league
                          } games found for Week ${week}, ${CURRENT_SEASON}.`}
                  </p>

                </div>
              )}


              {/* =============================================
                  Games
              ============================================= */}

              <div className="space-y-3">

                {displayed.map(
                  (
                    game,
                    i
                  ) => {

                    const d =
                      game.disparity;

                    const p =
                      game.prediction;

                    const gamePickKey =
                      `${game.home_team}|${game.away_team}`;

                    const isOfficialPick =
                      officialPicks.has(
                        gamePickKey
                      );

                    const rankingScore =
                      d.ranking_score ??
                      d.edge_score ??
                      0;

                    const isSteam =
                      d.steam_move;

                    const isSharp =
                      d.sharp_signal >
                      0.4;

                    const expanded =
                      expandedGame ===
                      i;

                    const backendEdgeLabel =
                      d.edge_label ||
                      '';

                    const showEdgeBadge =
                      backendEdgeLabel !==
                        '' &&
                      backendEdgeLabel !==
                        '— Neutral';

                    const edgeBadgeStyle =
                      getEdgeBadgeStyle(
                        game
                      );

                    const isFallback =
                      game.prediction_mode ===
                        'opponent_adjusted_fallback' ||
                      game.prediction_mode ===
                        'historical_fallback';

                    const lowConfidence =
                      game.confidence ===
                        'low' ||
                      d.confidence ===
                        'low' ||
                      d.low_confidence ===
                        true;

                    const kickoff =
                      formatKickoff(
                        game.date
                      );


                    return (
                      <div
                        key={
                          game.game_id ??
                          i
                        }

                        className="panel rounded-xl overflow-hidden"

                        style={{
                          borderColor:
                            getBorderColor(
                              game
                            ),
                        }}
                      >

                        {/* ===================================
                            Main row
                        =================================== */}

                        <div
                          className="p-5 cursor-pointer"

                          onClick={() =>
                            setExpandedGame(
                              expanded
                                ? null
                                : i
                            )
                          }
                        >

                          <div className="flex items-start justify-between gap-4 flex-wrap">

                            {/* ===============================
                                Teams
                            =============================== */}

                            <div className="flex-1 min-w-0">

                              <div className="flex gap-2 flex-wrap mb-2">

                                {showEdgeBadge && (
                                  <span
                                    className="score-display text-xs px-2 py-0.5 rounded"

                                    style={{
                                      ...edgeBadgeStyle,

                                      letterSpacing:
                                        '0.08em',
                                    }}
                                  >
                                    {backendEdgeLabel}
                                  </span>
                                )}


                                {lowConfidence && (
                                  <span
                                    className="text-xs px-2 py-0.5 rounded"

                                    style={{
                                      background:
                                        'rgba(139,155,180,0.08)',

                                      color:
                                        '#8B9BB4',

                                      border:
                                        '1px solid rgba(139,155,180,0.18)',

                                      fontFamily:
                                        'var(--font-mono)',
                                    }}
                                  >
                                    LOW CONFIDENCE
                                  </span>
                                )}


                                {isFallback && (
                                  <span
                                    className="text-xs px-2 py-0.5 rounded"

                                    style={{
                                      background:
                                        'rgba(201,168,76,0.06)',

                                      color:
                                        '#C9A84C',

                                      border:
                                        '1px solid rgba(201,168,76,0.16)',

                                      fontFamily:
                                        'var(--font-mono)',
                                    }}
                                  >
                                    SRS FALLBACK
                                  </span>
                                )}


                                {isSteam && (
                                  <span
                                    className="score-display text-xs px-2 py-0.5 rounded"

                                    style={{
                                      background:
                                        'rgba(217,64,64,0.12)',

                                      color:
                                        '#D94040',

                                      border:
                                        '1px solid rgba(217,64,64,0.3)',

                                      letterSpacing:
                                        '0.08em',
                                    }}
                                  >
                                    🔥 STEAM MOVE
                                  </span>
                                )}


                                {isOfficialPick && (
                                  <span
                                    className="score-display text-xs px-2 py-0.5 rounded"

                                    style={{
                                      background:
                                        'rgba(61,170,106,0.2)',

                                      color:
                                        '#3DAA6A',

                                      border:
                                        '1px solid rgba(61,170,106,0.5)',

                                      letterSpacing:
                                        '0.08em',

                                      fontSize:
                                        12,
                                    }}
                                  >
                                    ✅ OFFICIAL PICK
                                  </span>
                                )}


                                {isSharp &&
                                  !isSteam && (
                                  <span
                                    className="score-display text-xs px-2 py-0.5 rounded"

                                    style={{
                                      background:
                                        'rgba(201,168,76,0.1)',

                                      color:
                                        '#C9A84C',

                                      border:
                                        '1px solid rgba(201,168,76,0.25)',

                                      letterSpacing:
                                        '0.08em',
                                    }}
                                  >
                                    ⚡ SHARP ACTION
                                  </span>
                                )}


                                {d.sharp_aligned && (
                                  <span
                                    className="text-xs px-2 py-0.5 rounded"

                                    style={{
                                      background:
                                        'rgba(61,170,106,0.1)',

                                      color:
                                        '#3DAA6A',

                                      border:
                                        '1px solid rgba(61,170,106,0.25)',

                                      fontFamily:
                                        'var(--font-mono)',
                                    }}
                                  >
                                    ✓ aligned
                                  </span>
                                )}

                              </div>


                              <div
                                className="mb-3 flex items-center gap-2 flex-wrap"
                                style={{
                                  fontFamily:
                                    'var(--font-mono)',
                                }}
                              >
                                {kickoff.relative && (
                                  <span
                                    className="text-xs px-2 py-1 rounded"
                                    style={{
                                      background:
                                        kickoff.relative === 'TODAY'
                                          ? 'rgba(61,170,106,0.1)'
                                          : 'rgba(201,168,76,0.08)',

                                      color:
                                        kickoff.relative === 'TODAY'
                                          ? '#3DAA6A'
                                          : '#C9A84C',

                                      border:
                                        kickoff.relative === 'TODAY'
                                          ? '1px solid rgba(61,170,106,0.24)'
                                          : '1px solid rgba(201,168,76,0.2)',

                                      fontWeight:
                                        700,

                                      letterSpacing:
                                        '0.08em',
                                    }}
                                  >
                                    {kickoff.relative}
                                  </span>
                                )}

                                <span
                                  className="text-xs"
                                  style={{
                                    color:
                                      '#F0EEE6',

                                    fontWeight:
                                      600,
                                  }}
                                >
                                  {kickoff.dateLabel}
                                </span>

                                <span
                                  className="text-xs"
                                  style={{
                                    color:
                                      '#C9A84C',

                                    fontWeight:
                                      700,
                                  }}
                                >
                                  • {kickoff.timeLabel}
                                </span>

                                {game.venue && (
                                  <span
                                    className="text-xs"
                                    style={{
                                      color:
                                        '#596A80',
                                    }}
                                  >
                                    • {game.venue}
                                  </span>
                                )}
                              </div>


                              <div className="flex items-center gap-3">

                                <div className="text-center">

                                  <div
                                    className="text-slate text-xs mb-0.5"
                                    style={{
                                      fontFamily:
                                        'var(--font-mono)',
                                    }}
                                  >
                                    AWAY
                                  </div>

                                  <div className="text-chalk text-sm font-semibold">
                                    {game.away_team}
                                  </div>

                                  <div
                                    className="score-display"
                                    style={{
                                      fontSize:
                                        32,

                                      color:
                                        '#8B9BB4',
                                    }}
                                  >
                                    {Math.round(
                                      p.predicted_away_score
                                    )}
                                  </div>

                                </div>


                                <div
                                  className="text-slate score-display"
                                  style={{
                                    fontSize:
                                      18,
                                  }}
                                >
                                  @
                                </div>


                                <div className="text-center">

                                  <div
                                    className="text-slate text-xs mb-0.5"
                                    style={{
                                      fontFamily:
                                        'var(--font-mono)',
                                    }}
                                  >
                                    HOME
                                  </div>

                                  <div className="text-chalk text-sm font-semibold">
                                    {game.home_team}
                                  </div>

                                  <div
                                    className="score-display"
                                    style={{
                                      fontSize:
                                        32,

                                      color:
                                        '#F0EEE6',
                                    }}
                                  >
                                    {Math.round(
                                      p.predicted_home_score
                                    )}
                                  </div>

                                </div>

                              </div>

                            </div>


                            {/* ===============================
                                Lines
                            =============================== */}

                            <div className="shrink-0 min-w-[190px]">

                              <div className="grid grid-cols-3 gap-1 text-center mb-1">

                                <div />

                                <div
                                  className="text-xs text-slate"
                                  style={{
                                    fontFamily:
                                      'var(--font-mono)',
                                  }}
                                >
                                  Vegas
                                </div>

                                <div
                                  className="text-xs"
                                  style={{
                                    color:
                                      '#C9A84C',

                                    fontFamily:
                                      'var(--font-mono)',
                                  }}
                                >
                                  PP
                                </div>

                              </div>


                              <div className="grid grid-cols-3 gap-1 text-center mb-1">

                                <div
                                  className="text-xs text-slate"
                                  style={{
                                    fontFamily:
                                      'var(--font-mono)',
                                  }}
                                >
                                  Spread
                                </div>

                                <div className="text-sm font-mono text-chalk">
                                  {d.vegas_spread != null
                                    ? (
                                      d.vegas_spread >
                                      0
                                        ? `+${d.vegas_spread}`
                                        : d.vegas_spread
                                    )
                                    : '—'}
                                </div>

                                <div
                                  className="text-sm font-mono"
                                  style={{
                                    color:
                                      '#C9A84C',
                                  }}
                                >
                                  {p.predicted_margin >
                                  0
                                    ? `+${p.predicted_margin.toFixed(1)}`
                                    : p.predicted_margin.toFixed(1)}
                                </div>

                              </div>


                              <div className="grid grid-cols-3 gap-1 text-center mb-3">

                                <div
                                  className="text-xs text-slate"
                                  style={{
                                    fontFamily:
                                      'var(--font-mono)',
                                  }}
                                >
                                  Total
                                </div>

                                <div className="text-sm font-mono text-chalk">
                                  {d.vegas_total ??
                                    '—'}
                                </div>

                                <div
                                  className="text-sm font-mono"
                                  style={{
                                    color:
                                      '#C9A84C',
                                  }}
                                >
                                  {p.predicted_total.toFixed(
                                    1
                                  )}
                                </div>

                              </div>


                              <div className="flex flex-col gap-1">

                                {d.spread_edge_type &&
                                  d.spread_edge_type !==
                                    'neutral' && (
                                  <span
                                    className="text-xs px-2 py-0.5 rounded text-center"

                                    style={{
                                      fontFamily:
                                        'var(--font-mono)',

                                      background:
                                        `${EDGE_COLORS[
                                          d.spread_edge_type
                                        ]}18`,

                                      color:
                                        EDGE_COLORS[
                                          d.spread_edge_type
                                        ],

                                      border:
                                        `1px solid ${
                                          EDGE_COLORS[
                                            d.spread_edge_type
                                          ]
                                        }40`,
                                    }}
                                  >
                                    {EDGE_LABELS[
                                      d.spread_edge_type
                                    ]}{' '}
                                    (
                                    {d.spread_disparity !=
                                    null
                                      ? (
                                        d.spread_disparity >
                                        0
                                          ? `+${d.spread_disparity}`
                                          : d.spread_disparity
                                      )
                                      : ''}
                                    pts)
                                  </span>
                                )}


                                {d.total_edge_type &&
                                  d.total_edge_type !==
                                    'neutral' && (
                                  <span
                                    className="text-xs px-2 py-0.5 rounded text-center"

                                    style={{
                                      fontFamily:
                                        'var(--font-mono)',

                                      background:
                                        `${EDGE_COLORS[
                                          d.total_edge_type
                                        ]}18`,

                                      color:
                                        EDGE_COLORS[
                                          d.total_edge_type
                                        ],

                                      border:
                                        `1px solid ${
                                          EDGE_COLORS[
                                            d.total_edge_type
                                          ]
                                        }40`,
                                    }}
                                  >
                                    {EDGE_LABELS[
                                      d.total_edge_type
                                    ]}{' '}
                                    (
                                    {d.total_disparity !=
                                    null
                                      ? (
                                        d.total_disparity >
                                        0
                                          ? `+${d.total_disparity}`
                                          : d.total_disparity
                                      )
                                      : ''}
                                    pts)
                                  </span>
                                )}

                              </div>

                            </div>

                          </div>


                          {game.model_note && (
                            <div
                              className="mt-3 text-xs px-2 py-1.5 rounded"

                              style={{
                                background:
                                  'rgba(201,168,76,0.05)',

                                color:
                                  '#8B9BB4',

                                border:
                                  '1px solid rgba(201,168,76,0.12)',

                                fontFamily:
                                  'var(--font-mono)',
                              }}
                            >
                              {game.model_note}
                            </div>
                          )}


                          {(
                            game.home_injury_notes.length >
                              0 ||
                            game.away_injury_notes.length >
                              0 ||
                            game.roster_notes.length >
                              0
                          ) && (
                            <div className="mt-3 flex flex-wrap gap-1">

                              {[
                                ...game.home_injury_notes,
                                ...game.away_injury_notes,
                              ]
                                .slice(
                                  0,
                                  3
                                )
                                .map(
                                  (
                                    note,
                                    j
                                  ) => (
                                    <span
                                      key={j}
                                      className="text-xs px-2 py-0.5 rounded"

                                      style={{
                                        background:
                                          note.startsWith(
                                            '❌'
                                          )
                                            ? 'rgba(217,64,64,0.1)'
                                            : note.startsWith(
                                                '⚠'
                                              )
                                              ? 'rgba(201,168,76,0.1)'
                                              : 'rgba(139,155,180,0.1)',

                                        color:
                                          note.startsWith(
                                            '❌'
                                          )
                                            ? '#D94040'
                                            : note.startsWith(
                                                '⚠'
                                              )
                                              ? '#C9A84C'
                                              : '#8B9BB4',

                                        fontFamily:
                                          'var(--font-mono)',
                                      }}
                                    >
                                      {note}
                                    </span>
                                  )
                                )}


                              {game.home_injury_notes.length +
                                game.away_injury_notes.length >
                                3 && (
                                <span
                                  className="text-xs px-2 py-0.5 rounded"

                                  style={{
                                    color:
                                      '#4A5568',

                                    fontFamily:
                                      'var(--font-mono)',
                                  }}
                                >
                                  +
                                  {game.home_injury_notes.length +
                                    game.away_injury_notes.length -
                                    3}{' '}
                                  more
                                </span>
                              )}

                            </div>
                          )}


                          <div
                            className="mt-2 text-xs text-slate"

                            style={{
                              fontFamily:
                                'var(--font-mono)',

                              opacity:
                                0.5,
                            }}
                          >
                            {expanded
                              ? '▲ collapse'
                              : '▼ details'}
                          </div>

                        </div>


                        {/* ===================================
                            Expanded
                        =================================== */}

                        {expanded && (
                          <div
                            className="border-t px-5 py-4 space-y-4"

                            style={{
                              borderColor:
                                'rgba(201,168,76,0.1)',
                            }}
                          >

                            <ExplanationPanel
                              game={game}
                              lowConfidence={lowConfidence}
                              isFallback={isFallback}
                            />


                            <div>

                              <div
                                className="text-xs text-slate mb-2 uppercase tracking-widest"

                                style={{
                                  fontFamily:
                                    'var(--font-mono)',
                                }}
                              >
                                Prediction Model
                              </div>


                              <div
                                className="text-xs text-slate"

                                style={{
                                  fontFamily:
                                    'var(--font-mono)',
                                }}
                              >
                                Mode:{' '}
                                <span
                                  style={{
                                    color:
                                      '#F0EEE6',
                                  }}
                                >
                                  {game.prediction_mode ||
                                    'unknown'}
                                </span>
                              </div>


                              <div
                                className="text-xs text-slate"

                                style={{
                                  fontFamily:
                                    'var(--font-mono)',
                                }}
                              >
                                Confidence:{' '}

                                <span
                                  style={{
                                    color:
                                      lowConfidence
                                        ? '#C9A84C'
                                        : '#3DAA6A',
                                  }}
                                >
                                  {game.confidence ||
                                    d.confidence ||
                                    'unrated'}
                                </span>
                              </div>


                              {d.raw_edge_score != null && (
                                <div
                                  className="text-xs text-slate"

                                  style={{
                                    fontFamily:
                                      'var(--font-mono)',
                                  }}
                                >
                                  Raw edge score:{' '}
                                  {d.raw_edge_score.toFixed(
                                    1
                                  )}
                                </div>
                              )}


                              {d.ranking_score != null && (
                                <div
                                  className="text-xs text-slate"

                                  style={{
                                    fontFamily:
                                      'var(--font-mono)',
                                  }}
                                >
                                  Ranking score:{' '}
                                  {d.ranking_score.toFixed(
                                    1
                                  )}
                                </div>
                              )}

                            </div>


                            {game.line_movement && (
                              <div>

                                <div
                                  className="text-xs text-slate mb-2 uppercase tracking-widest"

                                  style={{
                                    fontFamily:
                                      'var(--font-mono)',
                                  }}
                                >
                                  Line Movement
                                  {' '}
                                  (Last{' '}
                                  {
                                    game.line_movement.hours_tracked
                                  }
                                  h)
                                </div>


                                <MovementBadge
                                  mv={
                                    game.line_movement
                                  }
                                />

                              </div>
                            )}


                            {game.home_injury_notes.length >
                              0 && (
                              <div>

                                <div
                                  className="text-xs mb-1 uppercase tracking-widest"

                                  style={{
                                    fontFamily:
                                      'var(--font-mono)',

                                    color:
                                      '#C9A84C',
                                  }}
                                >
                                  {game.home_team}{' '}
                                  Injuries
                                </div>


                                <div className="space-y-1">

                                  {game.home_injury_notes.map(
                                    (
                                      n,
                                      j
                                    ) => (
                                      <div
                                        key={j}
                                        className="text-xs text-slate"

                                        style={{
                                          fontFamily:
                                            'var(--font-mono)',
                                        }}
                                      >
                                        {n}
                                      </div>
                                    )
                                  )}

                                </div>

                              </div>
                            )}


                            {game.away_injury_notes.length >
                              0 && (
                              <div>

                                <div
                                  className="text-xs mb-1 uppercase tracking-widest"

                                  style={{
                                    fontFamily:
                                      'var(--font-mono)',

                                    color:
                                      '#C9A84C',
                                  }}
                                >
                                  {game.away_team}{' '}
                                  Injuries
                                </div>


                                <div className="space-y-1">

                                  {game.away_injury_notes.map(
                                    (
                                      n,
                                      j
                                    ) => (
                                      <div
                                        key={j}
                                        className="text-xs text-slate"

                                        style={{
                                          fontFamily:
                                            'var(--font-mono)',
                                        }}
                                      >
                                        {n}
                                      </div>
                                    )
                                  )}

                                </div>

                              </div>
                            )}


                            <div>

                              <div
                                className="text-xs text-slate mb-2 uppercase tracking-widest"

                                style={{
                                  fontFamily:
                                    'var(--font-mono)',
                                }}
                              >
                                Rating Adjustments Applied
                              </div>


                              <div
                                className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs"

                                style={{
                                  fontFamily:
                                    'var(--font-mono)',
                                }}
                              >

                                {[
                                  {
                                    label:
                                      `${game.home_team} Roster`,

                                    val:
                                      game.adjustments.home_roster,
                                  },

                                  {
                                    label:
                                      `${game.away_team} Roster`,

                                    val:
                                      game.adjustments.away_roster,
                                  },

                                  {
                                    label:
                                      `${game.home_team} Injuries`,

                                    val:
                                      game.adjustments.home_injury,
                                  },

                                  {
                                    label:
                                      `${game.away_team} Injuries`,

                                    val:
                                      game.adjustments.away_injury,
                                  },
                                ].map(
                                  a => (
                                    <div
                                      key={
                                        a.label
                                      }

                                      className="flex justify-between gap-3"
                                    >

                                      <span className="text-slate">
                                        {a.label}
                                      </span>


                                      <span
                                        style={{
                                          color:
                                            a.val >
                                            0
                                              ? '#3DAA6A'
                                              : a.val <
                                                  0
                                                ? '#D94040'
                                                : '#4A5568',
                                        }}
                                      >
                                        {a.val >
                                        0
                                          ? `+${a.val.toFixed(
                                              2
                                            )}`
                                          : a.val.toFixed(
                                              2
                                            )}{' '}
                                        pts
                                      </span>

                                    </div>
                                  )
                                )}

                              </div>

                            </div>


                            <div>

                              <div
                                className="text-xs text-slate mb-2 uppercase tracking-widest"

                                style={{
                                  fontFamily:
                                    'var(--font-mono)',
                                }}
                              >
                                80% Confidence Intervals
                              </div>


                              <div
                                className="text-xs text-slate"

                                style={{
                                  fontFamily:
                                    'var(--font-mono)',
                                }}
                              >

                                {p.margin_80_lo != null &&
                                p.margin_80_hi != null
                                  ? (
                                    <>
                                      Margin:{' '}
                                      {p.margin_80_lo.toFixed(
                                        1
                                      )}
                                      {' '}
                                      to{' '}
                                      {p.margin_80_hi.toFixed(
                                        1
                                      )}
                                    </>
                                  )
                                  : (
                                    <>
                                      Margin:{' '}
                                      {p.predicted_margin.toFixed(
                                        1
                                      )}{' '}
                                      pts
                                    </>
                                  )}

                                {' '}
                                |{' '}

                                {p.total_80_lo != null &&
                                p.total_80_hi != null
                                  ? (
                                    <>
                                      Total:{' '}
                                      {p.total_80_lo.toFixed(
                                        1
                                      )}
                                      {' '}
                                      to{' '}
                                      {p.total_80_hi.toFixed(
                                        1
                                      )}
                                    </>
                                  )
                                  : (
                                    <>
                                      Total:{' '}
                                      {p.predicted_total.toFixed(
                                        1
                                      )}{' '}
                                      pts
                                    </>
                                  )}

                              </div>

                            </div>

                          </div>
                        )}

                      </div>
                    );
                  }
                )}

              </div>


              <p
                className="text-center text-xs text-slate mt-6"

                style={{
                  fontFamily:
                    'var(--font-mono)',

                  opacity:
                    0.4,
                }}
              >
                Statistical model only.
                Not financial advice.
                Generated{' '}
                {new Date(
                  card.generated_at
                ).toLocaleString()}
              </p>

            </>
          )}


          {!card &&
            !loading && (
            <div className="panel rounded-xl p-10 text-center">

              <div
                className="score-display text-slate mb-2"

                style={{
                  fontSize:
                    24,

                  letterSpacing:
                    '0.1em',
                }}
              >
                SELECT WEEK & LOAD CARD
              </div>


              <p
                className="text-xs text-slate"

                style={{
                  fontFamily:
                    'var(--font-mono)',
                }}
              >
                Strong Edge and Sharp plays are
                prioritized first, then shown in
                chronological kickoff order · Game
                dates and times shown in Eastern Time ·
                Injuries and market movement included
              </p>

            </div>
          )}

        </div>

      </div>
    </>
  );
}
