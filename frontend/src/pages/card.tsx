import { useState, useEffect, useCallback } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import RecordWidget from '../components/RecordWidget';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const EDGE_COLORS: Record<string, string> = {
  fade_away: '#D94040', fade_home: '#D94040',
  lean_home: '#C9A84C', lean_away: '#C9A84C',
  lean_over: '#3DAA6A', lean_under: '#8B9BB4',
  neutral: '#4A5568',
};
const EDGE_LABELS: Record<string, string> = {
  fade_away: '🔄 FADE AWAY', fade_home: '🔄 FADE HOME',
  lean_home: '↑ LEAN HOME', lean_away: '↓ LEAN AWAY',
  lean_over: '⬆ LEAN OVER', lean_under: '⬇ LEAN UNDER',
  neutral: '— NEUTRAL',
};

type Movement = {
  spread_open: number | null; spread_current: number | null; spread_move: number | null;
  total_open: number | null; total_current: number | null; total_move: number | null;
  sharp_signal: number; steam_move: boolean; move_direction: string | null; hours_tracked: number;
};

type GameCard = {
  home_team: string; away_team: string; date: string;
  prediction: { predicted_home_score: number; predicted_away_score: number; predicted_margin: number; predicted_total: number; home_win_prob: number };
  disparity: { has_line: boolean; spread_disparity: number | null; total_disparity: number | null; edge_score: number | null; edge_label: string; vegas_spread: number | null; vegas_total: number | null; spread_edge_type: string | null; total_edge_type: string | null; sharp_signal: number; steam_move: boolean; sharp_aligned: boolean };
  line_movement: Movement | null;
  home_injury_notes: string[]; away_injury_notes: string[];
  roster_notes: string[];
  adjustments: { home_roster: number; away_roster: number; home_injury: number; away_injury: number };
};

type CardData = {
  league: string; week: number; total_games: number; games_with_lines: number;
  games_with_movement: number; games: GameCard[]; generated_at: string; error?: string;
};

function MovementBadge({ mv }: { mv: Movement }) {
  if (!mv || mv.spread_move === null) return null;
  const abs = Math.abs(mv.spread_move);
  const dir = mv.spread_move > 0 ? '→ HOME' : '→ AWAY';
  const color = mv.steam_move ? '#D94040' : mv.sharp_signal > 0.5 ? '#C9A84C' : '#8B9BB4';
  const label = mv.steam_move ? '🔥 STEAM' : mv.sharp_signal > 0.5 ? '⚡ SHARP' : '○ MOVE';
  return (
    <div className="rounded px-2 py-1.5 text-xs" style={{ background: `${color}12`, border: `1px solid ${color}30`, fontFamily: 'var(--font-mono)' }}>
      <div style={{ color }} className="font-semibold">{label}</div>
      <div className="text-slate mt-0.5">
        Spread: {mv.spread_open} → {mv.spread_current}
        <span style={{ color }} className="ml-1">({mv.spread_move > 0 ? '+' : ''}{mv.spread_move} {dir})</span>
      </div>
      {mv.total_move !== null && (
        <div className="text-slate">Total: {mv.total_open} → {mv.total_current} ({mv.total_move > 0 ? '+' : ''}{mv.total_move})</div>
      )}
      <div className="text-slate">Signal: {(mv.sharp_signal * 100).toFixed(0)}% · {mv.hours_tracked}h tracked</div>
    </div>
  );
}

export default function CardPage() {
  const { user, userStatus, loading: authLoading } = useAuth();
  const router = useRouter();
  const [league, setLeague] = useState<'NFL' | 'CFB'>('NFL');
  const [week, setWeek] = useState(1);
  const [card, setCard] = useState<CardData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [filter, setFilter] = useState<'all' | 'edges' | 'sharp'>('all');
  const [expandedGame, setExpandedGame] = useState<number | null>(null);
  const [officialPicks, setOfficialPicks] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (!authLoading && (!user || userStatus !== 'approved')) router.replace('/login');
  }, [user, userStatus, authLoading, router]);

  // Fetch this week's locked official picks
  const fetchOfficialPicks = useCallback(async () => {
    if (!card) return;
    try {
      const r = await axios.get(`/api/picks?week=${week}&league=${league}`);
      const picks = r.data.picks || [];
      const keys = new Set<string>(picks.map((p: any) => `${p.home_team}|${p.away_team}`));
      setOfficialPicks(keys);
    } catch {}
  }, [week, league, card]);

  useEffect(() => { if (card) fetchOfficialPicks(); }, [card, fetchOfficialPicks]);

  const fetchCard = useCallback(async () => {
    setLoading(true); setError(''); setCard(null);
    try {
      const r = await axios.get(`${API}/card/${league}`, { params: { week } });
      setCard(r.data);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to load card.');
    } finally {
      setLoading(false); }
  }, [league, week]);

  if (authLoading || !user || userStatus !== 'approved') return <div className="field-bg min-h-screen flex items-center justify-center"><div className="animate-pulse-gold score-display text-slate" style={{ fontSize: 18 }}>LOADING...</div></div>;

  const displayed = (card?.games || []).filter(g => {
    if (filter === 'edges') return (g.disparity.edge_score || 0) >= 8;
    if (filter === 'sharp') return g.disparity.sharp_signal > 0.4 || g.disparity.steam_move;
    return true;
  });

  return (
    <>
      <Head><title>Weekly Card — Prime Picks</title></Head>
      <div className="field-bg min-h-screen px-4 py-10">
        <div className="max-w-5xl mx-auto">
          <div className="flex items-center justify-between mb-2 flex-wrap gap-3">
            <div className="flex items-center gap-3">
              <span style={{ fontSize: 28 }}>🏈</span>
              <h1 className="score-display text-chalk" style={{ fontSize: 36, letterSpacing: '0.08em' }}>PRIME PICKS</h1>
            </div>
            <nav className="flex gap-4 text-xs" style={{ fontFamily: 'var(--font-mono)' }}>
              <Link href="/" style={{ color: '#8B9BB4' }}>Predict</Link>
              <Link href="/card" style={{ color: '#C9A84C' }}>Weekly Card</Link>
              <Link href="/roster" style={{ color: '#8B9BB4' }}>Roster Intel</Link>
              <Link href="/record" style={{ color: '#8B9BB4' }}>Record</Link>
            </nav>
          </div>
          <div className="gold-line mb-6" />

          {/* Record Widget */}
          <div className="mb-5">
            <RecordWidget compact={true} />
          </div>

          {/* Controls */}
          <div className="panel rounded-xl p-5 mb-5">
            <div className="flex flex-wrap gap-4 items-end">
              <div>
                <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>League</label>
                <div className="flex gap-2">
                  {(['NFL','CFB'] as const).map(l => (
                    <button key={l} onClick={() => { setLeague(l); setCard(null); }}
                      className="score-display px-4 py-2 rounded"
                      style={{ fontSize: 16, letterSpacing: '0.08em', background: league === l ? '#C9A84C' : 'rgba(15,44,71,0.5)', color: league === l ? '#030B14' : '#8B9BB4', border: '1px solid', borderColor: league === l ? '#C9A84C' : 'rgba(201,168,76,0.15)', cursor: 'pointer' }}>
                      {l === 'CFB' ? 'NCAAF' : l}
                    </button>
                  ))}
                </div>
              </div>
              <div>
                <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>Week</label>
                <select className="rounded px-3 py-2 text-sm" value={week} onChange={e => setWeek(Number(e.target.value))} style={{ minWidth: 90 }}>
                  {Array.from({ length: 18 }, (_, i) => i + 1).map(w => <option key={w} value={w}>Week {w}</option>)}
                </select>
              </div>
              <button onClick={fetchCard} disabled={loading}
                className="score-display px-6 py-2 rounded"
                style={{ fontSize: 16, letterSpacing: '0.1em', background: loading ? 'rgba(201,168,76,0.2)' : 'linear-gradient(135deg, #C9A84C, #E8C96A)', color: loading ? '#4A5568' : '#030B14', border: 'none', cursor: loading ? 'not-allowed' : 'pointer' }}>
                {loading ? 'LOADING...' : 'LOAD CARD'}
              </button>
            </div>
          </div>

          {error && <div className="mb-4 px-3 py-2 rounded text-xs" style={{ background: 'rgba(217,64,64,0.1)', color: '#D94040', border: '1px solid rgba(217,64,64,0.2)', fontFamily: 'var(--font-mono)' }}>{error}</div>}

          {card && (
            <>
              {/* Summary bar */}
              <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
                <div className="flex gap-4 flex-wrap">
                  <span className="text-chalk text-sm font-semibold">{card.league} Week {card.week}</span>
                  <span className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>{card.total_games} games</span>
                  <span className="text-xs" style={{ color: '#3DAA6A', fontFamily: 'var(--font-mono)' }}>{card.games_with_lines} with lines</span>
                  <span className="text-xs" style={{ color: '#C9A84C', fontFamily: 'var(--font-mono)' }}>{card.games_with_movement} with movement data</span>
                </div>
                <div className="flex gap-2">
                  {(['all', 'edges', 'sharp'] as const).map(f => (
                    <button key={f} onClick={() => setFilter(f)}
                      className="score-display px-3 py-1 rounded text-xs"
                      style={{ fontSize: 11, letterSpacing: '0.08em', background: filter === f ? '#C9A84C' : 'rgba(15,44,71,0.5)', color: filter === f ? '#030B14' : '#8B9BB4', border: '1px solid', borderColor: filter === f ? '#C9A84C' : 'rgba(201,168,76,0.15)', cursor: 'pointer' }}>
                      {f === 'all' ? 'ALL' : f === 'edges' ? '⚡ EDGES' : '🔥 SHARP'}
                    </button>
                  ))}
                </div>
              </div>

              {displayed.length === 0 && (
                <div className="panel rounded-xl p-8 text-center">
                  <p className="text-slate text-sm" style={{ fontFamily: 'var(--font-mono)' }}>
                    {filter === 'edges' ? 'No significant edges this week.' : filter === 'sharp' ? 'No sharp movement detected yet.' : 'No games found.'}
                  </p>
                </div>
              )}

              <div className="space-y-3">
                {displayed.map((game, i) => {
                  const d = game.disparity;
                  const p = game.prediction;
                  const gamePickKey = `${game.home_team}|${game.away_team}`;
                  const isOfficialPick = officialPicks.has(gamePickKey);
                  const isTopEdge = (d.edge_score || 0) >= 15;
                  const isMedEdge = (d.edge_score || 0) >= 8;
                  const isSteam = d.steam_move;
                  const isSharp = d.sharp_signal > 0.4;
                  const expanded = expandedGame === i;

                  return (
                    <div key={i} className="panel rounded-xl overflow-hidden" style={{ borderColor: isTopEdge ? 'rgba(201,168,76,0.5)' : isSteam ? 'rgba(217,64,64,0.4)' : isMedEdge ? 'rgba(201,168,76,0.2)' : 'rgba(201,168,76,0.08)' }}>
                      {/* Main row */}
                      <div className="p-5 cursor-pointer" onClick={() => setExpandedGame(expanded ? null : i)}>
                        <div className="flex items-start justify-between gap-4 flex-wrap">

                          {/* Left: teams + scores */}
                          <div className="flex-1 min-w-0">
                            <div className="flex gap-2 flex-wrap mb-2">
                              {isTopEdge && <span className="score-display text-xs px-2 py-0.5 rounded" style={{ background: 'rgba(201,168,76,0.15)', color: '#C9A84C', border: '1px solid rgba(201,168,76,0.3)', letterSpacing: '0.08em' }}>🔥 STRONG EDGE</span>}
                              {isSteam && <span className="score-display text-xs px-2 py-0.5 rounded" style={{ background: 'rgba(217,64,64,0.12)', color: '#D94040', border: '1px solid rgba(217,64,64,0.3)', letterSpacing: '0.08em' }}>🔥 STEAM MOVE</span>}
                              {isOfficialPick && <span className="score-display text-xs px-2 py-0.5 rounded" style={{ background: 'rgba(61,170,106,0.2)', color: '#3DAA6A', border: '1px solid rgba(61,170,106,0.5)', letterSpacing: '0.08em', fontSize: 12 }}>✅ OFFICIAL PICK</span>}
                          {isSharp && !isSteam && <span className="score-display text-xs px-2 py-0.5 rounded" style={{ background: 'rgba(201,168,76,0.1)', color: '#C9A84C', border: '1px solid rgba(201,168,76,0.25)', letterSpacing: '0.08em' }}>⚡ SHARP ACTION</span>}
                              {d.sharp_aligned && <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'rgba(61,170,106,0.1)', color: '#3DAA6A', border: '1px solid rgba(61,170,106,0.25)', fontFamily: 'var(--font-mono)' }}>✓ aligned</span>}
                            </div>
                            <div className="flex items-center gap-3">
                              <div className="text-center">
                                <div className="text-slate text-xs mb-0.5" style={{ fontFamily: 'var(--font-mono)' }}>AWAY</div>
                                <div className="text-chalk text-sm font-semibold">{game.away_team}</div>
                                <div className="score-display" style={{ fontSize: 32, color: '#8B9BB4' }}>{Math.round(p.predicted_away_score)}</div>
                              </div>
                              <div className="text-slate score-display" style={{ fontSize: 18 }}>@</div>
                              <div className="text-center">
                                <div className="text-slate text-xs mb-0.5" style={{ fontFamily: 'var(--font-mono)' }}>HOME</div>
                                <div className="text-chalk text-sm font-semibold">{game.home_team}</div>
                                <div className="score-display" style={{ fontSize: 32, color: '#F0EEE6' }}>{Math.round(p.predicted_home_score)}</div>
                              </div>
                            </div>
                          </div>

                          {/* Right: lines comparison + edges */}
                          <div className="shrink-0 min-w-[190px]">
                            <div className="grid grid-cols-3 gap-1 text-center mb-1">
                              <div />
                              <div className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>Vegas</div>
                              <div className="text-xs" style={{ color: '#C9A84C', fontFamily: 'var(--font-mono)' }}>PP</div>
                            </div>
                            <div className="grid grid-cols-3 gap-1 text-center mb-1">
                              <div className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>Spread</div>
                              <div className="text-sm font-mono text-chalk">{d.vegas_spread != null ? (d.vegas_spread > 0 ? `+${d.vegas_spread}` : d.vegas_spread) : '—'}</div>
                              <div className="text-sm font-mono" style={{ color: '#C9A84C' }}>{p.predicted_margin > 0 ? `+${p.predicted_margin.toFixed(1)}` : p.predicted_margin.toFixed(1)}</div>
                            </div>
                            <div className="grid grid-cols-3 gap-1 text-center mb-3">
                              <div className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>Total</div>
                              <div className="text-sm font-mono text-chalk">{d.vegas_total ?? '—'}</div>
                              <div className="text-sm font-mono" style={{ color: '#C9A84C' }}>{p.predicted_total.toFixed(1)}</div>
                            </div>
                            <div className="flex flex-col gap-1">
                              {d.spread_edge_type && d.spread_edge_type !== 'neutral' && (
                                <span className="text-xs px-2 py-0.5 rounded text-center" style={{ fontFamily: 'var(--font-mono)', background: `${EDGE_COLORS[d.spread_edge_type]}18`, color: EDGE_COLORS[d.spread_edge_type], border: `1px solid ${EDGE_COLORS[d.spread_edge_type]}40` }}>
                                  {EDGE_LABELS[d.spread_edge_type]} ({d.spread_disparity != null ? (d.spread_disparity > 0 ? `+${d.spread_disparity}` : d.spread_disparity) : ''}pts)
                                </span>
                              )}
                              {d.total_edge_type && d.total_edge_type !== 'neutral' && (
                                <span className="text-xs px-2 py-0.5 rounded text-center" style={{ fontFamily: 'var(--font-mono)', background: `${EDGE_COLORS[d.total_edge_type]}18`, color: EDGE_COLORS[d.total_edge_type], border: `1px solid ${EDGE_COLORS[d.total_edge_type]}40` }}>
                                  {EDGE_LABELS[d.total_edge_type]} ({d.total_disparity != null ? (d.total_disparity > 0 ? `+${d.total_disparity}` : d.total_disparity) : ''}pts)
                                </span>
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Quick injury/roster flags */}
                        {(game.home_injury_notes.length > 0 || game.away_injury_notes.length > 0 || game.roster_notes.length > 0) && (
                          <div className="mt-3 flex flex-wrap gap-1">
                            {[...game.home_injury_notes, ...game.away_injury_notes].slice(0, 3).map((note, j) => (
                              <span key={j} className="text-xs px-2 py-0.5 rounded" style={{ background: note.startsWith('❌') ? 'rgba(217,64,64,0.1)' : note.startsWith('⚠') ? 'rgba(201,168,76,0.1)' : 'rgba(139,155,180,0.1)', color: note.startsWith('❌') ? '#D94040' : note.startsWith('⚠') ? '#C9A84C' : '#8B9BB4', fontFamily: 'var(--font-mono)' }}>{note}</span>
                            ))}
                            {game.home_injury_notes.length + game.away_injury_notes.length > 3 && (
                              <span className="text-xs px-2 py-0.5 rounded" style={{ color: '#4A5568', fontFamily: 'var(--font-mono)' }}>+{game.home_injury_notes.length + game.away_injury_notes.length - 3} more</span>
                            )}
                          </div>
                        )}

                        <div className="mt-2 text-xs text-slate" style={{ fontFamily: 'var(--font-mono)', opacity: 0.5 }}>
                          {expanded ? '▲ collapse' : '▼ details'}
                        </div>
                      </div>

                      {/* Expanded detail panel */}
                      {expanded && (
                        <div className="border-t px-5 py-4 space-y-4" style={{ borderColor: 'rgba(201,168,76,0.1)' }}>
                          {/* Line movement */}
                          {game.line_movement && (
                            <div>
                              <div className="text-xs text-slate mb-2 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>Line Movement (Last {game.line_movement.hours_tracked}h)</div>
                              <MovementBadge mv={game.line_movement} />
                            </div>
                          )}

                          {/* Injuries detail */}
                          {game.home_injury_notes.length > 0 && (
                            <div>
                              <div className="text-xs mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)', color: '#C9A84C' }}>{game.home_team} Injuries</div>
                              <div className="space-y-1">
                                {game.home_injury_notes.map((n, j) => <div key={j} className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>{n}</div>)}
                              </div>
                            </div>
                          )}
                          {game.away_injury_notes.length > 0 && (
                            <div>
                              <div className="text-xs mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)', color: '#C9A84C' }}>{game.away_team} Injuries</div>
                              <div className="space-y-1">
                                {game.away_injury_notes.map((n, j) => <div key={j} className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>{n}</div>)}
                              </div>
                            </div>
                          )}

                          {/* Adjustments summary */}
                          <div>
                            <div className="text-xs text-slate mb-2 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>Rating Adjustments Applied</div>
                            <div className="grid grid-cols-2 gap-2 text-xs" style={{ fontFamily: 'var(--font-mono)' }}>
                              {[
                                { label: `${game.home_team} Roster`, val: game.adjustments.home_roster },
                                { label: `${game.away_team} Roster`, val: game.adjustments.away_roster },
                                { label: `${game.home_team} Injuries`, val: game.adjustments.home_injury },
                                { label: `${game.away_team} Injuries`, val: game.adjustments.away_injury },
                              ].map(a => (
                                <div key={a.label} className="flex justify-between">
                                  <span className="text-slate">{a.label}</span>
                                  <span style={{ color: a.val > 0 ? '#3DAA6A' : a.val < 0 ? '#D94040' : '#4A5568' }}>
                                    {a.val > 0 ? `+${a.val.toFixed(2)}` : a.val.toFixed(2)} pts
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Confidence intervals */}
                          <div>
                            <div className="text-xs text-slate mb-2 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>80% Confidence Intervals</div>
                            <div className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>
                              Margin: {p.predicted_margin.toFixed(1)} pts &nbsp;|&nbsp; Total: {p.predicted_total.toFixed(1)} pts
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <p className="text-center text-xs text-slate mt-6" style={{ fontFamily: 'var(--font-mono)', opacity: 0.4 }}>
                Statistical model only. Not financial advice. Generated {new Date(card.generated_at).toLocaleString()}
              </p>
            </>
          )}

          {!card && !loading && (
            <div className="panel rounded-xl p-10 text-center">
              <div className="score-display text-slate mb-2" style={{ fontSize: 24, letterSpacing: '0.1em' }}>SELECT WEEK & LOAD CARD</div>
              <p className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>Games ranked by Prime Picks edge vs Vegas line · Steam/sharp moves highlighted · Injuries cascaded through depth chart</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
