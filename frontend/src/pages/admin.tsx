import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';
import axios from 'axios';

const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const CURRENT_SEASON = new Date().getFullYear();

// ─────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────
type UserRecord = {
  uid: string; firstName: string; lastName: string;
  email: string; phone?: string; referral?: string;
  status: 'pending' | 'approved' | 'denied';
  createdAt?: { seconds: number };
};

type CardGame = {
  home_team: string; away_team: string; date: string;
  prediction: { predicted_home_score: number; predicted_away_score: number; predicted_margin: number; predicted_total: number };
  disparity: { vegas_spread: number | null; vegas_total: number | null; edge_score: number | null };
  league: string;
};

type LockedPick = {
  id: string; league: string; week: number; season: number;
  home_team: string; away_team: string;
  picked_team: string; spread_at_lock: number;
  predicted_home: number; predicted_away: number;
  predicted_margin: number; predicted_total: number;
  status: 'pending' | 'graded';
  result: 'win' | 'loss' | 'push' | null;
  actual_home: number | null; actual_away: number | null;
  locked_at_iso: string; verify_hash: string;
};

const statusColor = { pending: '#C9A84C', approved: '#3DAA6A', denied: '#D94040' };
const statusLabel = { pending: '⏳ Pending', approved: '✓ Approved', denied: '✗ Denied' };
const resultColor = { win: '#3DAA6A', loss: '#D94040', push: '#C9A84C' };
const resultLabel = { win: '✓ WIN', loss: '✗ LOSS', push: '— PUSH' };

// ─────────────────────────────────────────────
// Auth Gate
// ─────────────────────────────────────────────
function AuthGate({ onAuth }: { onAuth: (s: string) => void }) {
  const [secret, setSecret] = useState('');
  const [err, setErr] = useState('');
  return (
    <div className="field-bg min-h-screen flex items-center justify-center px-4">
      <div className="panel rounded-xl p-8 max-w-xs w-full">
        <h1 className="score-display text-chalk mb-6" style={{ fontSize: 22, letterSpacing: '0.1em' }}>🔐 ADMIN ACCESS</h1>
        <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>Admin Password</label>
        <input type="password" className="w-full rounded px-3 py-2.5 text-sm mb-3" value={secret}
          onChange={e => setSecret(e.target.value)} onKeyDown={e => e.key === 'Enter' && onAuth(secret)}
          placeholder="Enter admin password" />
        {err && <p className="text-xs mb-3" style={{ color: '#D94040', fontFamily: 'var(--font-mono)' }}>{err}</p>}
        <button onClick={() => onAuth(secret)}
          className="w-full score-display py-2.5 rounded"
          style={{ fontSize: 16, letterSpacing: '0.1em', background: 'linear-gradient(135deg, #C9A84C, #E8C96A)', color: '#030B14', border: 'none', cursor: 'pointer' }}>
          ENTER
        </button>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Users Tab
// ─────────────────────────────────────────────
function UsersTab({ secret }: { secret: string }) {
  const router = useRouter();
  const [users, setUsers] = useState<UserRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [filter, setFilter] = useState<'all'|'pending'|'approved'|'denied'>('all');
  const [toast, setToast] = useState('');

  const showToast = (m: string) => { setToast(m); setTimeout(() => setToast(''), 3000); };

  const fetchUsers = useCallback(async () => {
    setLoading(true);
    try {
      const r = await axios.post('/api/list-users', { secret });
      setUsers(r.data.users);
    } finally { setLoading(false); }
  }, [secret]);

  useEffect(() => { fetchUsers(); }, [fetchUsers]);

  // Handle email link actions
  const { action, uid: queryUid } = router.query;
  useEffect(() => {
    if (action && queryUid && typeof queryUid === 'string') {
      axios.post('/api/approve-user', { secret, uid: queryUid, action })
        .then(() => { showToast(`User ${action}d`); fetchUsers(); router.replace('/admin'); })
        .catch(() => {});
    }
  }, [action, queryUid]);

  const doAction = async (uid: string, act: 'approve'|'deny') => {
    setActionLoading(uid + act);
    try {
      await axios.post('/api/approve-user', { secret, uid, action: act });
      showToast(`User ${act}d successfully.`);
      fetchUsers();
    } finally { setActionLoading(null); }
  };

  const filtered = users.filter(u => filter === 'all' || u.status === filter);

  return (
    <div>
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2">
        <p className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>
          {users.filter(u => u.status === 'pending').length} pending · {users.filter(u => u.status === 'approved').length} approved
        </p>
        <div className="flex gap-2">
          {(['all','pending','approved','denied'] as const).map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className="score-display px-3 py-1 rounded text-xs"
              style={{ fontSize: 12, letterSpacing: '0.06em', background: filter === f ? '#C9A84C' : 'rgba(15,44,71,0.5)', color: filter === f ? '#030B14' : '#8B9BB4', border: '1px solid', borderColor: filter === f ? '#C9A84C' : 'rgba(201,168,76,0.15)', cursor: 'pointer' }}>
              {f.toUpperCase()}
            </button>
          ))}
        </div>
      </div>
      {toast && <div className="mb-4 px-4 py-2 rounded text-sm" style={{ background: 'rgba(61,170,106,0.12)', color: '#3DAA6A', border: '1px solid rgba(61,170,106,0.3)', fontFamily: 'var(--font-mono)' }}>{toast}</div>}
      {loading && <p className="text-slate text-sm" style={{ fontFamily: 'var(--font-mono)' }}>Loading...</p>}
      <div className="space-y-3">
        {filtered.map(u => (
          <div key={u.uid} className="panel rounded-xl p-5">
            <div className="flex items-start justify-between gap-4 flex-wrap">
              <div className="flex-1">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="text-chalk font-semibold text-sm">{u.firstName} {u.lastName}</span>
                  <span className="text-xs px-2 py-0.5 rounded" style={{ fontFamily: 'var(--font-mono)', background: `${statusColor[u.status]}18`, color: statusColor[u.status], border: `1px solid ${statusColor[u.status]}40` }}>
                    {statusLabel[u.status]}
                  </span>
                </div>
                <p className="text-xs text-slate mt-1" style={{ fontFamily: 'var(--font-mono)' }}>📧 {u.email}</p>
                {u.phone && <p className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>📱 {u.phone}</p>}
                {u.referral && <p className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>🔍 {u.referral}</p>}
                {u.createdAt && <p className="text-xs" style={{ color: '#4A5568', fontFamily: 'var(--font-mono)' }}>{new Date(u.createdAt.seconds * 1000).toLocaleDateString()}</p>}
              </div>
              <div className="flex gap-2 shrink-0">
                {u.status !== 'approved' && (
                  <button onClick={() => doAction(u.uid, 'approve')} disabled={actionLoading === u.uid+'approve'}
                    className="score-display px-3 py-1.5 rounded text-xs"
                    style={{ fontSize: 12, background: 'rgba(61,170,106,0.15)', color: '#3DAA6A', border: '1px solid rgba(61,170,106,0.4)', cursor: 'pointer' }}>
                    {actionLoading === u.uid+'approve' ? '...' : '✓ APPROVE'}
                  </button>
                )}
                {u.status !== 'denied' && (
                  <button onClick={() => doAction(u.uid, 'deny')} disabled={actionLoading === u.uid+'deny'}
                    className="score-display px-3 py-1.5 rounded text-xs"
                    style={{ fontSize: 12, background: 'rgba(217,64,64,0.1)', color: '#D94040', border: '1px solid rgba(217,64,64,0.3)', cursor: 'pointer' }}>
                    {actionLoading === u.uid+'deny' ? '...' : '✗ DENY'}
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────
// Picks Tab
// ─────────────────────────────────────────────
function PicksTab({ secret }: { secret: string }) {
  const [week, setWeek] = useState(1);
  const [season, setSeason] = useState(CURRENT_SEASON);
  const [league, setLeague] = useState<'NFL'|'CFB'>('NFL');
  const [cardGames, setCardGames] = useState<CardGame[]>([]);
  const [lockedPicks, setLockedPicks] = useState<LockedPick[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [loadingCard, setLoadingCard] = useState(false);
  const [locking, setLocking] = useState(false);
  const [toast, setToast] = useState('');
  const [toastType, setToastType] = useState<'ok'|'err'>('ok');
  const [gradeMode, setGradeMode] = useState<string | null>(null);
  const [gradeScores, setGradeScores] = useState<Record<string, { home: string; away: string }>>({});

  const MAX_PICKS = 5;
  const showToast = (m: string, t: 'ok'|'err' = 'ok') => { setToast(m); setToastType(t); setTimeout(() => setToast(''), 4000); };

  const fetchCard = async () => {
    setLoadingCard(true); setCardGames([]);
    try {
      const r = await axios.get(`${API}/card/${league}`, { params: { week, season } });
      setCardGames(r.data.games || []);
    } catch { showToast('Failed to load card', 'err'); }
    finally { setLoadingCard(false); }
  };

  const fetchLocked = useCallback(async () => {
    try {
      const r = await axios.get(`/api/picks?week=${week}&season=${season}&league=${league}`);
      setLockedPicks(r.data.picks || []);
    } catch {}
  }, [week, season, league]);

  useEffect(() => { fetchLocked(); }, [fetchLocked]);

  const toggleSelect = (gameKey: string) => {
    const next = new Set(selected);
    if (next.has(gameKey)) { next.delete(gameKey); }
    else if (next.size < MAX_PICKS) { next.add(gameKey); }
    else { showToast(`Max ${MAX_PICKS} picks per league`, 'err'); return; }
    setSelected(next);
  };

  const gameKey = (g: CardGame) => `${g.home_team}|${g.away_team}`;

  // For each selected game, track which team was picked
  const [pickedTeams, setPickedTeams] = useState<Record<string, string>>({});
  const setPick = (key: string, team: string) => setPickedTeams(p => ({ ...p, [key]: team }));

  const handleLock = async () => {
    if (selected.size === 0) { showToast('Select at least one game', 'err'); return; }
    for (const key of selected) {
      if (!pickedTeams[key]) { showToast('Select a team for every pick', 'err'); return; }
    }
    setLocking(true);
    try {
      const picks = Array.from(selected).map(key => {
        const g = cardGames.find(g => gameKey(g) === key)!;
        const pickedHome = pickedTeams[key] === g.home_team;
        const spread = g.disparity.vegas_spread ?? 0;
        return {
          league: g.league,
          home_team: g.home_team,
          away_team: g.away_team,
          picked_team: pickedTeams[key],
          spread_at_lock: pickedHome ? spread : -spread,
          predicted_home: g.prediction.predicted_home_score,
          predicted_away: g.prediction.predicted_away_score,
          predicted_margin: g.prediction.predicted_margin,
          predicted_total: g.prediction.predicted_total,
        };
      });
      await axios.post(`/api/picks?action=lock`, { secret, picks, week, season });
      showToast(`🔒 ${picks.length} picks locked for ${league} Week ${week}`);
      setSelected(new Set()); setPickedTeams({});
      fetchLocked();
    } catch (e: any) {
      showToast('Lock failed: ' + (e?.response?.data?.error || e.message), 'err');
    } finally { setLocking(false); }
  };

  const handleGrade = async (pickId: string) => {
    const scores = gradeScores[pickId];
    if (!scores?.home || !scores?.away) { showToast('Enter both scores', 'err'); return; }
    try {
      await axios.post(`/api/picks?action=grade`, {
        secret, pick_id: pickId,
        actual_home: parseFloat(scores.home),
        actual_away: parseFloat(scores.away),
      });
      showToast('Pick graded!'); setGradeMode(null);
      fetchLocked();
    } catch (e: any) {
      showToast('Grade failed: ' + (e?.response?.data?.error || e.message), 'err');
    }
  };

  const handleDelete = async (pickId: string) => {
    if (!confirm('Delete this pick?')) return;
    await axios.post(`/api/picks?action=delete`, { secret, pick_id: pickId });
    fetchLocked();
  };

  const pendingPicks = lockedPicks.filter(p => p.status === 'pending');
  const gradedPicks = lockedPicks.filter(p => p.status === 'graded');

  return (
    <div>
      {toast && (
        <div className="mb-4 px-4 py-2 rounded text-sm" style={{ background: toastType === 'ok' ? 'rgba(61,170,106,0.12)' : 'rgba(217,64,64,0.1)', color: toastType === 'ok' ? '#3DAA6A' : '#D94040', border: `1px solid ${toastType === 'ok' ? 'rgba(61,170,106,0.3)' : 'rgba(217,64,64,0.2)'}`, fontFamily: 'var(--font-mono)' }}>{toast}</div>
      )}

      {/* Controls */}
      <div className="panel rounded-xl p-5 mb-5">
        <div className="flex flex-wrap gap-4 items-end">
          <div>
            <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>League</label>
            <div className="flex gap-2">
              {(['NFL','CFB'] as const).map(l => (
                <button key={l} onClick={() => { setLeague(l); setCardGames([]); setSelected(new Set()); }}
                  className="score-display px-4 py-1.5 rounded"
                  style={{ fontSize: 14, background: league === l ? '#C9A84C' : 'rgba(15,44,71,0.5)', color: league === l ? '#030B14' : '#8B9BB4', border: '1px solid', borderColor: league === l ? '#C9A84C' : 'rgba(201,168,76,0.15)', cursor: 'pointer' }}>
                  {l}
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
          <div>
            <label className="block text-xs text-slate mb-1 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>Season</label>
            <input type="number" className="rounded px-3 py-2 text-sm" value={season} onChange={e => setSeason(Number(e.target.value))} style={{ width: 90 }} />
          </div>
          <button onClick={fetchCard} disabled={loadingCard}
            className="score-display px-5 py-2 rounded"
            style={{ fontSize: 14, background: loadingCard ? 'rgba(201,168,76,0.2)' : 'linear-gradient(135deg,#C9A84C,#E8C96A)', color: loadingCard ? '#4A5568' : '#030B14', border: 'none', cursor: 'pointer' }}>
            {loadingCard ? 'LOADING...' : 'LOAD GAMES'}
          </button>
        </div>
      </div>

      {/* Game selection */}
      {cardGames.length > 0 && (
        <div className="mb-6">
          <div className="flex items-center justify-between mb-3">
            <h3 className="score-display text-chalk" style={{ fontSize: 16, letterSpacing: '0.08em' }}>
              SELECT OFFICIAL PICKS — {selected.size}/{MAX_PICKS}
            </h3>
            {selected.size > 0 && (
              <button onClick={handleLock} disabled={locking}
                className="score-display px-5 py-2 rounded"
                style={{ fontSize: 14, background: locking ? 'rgba(201,168,76,0.2)' : 'linear-gradient(135deg,#C9A84C,#E8C96A)', color: locking ? '#4A5568' : '#030B14', border: 'none', cursor: 'pointer' }}>
                🔒 {locking ? 'LOCKING...' : `LOCK ${selected.size} PICK${selected.size > 1 ? 'S' : ''}`}
              </button>
            )}
          </div>
          <p className="text-xs text-slate mb-3" style={{ fontFamily: 'var(--font-mono)' }}>
            Click a game to select it, then choose which team you're picking. Locking is permanent and timestamped.
          </p>
          <div className="space-y-2">
            {cardGames.map((game, i) => {
              const key = gameKey(game);
              const isSelected = selected.has(key);
              const isLocked = lockedPicks.some(p => p.home_team === game.home_team && p.away_team === game.away_team);
              const spread = game.disparity.vegas_spread;
              return (
                <div key={i}
                  className="panel rounded-xl p-4 cursor-pointer transition-all"
                  style={{ borderColor: isSelected ? 'rgba(61,170,106,0.6)' : isLocked ? 'rgba(201,168,76,0.4)' : 'rgba(201,168,76,0.1)', opacity: isLocked ? 0.6 : 1 }}
                  onClick={() => !isLocked && toggleSelect(key)}>
                  <div className="flex items-center justify-between gap-4 flex-wrap">
                    <div className="flex items-center gap-3">
                      {isSelected && <span style={{ color: '#3DAA6A', fontSize: 18 }}>✓</span>}
                      {isLocked && <span style={{ color: '#C9A84C', fontSize: 14, fontFamily: 'var(--font-mono)' }}>🔒</span>}
                      {!isSelected && !isLocked && <span style={{ color: '#4A5568', fontSize: 18 }}>○</span>}
                      <div>
                        <span className="text-chalk text-sm font-semibold">{game.away_team}</span>
                        <span className="text-slate text-xs mx-2">@</span>
                        <span className="text-chalk text-sm font-semibold">{game.home_team}</span>
                        {spread !== null && (
                          <span className="text-xs ml-2" style={{ color: '#8B9BB4', fontFamily: 'var(--font-mono)' }}>
                            Home {spread > 0 ? `+${spread}` : spread}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="text-xs" style={{ color: '#C9A84C', fontFamily: 'var(--font-mono)' }}>
                      PP: {game.prediction.predicted_home_score.toFixed(0)}-{game.prediction.predicted_away_score.toFixed(0)}
                    </div>
                  </div>

                  {/* Team picker — only show when selected */}
                  {isSelected && (
                    <div className="mt-3 flex gap-2 flex-wrap" onClick={e => e.stopPropagation()}>
                      <p className="text-xs text-slate w-full mb-1" style={{ fontFamily: 'var(--font-mono)' }}>Pick which team to win & cover:</p>
                      {[game.home_team, game.away_team].map(team => (
                        <button key={team} onClick={() => setPick(key, team)}
                          className="score-display px-4 py-1.5 rounded text-xs"
                          style={{ fontSize: 12, letterSpacing: '0.06em', background: pickedTeams[key] === team ? '#3DAA6A' : 'rgba(15,44,71,0.6)', color: pickedTeams[key] === team ? '#fff' : '#8B9BB4', border: '1px solid', borderColor: pickedTeams[key] === team ? '#3DAA6A' : 'rgba(201,168,76,0.2)', cursor: 'pointer' }}>
                          {team}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Locked pending picks — grade these */}
      {pendingPicks.length > 0 && (
        <div className="mb-6">
          <h3 className="score-display text-chalk mb-3" style={{ fontSize: 16, letterSpacing: '0.08em' }}>🔒 LOCKED — AWAITING RESULTS</h3>
          <div className="space-y-2">
            {pendingPicks.map(pick => (
              <div key={pick.id} className="panel rounded-xl p-4">
                <div className="flex items-start justify-between flex-wrap gap-3">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-chalk text-sm font-semibold">
                        ✅ {pick.picked_team}
                      </span>
                      <span className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>
                        ({pick.away_team} @ {pick.home_team})
                      </span>
                    </div>
                    <div className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>
                      Spread at lock: {pick.spread_at_lock > 0 ? `+${pick.spread_at_lock}` : pick.spread_at_lock} ·
                      PP: {pick.predicted_home}-{pick.predicted_away} ·
                      Locked: {new Date(pick.locked_at_iso).toLocaleString()}
                    </div>
                  </div>
                  <div className="flex gap-2">
                    {gradeMode === pick.id ? (
                      <div className="flex items-center gap-2" onClick={e => e.stopPropagation()}>
                        <input type="number" placeholder={pick.home_team.split(' ').pop()} className="rounded px-2 py-1 text-xs w-16"
                          value={gradeScores[pick.id]?.home || ''}
                          onChange={e => setGradeScores(g => ({ ...g, [pick.id]: { ...g[pick.id], home: e.target.value } }))} />
                        <span className="text-slate text-xs">-</span>
                        <input type="number" placeholder={pick.away_team.split(' ').pop()} className="rounded px-2 py-1 text-xs w-16"
                          value={gradeScores[pick.id]?.away || ''}
                          onChange={e => setGradeScores(g => ({ ...g, [pick.id]: { ...g[pick.id], away: e.target.value } }))} />
                        <button onClick={() => handleGrade(pick.id)}
                          className="score-display px-3 py-1 rounded text-xs"
                          style={{ fontSize: 11, background: '#3DAA6A', color: '#fff', border: 'none', cursor: 'pointer' }}>SAVE</button>
                        <button onClick={() => setGradeMode(null)}
                          className="text-xs text-slate" style={{ background: 'none', border: 'none', cursor: 'pointer' }}>×</button>
                      </div>
                    ) : (
                      <>
                        <button onClick={() => setGradeMode(pick.id)}
                          className="score-display px-3 py-1.5 rounded text-xs"
                          style={{ fontSize: 11, background: 'rgba(61,170,106,0.12)', color: '#3DAA6A', border: '1px solid rgba(61,170,106,0.3)', cursor: 'pointer' }}>
                          ENTER RESULT
                        </button>
                        <button onClick={() => handleDelete(pick.id)}
                          className="text-xs px-2 py-1 rounded"
                          style={{ color: '#D94040', background: 'rgba(217,64,64,0.1)', border: '1px solid rgba(217,64,64,0.2)', cursor: 'pointer', fontFamily: 'var(--font-mono)' }}>
                          ✕
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Graded picks history */}
      {gradedPicks.length > 0 && (
        <div>
          <h3 className="score-display text-chalk mb-3" style={{ fontSize: 16, letterSpacing: '0.08em' }}>GRADED PICKS</h3>
          <div className="space-y-2">
            {gradedPicks.map(pick => (
              <div key={pick.id} className="panel rounded-xl p-4">
                <div className="flex items-center justify-between flex-wrap gap-2">
                  <div>
                    <span className="text-chalk text-sm font-semibold">✅ {pick.picked_team}</span>
                    <span className="text-xs text-slate ml-2" style={{ fontFamily: 'var(--font-mono)' }}>({pick.away_team} @ {pick.home_team})</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)' }}>
                      Final: {pick.actual_home}-{pick.actual_away}
                    </span>
                    <span className="score-display text-sm px-3 py-1 rounded"
                      style={{ background: `${resultColor[pick.result!]}18`, color: resultColor[pick.result!], border: `1px solid ${resultColor[pick.result!]}40` }}>
                      {resultLabel[pick.result!]}
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {pendingPicks.length === 0 && gradedPicks.length === 0 && cardGames.length === 0 && (
        <div className="panel rounded-xl p-8 text-center">
          <p className="text-slate text-sm" style={{ fontFamily: 'var(--font-mono)' }}>Load games to select and lock this week's official picks.</p>
        </div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────
// Main Admin Page
// ─────────────────────────────────────────────
export default function Admin() {
  const [authed, setAuthed] = useState(false);
  const [secret, setSecret] = useState('');
  const [tab, setTab] = useState<'users'|'picks'>('users');

  const handleAuth = async (s: string) => {
    try {
      await axios.post('/api/list-users', { secret: s });
      setSecret(s); setAuthed(true);
    } catch { /* AuthGate handles error display */ }
  };

  if (!authed) return (
    <>
      <Head><title>Admin — Prime Picks</title></Head>
      <AuthGate onAuth={handleAuth} />
    </>
  );

  return (
    <>
      <Head><title>Admin — Prime Picks</title></Head>
      <div className="field-bg min-h-screen px-4 py-10">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-6 flex-wrap gap-3">
            <h1 className="score-display text-chalk" style={{ fontSize: 28, letterSpacing: '0.08em' }}>🏈 PRIME PICKS ADMIN</h1>
          </div>

          {/* Tabs */}
          <div className="flex gap-2 mb-6">
            {(['users','picks'] as const).map(t => (
              <button key={t} onClick={() => setTab(t)}
                className="score-display px-5 py-2 rounded"
                style={{ fontSize: 16, letterSpacing: '0.08em', background: tab === t ? '#C9A84C' : 'rgba(15,44,71,0.5)', color: tab === t ? '#030B14' : '#8B9BB4', border: '1px solid', borderColor: tab === t ? '#C9A84C' : 'rgba(201,168,76,0.15)', cursor: 'pointer' }}>
                {t === 'users' ? '👥 USERS' : '🏈 PICKS'}
              </button>
            ))}
          </div>

          {tab === 'users' && <UsersTab secret={secret} />}
          {tab === 'picks' && <PicksTab secret={secret} />}
        </div>
      </div>
    </>
  );
}
