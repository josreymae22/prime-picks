import { useState, useEffect } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import axios from 'axios';
import RecordWidget from '../components/RecordWidget';

type Pick = {
  id: string; league: string; week: number; season: number;
  home_team: string; away_team: string;
  picked_team: string; spread_at_lock: number;
  predicted_home: number; predicted_away: number;
  predicted_margin: number; predicted_total: number;
  status: 'pending' | 'graded';
  result: 'win' | 'loss' | 'push' | null;
  actual_home: number | null; actual_away: number | null;
  locked_at_iso: string; verify_hash: string;
  hash_source?: object;
};

type Record = {
  overall: { wins: number; losses: number; pushes: number; total: number };
  nfl: { wins: number; losses: number; pushes: number };
  cfb: { wins: number; losses: number; pushes: number };
  recent: Pick[];
};

const resultColor = { win: '#3DAA6A', loss: '#D94040', push: '#C9A84C' };
const resultLabel = { win: '✓ WIN', loss: '✗ LOSS', push: '— PUSH' };
const resultBg    = { win: 'rgba(61,170,106,0.12)', loss: 'rgba(217,64,64,0.1)', push: 'rgba(201,168,76,0.1)' };

function WinPct(w: number, l: number) {
  if (w + l === 0) return '—';
  return ((w / (w + l)) * 100).toFixed(1) + '%';
}

function RecordBadge({ wins, losses, pushes, label }: { wins: number; losses: number; pushes: number; label: string }) {
  return (
    <div className="panel-bright rounded-xl p-5 text-center">
      <div className="text-xs text-slate mb-2 uppercase tracking-widest" style={{ fontFamily: 'var(--font-mono)' }}>{label}</div>
      <div className="score-display" style={{ fontSize: 36, color: '#F0EEE6', letterSpacing: '0.05em' }}>
        <span style={{ color: '#3DAA6A' }}>{wins}</span>
        <span className="text-slate mx-1" style={{ fontSize: 24 }}>-</span>
        <span style={{ color: '#D94040' }}>{losses}</span>
        {pushes > 0 && <><span className="text-slate mx-1" style={{ fontSize: 24 }}>-</span><span style={{ color: '#C9A84C' }}>{pushes}</span></>}
      </div>
      <div className="text-xs mt-2" style={{ color: '#C9A84C', fontFamily: 'var(--font-mono)' }}>
        {WinPct(wins, losses)} ATS
      </div>
    </div>
  );
}

export default function RecordPage() {
  const [record, setRecord] = useState<Record | null>(null);
  const [allPicks, setAllPicks] = useState<Pick[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<'all'|'NFL'|'CFB'>('all');
  const [verifying, setVerifying] = useState<string | null>(null);
  const [seasonFilter, setSeasonFilter] = useState<number | 'all'>('all');

  useEffect(() => {
    Promise.all([
      axios.post('/api/picks?action=record', { secret: '' }).catch(() => null),
      axios.get('/api/picks').catch(() => null),
    ]).then(([rec, picks]) => {
      if (rec) setRecord(rec.data);
      if (picks) setAllPicks(picks.data.picks || []);
    }).finally(() => setLoading(false));
  }, []);

  const seasons = Array.from(new Set(allPicks.map(p => p.season))).sort((a, b) => b - a);

  const displayed = allPicks
    .filter(p => filter === 'all' || p.league === filter)
    .filter(p => seasonFilter === 'all' || p.season === seasonFilter)
    .filter(p => p.status === 'graded');

  return (
    <>
      <Head>
        <title>Pick Record & Verification — Prime Picks</title>
        <meta name="description" content="Prime Picks official ATS record — every pick timestamped and verifiable." />
      </Head>

      <div className="field-bg min-h-screen px-4 py-10">
        <div className="max-w-4xl mx-auto">

          {/* Header */}
          <div className="text-center mb-10">
            <div className="flex items-center justify-center gap-3 mb-3">
              <span style={{ fontSize: 32 }}>🏈</span>
              <h1 className="score-display text-chalk" style={{ fontSize: 38, letterSpacing: '0.08em' }}>PRIME PICKS</h1>
            </div>
            <h2 className="score-display" style={{ fontSize: 20, color: '#C9A84C', letterSpacing: '0.12em' }}>OFFICIAL RECORD</h2>
            <p className="text-slate text-xs mt-2" style={{ fontFamily: 'var(--font-mono)' }}>
              Every pick cryptographically timestamped before kickoff · ATS win/loss/push
            </p>
            <div className="gold-line mt-4" />
          </div>

          {loading && (
            <div className="text-center py-20">
              <p className="animate-pulse-gold score-display text-slate" style={{ fontSize: 18 }}>LOADING...</p>
            </div>
          )}

          {/* Record widget — full with season toggle and breakdown table */}
          <div className="mb-8">
            <RecordWidget compact={false} />
          </div>

          {/* Filters */}
          <div className="flex flex-wrap gap-2 mb-5 items-center">
            <div className="flex gap-2">
              {(['all','NFL','CFB'] as const).map(f => (
                <button key={f} onClick={() => setFilter(f)}
                  className="score-display px-4 py-1.5 rounded text-xs"
                  style={{ fontSize: 12, letterSpacing: '0.08em', background: filter === f ? '#C9A84C' : 'rgba(15,44,71,0.5)', color: filter === f ? '#030B14' : '#8B9BB4', border: '1px solid', borderColor: filter === f ? '#C9A84C' : 'rgba(201,168,76,0.15)', cursor: 'pointer' }}>
                  {f === 'CFB' ? 'NCAAF' : f}
                </button>
              ))}
            </div>
            {seasons.length > 0 && (
              <select className="rounded px-3 py-1.5 text-xs" value={String(seasonFilter)} onChange={e => setSeasonFilter(e.target.value === 'all' ? 'all' : Number(e.target.value))}>
                <option value="all">All Seasons</option>
                {seasons.map(s => <option key={s} value={s}>{s}</option>)}
              </select>
            )}
          </div>

          {/* Pick history */}
          {displayed.length === 0 && !loading && (
            <div className="panel rounded-xl p-10 text-center">
              <p className="text-slate text-sm" style={{ fontFamily: 'var(--font-mono)' }}>No graded picks yet.</p>
            </div>
          )}

          <div className="space-y-3">
            {displayed.map((pick, i) => (
              <div key={pick.id} className="panel rounded-xl overflow-hidden animate-fade-up" style={{ animationDelay: `${i * 30}ms` }}>
                <div className="p-5">
                  <div className="flex items-start justify-between flex-wrap gap-3">
                    <div className="flex-1 min-w-0">

                      {/* Pick header */}
                      <div className="flex items-center gap-2 flex-wrap mb-2">
                        <span className="score-display text-xs px-2 py-0.5 rounded" style={{ background: 'rgba(201,168,76,0.1)', color: '#C9A84C', border: '1px solid rgba(201,168,76,0.2)', letterSpacing: '0.08em' }}>
                          {pick.league} · Wk {pick.week} · {pick.season}
                        </span>
                        {pick.result && (
                          <span className="score-display text-xs px-3 py-0.5 rounded" style={{ background: resultBg[pick.result], color: resultColor[pick.result], border: `1px solid ${resultColor[pick.result]}40`, letterSpacing: '0.1em' }}>
                            {resultLabel[pick.result]}
                          </span>
                        )}
                      </div>

                      {/* Teams + pick */}
                      <div className="mb-2">
                        <span className="text-chalk text-sm font-semibold">✅ {pick.picked_team}</span>
                        <span className="text-slate text-xs ml-2" style={{ fontFamily: 'var(--font-mono)' }}>
                          to win & cover ({pick.away_team} @ {pick.home_team})
                        </span>
                      </div>

                      {/* Stats row */}
                      <div className="flex flex-wrap gap-4 text-xs" style={{ fontFamily: 'var(--font-mono)' }}>
                        <span className="text-slate">
                          Spread at lock: <span style={{ color: '#F0EEE6' }}>{pick.spread_at_lock > 0 ? `+${pick.spread_at_lock}` : pick.spread_at_lock}</span>
                        </span>
                        <span className="text-slate">
                          PP predicted: <span style={{ color: '#C9A84C' }}>{pick.predicted_home}-{pick.predicted_away}</span>
                        </span>
                        {pick.actual_home !== null && (
                          <span className="text-slate">
                            Actual: <span style={{ color: '#F0EEE6' }}>{pick.actual_home}-{pick.actual_away}</span>
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Verify button */}
                    <button
                      onClick={() => setVerifying(verifying === pick.id ? null : pick.id)}
                      className="shrink-0 text-xs px-3 py-1.5 rounded"
                      style={{ fontFamily: 'var(--font-mono)', background: 'rgba(139,155,180,0.1)', color: '#8B9BB4', border: '1px solid rgba(139,155,180,0.2)', cursor: 'pointer' }}>
                      {verifying === pick.id ? 'HIDE' : '🔍 VERIFY'}
                    </button>
                  </div>

                  {/* Verification panel */}
                  {verifying === pick.id && (
                    <div className="mt-4 rounded-lg p-4" style={{ background: 'rgba(7,21,36,0.8)', border: '1px solid rgba(201,168,76,0.15)' }}>
                      <p className="text-xs mb-2" style={{ color: '#C9A84C', fontFamily: 'var(--font-mono)', letterSpacing: '0.1em' }}>VERIFICATION RECORD</p>
                      <div className="space-y-1.5 text-xs" style={{ fontFamily: 'var(--font-mono)' }}>
                        <div className="flex gap-2">
                          <span className="text-slate w-28 shrink-0">Locked at:</span>
                          <span className="text-chalk">{new Date(pick.locked_at_iso).toUTCString()}</span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-slate w-28 shrink-0">Pick ID:</span>
                          <span style={{ color: '#8B9BB4' }}>{pick.id}</span>
                        </div>
                        <div className="flex gap-2">
                          <span className="text-slate w-28 shrink-0">SHA-256:</span>
                          <span style={{ color: '#E8C96A', wordBreak: 'break-all' }}>{pick.verify_hash}</span>
                        </div>
                        {pick.hash_source && (
                          <div className="mt-2">
                            <p className="text-slate mb-1">Hash source data (verify independently):</p>
                            <pre className="rounded p-2 text-xs overflow-x-auto" style={{ background: 'rgba(0,0,0,0.3)', color: '#8B9BB4' }}>
                              {JSON.stringify(pick.hash_source, null, 2)}
                            </pre>
                          </div>
                        )}
                        <p className="text-slate mt-2" style={{ opacity: 0.6 }}>
                          To verify: SHA-256 hash the JSON above with no whitespace. Result must match the hash shown.
                          Lock timestamp confirms pick was recorded before kickoff.
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Footer */}
          <div className="text-center mt-10">
            <div className="gold-line mb-4" />
            <p className="text-xs text-slate" style={{ fontFamily: 'var(--font-mono)', opacity: 0.5 }}>
              primepicks.ai · All picks cryptographically timestamped at lock time
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
