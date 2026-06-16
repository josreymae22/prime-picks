import { useState, useEffect } from 'react';
import Link from 'next/link';
import axios from 'axios';

type LeagueRecord = { wins: number; losses: number; pushes: number; total: number };
type RecordData = {
  overall: LeagueRecord;
  nfl: LeagueRecord;
  cfb: LeagueRecord;
  current_season: number;
  season_to_date: LeagueRecord;
  nfl_season: LeagueRecord;
  cfb_season: LeagueRecord;
  seasons: number[];
  by_season_nfl: Record<string, LeagueRecord>;
  by_season_cfb: Record<string, LeagueRecord>;
};

function pct(w: number, l: number) {
  if (w + l === 0) return '—';
  return ((w / (w + l)) * 100).toFixed(1) + '%';
}

function RecordPill({ rec, label, color }: { rec: LeagueRecord; label: string; color: string }) {
  const winPct = pct(rec.wins, rec.losses);
  return (
    <div className="text-center px-4 py-3 rounded-lg flex-1 min-w-0"
      style={{ background: 'rgba(7,21,36,0.6)', border: `1px solid ${color}25` }}>
      <div className="text-xs mb-1 uppercase tracking-widest" style={{ color, fontFamily: 'var(--font-mono)', fontSize: 9 }}>
        {label}
      </div>
      <div className="score-display" style={{ fontSize: 22, letterSpacing: '0.04em' }}>
        <span style={{ color: '#3DAA6A' }}>{rec.wins}</span>
        <span className="text-slate" style={{ fontSize: 16, margin: '0 3px' }}>-</span>
        <span style={{ color: '#D94040' }}>{rec.losses}</span>
        {rec.pushes > 0 && <>
          <span className="text-slate" style={{ fontSize: 16, margin: '0 3px' }}>-</span>
          <span style={{ color: '#C9A84C' }}>{rec.pushes}</span>
        </>}
      </div>
      <div className="text-xs mt-1" style={{ color, fontFamily: 'var(--font-mono)', opacity: 0.8 }}>
        {winPct} ATS
      </div>
    </div>
  );
}

export default function RecordWidget({ compact = false }: { compact?: boolean }) {
  const [data, setData] = useState<RecordData | null>(null);
  const [view, setView] = useState<'season' | 'alltime'>('season');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.post('/api/picks?action=record', { secret: '' })
      .then(r => setData(r.data))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="panel rounded-xl p-4 text-center">
      <p className="text-xs text-slate animate-pulse-gold" style={{ fontFamily: 'var(--font-mono)' }}>Loading record...</p>
    </div>
  );

  if (!data) return null;

  const isSeasonView = view === 'season';
  const nflRec  = isSeasonView ? data.nfl_season  : data.nfl;
  const cfbRec  = isSeasonView ? data.cfb_season  : data.cfb;
  const totRec  = isSeasonView ? data.season_to_date : data.overall;

  return (
    <div className="panel rounded-xl overflow-hidden">
      {/* Header row */}
      <div className="flex items-center justify-between px-5 pt-4 pb-3"
        style={{ borderBottom: '1px solid rgba(201,168,76,0.1)' }}>
        <div className="flex items-center gap-2">
          <span style={{ fontSize: 16 }}>📊</span>
          <span className="score-display text-chalk" style={{ fontSize: 15, letterSpacing: '0.08em' }}>
            PRIME PICKS RECORD
          </span>
          {isSeasonView && (
            <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'rgba(201,168,76,0.12)', color: '#C9A84C', border: '1px solid rgba(201,168,76,0.25)', fontFamily: 'var(--font-mono)' }}>
              {data.current_season} SEASON
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Toggle */}
          <div className="flex rounded overflow-hidden" style={{ border: '1px solid rgba(201,168,76,0.2)' }}>
            {(['season', 'alltime'] as const).map(v => (
              <button key={v} onClick={() => setView(v)}
                className="text-xs px-3 py-1"
                style={{ fontFamily: 'var(--font-mono)', background: view === v ? 'rgba(201,168,76,0.2)' : 'transparent', color: view === v ? '#C9A84C' : '#4A5568', border: 'none', cursor: 'pointer' }}>
                {v === 'season' ? 'THIS SEASON' : 'ALL TIME'}
              </button>
            ))}
          </div>
          <Link href="/record"
            className="text-xs"
            style={{ color: '#8B9BB4', fontFamily: 'var(--font-mono)', textDecoration: 'none' }}>
            Full record →
          </Link>
        </div>
      </div>

      {/* Record pills */}
      <div className="flex gap-3 p-4">
        <RecordPill rec={totRec}  label="Overall"    color="#C9A84C" />
        <RecordPill rec={nflRec}  label="NFL"        color="#8B9BB4" />
        <RecordPill rec={cfbRec}  label="NCAAF"      color="#8B9BB4" />
      </div>

      {/* Season-by-season table — only in alltime view */}
      {!compact && view === 'alltime' && data.seasons.length > 1 && (
        <div className="px-4 pb-4">
          <div className="rounded-lg overflow-hidden" style={{ border: '1px solid rgba(201,168,76,0.1)' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: 'var(--font-mono)', fontSize: 11 }}>
              <thead>
                <tr style={{ background: 'rgba(201,168,76,0.08)' }}>
                  <th className="text-left px-3 py-2 text-slate uppercase tracking-widest" style={{ fontSize: 9 }}>Season</th>
                  <th className="text-center px-3 py-2 text-slate uppercase tracking-widest" style={{ fontSize: 9 }}>NFL W-L-P</th>
                  <th className="text-center px-3 py-2 text-slate uppercase tracking-widest" style={{ fontSize: 9 }}>NFL %</th>
                  <th className="text-center px-3 py-2 text-slate uppercase tracking-widest" style={{ fontSize: 9 }}>NCAAF W-L-P</th>
                  <th className="text-center px-3 py-2 text-slate uppercase tracking-widest" style={{ fontSize: 9 }}>NCAAF %</th>
                </tr>
              </thead>
              <tbody>
                {data.seasons.map((s, i) => {
                  const nr = data.by_season_nfl[s] || { wins: 0, losses: 0, pushes: 0 };
                  const cr = data.by_season_cfb[s] || { wins: 0, losses: 0, pushes: 0 };
                  const isCurrentSeason = s === data.current_season;
                  return (
                    <tr key={s} style={{ background: i % 2 === 0 ? 'rgba(7,21,36,0.4)' : 'transparent', borderTop: '1px solid rgba(201,168,76,0.06)' }}>
                      <td className="px-3 py-2">
                        <span style={{ color: isCurrentSeason ? '#C9A84C' : '#F0EEE6', fontWeight: isCurrentSeason ? 700 : 400 }}>
                          {s}{isCurrentSeason ? ' ★' : ''}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-center">
                        <span style={{ color: '#3DAA6A' }}>{nr.wins}</span>
                        <span className="text-slate">-</span>
                        <span style={{ color: '#D94040' }}>{nr.losses}</span>
                        {nr.pushes > 0 && <><span className="text-slate">-</span><span style={{ color: '#C9A84C' }}>{nr.pushes}</span></>}
                      </td>
                      <td className="px-3 py-2 text-center" style={{ color: '#8B9BB4' }}>{pct(nr.wins, nr.losses)}</td>
                      <td className="px-3 py-2 text-center">
                        <span style={{ color: '#3DAA6A' }}>{cr.wins}</span>
                        <span className="text-slate">-</span>
                        <span style={{ color: '#D94040' }}>{cr.losses}</span>
                        {cr.pushes > 0 && <><span className="text-slate">-</span><span style={{ color: '#C9A84C' }}>{cr.pushes}</span></>}
                      </td>
                      <td className="px-3 py-2 text-center" style={{ color: '#8B9BB4' }}>{pct(cr.wins, cr.losses)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
