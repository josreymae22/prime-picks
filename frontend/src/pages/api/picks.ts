import type { NextApiRequest, NextApiResponse } from 'next';
import { initializeApp, getApps, cert } from 'firebase-admin/app';
import { getFirestore, FieldValue } from 'firebase-admin/firestore';
import crypto from 'crypto';

if (!getApps().length) {
  initializeApp({
    credential: cert({
      projectId: process.env.FIREBASE_PROJECT_ID,
      clientEmail: process.env.FIREBASE_CLIENT_EMAIL,
      privateKey: process.env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, '\n'),
    }),
  });
}

const db = getFirestore();
const ADMIN_SECRET = process.env.ADMIN_SECRET || 'changeme';

function hashPick(pick: object): string {
  return crypto.createHash('sha256').update(JSON.stringify(pick)).digest('hex');
}

function gradeResult(pick: any, actualHome: number, actualAway: number): 'win' | 'loss' | 'push' {
  const margin = actualHome - actualAway;
  const spread = pick.spread_at_lock;
  const pickedHome = pick.picked_team === pick.home_team;
  const pickedMargin = pickedHome ? margin : -margin;
  const spreadAbs = Math.abs(spread);
  const didWin = pickedMargin > 0;
  const didCover = pickedMargin > spreadAbs;
  const isPush = Math.abs(pickedMargin - spreadAbs) < 0.1;

  if (isPush) return 'push';
  if (didWin && didCover) return 'win';
  return 'loss';
}

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  const { action } = req.query;

  // ── PUBLIC RECORD SUMMARY ───────────────────────────────
  if (action === 'record') {
    try {
      const currentYear = new Date().getFullYear();
      const requestedSeason =
        req.method === 'POST' && req.body?.season ? Number(req.body.season) : null;

      let query: any = db.collection('picks').where('status', '==', 'graded');
      if (requestedSeason) query = query.where('season', '==', requestedSeason);
      query = query.orderBy('locked_at', 'desc').limit(500);

      const snap = await query.get();
      const allPicks = snap.docs.map((d: any) => ({ id: d.id, ...d.data() }));

      const buildRecord = (picks: any[]) => ({
        wins: picks.filter((p: any) => p.result === 'win').length,
        losses: picks.filter((p: any) => p.result === 'loss').length,
        pushes: picks.filter((p: any) => p.result === 'push').length,
        total: picks.length,
      });

      const seasonPicks = allPicks.filter((p: any) => p.season === currentYear);
      const nfl = allPicks.filter((p: any) => p.league === 'NFL');
      const cfb = allPicks.filter((p: any) => p.league === 'CFB');
      const nflSeason = seasonPicks.filter((p: any) => p.league === 'NFL');
      const cfbSeason = seasonPicks.filter((p: any) => p.league === 'CFB');

      const allSeasons = Array.from(new Set(allPicks.map((p: any) => p.season))).sort(
        (a: any, b: any) => b - a
      );

      const bySeasonNfl: Record<number, any> = {};
      const bySeasonCfb: Record<number, any> = {};

      for (const s of allSeasons as number[]) {
        bySeasonNfl[s] = buildRecord(
          allPicks.filter((p: any) => p.league === 'NFL' && p.season === s)
        );
        bySeasonCfb[s] = buildRecord(
          allPicks.filter((p: any) => p.league === 'CFB' && p.season === s)
        );
      }

      return res.status(200).json({
        overall: buildRecord(allPicks),
        nfl: buildRecord(nfl),
        cfb: buildRecord(cfb),
        current_season: currentYear,
        season_to_date: buildRecord(seasonPicks),
        nfl_season: buildRecord(nflSeason),
        cfb_season: buildRecord(cfbSeason),
        seasons: allSeasons,
        by_season_nfl: bySeasonNfl,
        by_season_cfb: bySeasonCfb,
        recent: allPicks.slice(0, 10),
      });
    } catch (err: any) {
      return res.status(500).json({ error: err.message });
    }
  }

  // ── GET PICKS PUBLIC ────────────────────────────────────
  if (req.method === 'GET') {
    const { week, season, league, status } = req.query;

    try {
      let query: any = db.collection('picks');
      if (week) query = query.where('week', '==', Number(week));
      if (season) query = query.where('season', '==', Number(season));
      if (league) query = query.where('league', '==', String(league).toUpperCase());
      if (status) query = query.where('status', '==', String(status));

      query = query.orderBy('locked_at', 'desc').limit(200);

      const snap = await query.get();
      const picks = snap.docs.map((d: any) => ({ id: d.id, ...d.data() }));

      return res.status(200).json({ picks });
    } catch (err: any) {
      return res.status(500).json({ error: err.message });
    }
  }

  if (req.method !== 'POST') return res.status(405).end();

  const { secret } = req.body;
  if (secret !== ADMIN_SECRET) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  if (action === 'lock') {
    const { picks, week, season } = req.body;

    if (!picks || !Array.isArray(picks) || picks.length === 0) {
      return res.status(400).json({ error: 'picks array required' });
    }

    const batch = db.batch();
    const lockedPicks = [];

    for (const pick of picks) {
      const lockData = {
        league: pick.league,
        week: Number(week),
        season: Number(season),
        home_team: pick.home_team,
        away_team: pick.away_team,
        picked_team: pick.picked_team,
        spread_at_lock: pick.spread_at_lock,
        predicted_home: pick.predicted_home,
        predicted_away: pick.predicted_away,
        predicted_margin: pick.predicted_margin,
        predicted_total: pick.predicted_total,
        status: 'pending',
        result: null,
        actual_home: null,
        actual_away: null,
        locked_at: FieldValue.serverTimestamp(),
        locked_at_iso: new Date().toISOString(),
      };

      const hashSource = {
        league: lockData.league,
        week: lockData.week,
        season: lockData.season,
        home_team: lockData.home_team,
        away_team: lockData.away_team,
        picked_team: lockData.picked_team,
        spread_at_lock: lockData.spread_at_lock,
        locked_at_iso: lockData.locked_at_iso,
      };

      const verifyHash = hashPick(hashSource);
      const docRef = db.collection('picks').doc();

      batch.set(docRef, {
        ...lockData,
        verify_hash: verifyHash,
        hash_source: hashSource,
      });

      lockedPicks.push({
        id: docRef.id,
        ...lockData,
        verify_hash: verifyHash,
      });
    }

    await batch.commit();

    return res.status(200).json({
      ok: true,
      locked: lockedPicks.length,
      picks: lockedPicks,
    });
  }

  if (action === 'grade') {
    const { pick_id, actual_home, actual_away } = req.body;

    if (!pick_id || actual_home == null || actual_away == null) {
      return res.status(400).json({ error: 'pick_id, actual_home, actual_away required' });
    }

    const ref = db.collection('picks').doc(pick_id);
    const snap = await ref.get();

    if (!snap.exists) {
      return res.status(404).json({ error: 'Pick not found' });
    }

    const pick = snap.data()!;
    const result = gradeResult(pick, Number(actual_home), Number(actual_away));

    await ref.update({
      actual_home: Number(actual_home),
      actual_away: Number(actual_away),
      result,
      status: 'graded',
      graded_at: FieldValue.serverTimestamp(),
    });

    return res.status(200).json({ ok: true, result, pick_id });
  }

  if (action === 'delete') {
    const { pick_id } = req.body;
    await db.collection('picks').doc(pick_id).delete();
    return res.status(200).json({ ok: true });
  }

  return res.status(400).json({ error: 'Unknown action' });
}