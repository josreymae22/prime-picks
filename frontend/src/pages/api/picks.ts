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
  return crypto
    .createHash('sha256')
    .update(JSON.stringify(pick))
    .digest('hex');
}

/**
 * Grade an ATS (against-the-spread) pick.
 *
 * spread_at_lock is stored from the perspective of the PICKED TEAM.
 *
 * Examples:
 *
 * Tennessee -1.5
 * Final: Tennessee 24, Jets 20
 * 24 + (-1.5) = 22.5 > 20
 * WIN
 *
 * Tennessee -1.5
 * Final: Tennessee 21, Jets 20
 * 21 + (-1.5) = 19.5 < 20
 * LOSS
 *
 * Jets +3
 * Final: Tennessee 24, Jets 22
 * 22 + 3 = 25 > 24
 * WIN
 *
 * Jets +3
 * Final: Tennessee 24, Jets 21
 * 21 + 3 = 24
 * PUSH
 */
function gradeResult(
  pick: any,
  actualHome: number,
  actualAway: number
): 'win' | 'loss' | 'push' {
  const spread = Number(pick.spread_at_lock ?? 0);

  const pickedHome = pick.picked_team === pick.home_team;
  const pickedAway = pick.picked_team === pick.away_team;

  if (!pickedHome && !pickedAway) {
    throw new Error(
      `Picked team "${pick.picked_team}" does not match home or away team`
    );
  }

  const pickedTeamScore = pickedHome ? actualHome : actualAway;
  const opponentScore = pickedHome ? actualAway : actualHome;

  const adjustedPickedScore = pickedTeamScore + spread;

  // Small tolerance protects against floating-point edge cases.
  const difference = adjustedPickedScore - opponentScore;

  if (Math.abs(difference) < 0.0001) {
    return 'push';
  }

  return difference > 0 ? 'win' : 'loss';
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  const { action } = req.query;

  // ─────────────────────────────────────────────
  // PUBLIC RECORD SUMMARY
  // ─────────────────────────────────────────────
  if (action === 'record') {
    try {
      const currentYear = new Date().getFullYear();

      const requestedSeason =
        req.method === 'POST' && req.body?.season
          ? Number(req.body.season)
          : null;

      let query: any = db
        .collection('picks')
        .where('status', '==', 'graded');

      if (requestedSeason) {
        query = query.where('season', '==', requestedSeason);
      }

      query = query
        .orderBy('locked_at', 'desc')
        .limit(500);

      const snap = await query.get();

      const allPicks = snap.docs.map((d: any) => ({
        id: d.id,
        ...d.data(),
      }));

      const buildRecord = (picks: any[]) => ({
        wins: picks.filter((p: any) => p.result === 'win').length,
        losses: picks.filter((p: any) => p.result === 'loss').length,
        pushes: picks.filter((p: any) => p.result === 'push').length,
        total: picks.length,
      });

      const seasonPicks = allPicks.filter(
        (p: any) => p.season === currentYear
      );

      const nfl = allPicks.filter(
        (p: any) => p.league === 'NFL'
      );

      const cfb = allPicks.filter(
        (p: any) => p.league === 'CFB'
      );

      const nflSeason = seasonPicks.filter(
        (p: any) => p.league === 'NFL'
      );

      const cfbSeason = seasonPicks.filter(
        (p: any) => p.league === 'CFB'
      );

      const allSeasons = Array.from(
        new Set(allPicks.map((p: any) => p.season))
      ).sort((a: any, b: any) => b - a);

      const bySeasonNfl: Record<number, any> = {};
      const bySeasonCfb: Record<number, any> = {};

      for (const s of allSeasons as number[]) {
        bySeasonNfl[s] = buildRecord(
          allPicks.filter(
            (p: any) =>
              p.league === 'NFL' &&
              p.season === s
          )
        );

        bySeasonCfb[s] = buildRecord(
          allPicks.filter(
            (p: any) =>
              p.league === 'CFB' &&
              p.season === s
          )
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
      console.error('Record API error:', err);

      return res.status(500).json({
        error: err.message,
      });
    }
  }

  // ─────────────────────────────────────────────
  // GET PICKS PUBLIC
  // ─────────────────────────────────────────────
  if (req.method === 'GET') {
    const {
      week,
      season,
      league,
      status,
    } = req.query;

    try {
      let query: any = db.collection('picks');

      if (week) {
        query = query.where(
          'week',
          '==',
          Number(week)
        );
      }

      if (season) {
        query = query.where(
          'season',
          '==',
          Number(season)
        );
      }

      if (league) {
        query = query.where(
          'league',
          '==',
          String(league).toUpperCase()
        );
      }

      if (status) {
        query = query.where(
          'status',
          '==',
          String(status)
        );
      }

      query = query
        .orderBy('locked_at', 'desc')
        .limit(200);

      const snap = await query.get();

      const picks = snap.docs.map((d: any) => ({
        id: d.id,
        ...d.data(),
      }));

      return res.status(200).json({
        picks,
      });
    } catch (err: any) {
      console.error('Get picks API error:', err);

      return res.status(500).json({
        error: err.message,
      });
    }
  }

  // Everything below requires POST.
  if (req.method !== 'POST') {
    return res.status(405).json({
      error: 'Method not allowed',
    });
  }

  // ─────────────────────────────────────────────
  // ADMIN AUTH
  // ─────────────────────────────────────────────
  const { secret } = req.body;

  if (secret !== ADMIN_SECRET) {
    return res.status(401).json({
      error: 'Unauthorized',
    });
  }

  // ─────────────────────────────────────────────
  // LOCK PICKS
  // ─────────────────────────────────────────────
  if (action === 'lock') {
    const {
      picks,
      week,
      season,
    } = req.body;

    if (
      !picks ||
      !Array.isArray(picks) ||
      picks.length === 0
    ) {
      return res.status(400).json({
        error: 'picks array required',
      });
    }

    if (!week || !season) {
      return res.status(400).json({
        error: 'week and season required',
      });
    }

    const batch = db.batch();
    const lockedPicks: any[] = [];

    for (const pick of picks) {
      if (
        !pick.league ||
        !pick.home_team ||
        !pick.away_team ||
        !pick.picked_team
      ) {
        return res.status(400).json({
          error:
            'Each pick requires league, home_team, away_team and picked_team',
        });
      }

      if (
        pick.picked_team !== pick.home_team &&
        pick.picked_team !== pick.away_team
      ) {
        return res.status(400).json({
          error:
            `Picked team "${pick.picked_team}" does not match the game's teams`,
        });
      }

      const lockedAtIso = new Date().toISOString();

      const lockData = {
        league: String(pick.league).toUpperCase(),

        week: Number(week),
        season: Number(season),

        home_team: pick.home_team,
        away_team: pick.away_team,
        picked_team: pick.picked_team,

        // IMPORTANT:
        // This spread is from the perspective
        // of the selected/picked team.
        spread_at_lock: Number(
          pick.spread_at_lock ?? 0
        ),

        predicted_home:
          pick.predicted_home ?? null,

        predicted_away:
          pick.predicted_away ?? null,

        predicted_margin:
          pick.predicted_margin ?? null,

        predicted_total:
          pick.predicted_total ?? null,

        status: 'pending',

        result: null,

        actual_home: null,
        actual_away: null,

        locked_at:
          FieldValue.serverTimestamp(),

        locked_at_iso:
          lockedAtIso,
      };

      const hashSource = {
        league: lockData.league,
        week: lockData.week,
        season: lockData.season,

        home_team:
          lockData.home_team,

        away_team:
          lockData.away_team,

        picked_team:
          lockData.picked_team,

        spread_at_lock:
          lockData.spread_at_lock,

        locked_at_iso:
          lockData.locked_at_iso,
      };

      const verifyHash =
        hashPick(hashSource);

      const docRef =
        db.collection('picks').doc();

      batch.set(docRef, {
        ...lockData,

        verify_hash:
          verifyHash,

        hash_source:
          hashSource,
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

  // ─────────────────────────────────────────────
  // MANUAL GRADE PICK
  // ─────────────────────────────────────────────
  if (action === 'grade') {
    const {
      pick_id,
      actual_home,
      actual_away,
    } = req.body;

    if (
      !pick_id ||
      actual_home == null ||
      actual_away == null
    ) {
      return res.status(400).json({
        error:
          'pick_id, actual_home, actual_away required',
      });
    }

    const actualHome =
      Number(actual_home);

    const actualAway =
      Number(actual_away);

    if (
      !Number.isFinite(actualHome) ||
      !Number.isFinite(actualAway)
    ) {
      return res.status(400).json({
        error:
          'Scores must be valid numbers',
      });
    }

    const ref =
      db.collection('picks').doc(
        String(pick_id)
      );

    const snap =
      await ref.get();

    if (!snap.exists) {
      return res.status(404).json({
        error: 'Pick not found',
      });
    }

    const pick =
      snap.data()!;

    // Don't silently overwrite a result
    // that has already been graded.
    if (pick.status === 'graded') {
      return res.status(409).json({
        error: 'Pick is already graded',
        pick_id,
        result: pick.result,
      });
    }

    let result:
      | 'win'
      | 'loss'
      | 'push';

    try {
      result = gradeResult(
        pick,
        actualHome,
        actualAway
      );
    } catch (err: any) {
      return res.status(400).json({
        error: err.message,
      });
    }

    await ref.update({
      actual_home:
        actualHome,

      actual_away:
        actualAway,

      result,

      status:
        'graded',

      graded_at:
        FieldValue.serverTimestamp(),

      graded_at_iso:
        new Date().toISOString(),

      grading_source:
        'manual',
    });

    return res.status(200).json({
      ok: true,

      result,

      pick_id,

      picked_team:
        pick.picked_team,

      spread_at_lock:
        Number(
          pick.spread_at_lock ?? 0
        ),

      actual_home:
        actualHome,

      actual_away:
        actualAway,

      grading_source:
        'manual',
    });
  }

  // ─────────────────────────────────────────────
  // DELETE PICK
  // ─────────────────────────────────────────────
  if (action === 'delete') {
    const { pick_id } = req.body;

    if (!pick_id) {
      return res.status(400).json({
        error: 'pick_id required',
      });
    }

    const ref =
      db.collection('picks').doc(
        String(pick_id)
      );

    const snap =
      await ref.get();

    if (!snap.exists) {
      return res.status(404).json({
        error: 'Pick not found',
      });
    }

    await ref.delete();

    return res.status(200).json({
      ok: true,
      deleted: pick_id,
    });
  }

  return res.status(400).json({
    error: 'Unknown action',
  });
}
