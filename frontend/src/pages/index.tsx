import { useState, useEffect, useCallback } from "react";
import axios from "axios";
import Head from "next/head";
import { useRouter } from "next/router";
import { useAuth } from "../context/AuthContext";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const NFL_TEAMS = [
  "Arizona Cardinals",
  "Atlanta Falcons",
  "Baltimore Ravens",
  "Buffalo Bills",
  "Carolina Panthers",
  "Chicago Bears",
  "Cincinnati Bengals",
  "Cleveland Browns",
  "Dallas Cowboys",
  "Denver Broncos",
  "Detroit Lions",
  "Green Bay Packers",
  "Houston Texans",
  "Indianapolis Colts",
  "Jacksonville Jaguars",
  "Kansas City Chiefs",
  "Las Vegas Raiders",
  "Los Angeles Chargers",
  "Los Angeles Rams",
  "Miami Dolphins",
  "Minnesota Vikings",
  "New England Patriots",
  "New Orleans Saints",
  "New York Giants",
  "New York Jets",
  "Philadelphia Eagles",
  "Pittsburgh Steelers",
  "San Francisco 49ers",
  "Seattle Seahawks",
  "Tampa Bay Buccaneers",
  "Tennessee Titans",
  "Washington Commanders",
];

const CFB_TOP_TEAMS = [
  "Alabama",
  "Georgia",
  "Ohio State",
  "Michigan",
  "Texas",
  "Penn State",
  "Notre Dame",
  "LSU",
  "USC",
  "Oregon",
  "Florida State",
  "Clemson",
  "Oklahoma",
  "Tennessee",
  "Utah",
  "Washington",
  "Missouri",
  "Ole Miss",
  "Kansas State",
  "Iowa",
  "Wisconsin",
  "Auburn",
  "Florida",
  "Miami",
  "Texas A&M",
  "Colorado",
  "Louisville",
  "Pittsburgh",
  "NC State",
  "North Carolina",
  "Duke",
  "Virginia Tech",
  "TCU",
  "Baylor",
  "Oklahoma State",
  "Iowa State",
  "Cincinnati",
  "Houston",
  "UCF",
  "Tulane",
  "Air Force",
  "Boise State",
  "UNLV",
  "San Diego State",
  "Wyoming",
  "Marshall",
  "James Madison",
  "Liberty",
  "Appalachian State",
  "Coastal Carolina",
];

type Prediction = {
  predicted_home_score: number;
  predicted_away_score: number;
  predicted_margin: number;
  predicted_total: number;
  margin_80_lo: number;
  margin_80_hi: number;
  total_80_lo: number;
  total_80_hi: number;
  home_win_prob: number;
  model_trained: boolean;
};

type Factor = {
  label: string;
  detail: string;
  impact: "high" | "medium" | "low";
};

type PredictResult = {
  home_team: string;
  away_team: string;
  league: string;
  prediction: Prediction;
  key_factors: Factor[];
};

const impactColor = {
  high: "#C9A84C",
  medium: "#8B9BB4",
  low: "#4A5568",
};

const impactLabel = {
  high: "↑↑ HIGH",
  medium: "→ MED",
  low: "↓ LOW",
};

export default function Home() {
  const { user, userStatus, loading: authLoading, signOut } = useAuth();
  const router = useRouter();

  const [league, setLeague] = useState<"NFL" | "CFB">("NFL");
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");
  const [neutral, setNeutral] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResult | null>(null);
  const [error, setError] = useState("");
  const [apiReady, setApiReady] = useState<boolean | null>(null);

  useEffect(() => {
    if (!authLoading && (!user || userStatus !== "approved")) {
      router.replace("/login");
    }
  }, [user, userStatus, authLoading, router]);

  useEffect(() => {
    axios
      .get(`${API}/health`)
      .then((r) => setApiReady(r.data.ready))
      .catch(() => setApiReady(false));
  }, []);

  const teams = league === "NFL" ? NFL_TEAMS : CFB_TOP_TEAMS;

  const handlePredict = useCallback(async () => {
    if (!homeTeam || !awayTeam) {
      setError("Select both teams.");
      return;
    }

    if (homeTeam === awayTeam) {
      setError("Select two different teams.");
      return;
    }

    setError("");
    setLoading(true);
    setResult(null);

    try {
      const r = await axios.post(`${API}/predict`, {
        league,
        home_team: homeTeam,
        away_team: awayTeam,
        neutral_site: neutral,
      });

      setResult(r.data);
    } catch (e: any) {
      setError(
        e?.response?.data?.detail || "Prediction failed. Is the API running?",
      );
    } finally {
      setLoading(false);
    }
  }, [league, homeTeam, awayTeam, neutral]);

  if (authLoading || !user || userStatus !== "approved") {
    return (
      <div className="field-bg min-h-screen flex items-center justify-center">
        <div
          className="animate-pulse-gold score-display text-slate"
          style={{ fontSize: 18, letterSpacing: "0.1em" }}
        >
          LOADING...
        </div>
      </div>
    );
  }

  const winProb = result?.prediction.home_win_prob ?? 0.5;
  const homeProb = Math.round(winProb * 100);
  const awayProb = 100 - homeProb;

  return (
    <>
      <Head>
        <title>Prime Picks</title>
        <meta name="description" content="NFL & CFB score prediction engine" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div className="field-bg min-h-screen px-4 py-10 md:py-16">
        <div className="max-w-4xl mx-auto mb-10">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-3">
              <span style={{ fontSize: 28 }}>🏈</span>
              <h1
                className="score-display text-chalk"
                style={{ fontSize: 42, letterSpacing: "0.08em", lineHeight: 1 }}
              >
                PRIME PICKS
              </h1>
            </div>

            <nav
              className="flex gap-4 text-xs"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              <a href="/" style={{ color: "#C9A84C" }}>
                Predict
              </a>
              <a href="/card" style={{ color: "#8B9BB4" }}>
                Weekly Card
              </a>
              <a href="/roster" style={{ color: "#8B9BB4" }}>
                Roster Intel
              </a>
              <a href="/record" style={{ color: "#8B9BB4" }}>
                Record
              </a>
            </nav>

            <div className="flex items-center gap-3">
              <span
                className="text-xs text-slate hidden md:block"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                {user?.email}
              </span>
              <button
                onClick={() => signOut().then(() => router.replace("/login"))}
                className="text-xs px-3 py-1.5 rounded"
                style={{
                  fontFamily: "var(--font-mono)",
                  color: "#8B9BB4",
                  border: "1px solid rgba(201,168,76,0.2)",
                  background: "transparent",
                  cursor: "pointer",
                }}
              >
                Sign Out
              </button>
            </div>
          </div>

          <p
            className="text-slate text-sm"
            style={{ fontFamily: "var(--font-body)" }}
          >
            Score margin & total prediction engine — NFL & NCAAF
          </p>

          <div className="gold-line mt-4" />

          <div className="mt-3 flex items-center gap-2">
            <span
              className="inline-block rounded-full"
              style={{
                width: 8,
                height: 8,
                background:
                  apiReady === null
                    ? "#8B9BB4"
                    : apiReady
                      ? "#3DAA6A"
                      : "#D94040",
              }}
            />
            <span
              className="text-xs text-slate"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              {apiReady === null
                ? "Checking API..."
                : apiReady
                  ? "Models ready"
                  : "API offline — start backend"}
            </span>
          </div>
        </div>

        <div className="max-w-4xl mx-auto panel rounded-xl p-6 md:p-8">
          <div className="flex gap-2 mb-6">
            {(["NFL", "CFB"] as const).map((l) => (
              <button
                key={l}
                onClick={() => {
                  setLeague(l);
                  setHomeTeam("");
                  setAwayTeam("");
                  setResult(null);
                }}
                className="score-display px-6 py-2 rounded text-sm transition-all"
                style={{
                  background: league === l ? "#C9A84C" : "rgba(15,44,71,0.4)",
                  color: league === l ? "#030B14" : "#8B9BB4",
                  letterSpacing: "0.1em",
                  border: "1px solid",
                  borderColor:
                    league === l ? "#C9A84C" : "rgba(201,168,76,0.15)",
                  cursor: "pointer",
                  fontSize: 18,
                }}
              >
                {l === "NFL" ? "NFL" : "NCAAF"}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            <div>
              <label
                className="block text-xs text-slate mb-1 uppercase tracking-widest"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                Home Team
              </label>
              <select
                className="w-full rounded px-3 py-2.5 text-sm"
                value={homeTeam}
                onChange={(e) => setHomeTeam(e.target.value)}
              >
                <option value="">Select team...</option>
                {teams.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label
                className="block text-xs text-slate mb-1 uppercase tracking-widest"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                Away Team
              </label>
              <select
                className="w-full rounded px-3 py-2.5 text-sm"
                value={awayTeam}
                onChange={(e) => setAwayTeam(e.target.value)}
              >
                <option value="">Select team...</option>
                {teams.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex items-center gap-2 mb-6">
            <input
              type="checkbox"
              id="neutral"
              checked={neutral}
              onChange={(e) => setNeutral(e.target.checked)}
              className="w-4 h-4"
              style={{ accentColor: "#C9A84C" }}
            />
            <label
              htmlFor="neutral"
              className="text-sm text-slate cursor-pointer"
            >
              Neutral site (no home field adjustment)
            </label>
          </div>

          {error && (
            <div
              className="mb-4 px-3 py-2 rounded text-sm"
              style={{
                background: "rgba(217,64,64,0.1)",
                color: "#D94040",
                border: "1px solid rgba(217,64,64,0.2)",
              }}
            >
              {error}
            </div>
          )}

          <button
            onClick={handlePredict}
            disabled={loading || !homeTeam || !awayTeam}
            className="w-full score-display py-3 rounded transition-all"
            style={{
              fontSize: 22,
              letterSpacing: "0.12em",
              background:
                loading || !homeTeam || !awayTeam
                  ? "rgba(201,168,76,0.2)"
                  : "linear-gradient(135deg, #C9A84C, #E8C96A)",
              color: loading || !homeTeam || !awayTeam ? "#4A5568" : "#030B14",
              cursor:
                loading || !homeTeam || !awayTeam ? "not-allowed" : "pointer",
              border: "none",
            }}
          >
            {loading ? "⚡ COMPUTING..." : "PREDICT SCORE"}
          </button>
        </div>

        {result && (
          <div className="max-w-4xl mx-auto mt-6 animate-fade-up">
            <div className="panel-bright rounded-xl p-6 md:p-8 mb-4">
              <div
                className="text-xs text-slate mb-4 uppercase tracking-widest"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                Predicted Final Score
                {!result.prediction.model_trained && (
                  <span className="ml-2 text-gold" style={{ color: "#C9A84C" }}>
                    · stat-based fallback (model training)
                  </span>
                )}
              </div>

              <div className="flex items-center justify-center gap-6 md:gap-12 mb-6">
                <div className="text-center flex-1">
                  <div
                    className="text-slate text-xs uppercase tracking-widest mb-1"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    {result.home_team}
                  </div>
                  <div
                    className="score-display"
                    style={{ fontSize: 72, color: "#F0EEE6", lineHeight: 1 }}
                  >
                    {Math.round(result.prediction.predicted_home_score)}
                  </div>
                  <div
                    className="text-xs mt-1"
                    style={{ color: "#C9A84C", fontFamily: "var(--font-mono)" }}
                  >
                    HOME · {homeProb}%
                  </div>
                </div>

                <div>
                  <div
                    className="score-display text-slate"
                    style={{ fontSize: 36 }}
                  >
                    VS
                  </div>
                </div>

                <div className="text-center flex-1">
                  <div
                    className="text-slate text-xs uppercase tracking-widest mb-1"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    {result.away_team}
                  </div>
                  <div
                    className="score-display"
                    style={{ fontSize: 72, color: "#F0EEE6", lineHeight: 1 }}
                  >
                    {Math.round(result.prediction.predicted_away_score)}
                  </div>
                  <div
                    className="text-xs mt-1"
                    style={{ color: "#8B9BB4", fontFamily: "var(--font-mono)" }}
                  >
                    AWAY · {awayProb}%
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="confidence-band rounded px-3 py-3">
                  <div
                    className="text-xs text-slate uppercase tracking-widest mb-1"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    Score Margin
                  </div>
                  <div
                    className="score-display"
                    style={{ fontSize: 28, color: "#C9A84C" }}
                  >
                    {result.prediction.predicted_margin > 0
                      ? `${result.home_team.split(" ").pop()} +${Math.abs(result.prediction.predicted_margin).toFixed(1)}`
                      : `${result.away_team.split(" ").pop()} +${Math.abs(result.prediction.predicted_margin).toFixed(1)}`}
                  </div>
                  <div
                    className="text-xs text-slate mt-1"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    80% CI: {result.prediction.margin_80_lo.toFixed(1)} to +
                    {result.prediction.margin_80_hi.toFixed(1)}
                  </div>
                </div>

                <div className="confidence-band rounded px-3 py-3">
                  <div
                    className="text-xs text-slate uppercase tracking-widest mb-1"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    Predicted Total
                  </div>
                  <div
                    className="score-display"
                    style={{ fontSize: 28, color: "#C9A84C" }}
                  >
                    {result.prediction.predicted_total.toFixed(1)}
                  </div>
                  <div
                    className="text-xs text-slate mt-1"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    80% CI: {result.prediction.total_80_lo.toFixed(1)} —{" "}
                    {result.prediction.total_80_hi.toFixed(1)}
                  </div>
                </div>
              </div>
            </div>

            {result.key_factors.length > 0 && (
              <div className="panel rounded-xl p-6">
                <div
                  className="text-xs text-slate mb-4 uppercase tracking-widest"
                  style={{ fontFamily: "var(--font-mono)" }}
                >
                  Key Factors
                </div>
                <div className="space-y-3">
                  {result.key_factors.map((f, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <span
                        className="text-xs rounded px-1.5 py-0.5 mt-0.5 shrink-0"
                        style={{
                          fontFamily: "var(--font-mono)",
                          background: `${impactColor[f.impact]}18`,
                          color: impactColor[f.impact],
                          border: `1px solid ${impactColor[f.impact]}40`,
                        }}
                      >
                        {impactLabel[f.impact]}
                      </span>
                      <div>
                        <div className="text-sm font-medium text-chalk">
                          {f.label}
                        </div>
                        <div className="text-xs text-slate mt-0.5">
                          {f.detail}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <p
              className="text-center text-xs text-slate mt-4"
              style={{ fontFamily: "var(--font-mono)", opacity: 0.5 }}
            >
              Statistical model only. Confidence intervals reflect typical
              prediction error (~9–14 pt RMSE). Not financial advice.
            </p>
          </div>
        )}
      </div>
    </>
  );
}
