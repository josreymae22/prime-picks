import { useState, useEffect, useCallback } from "react";
import Head from "next/head";
import Link from "next/link";
import { useRouter } from "next/router";
import axios from "axios";
import { useAuth } from "../context/AuthContext";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const POSITION_GROUPS = [
  "QB",
  "RB",
  "WR",
  "TE",
  "OL",
  "DL",
  "LB",
  "CB",
  "S",
  "K",
];
const MOVE_TYPES = [
  "trade",
  "free_agency",
  "transfer_portal",
  "waiver",
  "injury_return",
];

const IMPACT_GUIDE = [
  {
    tier: "Elite",
    range: "85–100",
    desc: "Pro Bowl / All-American caliber",
    color: "#C9A84C",
  },
  { tier: "Good", range: "70–84", desc: "Solid starter", color: "#3DAA6A" },
  {
    tier: "Average",
    range: "50–69",
    desc: "League average starter",
    color: "#8B9BB4",
  },
  { tier: "Backup", range: "30–49", desc: "Depth player", color: "#4A5568" },
];

type Player = {
  id?: string;
  player_id: string;
  name: string;
  team: string;
  position_group: string;
  impact_score: number;
  notes?: string;
  league: string;
};

type Move = {
  id?: string;
  player_name: string;
  from_team: string;
  to_team: string;
  position_group: string;
  impact_score: number;
  move_type: string;
  timestamp: string;
  notes?: string;
};

type RosterStatus = {
  db_stats: {
    teams: number;
    players: number;
    moves_logged: number;
  };
  data_sources: {
    active_source: string;
    sportsdata_io: boolean;
    mysportsfeeds: boolean;
  };
};

export default function RosterPage() {
  const { user, userStatus, loading: authLoading } = useAuth();
  const router = useRouter();

  const [tab, setTab] = useState<"add" | "transfer" | "moves" | "players">(
    "players",
  );
  const [status, setStatus] = useState<RosterStatus | null>(null);
  const [players, setPlayers] = useState<Player[]>([]);
  const [moves, setMoves] = useState<Move[]>([]);
  const [adminSecret, setAdminSecret] = useState("");
  const [authed, setAuthed] = useState(false);
  const [secretError, setSecretError] = useState("");
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState("");
  const [searchQ, setSearchQ] = useState("");
  const [searchResults, setSearchResults] = useState<Player[]>([]);

  const [addForm, setAddForm] = useState<Record<string, string>>({});
  const [xferPlayerId, setXferPlayerId] = useState("");
  const [xferTeam, setXferTeam] = useState("");
  const [xferType, setXferType] = useState("trade");
  const [xferNotes, setXferNotes] = useState("");

  useEffect(() => {
    if (!authLoading && (!user || userStatus !== "approved")) {
      router.replace("/login");
    }
  }, [user, userStatus, authLoading, router]);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(""), 3500);
  };

  const loadStatus = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/roster/status`);
      setStatus(r.data);
    } catch {
      // Keep page usable if status fails.
    }
  }, []);

  const loadPlayers = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/roster/players`);
      const data = r.data;
      setPlayers(Array.isArray(data) ? data : data.players || []);
    } catch {
      setPlayers([]);
    }
  }, []);

  const loadMoves = useCallback(async () => {
    try {
      const r = await axios.get(`${API}/roster/moves`);
      const data = r.data;
      setMoves(Array.isArray(data) ? data : data.moves || []);
    } catch {
      setMoves([]);
    }
  }, []);

  useEffect(() => {
    loadStatus();
    loadPlayers();
    loadMoves();
  }, [loadStatus, loadPlayers, loadMoves]);

  const handleAuth = async () => {
    try {
      const r = await axios.get(`${API}/roster/players`, {
        params: { secret: adminSecret },
      });
      const data = r.data;
      setPlayers(Array.isArray(data) ? data : data.players || []);
      setAuthed(true);
      setSecretError("");
      loadMoves();
    } catch {
      setSecretError("Invalid admin password.");
    }
  };

  const handleSearch = async () => {
    if (!searchQ.trim()) {
      setSearchResults([]);
      loadPlayers();
      return;
    }

    try {
      const r = await axios.get(`${API}/roster/players/search`, {
        params: { q: searchQ },
      });
      const data = r.data;
      setSearchResults(Array.isArray(data) ? data : data.players || []);
    } catch {
      setSearchResults([]);
    }
  };

  const handleAddPlayer = async () => {
    if (
      !addForm.name ||
      !addForm.team ||
      !addForm.position_group ||
      !addForm.impact_score
    ) {
      showToast("Name, team, position group, and impact score are required.");
      return;
    }

    setLoading(true);

    try {
      await axios.post(
        `${API}/roster/player/add`,
        {
          player_id: addForm.player_id || `manual_${Date.now()}`,
          name: addForm.name,
          team: addForm.team,
          position_group: addForm.position_group,
          impact_score: parseFloat(addForm.impact_score),
          league: addForm.league || "NFL",
          notes: addForm.notes || "",
        },
        { params: { secret: adminSecret } },
      );

      showToast(`✓ ${addForm.name} added to ${addForm.team}`);
      setAddForm({});
      setSearchResults([]);
      await loadPlayers();
      await loadStatus();
    } catch (e: any) {
      showToast("Error: " + (e?.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  };

  const handleTransfer = async () => {
    if (!xferPlayerId || !xferTeam) {
      showToast("Player ID and new team required.");
      return;
    }

    setLoading(true);

    try {
      const r = await axios.post(
        `${API}/roster/player/transfer`,
        {
          player_id: xferPlayerId,
          new_team: xferTeam,
          move_type: xferType,
          notes: xferNotes,
        },
        { params: { secret: adminSecret } },
      );

      const mv = r.data.move;

      showToast(`✓ ${mv.player_name} → ${xferTeam}`);
      setXferPlayerId("");
      setXferTeam("");
      setXferNotes("");
      setSearchQ("");
      setSearchResults([]);

      await loadPlayers();
      await loadMoves();
      await loadStatus();
    } catch (e: any) {
      showToast("Error: " + (e?.response?.data?.detail || e.message));
    } finally {
      setLoading(false);
    }
  };

  const visiblePlayers = searchResults.length > 0 ? searchResults : players;

  if (authLoading || !user || userStatus !== "approved") {
    return (
      <div className="field-bg min-h-screen flex items-center justify-center">
        <div
          className="animate-pulse-gold score-display text-slate"
          style={{ fontSize: 18 }}
        >
          LOADING...
        </div>
      </div>
    );
  }

  return (
    <>
      <Head>
        <title>Roster Intel — Prime Picks</title>
      </Head>

      <div className="field-bg min-h-screen px-4 py-10">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-2 flex-wrap gap-3">
            <div className="flex items-center gap-3">
              <span style={{ fontSize: 28 }}>👤</span>
              <h1
                className="score-display text-chalk"
                style={{ fontSize: 36, letterSpacing: "0.08em" }}
              >
                ROSTER INTEL
              </h1>
            </div>

            <nav
              className="flex gap-4 text-xs"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              <Link href="/" style={{ color: "#8B9BB4" }}>
                Predict
              </Link>
              <Link href="/card" style={{ color: "#8B9BB4" }}>
                Weekly Card
              </Link>
              <Link href="/roster" style={{ color: "#C9A84C" }}>
                Roster Intel
              </Link>
            </nav>
          </div>

          <div className="gold-line mb-5" />

          {status && (
            <div className="panel-bright rounded-xl px-5 py-3 mb-5 flex flex-wrap gap-5 items-center">
              <div
                className="text-xs"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                <span className="text-slate">Teams tracked: </span>
                <span className="text-chalk">{status.db_stats.teams}</span>
              </div>

              <div
                className="text-xs"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                <span className="text-slate">Players: </span>
                <span className="text-chalk">{status.db_stats.players}</span>
              </div>

              <div
                className="text-xs"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                <span className="text-slate">Moves logged: </span>
                <span className="text-chalk">
                  {status.db_stats.moves_logged}
                </span>
              </div>

              <div
                className="text-xs"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                <span className="text-slate">Data source: </span>
                <span
                  style={{
                    color:
                      status.data_sources.sportsdata_io ||
                      status.data_sources.mysportsfeeds
                        ? "#3DAA6A"
                        : "#C9A84C",
                  }}
                >
                  {status.data_sources.active_source}
                </span>
              </div>

              {!status.data_sources.sportsdata_io &&
                !status.data_sources.mysportsfeeds && (
                  <span
                    className="text-xs"
                    style={{ color: "#C9A84C", fontFamily: "var(--font-mono)" }}
                  >
                    ⚠ Add SPORTSDATA_API_KEY for auto-sync
                  </span>
                )}
            </div>
          )}

          {toast && (
            <div
              className="mb-4 px-4 py-2 rounded text-sm animate-fade-up"
              style={{
                background: "rgba(61,170,106,0.12)",
                color: "#3DAA6A",
                border: "1px solid rgba(61,170,106,0.3)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {toast}
            </div>
          )}

          <div className="panel rounded-xl p-4 mb-5">
            <div
              className="text-xs text-slate mb-3 uppercase tracking-widest"
              style={{ fontFamily: "var(--font-mono)" }}
            >
              Impact Score Guide
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {IMPACT_GUIDE.map((t) => (
                <div
                  key={t.tier}
                  className="rounded px-3 py-2"
                  style={{
                    background: `${t.color}12`,
                    border: `1px solid ${t.color}30`,
                  }}
                >
                  <div
                    className="text-xs font-semibold"
                    style={{ color: t.color, fontFamily: "var(--font-mono)" }}
                  >
                    {t.tier} ({t.range})
                  </div>
                  <div className="text-xs text-slate mt-0.5">{t.desc}</div>
                </div>
              ))}
            </div>
          </div>

          <div className="flex gap-2 mb-5 flex-wrap">
            {(["players", "add", "transfer", "moves"] as const).map((t) => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="score-display px-4 py-1.5 rounded text-sm capitalize"
                style={{
                  fontSize: 14,
                  letterSpacing: "0.08em",
                  background: tab === t ? "#C9A84C" : "rgba(15,44,71,0.5)",
                  color: tab === t ? "#030B14" : "#8B9BB4",
                  border: "1px solid",
                  borderColor: tab === t ? "#C9A84C" : "rgba(201,168,76,0.15)",
                  cursor: "pointer",
                }}
              >
                {t === "add"
                  ? "ADD PLAYER"
                  : t === "transfer"
                    ? "LOG MOVE"
                    : t === "moves"
                      ? "MOVE LOG"
                      : "PLAYERS"}
              </button>
            ))}
          </div>

          {(tab === "add" || tab === "transfer") && !authed && (
            <div className="panel rounded-xl p-6 max-w-xs">
              <label
                className="block text-xs text-slate mb-1 uppercase tracking-widest"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                Admin Password
              </label>

              <input
                type="password"
                className="w-full rounded px-3 py-2 text-sm mb-2"
                value={adminSecret}
                onChange={(e) => setAdminSecret(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAuth()}
                placeholder="Enter admin password"
              />

              {secretError && (
                <p
                  className="text-xs mb-2"
                  style={{ color: "#D94040", fontFamily: "var(--font-mono)" }}
                >
                  {secretError}
                </p>
              )}

              <button
                onClick={handleAuth}
                className="score-display w-full py-2 rounded"
                style={{
                  fontSize: 14,
                  letterSpacing: "0.08em",
                  background: "linear-gradient(135deg, #C9A84C, #E8C96A)",
                  color: "#030B14",
                  border: "none",
                  cursor: "pointer",
                }}
              >
                UNLOCK
              </button>
            </div>
          )}

          {tab === "players" && (
            <div>
              <div className="flex gap-2 mb-4">
                <input
                  className="flex-1 rounded px-3 py-2 text-sm"
                  placeholder="Search by name or team..."
                  value={searchQ}
                  onChange={(e) => setSearchQ(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                />

                <button
                  onClick={handleSearch}
                  className="score-display px-4 py-2 rounded"
                  style={{
                    fontSize: 13,
                    background: "rgba(201,168,76,0.15)",
                    color: "#C9A84C",
                    border: "1px solid rgba(201,168,76,0.3)",
                    cursor: "pointer",
                  }}
                >
                  SEARCH
                </button>
              </div>

              <div className="space-y-2">
                {visiblePlayers.slice(0, 50).map((p, i) => (
                  <div
                    key={p.player_id || p.id || i}
                    className="panel rounded-xl px-4 py-3 flex items-center justify-between gap-3"
                  >
                    <div>
                      <span className="text-chalk text-sm font-semibold">
                        {p.name}
                      </span>
                      <span
                        className="text-slate text-xs ml-2"
                        style={{ fontFamily: "var(--font-mono)" }}
                      >
                        {p.position_group} · {p.team}
                      </span>
                      {p.notes && (
                        <span
                          className="text-xs ml-2"
                          style={{
                            color: "#4A5568",
                            fontFamily: "var(--font-mono)",
                          }}
                        >
                          {p.notes}
                        </span>
                      )}
                    </div>

                    <div className="shrink-0">
                      <span
                        className="score-display text-lg"
                        style={{
                          color:
                            p.impact_score >= 85
                              ? "#C9A84C"
                              : p.impact_score >= 70
                                ? "#3DAA6A"
                                : p.impact_score >= 50
                                  ? "#8B9BB4"
                                  : "#4A5568",
                        }}
                      >
                        {p.impact_score}
                      </span>
                    </div>
                  </div>
                ))}

                {visiblePlayers.length === 0 && (
                  <div className="panel rounded-xl p-8 text-center">
                    <p
                      className="text-slate text-sm"
                      style={{ fontFamily: "var(--font-mono)" }}
                    >
                      No players in database yet. Add players using the Add
                      Player tab.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {tab === "add" && authed && (
            <div className="panel rounded-xl p-6 max-w-lg">
              <div className="space-y-4">
                {[
                  {
                    key: "name",
                    label: "Player Name",
                    placeholder: "Patrick Mahomes",
                  },
                  {
                    key: "team",
                    label: "Current Team",
                    placeholder: "Kansas City Chiefs",
                  },
                  {
                    key: "player_id",
                    label: "Player ID (optional)",
                    placeholder: "Auto-generated if blank",
                  },
                  {
                    key: "notes",
                    label: "Notes (optional)",
                    placeholder: "Injury, contract year, etc.",
                  },
                ].map((f) => (
                  <div key={f.key}>
                    <label
                      className="block text-xs text-slate mb-1 uppercase tracking-widest"
                      style={{ fontFamily: "var(--font-mono)" }}
                    >
                      {f.label}
                    </label>
                    <input
                      className="w-full rounded px-3 py-2.5 text-sm"
                      placeholder={f.placeholder}
                      value={addForm[f.key] || ""}
                      onChange={(e) =>
                        setAddForm((prev) => ({
                          ...prev,
                          [f.key]: e.target.value,
                        }))
                      }
                    />
                  </div>
                ))}

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label
                      className="block text-xs text-slate mb-1 uppercase tracking-widest"
                      style={{ fontFamily: "var(--font-mono)" }}
                    >
                      Position Group
                    </label>

                    <select
                      className="w-full rounded px-3 py-2.5 text-sm"
                      value={addForm.position_group || ""}
                      onChange={(e) =>
                        setAddForm((prev) => ({
                          ...prev,
                          position_group: e.target.value,
                        }))
                      }
                    >
                      <option value="">Select...</option>
                      {POSITION_GROUPS.map((g) => (
                        <option key={g} value={g}>
                          {g}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label
                      className="block text-xs text-slate mb-1 uppercase tracking-widest"
                      style={{ fontFamily: "var(--font-mono)" }}
                    >
                      League
                    </label>

                    <select
                      className="w-full rounded px-3 py-2.5 text-sm"
                      value={addForm.league || "NFL"}
                      onChange={(e) =>
                        setAddForm((prev) => ({
                          ...prev,
                          league: e.target.value,
                        }))
                      }
                    >
                      <option value="NFL">NFL</option>
                      <option value="CFB">NCAAF</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label
                    className="block text-xs text-slate mb-1 uppercase tracking-widest"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    Impact Score (0–100)
                  </label>

                  <input
                    type="number"
                    min="0"
                    max="100"
                    className="w-full rounded px-3 py-2.5 text-sm"
                    placeholder="e.g. 88 for elite QB"
                    value={addForm.impact_score || ""}
                    onChange={(e) =>
                      setAddForm((prev) => ({
                        ...prev,
                        impact_score: e.target.value,
                      }))
                    }
                  />
                </div>
              </div>

              <button
                onClick={handleAddPlayer}
                disabled={loading}
                className="w-full mt-5 score-display py-3 rounded"
                style={{
                  fontSize: 16,
                  letterSpacing: "0.1em",
                  background: loading
                    ? "rgba(201,168,76,0.2)"
                    : "linear-gradient(135deg, #C9A84C, #E8C96A)",
                  color: loading ? "#4A5568" : "#030B14",
                  border: "none",
                  cursor: loading ? "not-allowed" : "pointer",
                }}
              >
                {loading ? "SAVING..." : "ADD PLAYER"}
              </button>
            </div>
          )}

          {tab === "transfer" && authed && (
            <div className="panel rounded-xl p-6 max-w-lg">
              <p
                className="text-xs text-slate mb-4"
                style={{ fontFamily: "var(--font-mono)", lineHeight: 1.6 }}
              >
                Log a player move. This updates both teams&apos; rating
                adjustments immediately, affecting all future predictions.
              </p>

              <div className="space-y-4">
                <div>
                  <label
                    className="block text-xs text-slate mb-1 uppercase tracking-widest"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    Player ID or Search
                  </label>

                  <input
                    className="w-full rounded px-3 py-2.5 text-sm"
                    placeholder="Search name first →"
                    value={xferPlayerId}
                    onChange={(e) => setXferPlayerId(e.target.value)}
                  />

                  <div className="flex gap-2 mt-2">
                    <input
                      className="flex-1 rounded px-3 py-2 text-xs"
                      placeholder="Search player name..."
                      value={searchQ}
                      onChange={(e) => setSearchQ(e.target.value)}
                    />

                    <button
                      onClick={handleSearch}
                      className="score-display px-3 py-2 rounded text-xs"
                      style={{
                        fontSize: 12,
                        background: "rgba(201,168,76,0.1)",
                        color: "#C9A84C",
                        border: "1px solid rgba(201,168,76,0.2)",
                        cursor: "pointer",
                      }}
                    >
                      FIND
                    </button>
                  </div>

                  {searchResults.slice(0, 5).map((p) => (
                    <button
                      key={p.player_id}
                      onClick={() => {
                        setXferPlayerId(p.player_id || p.name);
                        setSearchResults([]);
                        setSearchQ("");
                      }}
                      className="w-full text-left rounded px-3 py-2 mt-1 text-xs"
                      style={{
                        background: "rgba(15,44,71,0.6)",
                        color: "#F0EEE6",
                        border: "1px solid rgba(201,168,76,0.15)",
                        cursor: "pointer",
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      {p.name} · {p.position_group} · {p.team} · Impact:{" "}
                      {p.impact_score}
                    </button>
                  ))}
                </div>

                <div>
                  <label
                    className="block text-xs text-slate mb-1 uppercase tracking-widest"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    New Team
                  </label>

                  <input
                    className="w-full rounded px-3 py-2.5 text-sm"
                    placeholder="e.g. Dallas Cowboys"
                    value={xferTeam}
                    onChange={(e) => setXferTeam(e.target.value)}
                  />
                </div>

                <div>
                  <label
                    className="block text-xs text-slate mb-1 uppercase tracking-widest"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    Move Type
                  </label>

                  <select
                    className="w-full rounded px-3 py-2.5 text-sm"
                    value={xferType}
                    onChange={(e) => setXferType(e.target.value)}
                  >
                    {MOVE_TYPES.map((t) => (
                      <option key={t} value={t}>
                        {t.replace("_", " ")}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label
                    className="block text-xs text-slate mb-1 uppercase tracking-widest"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    Notes (optional)
                  </label>

                  <input
                    className="w-full rounded px-3 py-2.5 text-sm"
                    placeholder="e.g. 3-year, $90M deal"
                    value={xferNotes}
                    onChange={(e) => setXferNotes(e.target.value)}
                  />
                </div>
              </div>

              <button
                onClick={handleTransfer}
                disabled={loading}
                className="w-full mt-5 score-display py-3 rounded"
                style={{
                  fontSize: 16,
                  letterSpacing: "0.1em",
                  background: loading
                    ? "rgba(201,168,76,0.2)"
                    : "linear-gradient(135deg, #C9A84C, #E8C96A)",
                  color: loading ? "#4A5568" : "#030B14",
                  border: "none",
                  cursor: loading ? "not-allowed" : "pointer",
                }}
              >
                {loading ? "LOGGING..." : "LOG MOVE"}
              </button>
            </div>
          )}

          {tab === "moves" && (
            <div className="space-y-2">
              {moves.length === 0 && (
                <div className="panel rounded-xl p-8 text-center">
                  <p
                    className="text-slate text-sm"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    No moves logged yet.
                  </p>
                </div>
              )}

              {moves.map((m, i) => (
                <div key={m.id || i} className="panel rounded-xl px-4 py-3">
                  <div className="flex items-center justify-between flex-wrap gap-2">
                    <div>
                      <span className="text-chalk text-sm font-semibold">
                        {m.player_name}
                      </span>
                      <span
                        className="text-slate text-xs ml-2"
                        style={{ fontFamily: "var(--font-mono)" }}
                      >
                        {m.position_group} · Impact {m.impact_score}
                      </span>
                    </div>

                    <span
                      className="text-xs px-2 py-0.5 rounded"
                      style={{
                        background: "rgba(201,168,76,0.1)",
                        color: "#C9A84C",
                        border: "1px solid rgba(201,168,76,0.2)",
                        fontFamily: "var(--font-mono)",
                      }}
                    >
                      {m.move_type.replace("_", " ")}
                    </span>
                  </div>

                  <div
                    className="mt-1 text-xs text-slate"
                    style={{ fontFamily: "var(--font-mono)" }}
                  >
                    {m.from_team} →{" "}
                    <span className="text-chalk">{m.to_team}</span>
                    {m.notes && (
                      <span className="ml-2 opacity-60">· {m.notes}</span>
                    )}
                    <span className="ml-2 opacity-40">
                      · {new Date(m.timestamp).toLocaleDateString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
