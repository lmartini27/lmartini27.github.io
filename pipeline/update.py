"""
update.py -- weekly pipeline for the live projections page.

Every run it:
  1. Downloads the latest nflverse weekly player stats and schedule.
  2. GRADES any past projections whose games have now been played
     (model vs. the naive "recent average" baseline).
  3. Fills in a retroactive BACKTEST for earlier weeks of the season
     that were never projected live (clearly labeled as such).
  4. Trains the model on everything available and PROJECTS the next
     week's games, saving a timestamped file.

Projections saved before the week's first kickoff are marked "live".
Git commit times provide an independent record of when each file was
published.

Players who aren't on their team's active roster that week (injured
reserve, practice squad, released) are left out. A separate file,
data/injuries.json, carries the official injury report (Out, Doubtful,
Questionable) and is refreshed later in the week with --injuries-only,
so the site can flag or hide injured players without changing the
locked projections.

Run locally:   python pipeline/update.py
Options:       --season 2026   --cache DIR (reuse downloaded CSVs)
               --injuries-only (refresh data/injuries.json and nothing else)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
PROJ_DIR = os.path.join(DATA_DIR, "projections")

STATS_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.csv"
SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
ROSTER_URL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{year}.csv"
INJURY_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{year}.csv"
REPORT_STATUSES = ["Out", "Doubtful", "Questionable"]

FIRST_SEASON = 2021
POSITIONS = ["QB", "RB", "WR", "TE"]
STAT_COLS = ["fantasy_points_ppr", "targets", "carries", "receiving_yards",
             "rushing_yards", "passing_yards", "receptions"]
WINDOW = 3
MAX_WEEK_GAP = 2           # allows one bye week between games
BACKTEST_FIRST_WEEK = 4    # earliest week with 3 games of in-season history
ROLL_COLS = [f"{c}_avg_last{WINDOW}" for c in STAT_COLS]
OPP_COL = "next_opp_def_allowed"
FEATURES = ROLL_COLS + [f"pos_{p}" for p in POSITIONS] + [OPP_COL]
BASELINE_COL = f"fantasy_points_ppr_avg_last{WINDOW}"
EASTERN = ZoneInfo("America/New_York")


# ---------------------------------------------------------------- data

def read_csv(url, cache_dir, name):
    if cache_dir:
        path = os.path.join(cache_dir, name)
        if os.path.exists(path):
            return pd.read_csv(path, low_memory=False)
    df = pd.read_csv(url, low_memory=False)
    if cache_dir:
        df.to_csv(os.path.join(cache_dir, name), index=False)
    return df


def load_stats(season, cache_dir):
    frames = []
    for year in range(FIRST_SEASON, season + 1):
        df = read_csv(STATS_URL.format(year=year), cache_dir, f"stats_{year}.csv")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["season_type"] == "REG") & (df["position"].isin(POSITIONS))]
    keep = ["player_id", "player_display_name", "position", "team", "opponent_team",
            "season", "week"] + STAT_COLS
    df = df[keep].copy()
    df[STAT_COLS] = df[STAT_COLS].fillna(0)
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def load_schedule(season, cache_dir):
    g = read_csv(SCHEDULE_URL, cache_dir, "games.csv")
    g = g[(g["season"] == season) & (g["game_type"] == "REG")].copy()
    times = pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"))
    g["kickoff_utc"] = times.dt.tz_localize(EASTERN).dt.tz_convert("UTC")
    return g


def load_rosters(season, cache_dir):
    r = read_csv(ROSTER_URL.format(year=season), cache_dir, f"roster_weekly_{season}.csv")
    return r[["week", "gsis_id", "team", "status"]].dropna(subset=["gsis_id"])


def roster_for_week(rosters, week):
    """Roster snapshot for `week`, or the latest one published before it."""
    avail = rosters.loc[rosters["week"] <= week, "week"]
    if avail.empty:
        return None
    return rosters[rosters["week"] == avail.max()].drop_duplicates("gsis_id", keep="last")


def load_injuries(season, cache_dir):
    try:
        return read_csv(INJURY_URL.format(year=season), cache_dir, f"injuries_{season}.csv")
    except Exception as e:  # injury file can lag early in a season
        print(f"  (injury report unavailable: {e})")
        return None


# ---------------------------------------------------------------- features

class DefenseTable:
    """Fantasy points each defense allowed to each position, by week."""

    def __init__(self, stats):
        t = (stats.groupby(["season", "week", "opponent_team", "position"])["fantasy_points_ppr"]
             .sum().reset_index().rename(columns={"opponent_team": "team"}))
        t = t.sort_values(["team", "position", "season", "week"])
        self.groups = {k: (g["week"].to_numpy(), g["fantasy_points_ppr"].to_numpy())
                       for k, g in t.groupby(["team", "position", "season"])}

    def before(self, team, position, season, week):
        """Average allowed over the defense's last WINDOW games before `week`."""
        weeks, pts = self.groups.get((team, position, season), (np.array([]), np.array([])))
        prior = pts[weeks < week][-WINDOW:]
        return float(prior.mean()) if len(prior) else np.nan


def add_rolling_and_next_game(stats):
    df = stats.copy()
    g = df.groupby(["player_id", "season"])
    for col in STAT_COLS:
        df[f"{col}_avg_last{WINDOW}"] = g[col].transform(
            lambda s: s.rolling(WINDOW, min_periods=1).mean())
    df["next_week"] = g["week"].shift(-1)
    df["next_opponent"] = g["opponent_team"].shift(-1)
    df["target"] = g["fantasy_points_ppr"].shift(-1)
    too_far = (df["next_week"] - df["week"]) > MAX_WEEK_GAP
    df.loc[too_far, ["next_week", "next_opponent", "target"]] = np.nan
    for p in POSITIONS:
        df[f"pos_{p}"] = (df["position"] == p).astype(int)
    return df


def attach_opponent(df, defense, week_col, opp_col):
    df = df.copy()
    df[OPP_COL] = [defense.before(o, p, s, w) if pd.notna(o) else np.nan
                   for o, p, s, w in zip(df[opp_col], df["position"], df["season"], df[week_col])]
    return df


def training_rows(feat, defense, season, before_week):
    """Rows whose TARGET game happened before (season, before_week)."""
    rows = feat.dropna(subset=["target"])
    rows = rows[(rows["season"] < season) |
                ((rows["season"] == season) & (rows["next_week"] < before_week))]
    rows = attach_opponent(rows, defense, "next_week", "next_opponent")
    return rows.dropna(subset=FEATURES + ["target"])


def latest_rows_before(feat, season, week):
    """Each player's most recent game in `season` before `week` (within the gap limit)."""
    cur = feat[(feat["season"] == season) & (feat["week"] < week)]
    last = cur.sort_values("week").groupby("player_id").tail(1)
    return last[(week - last["week"]) <= MAX_WEEK_GAP]


def matchup_label(value, pos_values):
    pct = (pos_values < value).mean()
    if pct >= 0.75:
        return "Great"
    if pct >= 0.5:
        return "Good"
    if pct >= 0.25:
        return "Tough"
    return "Very tough"


def project_week(feat, defense, schedule, season, week, rosters=None):
    """Train on data available before `week`, project every eligible player."""
    train = training_rows(feat, defense, season, week)
    model = LinearRegression().fit(train[FEATURES], train["target"])

    games = schedule[schedule["week"] == week]
    opp_of = {}
    for _, gm in games.iterrows():
        opp_of[gm["home_team"]] = gm["away_team"]
        opp_of[gm["away_team"]] = gm["home_team"]

    cand = latest_rows_before(feat, season, week).copy()
    roster = roster_for_week(rosters, week) if rosters is not None else None
    if roster is not None:
        # keep only players on an active roster; use their CURRENT team (handles trades)
        cand = cand.merge(roster[["gsis_id", "team", "status"]].rename(
            columns={"gsis_id": "player_id", "team": "roster_team"}), on="player_id", how="inner")
        cand = cand[cand["status"] == "ACT"]
        cand["team"] = cand["roster_team"]
    cand = cand[cand["team"].isin(opp_of)]
    cand["proj_opponent"] = cand["team"].map(opp_of)
    cand["proj_week"] = week
    cand = attach_opponent(cand, defense, "proj_week", "proj_opponent").dropna(subset=FEATURES)
    cand["projection"] = model.predict(cand[FEATURES]).clip(min=0)

    # matchup label: compare this defense's recent allowance with all defenses at that position
    labels = []
    for pos, grp in cand.groupby("position"):
        defs = pd.Series({t: defense.before(t, pos, season, week) for t in opp_of}).dropna()
        for idx, v in grp[OPP_COL].items():
            labels.append((idx, matchup_label(v, defs.to_numpy())))
    cand["matchup"] = pd.Series(dict(labels))
    return cand, model, len(train)


# ---------------------------------------------------------------- grading

def grade(proj_players, actual_week):
    """Compare saved projections with actual results for players who played."""
    p = pd.DataFrame(proj_players)
    a = actual_week[["player_id", "fantasy_points_ppr"]].rename(columns={"fantasy_points_ppr": "actual"})
    m = p.merge(a, on="player_id", how="inner")
    if len(m) < 20:
        return None
    return {
        "n_graded": int(len(m)),
        "n_projected": int(len(p)),
        "model_mae": round(mean_absolute_error(m["actual"], m["projection"]), 3),
        "baseline_mae": round(mean_absolute_error(m["actual"], m["recent_avg"]), 3),
        "model_r2": round(r2_score(m["actual"], m["projection"]), 3),
        "baseline_r2": round(r2_score(m["actual"], m["recent_avg"]), 3),
        "model_closer_pct": round(float(((m["projection"] - m["actual"]).abs()
                                         < (m["recent_avg"] - m["actual"]).abs()).mean() * 100), 1),
    }


def players_payload(cand):
    out = cand.sort_values("projection", ascending=False)
    return [{
        "player_id": r.player_id, "name": r.player_display_name, "position": r.position,
        "team": r.team, "opponent": r.proj_opponent,
        "projection": round(float(r.projection), 2),
        "recent_avg": round(float(getattr(r, BASELINE_COL)), 2),
        "opp_allowed": round(float(getattr(r, OPP_COL)), 2),
        "matchup": r.matchup,
    } for r in out.itertuples()]


# ---------------------------------------------------------------- injuries

def write_injuries(season, week, injuries, rosters, now):
    """Official injury report for `week`, plus anyone moved to injured reserve."""
    out = {}
    if injuries is not None:
        inj = injuries[(injuries["season"] == season) & (injuries["week"] == week)
                       & (injuries["report_status"].isin(REPORT_STATUSES))]
        for r in inj.itertuples():
            out[r.gsis_id] = {"status": r.report_status,
                              "injury": r.report_primary_injury if isinstance(r.report_primary_injury, str) else ""}
    roster = roster_for_week(rosters, week) if rosters is not None else None
    if roster is not None:
        for pid in roster.loc[roster["status"] == "RES", "gsis_id"]:
            out.setdefault(pid, {"status": "IR", "injury": ""})
    write_json(os.path.join(DATA_DIR, "injuries.json"),
               {"season": season, "week": int(week), "updated_at": now.isoformat(timespec="seconds"),
                "players": out})
    counts = pd.Series([v["status"] for v in out.values()]).value_counts().to_dict() if out else {}
    print(f"  injury report for week {week}: {counts}")


# ---------------------------------------------------------------- main

def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=datetime.now().year)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--injuries-only", action="store_true")
    args = ap.parse_args()
    season = args.season
    now = datetime.now(timezone.utc)

    stats = load_stats(season, args.cache)
    schedule = load_schedule(season, args.cache)
    rosters = load_rosters(season, args.cache)
    injuries = load_injuries(season, args.cache)

    # A week counts as complete only when every game has a final score AND its stats are
    # published. (A Thursday game's stats appear mid-week; that must not end the week early.)
    all_scored = schedule.groupby("week")["home_score"].apply(lambda x: x.notna().all())
    teams_sched = schedule.groupby("week").size() * 2
    teams_stats = stats[stats["season"] == season].groupby("week")["team"].nunique()
    completed = [int(w) for w in all_scored.index
                 if all_scored[w] and teams_stats.get(w, 0) >= teams_sched[w] - 2]
    remaining = [int(w) for w in all_scored.index if int(w) not in completed]
    upcoming = min(remaining) if remaining else int(all_scored.index.max()) + 1
    kickoff = schedule.groupby("week")["kickoff_utc"].min()
    print(f"Season {season}: weeks completed = {completed[-1] if completed else 0}, current week {upcoming}")

    write_injuries(season, upcoming, injuries, rosters, now)
    if args.injuries_only:
        return

    feat = add_rolling_and_next_game(stats)
    defense = DefenseTable(stats)

    scorecard = read_json(os.path.join(DATA_DIR, "scorecard.json"), {"season": season, "weeks": []})
    graded = {(w["week"]) for w in scorecard["weeks"]}

    # 1) grade live projections whose games are now complete
    for w in completed:
        if w in graded:
            continue
        path = os.path.join(PROJ_DIR, f"{season}_week{w:02d}.json")
        saved = read_json(path)
        if not saved:
            continue
        res = grade(saved["players"], stats[(stats["season"] == season) & (stats["week"] == w)])
        if res:
            res.update({"week": int(w), "type": saved.get("type", "live")})
            scorecard["weeks"].append(res)
            graded.add(w)
            print(f"  graded week {w} ({res['type']}): model MAE {res['model_mae']} vs baseline {res['baseline_mae']}")

    # 2) retroactive backtest for completed weeks that were never projected live
    for w in completed:
        if w in graded or w < BACKTEST_FIRST_WEEK:
            continue
        cand, _, _ = project_week(feat, defense, schedule, season, w, rosters)
        res = grade(players_payload(cand), stats[(stats["season"] == season) & (stats["week"] == w)])
        if res:
            res.update({"week": int(w), "type": "backtest"})
            scorecard["weeks"].append(res)
            print(f"  backtest week {w}: model MAE {res['model_mae']} vs baseline {res['baseline_mae']}")

    scorecard["weeks"].sort(key=lambda r: r["week"])
    for kind in ["live", "backtest"]:
        rows = [r for r in scorecard["weeks"] if r["type"] == kind]
        if rows:
            scorecard[f"{kind}_summary"] = {
                "weeks": len(rows),
                "model_wins": sum(r["model_mae"] < r["baseline_mae"] for r in rows),
                "model_mae": round(float(np.mean([r["model_mae"] for r in rows])), 3),
                "baseline_mae": round(float(np.mean([r["baseline_mae"] for r in rows])), 3),
                "model_closer_pct": round(float(np.mean([r["model_closer_pct"] for r in rows])), 1),
            }
    scorecard["updated_at"] = now.isoformat(timespec="seconds")
    write_json(os.path.join(DATA_DIR, "scorecard.json"), scorecard)

    # 3) project the upcoming week (never overwrite a week once its games have started)
    if upcoming not in kickoff.index:
        print("No upcoming regular-season week; season complete.")
        return
    first_kick = kickoff[upcoming]
    out_path = os.path.join(PROJ_DIR, f"{season}_week{upcoming:02d}.json")
    if os.path.exists(out_path) and now >= first_kick:
        print(f"Week {upcoming} has already kicked off; keeping the saved projections.")
        return
    cand, model, n_train = project_week(feat, defense, schedule, season, upcoming, rosters)
    payload = {
        "season": season, "week": upcoming,
        "generated_at": now.isoformat(timespec="seconds"),
        "first_kickoff": first_kick.isoformat(),
        "type": "live" if now < first_kick else "late",
        "n_training_rows": n_train,
        "players": players_payload(cand),
    }
    write_json(out_path, payload)
    write_json(os.path.join(PROJ_DIR, "latest.json"), payload)
    print(f"  saved {len(payload['players'])} projections for week {upcoming} "
          f"({payload['type']}; first kickoff {first_kick:%a %b %d %H:%M} UTC)")


if __name__ == "__main__":
    sys.exit(main())
