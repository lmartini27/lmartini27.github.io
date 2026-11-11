"""
update.py -- weekly pipeline for the live projections page.

Every run it:
  1. Downloads the latest nflverse weekly player stats and schedule.
  2. GRADES any past projections whose games have now been played
     (model vs. the naive "recent average" baseline), and scores the model
     against several simple pre-game baselines for the public leaderboard.
  3. Fills in a retroactive BACKTEST for earlier weeks of the season
     that were never projected live (clearly labeled as such).
  4. Trains the model on everything available and PROJECTS the next
     week's games, saving a timestamped file.

Model versions:
  v1  linear regression on the last three games, usage and the next opponent.
  v2  (published from week 10 of 2026) adds season-to-date averages and last
      season's average, and predicts each player's MIDDLE outcome (median)
      rather than his average outcome. In a walk-forward test on 2022-2025 it
      cut the average miss from 4.70 to 4.42 points. Two more median-style
      models give an 80% range (10th to 90th percentile) for every player.

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
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
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
RETURN_MAX_GAP = 10        # players back within this many weeks are projected, flagged as returning
BACKTEST_FIRST_WEEK = 4    # earliest week with 3 games of in-season history
ROLL_COLS = [f"{c}_avg_last{WINDOW}" for c in STAT_COLS]
OPP_COL = "next_opp_def_allowed"
FEATURES = ROLL_COLS + [f"pos_{p}" for p in POSITIONS] + [OPP_COL]          # model v1
SEASON_COLS = ["fantasy_points_ppr", "targets", "carries"]
SEAS_FEATURES = [f"{c}_season_avg" for c in SEASON_COLS] + ["games_played"]
PRIOR_FEATURES = ["prev_season_avg", "has_prev"]
AV_FEATURES = ["vac_same", "ret_same", "vac_same_pts", "ret_same_pts", "vac_tgt_rec_pts",
               "qb_out_rec", "qb_out_rb", "qb_out_rec_pts", "qb_out_rb_pts", "qb_ret_pts"]
RETURN_FEATURES = ["returning", "returning_pts"]
FEATURES_V2 = FEATURES + SEAS_FEATURES + PRIOR_FEATURES + AV_FEATURES + RETURN_FEATURES   # model v2
V2_START = {2026: 10}       # first week each season published with model v2
RANGE_QUANTILES = (0.10, 0.90)
# groups of v2 inputs used to explain each projection
WHY_GROUPS = [
    ("Recent points", ["fantasy_points_ppr_avg_last3"]),
    ("Recent usage", [c for c in ROLL_COLS if c != "fantasy_points_ppr_avg_last3"]),
    ("Season so far", SEAS_FEATURES),
    ("Last season", PRIOR_FEATURES),
    ("Teammates and QB", AV_FEATURES),
    ("First game back", RETURN_FEATURES),
    ("Opponent", [OPP_COL]),
    ("Position", [f"pos_{p}" for p in POSITIONS]),
]
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
            "season", "week", "attempts"] + STAT_COLS
    df = df[keep].copy()
    df[STAT_COLS + ["attempts"]] = df[STAT_COLS + ["attempts"]].fillna(0)
    return df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def load_schedule(season, cache_dir):
    g = read_csv(SCHEDULE_URL, cache_dir, "games.csv")
    g = g[(g["season"] == season) & (g["game_type"] == "REG")].copy()
    times = pd.to_datetime(g["gameday"] + " " + g["gametime"].fillna("13:00"))
    g["kickoff_utc"] = times.dt.tz_localize(EASTERN).dt.tz_convert("UTC")
    return g


def load_rosters(season, cache_dir):
    r = read_csv(ROSTER_URL.format(year=season), cache_dir, f"roster_weekly_{season}.csv")
    r = r[["week", "gsis_id", "team", "status"]].dropna(subset=["gsis_id"]).copy()
    r["season"] = season
    return r


def load_history(season, cache_dir, rosters_now, injuries_now):
    """Injury reports and weekly rosters for every season the model trains on."""
    inj, ros = [], [rosters_now]
    for year in range(FIRST_SEASON, season):
        try:
            inj.append(read_csv(INJURY_URL.format(year=year), cache_dir, f"injuries_{year}.csv"))
        except Exception as e:
            print(f"  (injury report {year} unavailable: {e})")
        r = read_csv(ROSTER_URL.format(year=year), cache_dir, f"roster_weekly_{year}.csv")
        r = r[["week", "gsis_id", "team", "status"]].dropna(subset=["gsis_id"]).copy()
        r["season"] = year
        ros.append(r)
    if injuries_now is not None:
        inj.append(injuries_now)
    inj = pd.concat(inj, ignore_index=True) if inj else None
    if inj is not None and "game_type" in inj:
        inj = inj[inj["game_type"] == "REG"]
    return inj, pd.concat(ros, ignore_index=True)


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
    too_far = (df["next_week"] - df["week"]) > RETURN_MAX_GAP
    df.loc[too_far, ["next_week", "next_opponent", "target"]] = np.nan
    for p in POSITIONS:
        df[f"pos_{p}"] = (df["position"] == p).astype(int)
    # v2 inputs: season-to-date averages (through this game) and last season's average
    for col in SEASON_COLS:
        df[f"{col}_season_avg"] = g[col].transform(lambda s: s.expanding().mean())
    df["games_played"] = g.cumcount() + 1
    prev = (stats.groupby(["player_id", "season"])["fantasy_points_ppr"].mean()
            .rename("prev_season_avg").reset_index())
    prev["season"] += 1
    df = df.merge(prev, on=["player_id", "season"], how="left")
    df["has_prev"] = df["prev_season_avg"].notna().astype(int)
    df["prev_season_avg"] = df["prev_season_avg"].fillna(0)
    return add_return_terms(df, "next_week")


def add_return_terms(df, week_col):
    """Flag a player's first game back after missing time (more than one bye's worth of weeks)."""
    gap = df[week_col] - df["week"]
    df["returning"] = (gap > MAX_WEEK_GAP).astype(int)
    df["returning_pts"] = df["returning"] * df["fantasy_points_ppr_avg_last3"]
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
    return rows.dropna(subset=FEATURES_V2 + ["target"])


def latest_rows_before(feat, season, week):
    """Each player's most recent game in `season` before `week` (within the gap limit)."""
    cur = feat[(feat["season"] == season) & (feat["week"] < week)]
    last = cur.sort_values("week").groupby("player_id").tail(1)
    return last[(week - last["week"]) <= RETURN_MAX_GAP]


# ---------------------------------------------------------------- teammates and quarterbacks
#
# Uses only what is known before kickoff: the injury report (Out / Doubtful) and the
# weekly roster (anyone not on the active roster). "Game-day inactive" is decided 90
# minutes before kickoff, so it is NOT treated as known.

ACTIVE_STATUSES = {"ACT", "INA"}
OUT_STATUSES = {"Out", "Doubtful"}
AV_RAW = ["vac_same", "ret_same", "vac_tgt", "qb_out", "qb_ret"]
AV_CACHE = os.path.join(DATA_DIR, "cache", "availability.csv")


def group_of(pos):
    return "QB" if pos == "QB" else ("RB" if pos == "RB" else "REC")


class Availability:
    def __init__(self, stats, injuries, rosters):
        d = stats[["player_id", "player_display_name", "position", "team", "season", "week",
                   "targets", "carries", "attempts"]].copy()
        d["use"] = d["targets"] + d["carries"]
        tot = d.groupby(["season", "week", "team"])[["use", "targets"]].transform("sum")
        d["use_share"] = (d["use"] / tot["use"].replace(0, np.nan)).fillna(0)
        d["tgt_share"] = (d["targets"] / tot["targets"].replace(0, np.nan)).fillna(0)
        d["grp"] = d["position"].map(group_of)
        self.names = d.drop_duplicates("player_id", keep="last").set_index("player_id")[["player_display_name", "position"]]
        self.by_tw = {k: g for k, g in d.groupby(["season", "team", "week"])}
        self.team_weeks = d.groupby(["season", "team"])["week"].apply(lambda x: sorted(set(x))).to_dict()
        self.pgames = d.groupby(["player_id", "season"])["week"].apply(lambda x: sorted(set(x))).to_dict()
        self.on_team, self.out = {}, {}
        for (s, w, t), g in rosters.groupby(["season", "week", "team"]):
            self.on_team[(s, w, t)] = set(g.loc[g["status"].isin(ACTIVE_STATUSES), "gsis_id"])
        if injuries is not None:
            inj = injuries[injuries["report_status"].isin(OUT_STATUSES)]
            for (s, w, t), g in inj.groupby(["season", "week", "team"]):
                self.out[(s, w, t)] = set(g["gsis_id"])

    def use_roster_snapshot(self, season, week, snapshot):
        """For an upcoming week whose roster isn't published yet, use the latest snapshot."""
        if snapshot is None:
            return
        for t, g in snapshot.groupby("team"):
            self.on_team.setdefault((season, week, t), set(g.loc[g["status"].isin(ACTIVE_STATUSES), "gsis_id"]))

    def available(self, pid, s, w, t):
        act = self.on_team.get((s, w, t))
        out = self.out.get((s, w, t), set())
        return (pid not in out) if act is None else (pid in act and pid not in out)

    def name(self, pid):
        return self.names.loc[pid, "player_display_name"] if pid in self.names.index else pid

    def features(self, rows, week_col, with_context=False):
        """Teammate and QB situation for the game in `week_col`, based on the player's last 3 games."""
        feats, contexts = [], []
        for r in rows.itertuples():
            s, t, w, pid = r.season, r.team, r.week, r.player_id
            T = getattr(r, week_col)
            grp = group_of(r.position)
            f = dict(vac_same=0.0, ret_same=0.0, vac_tgt=0.0, qb_out=0, qb_ret=0)
            ctx = []
            mine = [x for x in self.pgames.get((pid, s), []) if x <= w][-WINDOW:]
            win = [self.by_tw[(s, t, x)] for x in mine if (s, t, x) in self.by_tw]
            if pd.isna(T) or not win:
                feats.append(f); contexts.append(ctx); continue
            T = int(T)
            wdf = pd.concat(win)
            played = set(wdf["player_id"])
            agg = wdf.groupby(["player_id", "grp"])[["use_share", "tgt_share"]].sum() / len(win)
            for (q, g), v in agg.iterrows():
                if q == pid or g == "QB" or self.available(q, s, T, t):
                    continue
                if g == grp:
                    f["vac_same"] += v["use_share"]
                f["vac_tgt"] += v["tgt_share"]
                # list only news the model uses for this player: same-position teammates,
                # plus any big target-earner for wide receivers and tight ends
                relevant = (g == grp and v["use_share"] >= 0.08) or (grp == "REC" and v["tgt_share"] >= 0.08)
                if with_context and relevant:
                    ctx.append(f"{self.name(q)} ({self.names.loc[q, 'position']}) out")
            season_rows = [self.by_tw[(s, t, x)] for x in self.team_weeks.get((s, t), []) if x <= w and (s, t, x) in self.by_tw]
            sdf = pd.concat(season_rows)
            if grp != "QB":
                sg = sdf[(sdf["grp"] == grp) & (sdf["player_id"] != pid) & (~sdf["player_id"].isin(played))]
                for q, sh in sg.groupby("player_id")["use_share"].mean().items():
                    if self.available(q, s, T, t):
                        f["ret_same"] += sh
                        if with_context and sh >= 0.08:
                            ctx.append(f"{self.name(q)} ({self.names.loc[q, 'position']}) back")
                qbw = wdf[wdf["grp"] == "QB"].groupby("player_id")["attempts"].sum()
                qbs = sdf[sdf["grp"] == "QB"].groupby("player_id")["attempts"].sum()
                if len(qbw) and qbw.max() > 0:
                    prim = qbw.idxmax()
                    if not self.available(prim, s, T, t):
                        f["qb_out"] = 1
                        if with_context:
                            ctx.append(f"QB {self.name(prim)} out")
                if len(qbs) and qbs.max() > 0:
                    sprim = qbs.idxmax()
                    if sprim not in played and self.available(sprim, s, T, t):
                        f["qb_ret"] = 1
                        if with_context:
                            ctx.append(f"QB {self.name(sprim)} back")
            feats.append(f); contexts.append(ctx)
        out = pd.DataFrame(feats, index=rows.index)
        if with_context:
            out["context"] = contexts
        return out


def team_context(avail, season, week, team, members):
    """What the What-if simulator needs about one team: each skill player's recent and season
    share of the team's targets and carries (last 3 team games before `week`), the recent and
    season-long starting quarterback, and who is expected to play right now."""
    weeks = [x for x in avail.team_weeks.get((season, team), []) if x < week]
    win_weeks = weeks[-WINDOW:]
    win = [avail.by_tw[(season, team, x)] for x in win_weeks if (season, team, x) in avail.by_tw]
    sea = [avail.by_tw[(season, team, x)] for x in weeks if (season, team, x) in avail.by_tw]
    if not win:
        return None
    wdf, sdf = pd.concat(win), pd.concat(sea)
    n = len(win)
    wsum = wdf.groupby("player_id")[["use_share", "tgt_share", "attempts"]].sum()
    sgrp = sdf.groupby("player_id")
    smean, sgames, satt = sgrp["use_share"].mean(), sgrp["week"].nunique(), sgrp["attempts"].sum()
    qbw = wdf[wdf["grp"] == "QB"].groupby("player_id")["attempts"].sum()
    qbs = sdf[sdf["grp"] == "QB"].groupby("player_id")["attempts"].sum()
    players = []
    for pid in sorted(set(sdf["player_id"]) & set(members)):
        pos = avail.names.loc[pid, "position"] if pid in avail.names.index else ""
        players.append({
            "id": pid, "name": avail.name(pid), "pos": pos, "grp": group_of(pos),
            "win_use": round(float(wsum["use_share"].get(pid, 0)) / n, 4),
            "win_tgt": round(float(wsum["tgt_share"].get(pid, 0)) / n, 4),
            "in_window": bool(pid in wsum.index),
            "season_use": round(float(smean.get(pid, 0)), 4), "season_games": int(sgames.get(pid, 0)),
            "available": bool(avail.available(pid, season, week, team)),
        })
    return {"qb_window": qbw.idxmax() if len(qbw) and qbw.max() > 0 else None,
            "qb_season": qbs.idxmax() if len(qbs) and qbs.max() > 0 else None,
            "players": players}


def simulator_payload(cand, models, avail, season, week, rosters):
    """Model coefficients and team context so the site can recompute projections in the
    browser when a visitor switches teammates in or out (the What-if page)."""
    if avail is None or not {"v2", "q10", "q90"} <= set(models):
        return None
    roster = roster_for_week(rosters, week) if rosters is not None else None
    teams = {}
    for t in sorted(cand["team"].unique()):
        members = set(roster.loc[roster["team"] == t, "gsis_id"]) if roster is not None else set(cand.loc[cand["team"] == t, "player_id"])
        ctx = team_context(avail, season, week, t, members)
        if ctx:
            teams[t] = ctx
    return {"features": FEATURES_V2, "av_raw": AV_RAW,
            "intercept": {k: float(models[k].intercept_) for k in ["v2", "q10", "q90"]},
            "coef": {k: [round(float(c), 6) for c in models[k].coef_] for k in ["v2", "q10", "q90"]},
            "teams": teams}


def add_av_terms(df):
    """The model's availability inputs: main effects plus interactions with recent scoring."""
    pts = df["fantasy_points_ppr_avg_last3"]
    rec = df["position"].isin(["WR", "TE"]).astype(int)
    rb = (df["position"] == "RB").astype(int)
    df["vac_same_pts"] = df["vac_same"] * pts
    df["ret_same_pts"] = df["ret_same"] * pts
    df["vac_tgt_rec_pts"] = df["vac_tgt"] * pts * rec
    df["qb_out_rec"] = df["qb_out"] * rec
    df["qb_out_rb"] = df["qb_out"] * rb
    df["qb_out_rec_pts"] = df["qb_out"] * rec * pts
    df["qb_out_rb_pts"] = df["qb_out"] * rb * pts
    df["qb_ret_pts"] = df["qb_ret"] * pts
    return df


def attach_training_availability(feat, avail):
    """Availability features for every training row (target game already played), cached on disk
    because a past game's features never change."""
    keys = ["player_id", "season", "week", "next_week"]
    cache = pd.read_csv(AV_CACHE) if os.path.exists(AV_CACHE) else pd.DataFrame(columns=keys + AV_RAW)
    need = feat.dropna(subset=["target"])
    merged = need[keys].merge(cache, on=keys, how="left", indicator=True)
    missing = need[(merged["_merge"] == "left_only").to_numpy()]
    if len(missing):
        new = pd.concat([missing[keys], avail.features(missing, "next_week")], axis=1)
        cache = pd.concat([cache, new], ignore_index=True).drop_duplicates(keys, keep="last")
        os.makedirs(os.path.dirname(AV_CACHE), exist_ok=True)
        cache.to_csv(AV_CACHE, index=False, float_format="%.4f")
        print(f"  availability features: {len(missing)} new rows cached")
    feat = feat.merge(cache, on=keys, how="left")
    feat[AV_RAW] = feat[AV_RAW].fillna(0)
    return add_av_terms(feat)


def matchup_label(value, pos_values):
    pct = (pos_values < value).mean()
    if pct >= 0.75:
        return "Great"
    if pct >= 0.5:
        return "Good"
    if pct >= 0.25:
        return "Tough"
    return "Very tough"


def published_version(season, week):
    return "v2" if week >= V2_START.get(season, 99) else "v1"


def fit_models(train, need_v2=True, ranges=False):
    y = train["target"]
    regular = train["returning"] == 0          # v1 never saw players returning from absences
    models = {"v1": LinearRegression().fit(train.loc[regular, FEATURES], y[regular])}
    if need_v2:
        models["v2"] = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(train[FEATURES_V2], y)
    if ranges:
        for q in RANGE_QUANTILES:
            models[f"q{int(q * 100)}"] = QuantileRegressor(quantile=q, alpha=0, solver="highs").fit(train[FEATURES_V2], y)
    return models


def explain(model, train, rows):
    """Split each v2 projection into a starting point plus grouped contributions.
    contribution = coefficient x (player's value - reference value). The reference is the
    training-data average, except for teammate news and returns, where it is "no news" (zero),
    so those bars only appear when something actually changed."""
    coef = pd.Series(model.coef_, index=FEATURES_V2)
    means = train[FEATURES_V2].mean()
    means[AV_FEATURES + RETURN_FEATURES] = 0.0
    base = float(model.intercept_ + (coef * means).sum())
    out = []
    for _, r in rows[FEATURES_V2].iterrows():
        c = coef * (r - means)
        out.append([[label, round(float(c[cols].sum()), 2)] for label, cols in WHY_GROUPS])
    return round(base, 2), out


def project_week(feat, defense, schedule, season, week, rosters=None, version=None, ranges=False, avail=None):
    """Train on data available before `week`, project every eligible player.
    `version` picks the published projection (default: whichever was live that week).
    v1 is always computed too, so the two versions can be compared."""
    version = version or published_version(season, week)
    train = training_rows(feat, defense, season, week)
    models = fit_models(train, need_v2=(version == "v2"), ranges=ranges)

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
    cand = add_return_terms(cand, "proj_week")
    cand = attach_opponent(cand, defense, "proj_week", "proj_opponent")
    # teammates and quarterback for the game being projected, as known right now
    if avail is not None:
        avail.use_roster_snapshot(season, week, roster)
        av = avail.features(cand, "proj_week", with_context=True)
        cand = pd.concat([cand.drop(columns=[c for c in AV_RAW if c in cand.columns]), av], axis=1)
    else:
        for c in AV_RAW:
            cand[c] = 0.0
        cand["context"] = [[] for _ in range(len(cand))]
    cand = add_av_terms(cand).dropna(subset=FEATURES_V2)
    cand["proj_v1"] = models["v1"].predict(cand[FEATURES]).clip(min=0)
    if "v2" in models:
        cand["proj_v2"] = models["v2"].predict(cand[FEATURES_V2]).clip(min=0)
    cand["projection"] = cand["proj_v2"] if version == "v2" else cand["proj_v1"]
    cand["version"] = version
    if ranges:
        lo = models["q10"].predict(cand[FEATURES_V2])
        hi = models["q90"].predict(cand[FEATURES_V2])
        cand["low"] = np.maximum(np.minimum(lo, cand["projection"].to_numpy()), 0)
        cand["high"] = np.maximum(hi, cand["projection"].to_numpy())
    if version == "v2":
        base, why = explain(models["v2"], train, cand)
        cand["why"] = why
        cand["why_base"] = base

    # matchup label: compare this defense's recent allowance with all defenses at that position
    labels = []
    for pos, grp in cand.groupby("position"):
        defs = pd.Series({t: defense.before(t, pos, season, week) for t in opp_of}).dropna()
        for idx, v in grp[OPP_COL].items():
            labels.append((idx, matchup_label(v, defs.to_numpy())))
    cand["matchup"] = pd.Series(dict(labels))
    # opponent rank at the player's position among defenses playing this week (1 = allows the most)
    ranks = {}
    for pos in cand["position"].unique():
        defs = pd.Series({t: defense.before(t, pos, season, week) for t in opp_of}).dropna()
        r = defs.rank(ascending=False, method="min")
        ranks.update({(t, pos): (int(r[t]), int(len(defs))) for t in r.index})
    cand["opp_rank"] = [ranks.get((o, p), (None, None)) for o, p in zip(cand["proj_opponent"], cand["position"])]
    return cand, models, len(train)


# ---------------------------------------------------------------- grading

def grade(proj_players, actual_week):
    """Compare saved projections with actual results for players who played."""
    p = pd.DataFrame(proj_players)
    a = actual_week[["player_id", "fantasy_points_ppr"]].rename(columns={"fantasy_points_ppr": "actual"})
    m = p.merge(a, on="player_id", how="inner")
    if len(m) < 20:
        return None
    return {
        "report": week_report(m),
        "n_graded": int(len(m)),
        "n_projected": int(len(p)),
        "model_mae": round(mean_absolute_error(m["actual"], m["projection"]), 3),
        "baseline_mae": round(mean_absolute_error(m["actual"], m["recent_avg"]), 3),
        "model_r2": round(r2_score(m["actual"], m["projection"]), 3),
        "baseline_r2": round(r2_score(m["actual"], m["recent_avg"]), 3),
        "model_closer_pct": round(float(((m["projection"] - m["actual"]).abs()
                                         < (m["recent_avg"] - m["actual"]).abs()).mean() * 100), 1),
    }


def week_report(m):
    """Details for the weekly report card: accuracy by position, best calls, biggest misses."""
    m = m.assign(err=m["actual"] - m["projection"])

    def entry(r):
        return {"name": r["name"], "position": r["position"], "team": r["team"],
                "projection": round(float(r["projection"]), 1), "actual": round(float(r["actual"]), 1)}

    by_pos = {}
    for pos, g in m.groupby("position"):
        by_pos[pos] = {"n": int(len(g)),
                       "model_mae": round(float(g["err"].abs().mean()), 2),
                       "baseline_mae": round(float((g["actual"] - g["recent_avg"]).abs().mean()), 2)}
    starters = m[m["projection"] >= 12]          # fantasy-relevant players only
    return {
        "by_position": by_pos,
        "best_calls": [entry(r) for _, r in starters.assign(a=starters["err"].abs()).nsmallest(3, "a").iterrows()],
        "outscored": [entry(r) for _, r in m.nlargest(3, "err").iterrows()],
        "letdowns": [entry(r) for _, r in starters.nsmallest(3, "err").iterrows()],
    }


# ---------------------------------------------------------------- leaderboard

# Every method uses only information available before the week's games.
METHODS = [
    {"key": "model", "label": "My model",
     "about": "The projection published before kickoff that week (model v1 through week 9, v2 from week 10)."},
    {"key": "last3", "label": "Last 3 games average",
     "about": "The player's average over his last three games. The site's main benchmark."},
    {"key": "season", "label": "Season average",
     "about": "The player's average over every game he has played this season."},
    {"key": "last_game", "label": "Last game",
     "about": "Whatever the player scored in his most recent game."},
    {"key": "pos_avg", "label": "Position average",
     "about": "Every player at a position gets the same number: the typical season average at that position."},
]
BIG_MISS = 10          # points
RELEVANT_MIN = 8       # season average needed to count in the "pick the better player" test


def method_predictions(proj_players, stats, season, week):
    """Each method's pre-game prediction for every projected player."""
    p = pd.DataFrame(proj_players)
    hist = stats[(stats["season"] == season) & (stats["week"] < week)].sort_values("week")
    g = hist.groupby("player_id")["fantasy_points_ppr"]
    p = p.merge(g.mean().rename("season").reset_index(), on="player_id", how="left")
    p = p.merge(g.last().rename("last_game").reset_index(), on="player_id", how="left")
    p["model"] = p["projection"]
    p["last3"] = p["recent_avg"]
    p["season"] = p["season"].fillna(p["recent_avg"])
    p["last_game"] = p["last_game"].fillna(p["recent_avg"])
    p["pos_avg"] = p.groupby("position")["season"].transform("mean")
    return p


def pick_counts(df, col):
    """Pairs of same-position relevant players, and how many the method ordered correctly."""
    pairs = correct = 0.0
    for _, g in df[df["season"] >= RELEVANT_MIN].groupby("position"):
        a, pr = g["actual"].to_numpy(), g[col].to_numpy()
        da, dp = np.sign(a[:, None] - a[None, :]), np.sign(pr[:, None] - pr[None, :])
        upper = np.triu(np.ones_like(da, dtype=bool), 1) & (da != 0)
        pairs += upper.sum()
        correct += ((dp == da) & upper).sum() + 0.5 * ((dp == 0) & upper).sum()
    return int(pairs), float(correct)


def leaderboard_week(proj_players, actual_week, stats, season, week, relevant_only=False):
    p = method_predictions(proj_players, stats, season, week)
    a = actual_week[["player_id", "fantasy_points_ppr"]].rename(columns={"fantasy_points_ppr": "actual"})
    m = p.merge(a, on="player_id", how="inner")
    if relevant_only:
        m = m[m["season"] >= RELEVANT_MIN]
    if len(m) < 20:
        return None
    out = {}
    for meth in METHODS:
        k = meth["key"]
        err = (m["actual"] - m[k]).abs()
        pairs, correct = pick_counts(m, k)
        out[k] = {
            "n": int(len(m)), "abs_sum": round(float(err.sum()), 2),
            "mae": round(float(err.mean()), 3),
            "r2": round(float(r2_score(m["actual"], m[k])), 3),
            "big": int((err >= BIG_MISS).sum()),
            "pairs": pairs, "correct": round(correct, 1),
            "pos": {pos: [int(len(e)), round(float(e.sum()), 2)]
                    for pos, e in err.groupby(m["position"])},
        }
    return out


def summarize_leaderboard(rows, key="leaderboard"):
    """Combine weekly results: player-weighted misses, pooled pairs, weeks each method had the lowest miss."""
    rows = [{"leaderboard": r[key]} for r in rows if r.get(key)]
    if not rows:
        return None
    best = {}
    for r in rows:
        lb = r["leaderboard"]
        k = min(lb, key=lambda key: lb[key]["mae"])
        best[k] = best.get(k, 0) + 1
    out = []
    for meth in METHODS:
        k = meth["key"]
        ws = [r["leaderboard"][k] for r in rows]
        n = sum(w["n"] for w in ws)
        pairs = sum(w["pairs"] for w in ws)
        pos = {}
        for w in ws:
            for ps, (pn, pa) in w["pos"].items():
                pn0, pa0 = pos.get(ps, (0, 0.0))
                pos[ps] = (pn0 + pn, pa0 + pa)
        out.append({
            "key": k, "label": meth["label"], "about": meth["about"],
            "mae": round(sum(w["abs_sum"] for w in ws) / n, 3),
            "r2": round(float(np.mean([w["r2"] for w in ws])), 3),
            "pick_pct": round(100 * sum(w["correct"] for w in ws) / pairs, 1) if pairs else None,
            "big_pct": round(100 * sum(w["big"] for w in ws) / n, 1),
            "weeks_best": best.get(k, 0),
            "by_position": {ps: round(a / c, 2) for ps, (c, a) in pos.items()},
        })
    out.sort(key=lambda x: x["mae"])
    return {"weeks": len(rows), "players_graded": sum(r["leaderboard"]["model"]["n"] for r in rows),
            "methods": out}


def versions_week(v1_players, v2_players, actual_week, v2_live):
    """v1 vs v2 on the same graded players. v2 before its launch week is a backtest."""
    a = actual_week[["player_id", "fantasy_points_ppr"]].rename(columns={"fantasy_points_ppr": "actual"})
    m = (pd.DataFrame(v1_players)[["player_id", "projection"]].rename(columns={"projection": "v1"})
         .merge(pd.DataFrame(v2_players)[["player_id", "projection"]].rename(columns={"projection": "v2"}), on="player_id")
         .merge(a, on="player_id"))
    if len(m) < 20:
        return None
    return {"n": int(len(m)), "v2_live": bool(v2_live),
            "v1_abs": round(float((m["actual"] - m["v1"]).abs().sum()), 2),
            "v2_abs": round(float((m["actual"] - m["v2"]).abs().sum()), 2)}


def updates_week(locked_players, updated_players, actual_week):
    """Wednesday's locked projections vs the injury-updated ones, on the same graded players."""
    a = actual_week[["player_id", "fantasy_points_ppr"]].rename(columns={"fantasy_points_ppr": "actual"})
    m = (pd.DataFrame(locked_players)[["player_id", "projection"]].rename(columns={"projection": "locked"})
         .merge(pd.DataFrame(updated_players)[["player_id", "projection"]].rename(columns={"projection": "updated"}), on="player_id")
         .merge(a, on="player_id"))
    if len(m) < 20:
        return None
    ch = m[(m["updated"] - m["locked"]).abs() >= 0.5]
    return {"n": int(len(m)), "locked_abs": round(float((m["actual"] - m["locked"]).abs().sum()), 2),
            "updated_abs": round(float((m["actual"] - m["updated"]).abs().sum()), 2),
            "changed_n": int(len(ch)),
            "changed_locked_abs": round(float((ch["actual"] - ch["locked"]).abs().sum()), 2),
            "changed_updated_abs": round(float((ch["actual"] - ch["updated"]).abs().sum()), 2)}


def summarize_updates(rows):
    rows = [r["updates"] for r in rows if r.get("updates")]
    if not rows:
        return None
    n, cn = sum(r["n"] for r in rows), sum(r["changed_n"] for r in rows)
    return {"weeks": len(rows), "n": n, "changed_n": cn,
            "locked_mae": round(sum(r["locked_abs"] for r in rows) / n, 3),
            "updated_mae": round(sum(r["updated_abs"] for r in rows) / n, 3),
            "changed_locked_mae": round(sum(r["changed_locked_abs"] for r in rows) / cn, 3) if cn else None,
            "changed_updated_mae": round(sum(r["changed_updated_abs"] for r in rows) / cn, 3) if cn else None}


def summarize_versions(rows):
    rows = [r["versions"] for r in rows if r.get("versions")]
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    return {"weeks": len(rows), "n": n,
            "v1_mae": round(sum(r["v1_abs"] for r in rows) / n, 3),
            "v2_mae": round(sum(r["v2_abs"] for r in rows) / n, 3),
            "v2_better_weeks": sum(r["v2_abs"] < r["v1_abs"] for r in rows),
            "v2_live_weeks": sum(r["v2_live"] for r in rows)}


def players_payload(cand, with_x=False):
    out = cand.sort_values("projection", ascending=False)
    cols = set(out.columns)
    rows = []
    for r in out.itertuples():
        d = {
            "player_id": r.player_id, "name": r.player_display_name, "position": r.position,
            "team": r.team, "opponent": r.proj_opponent,
            "projection": round(float(r.projection), 2),
            "recent_avg": round(float(getattr(r, BASELINE_COL)), 2),
            "season_avg": round(float(r.fantasy_points_ppr_season_avg), 2),
            "games": int(r.games_played),
            "last_season_avg": round(float(r.prev_season_avg), 2) if r.has_prev else None,
            "opp_allowed": round(float(getattr(r, OPP_COL)), 2),
            "matchup": r.matchup,
            "proj_v1": round(float(r.proj_v1), 2),
        }
        if "opp_rank" in cols and r.opp_rank and r.opp_rank[0] is not None:
            d["opp_rank"], d["opp_n"] = r.opp_rank
        if "low" in cols:
            d["low"], d["high"] = round(float(r.low), 1), round(float(r.high), 1)
        if "why" in cols:
            d["why"] = r.why
        if "context" in cols and r.context:
            d["context"] = list(r.context)
        if getattr(r, "returning", 0):
            d["returning_weeks"] = int(r.proj_week - r.week)
        if with_x:
            d["x"] = [round(float(getattr(r, f)), 4) for f in FEATURES_V2]
            d["av"] = [round(float(getattr(r, f)), 4) for f in AV_RAW]
        rows.append(d)
    return rows


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


# ---------------------------------------------------------------- results and the weekly game

RESULTS_DIR = os.path.join(DATA_DIR, "results")
GAME_DIR = os.path.join(DATA_DIR, "game")
H2H_QUOTA = {"QB": 2, "RB": 3, "WR": 3, "TE": 2}
GAME_SIZE = 10


def write_results(season, week, actual_week):
    """Actual PPR points for every player who played, for the lineup grader."""
    pts = {r.player_id: round(float(r.fantasy_points_ppr), 2) for r in actual_week.itertuples()}
    write_json(os.path.join(RESULTS_DIR, f"{season}_week{week:02d}.json"), {"season": season, "week": int(week), "points": pts})


def quantile_curve(p, n=240):
    """A player's outcomes as a piecewise-linear quantile curve through 0, p10, median, p90 and a
    stretched upper tail. The site's Start/Sit tool uses exactly the same curve."""
    lo, md, hi = p["low"], p["projection"], p["high"]
    u = (np.arange(n) + 0.5) / n
    return np.interp(u, [0, 0.1, 0.5, 0.9, 1.0], [0, lo, md, hi, hi + 1.6 * (hi - md)])


def win_prob(a, b):
    qa, qb = quantile_curve(a), quantile_curve(b)
    d = qa[:, None] - qb[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def make_game(season, week, players, stats, schedule, injuries):
    """This week's two games, fixed once posted: 10 close head-to-heads and 10 over/unders
    against each player's typical (median) score this season."""
    path = os.path.join(GAME_DIR, f"{season}_week{week:02d}.json")
    if os.path.exists(path):
        return
    kick = {}
    for _, g in schedule[schedule["week"] == week].iterrows():
        kick[g["home_team"]] = kick[g["away_team"]] = g["kickoff_utc"]
    hurt = set()
    if injuries is not None:
        inj = injuries[(injuries["season"] == season) & (injuries["week"] == week)]
        hurt = set(inj.loc[inj["report_status"].isin(["Out", "Doubtful"]), "gsis_id"])
    pool = [p for p in players if p.get("low") is not None and p["season_avg"] >= RELEVANT_MIN
            and not p.get("returning_weeks") and p["player_id"] not in hurt and p["team"] in kick]
    slim = lambda p: {k: p[k] for k in ["player_id", "name", "position", "team", "opponent", "projection", "low", "high"]}
    # head-to-heads: close calls (model favorite at 50-66%) between different teams, same position
    rng = np.random.default_rng(season * 100 + week)
    used, h2h = set(), []
    for pos, quota in H2H_QUOTA.items():
        ps = sorted([p for p in pool if p["position"] == pos], key=lambda p: -p["projection"])[:24]
        pairs = []
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                a, b = ps[i], ps[j]
                if a["team"] == b["team"]:
                    continue
                pa = win_prob(a, b)
                fav = max(pa, 1 - pa)
                if 0.5 <= fav <= 0.66:
                    pairs.append((abs(fav - 0.57) + rng.random() * 0.02, a, b, pa))
        for _, a, b, pa in sorted(pairs, key=lambda x: x[0]):
            if len([x for x in h2h if x["position"] == pos]) >= quota:
                break
            if a["player_id"] in used or b["player_id"] in used:
                continue
            used |= {a["player_id"], b["player_id"]}
            if rng.random() < 0.5:          # shuffle sides so the favorite isn't always listed first
                a, b, pa = b, a, 1 - pa
            h2h.append({"position": pos, "a": slim(a), "b": slim(b), "model_pick": "a" if pa >= 0.5 else "b",
                        "model_prob": round(max(pa, 1 - pa), 3),
                        "kickoff": min(kick[a["team"]], kick[b["team"]]).isoformat()})
    # over/unders: line = the player's median score this season, where the model disagrees by 1.5+
    cur = stats[(stats["season"] == season) & (stats["week"] < week)]
    med = cur.groupby("player_id")["fantasy_points_ppr"].median()
    games_n = cur.groupby("player_id")["week"].nunique()
    cands = []
    for p in pool:
        if p["player_id"] in used or games_n.get(p["player_id"], 0) < 4:
            continue
        line = float(np.floor(med[p["player_id"]]) + 0.5)
        gap = p["projection"] - line
        if abs(gap) >= 1.5:
            cands.append((p, line, "over" if gap > 0 else "under"))
    cands.sort(key=lambda x: -x[0]["projection"])
    overs = [c for c in cands if c[2] == "over"][:GAME_SIZE // 2]
    unders = [c for c in cands if c[2] == "under"][:GAME_SIZE - len(overs)]
    def p_over(p, line):
        return float((quantile_curve(p) > line).mean())
    ou = [{"player": slim(p), "line": line, "model_pick": pick,
           "model_prob": round(p_over(p, line) if pick == "over" else 1 - p_over(p, line), 3),
           "kickoff": kick[p["team"]].isoformat()}
          for p, line, pick in sorted(overs + unders, key=lambda x: -x[0]["projection"])]
    for i, it in enumerate(h2h):
        it["id"] = i
    for i, it in enumerate(ou):
        it["id"] = i
    write_json(path, {"season": season, "week": int(week), "h2h": h2h, "ou": ou, "graded": False})
    print(f"  game for week {week}: {len(h2h)} head-to-heads, {len(ou)} over/unders")


def fetch_picks(cfg, season, week):
    """All picks for a week from the game database (read-only public view)."""
    url, key = cfg.get("supabase_url"), cfg.get("supabase_anon_key")
    if not url or not key:
        return None
    out, start = [], 0
    while True:
        q = urllib.parse.urlencode({"season": f"eq.{season}", "week": f"eq.{week}",
                                    "select": "player_id,nickname,game,item,choice,created_at", "order": "created_at.asc"})
        req = urllib.request.Request(f"{url.rstrip('/')}/rest/v1/picks_public?{q}",
                                     headers={"apikey": key, "Authorization": f"Bearer {key}",
                                              "Range-Unit": "items", "Range": f"{start}-{start + 999}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            rows = json.load(resp)
        out += rows
        if len(rows) < 1000:
            return out
        start += 1000


def grade_game(season, week, actual_week):
    """Score everyone's picks (only picks the database stamped before that game's kickoff count)."""
    path = os.path.join(GAME_DIR, f"{season}_week{week:02d}.json")
    game = read_json(path)
    if not game or game.get("graded"):
        return
    pts = dict(zip(actual_week["player_id"], actual_week["fantasy_points_ppr"]))
    for it in game["h2h"]:
        a, b = pts.get(it["a"]["player_id"]), pts.get(it["b"]["player_id"])
        it["actual"] = {"a": a, "b": b}
        it["result"] = None if a is None or b is None or a == b else ("a" if a > b else "b")
    for it in game["ou"]:
        x = pts.get(it["player"]["player_id"])
        it["actual"] = x
        it["result"] = None if x is None else ("over" if x > it["line"] else "under")
    try:
        picks = fetch_picks(read_json(os.path.join(DATA_DIR, "site.json"), {}), season, week)
    except Exception as e:
        print(f"  game picks unavailable ({e}); will try again next run")
        return
    items = {("h2h", it["id"]): it for it in game["h2h"]} | {("ou", it["id"]): it for it in game["ou"]}
    latest = {}
    for pk in picks or []:
        it = items.get((pk["game"], pk["item"]))
        if it and pd.Timestamp(pk["created_at"]) < pd.Timestamp(it["kickoff"]):
            latest[(pk["player_id"], pk["game"], pk["item"])] = (pk["nickname"], pk["choice"])
    scores, crowd = {}, {}
    for (pid, g, i), (nick, choice) in latest.items():
        res = items[(g, i)]["result"]
        sc = scores.setdefault(pid, {"player_id": pid, "nickname": nick, "h2h": 0, "ou": 0, "picks": 0})
        sc["picks"] += 1
        if res is not None and choice == res:
            sc[g] += 1
        crowd.setdefault((g, i), {}).setdefault(choice, 0)
        crowd[(g, i)][choice] += 1
    def score_of(choices):
        return {g: sum(1 for it in game[g] if it["result"] is not None and choices(g, it) == it["result"]) for g in ["h2h", "ou"]}
    model = score_of(lambda g, it: it["model_pick"])
    crowd_pick = lambda g, it: max(crowd[(g, it["id"])], key=crowd[(g, it["id"])].get) if crowd.get((g, it["id"])) else None
    crowd_sc = score_of(crowd_pick)
    for g in ["h2h", "ou"]:
        for it in game[g]:
            it["crowd"] = crowd.get((g, it["id"]), {})
    game["graded"] = True
    game["model_score"], game["crowd_score"] = model, crowd_sc
    results = sorted(scores.values(), key=lambda r: -(r["h2h"] + r["ou"]))
    game["players"] = results
    write_json(path, game)
    lb = read_json(os.path.join(GAME_DIR, "leaderboard.json"), {"season": season, "weeks": {}, "standings": []})
    lb["weeks"][str(week)] = {"players": len(results), "model": model, "crowd": crowd_sc,
                              "beat_model": sum(1 for r in results if r["h2h"] + r["ou"] > model["h2h"] + model["ou"])}
    table = {r["player_id"]: r for r in lb.get("standings", [])}
    for r in results:
        t = table.setdefault(r["player_id"], {"player_id": r["player_id"], "nickname": r["nickname"], "total": 0, "weeks": 0, "beat_model": 0})
        t["nickname"] = r["nickname"]
        t["total"] += r["h2h"] + r["ou"]
        t["weeks"] += 1
        t["beat_model"] += int(r["h2h"] + r["ou"] > model["h2h"] + model["ou"])
    lb["model_total"] = sum(w["model"]["h2h"] + w["model"]["ou"] for w in lb["weeks"].values())
    lb["standings"] = sorted(table.values(), key=lambda t: (-t["total"], -t["beat_model"]))
    lb["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(os.path.join(GAME_DIR, "leaderboard.json"), lb)
    print(f"  graded week {week} game: {len(results)} players; model scored {model['h2h']}+{model['ou']}")


# ---------------------------------------------------------------- injury updates

def update_path(season, week):
    return os.path.join(PROJ_DIR, f"{season}_week{week:02d}_update.json")


def injury_update(season, week, feat, defense, schedule, rosters, avail, now):
    """Rerun this week's projections with the latest injury report, for games not yet started.
    The Wednesday projections stay locked and are still the official grade; this update is
    saved separately, with its own timestamp, and graded separately."""
    locked = read_json(os.path.join(PROJ_DIR, f"{season}_week{week:02d}.json"))
    if not locked or locked.get("version") != "v2":
        print("  no v2 projections to update this week")
        return
    games = schedule[schedule["week"] == week]
    kick = {}
    for _, g in games.iterrows():
        kick[g["home_team"]] = kick[g["away_team"]] = g["kickoff_utc"]
    prev = read_json(update_path(season, week)) or locked
    prev_by = {p["player_id"]: p for p in prev["players"]}
    locked_by = {p["player_id"]: p for p in locked["players"]}
    cand, models, _ = project_week(feat, defense, schedule, season, week, rosters, ranges=True, avail=avail)
    fresh = players_payload(cand, with_x=True)
    players, seen = [], set()
    for p in fresh:
        started = p["team"] in kick and now >= kick[p["team"]]
        if started:   # game already kicked off: keep what was published before kickoff
            if p["player_id"] in prev_by:
                players.append(prev_by[p["player_id"]])
        else:
            lk = locked_by.get(p["player_id"])
            p["locked"] = lk["projection"] if lk else None
            players.append(p)
        seen.add(p["player_id"])
    players += [p for pid, p in prev_by.items() if pid not in seen and p["team"] in kick and now >= kick[p["team"]]]
    players.sort(key=lambda p: -p["projection"])
    changed = sum(1 for p in players if p.get("locked") is not None and abs(p["projection"] - p["locked"]) >= 0.5)
    payload = {"season": season, "week": week, "generated_at": now.isoformat(timespec="seconds"),
               "locked_at": locked["generated_at"], "version": "v2", "changed": changed,
               "why_base": float(cand["why_base"].iloc[0]) if len(cand) else None, "players": players,
               "sim": simulator_payload(cand, models, avail, season, week, rosters)}
    write_json(update_path(season, week), payload)
    write_json(os.path.join(PROJ_DIR, "latest_update.json"), payload)
    print(f"  injury update for week {week}: {changed} projections moved by 0.5+ points")


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

    feat = add_rolling_and_next_game(stats)
    defense = DefenseTable(stats)
    hist_inj, hist_ros = load_history(season, args.cache, rosters, injuries)
    avail = Availability(stats, hist_inj, hist_ros)
    feat = attach_training_availability(feat, avail)

    if args.injuries_only:
        if upcoming in kickoff.index:
            injury_update(season, upcoming, feat, defense, schedule, rosters, avail, now)
        return

    scorecard = read_json(os.path.join(DATA_DIR, "scorecard.json"), {"season": season, "weeks": []})
    graded = {(w["week"]) for w in scorecard["weeks"]}

    # 0) actual scores for the lineup grader, and the weekly game, for every completed week
    for w in completed:
        actual = stats[(stats["season"] == season) & (stats["week"] == w)]
        if not os.path.exists(os.path.join(RESULTS_DIR, f"{season}_week{w:02d}.json")):
            write_results(season, w, actual)
        grade_game(season, w, actual)

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
        cand, _, _ = project_week(feat, defense, schedule, season, w, rosters, avail=avail)
        res = grade(players_payload(cand), stats[(stats["season"] == season) & (stats["week"] == w)])
        if res:
            res.update({"week": int(w), "type": "backtest"})
            scorecard["weeks"].append(res)
            print(f"  backtest week {w}: model MAE {res['model_mae']} vs baseline {res['baseline_mae']}")

    # 3) add report-card, leaderboard and model-version details to any graded week missing them
    #    (the original grades are never changed)
    needed = ["report", "leaderboard", "leaderboard_rel", "versions"]
    for row in scorecard["weeks"]:
        if "updates" not in row:
            upd = read_json(update_path(season, row["week"]))
            locked = read_json(os.path.join(PROJ_DIR, f"{season}_week{row['week']:02d}.json"))
            if upd and locked:
                ur = updates_week(locked["players"], upd["players"], stats[(stats["season"] == season) & (stats["week"] == row["week"])])
                if ur:
                    row["updates"] = ur
        if all(k in row for k in needed):
            continue
        w = row["week"]
        actual = stats[(stats["season"] == season) & (stats["week"] == w)]
        saved = read_json(os.path.join(PROJ_DIR, f"{season}_week{w:02d}.json"))
        retro = None   # v1 and v2 recomputed for this week, using only data from before it

        def get_retro():
            nonlocal retro
            if retro is None:
                retro = project_week(feat, defense, schedule, season, w, rosters, version="v2", avail=avail)[0]
            return retro

        if saved:
            proj = saved["players"]
        elif row["type"] == "backtest":
            c = get_retro()
            c = c.assign(projection=c["proj_v2"] if published_version(season, w) == "v2" else c["proj_v1"])
            proj = players_payload(c)
        else:
            continue
        if "report" not in row:
            res = grade(proj, actual)
            if res:
                row["report"] = res["report"]
        if "leaderboard" not in row:
            lb = leaderboard_week(proj, actual, stats, season, w)
            if lb:
                row["leaderboard"] = lb
        if "leaderboard_rel" not in row:
            lb = leaderboard_week(proj, actual, stats, season, w, relevant_only=True)
            if lb:
                row["leaderboard_rel"] = lb
        if "versions" not in row:
            pub_v2 = bool(saved) and saved.get("version") == "v2"
            if pub_v2:
                v1 = [{"player_id": p["player_id"], "projection": p["proj_v1"]} for p in proj]
                v2, live = proj, row["type"] == "live"
            else:
                c = get_retro()
                v1 = proj if saved else [{"player_id": i, "projection": x} for i, x in zip(c["player_id"], c["proj_v1"])]
                v2 = [{"player_id": i, "projection": x} for i, x in zip(c["player_id"], c["proj_v2"])]
                live = False
            vr = versions_week(v1, v2, actual, live)
            if vr:
                row["versions"] = vr
        print(f"  week {w}: details added")

    scorecard["weeks"].sort(key=lambda r: r["week"])
    live_rows = [r for r in scorecard["weeks"] if r["type"] == "live"]
    scorecard["leaderboard"] = {
        "live": summarize_leaderboard(live_rows),
        "all": summarize_leaderboard(scorecard["weeks"]),
        "live_rel": summarize_leaderboard(live_rows, "leaderboard_rel"),
        "all_rel": summarize_leaderboard(scorecard["weeks"], "leaderboard_rel"),
        "big_miss_points": BIG_MISS, "relevant_min": RELEVANT_MIN,
    }
    scorecard["versions"] = summarize_versions(scorecard["weeks"])
    scorecard["updates"] = summarize_updates(scorecard["weeks"])
    scorecard["v2_start_week"] = V2_START.get(season)
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

    # 4) project the upcoming week (never overwrite a week once its games have started)
    if upcoming not in kickoff.index:
        print("No upcoming regular-season week; season complete.")
        return
    first_kick = kickoff[upcoming]
    out_path = os.path.join(PROJ_DIR, f"{season}_week{upcoming:02d}.json")
    if os.path.exists(out_path) and now >= first_kick:
        print(f"Week {upcoming} has already kicked off; keeping the saved projections.")
        return
    cand, models, n_train = project_week(feat, defense, schedule, season, upcoming, rosters, ranges=True, avail=avail)
    version = published_version(season, upcoming)
    payload = {
        "season": season, "week": upcoming,
        "generated_at": now.isoformat(timespec="seconds"),
        "first_kickoff": first_kick.isoformat(),
        "type": "live" if now < first_kick else "late",
        "version": version,
        "n_training_rows": n_train,
        "range_pct": int(round((RANGE_QUANTILES[1] - RANGE_QUANTILES[0]) * 100)),
        "players": players_payload(cand, with_x=(version == "v2")),
    }
    if version == "v2":
        payload["why_base"] = float(cand["why_base"].iloc[0])
        payload["sim"] = simulator_payload(cand, models, avail, season, upcoming, rosters)
    write_json(out_path, payload)
    write_json(os.path.join(PROJ_DIR, "latest.json"), payload)
    if version == "v2" and payload["type"] == "live":
        make_game(season, upcoming, payload["players"], stats, schedule, injuries)
    print(f"  saved {len(payload['players'])} projections for week {upcoming} "
          f"({payload['type']}; first kickoff {first_kick:%a %b %d %H:%M} UTC)")


if __name__ == "__main__":
    sys.exit(main())
