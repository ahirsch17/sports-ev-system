"""
Sports Edge — local betting board for NFL + MLB.

Run:
  streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without editable install quirks
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Sports Edge",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Instrument+Serif:ital@0;1&display=swap');
  html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
  .block-container { padding-top: 1.2rem; max-width: 1200px; }
  h1, h2, h3 { font-family: 'Instrument Serif', Georgia, serif !important; font-weight: 400 !important; letter-spacing: -0.02em; }
  .game-card {
    background: linear-gradient(145deg, #0f1c18 0%, #163028 55%, #1a3a2a 100%);
    border: 1px solid rgba(180, 220, 160, 0.18);
    border-radius: 14px;
    padding: 1.1rem 1.25rem;
    margin-bottom: 0.85rem;
    color: #e8f0e6;
  }
  .game-card.mlb {
    background: linear-gradient(145deg, #1a1520 0%, #2a1f35 55%, #1e2838 100%);
    border-color: rgba(200, 180, 230, 0.2);
  }
  .matchup { font-size: 1.35rem; margin: 0 0 0.35rem 0; }
  .meta { opacity: 0.72; font-size: 0.85rem; margin-bottom: 0.75rem; }
  .pill {
    display: inline-block;
    padding: 0.2rem 0.55rem;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
    font-size: 0.78rem;
  }
  .pill.hot { background: rgba(90, 200, 120, 0.22); color: #b8f0c4; }
  .pill.cool { background: rgba(120, 160, 220, 0.2); color: #c5d8f5; }
  .bet-row { font-size: 0.9rem; padding: 0.25rem 0; border-top: 1px solid rgba(255,255,255,0.06); }
  .disclaimer { opacity: 0.55; font-size: 0.8rem; margin-top: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)


def _pct(x) -> str:
    try:
        return f"{float(x):.0%}"
    except Exception:
        return "—"


def _num(x, fmt="{:+.1f}") -> str:
    try:
        return fmt.format(float(x))
    except Exception:
        return "—"


@st.cache_data(ttl=600, show_spinner=False)
def load_nfl(season: int | None, week: int | None):
    from nfl_edge.services.slate import build_nfl_slate

    return build_nfl_slate(season=season, week=week)


@st.cache_data(ttl=600, show_spinner=False)
def load_mlb(days_ahead: int):
    from mlb_edge.predict import predict_mlb_slate

    return predict_mlb_slate(days_ahead=days_ahead, days_back=50)


def render_factor_chart(breakdown: list[dict], value_key: str = "weight"):
    if not breakdown:
        return
    df = pd.DataFrame(breakdown)
    if value_key not in df.columns:
        return
    st.bar_chart(df.set_index("factor" if "factor" in df.columns else "key")[value_key], height=220)


def render_nfl_game(g: dict, edges_for_game: list[dict]):
    away, home = g.get("away_team"), g.get("home_team")
    st.markdown(
        f"""
        <div class="game-card">
          <div class="matchup">{away} @ {home}</div>
          <div class="meta">Model margin { _num(g.get('pred_margin')) } · Total { _num(g.get('pred_total'), '{:.1f}') } · Home win {_pct(g.get('home_win_prob'))}</div>
          <span class="pill hot">deep {_num(g.get('deep_net_margin_adj'))}</span>
          <span class="pill">continuity {_pct(g.get('home_sim_continuity'))}/{_pct(g.get('away_sim_continuity'))}</span>
          <span class="pill cool">{g.get('home_mm_note') or '—'}</span>
          <span class="pill cool">{g.get('away_mm_note') or '—'}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.caption("Suggested edges")
        if not edges_for_game:
            st.write("No +EV flags vs schedule lines.")
        else:
            for e in edges_for_game[:6]:
                st.markdown(
                    f"**{e.get('label')}** · fair {e.get('fair_american')} · "
                    f"DK {e.get('dk_price')} · EV {_pct(e.get('ev'))} · p={_pct(e.get('model_prob'))}"
                )
    with c2:
        st.caption("Factor weights (model design)")
        render_factor_chart(g.get("factor_breakdown") or [])
        with st.expander("Why this number"):
            for f in g.get("factor_breakdown") or []:
                st.write(f"**{f.get('factor')}** ({_pct(f.get('weight'))}) — {f.get('effect')}")


def render_mlb_game(g: dict):
    away, home = g.get("away_team"), g.get("home_team")
    meta1 = f"{g.get('gameday')} · {g.get('venue') or ''} · SP {g.get('away_sp_name') or '?'} vs {g.get('home_sp_name') or '?'}"
    meta2 = (
        f"Model {_num(g.get('pred_away_runs'), '{:.1f}')}–{_num(g.get('pred_home_runs'), '{:.1f}')} "
        f"(total {_num(g.get('pred_total'), '{:.1f}')}) · Home win {_pct(g.get('home_win_prob'))}"
    )
    wind = g.get("wind") or g.get("condition") or "weather n/a"
    st.markdown(
        f"""
        <div class="game-card mlb">
          <div class="matchup">{away} @ {home}</div>
          <div class="meta">{meta1}</div>
          <div class="meta">{meta2}</div>
          <span class="pill hot">margin {_num(g.get('pred_margin'))}</span>
          <span class="pill">SP ERA {g.get('away_sp_era') or '—'} / {g.get('home_sp_era') or '—'}</span>
          <span class="pill cool">{wind}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.caption("Intuitive bets (model vs −110)")
        bets = g.get("suggested_bets") or []
        if not bets:
            st.write("No sides cleared EV threshold at −110.")
        for b in bets[:8]:
            st.markdown(
                f"**{b.get('label')}** · fair {b.get('fair_american')} · "
                f"EV {_pct(b.get('ev'))} · p={_pct(b.get('model_prob'))}"
            )
    with c2:
        st.caption("Weighted contributing factors")
        bd = g.get("factor_breakdown") or []
        chart_df = pd.DataFrame(
            [{"factor": x["factor"], "weight": x["weight"]} for x in bd]
        )
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("factor")["weight"], height=220)
        with st.expander("Factor detail"):
            for f in bd:
                st.write(
                    f"**{f.get('factor')}** ({_pct(f.get('weight'))}) — {f.get('detail')} "
                    f"[home {_num(f.get('home_effect'))} / away {_num(f.get('away_effect'))}]"
                )


def main():
    st.title("Sports Edge")
    st.write("Upcoming games with model prices, factor weights, and intuitive bet ideas.")

    sport = st.sidebar.radio("Sport", ["NFL", "MLB"], index=0)
    st.sidebar.markdown("---")
    st.sidebar.caption("Research tool — not gambling advice. Edges are noisy.")

    if sport == "NFL":
        from nfl_edge.config import current_nfl_season
        from nfl_edge.services.slate import resolve_week

        season_default, week_default = resolve_week()
        season = st.sidebar.number_input("Season", 2006, 2030, season_default)
        week = st.sidebar.number_input("Week", 1, 22, week_default)
        refresh = st.sidebar.button("Refresh NFL slate", type="primary")
        if refresh:
            load_nfl.clear()

        with st.spinner("Building NFL slate (ratings + deep matchups)…"):
            slate = load_nfl(int(season), int(week))

        st.subheader(f"NFL {slate['season']} · Week {slate['week']}")
        games = slate.get("games") or []
        edges = slate.get("edges") or []
        parlays = slate.get("parlays") or []

        tab_games, tab_board, tab_parlay, tab_weights = st.tabs(
            ["Games", "Bet board", "Parlays", "Weights"]
        )

        with tab_games:
            if not games:
                st.warning("No games found for this week.")
            for g in games:
                gid = g.get("game_id")
                ge = [e for e in edges if e.get("game_id") == gid or (
                    e.get("home_team") and g.get("home_team") in str(e.get("home_team", ""))
                )]
                # also match by abbr inside full names
                if not ge:
                    ge = [
                        e
                        for e in edges
                        if str(g.get("home_team")) in str(e.get("home_team", ""))
                        or str(g.get("away_team")) in str(e.get("away_team", ""))
                    ]
                render_nfl_game(g, ge)

        with tab_board:
            if edges:
                df = pd.DataFrame(edges)
                show = df[df.get("passes_threshold") == True] if "passes_threshold" in df.columns else df
                if show.empty:
                    show = df
                cols = [c for c in ["away_team", "home_team", "label", "dk_price", "fair_american", "model_prob", "ev", "market"] if c in show.columns]
                st.dataframe(show[cols].sort_values("ev", ascending=False), use_container_width=True, hide_index=True)
            else:
                st.info("No edges yet — schedule lines may be missing.")

        with tab_parlay:
            if not parlays:
                st.info("No correlated SGP ideas for this slate.")
            for p in parlays:
                st.markdown(
                    f"**{' + '.join(p.get('legs') or [])}** · fair {p.get('fair_american')} · "
                    f"proxy {p.get('combined_model_ev_proxy')}"
                )
                st.caption(p.get("note"))

        with tab_weights:
            st.write("How the NFL model weights contributing factors:")
            for g in games[:1]:
                render_factor_chart(g.get("factor_breakdown") or [])
                for f in g.get("factor_breakdown") or []:
                    st.write(f"- **{f['factor']}** ({_pct(f['weight'])}): {f['effect']}")

    else:
        days = st.sidebar.slider("Days ahead", 1, 5, 2)
        refresh = st.sidebar.button("Refresh MLB slate", type="primary")
        if refresh:
            load_mlb.clear()

        with st.spinner("Pulling MLB schedule + pitcher/team factors…"):
            slate = load_mlb(int(days))

        st.subheader(f"MLB · next {days} day(s)")
        games = slate.get("games") or []
        factors = slate.get("factors") or []

        tab_games, tab_board, tab_weights = st.tabs(["Games", "Bet board", "Factor weights"])

        with tab_games:
            if not games:
                st.warning("No upcoming MLB games found.")
            for g in games:
                render_mlb_game(g)

        with tab_board:
            edges = slate.get("edges") or []
            if edges:
                df = pd.DataFrame(edges).sort_values("ev", ascending=False)
                cols = [c for c in ["away_team", "home_team", "label", "fair_american", "model_prob", "ev", "market"] if c in df.columns]
                st.dataframe(df[cols], use_container_width=True, hide_index=True)
            else:
                st.info("No model bets cleared thresholds.")

        with tab_weights:
            st.write("MLB factor weights (explicit, editable in `research/mlb/factor_weights.json`):")
            if factors:
                fdf = pd.DataFrame(factors)
                st.bar_chart(fdf.set_index("key")["weight"], height=280)
                for f in factors:
                    st.write(f"**{f['key']}** — {_pct(f['weight'])}: {f['description']}")

    st.markdown(
        '<p class="disclaimer">For research / entertainment. Past performance ≠ future results. '
        "Compare fair odds to DraftKings before risking money.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
