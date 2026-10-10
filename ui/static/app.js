const state = { sport: "NFL", data: null, filter: "all", view: "board", bets: null, draft: null };

const $ = (sel) => document.querySelector(sel);
const boardEl = $("#board");
const titleEl = $("#slate-title");
const subEl = $("#slate-sub");
const statsEl = $("#stats");
const toastEl = $("#toast");
const focusStrip = $("#focus-strip");

function clean(s) {
  if (s == null) return "";
  return String(s)
    .replace(/\u2014/g, ",")
    .replace(/\u2013/g, "-")
    .replace(/—/g, ",")
    .replace(/–/g, "-");
}

function pct(x) {
  const n = Number(x);
  if (Number.isNaN(n)) return "n/a";
  return `${Math.round(n * 100)}%`;
}

function confPct(x) {
  const n = Number(x);
  if (Number.isNaN(n)) return "n/a";
  const v = n > 1.5 ? n : n * 100;
  return `${Math.round(v)}%`;
}

function odds(x) {
  const n = Number(x);
  if (Number.isNaN(n) || x == null) return "—";
  return n > 0 ? `+${n}` : `${n}`;
}

function showToast(msg) {
  toastEl.textContent = clean(msg);
  toastEl.classList.remove("hidden");
  setTimeout(() => toastEl.classList.add("hidden"), 1600);
}

function actionClass(a) {
  return `action-${String(a || "PASS").toLowerCase()}`;
}

async function copyText(label, text) {
  await navigator.clipboard.writeText(clean(text));
  showToast(label);
}

function priceCell(fair, dk) {
  const fairTxt = odds(fair);
  const dkTxt = dk != null ? odds(dk) : null;
  return `
    <span class="price">
      <span class="fair" title="Model fair">${fairTxt}</span>
      ${dkTxt ? `<span class="book" title="Book">${dkTxt}</span>` : ""}
    </span>`;
}

function renderFocus(board) {
  const takes = (board || []).filter((c) => c.action === "BET").slice(0, 4);
  if (!takes.length) {
    focusStrip.hidden = true;
    focusStrip.innerHTML = "";
    return;
  }
  focusStrip.hidden = false;
  focusStrip.innerHTML = `
    <div class="focus-label">Take first</div>
    <div class="focus-row">
      ${takes
        .map(
          (c) => `
        <a class="focus-chip" href="#game-${c.game_id}">
          <span class="action-tag ${actionClass(c.action)}">${c.action}</span>
          <strong>${clean(c.winner?.pick)}</strong>
          <span class="muted">${clean(c.matchup)}</span>
        </a>`
        )
        .join("")}
    </div>`;
}

function unproven(row) {
  return String(row?.stake || "").startsWith("info only") ? "unproven" : "";
}

function logBtn(kind, gameId, idx = "") {
  return `<button type="button" class="log-btn" data-log="${kind}" data-game="${clean(gameId)}" data-idx="${idx}">Log bet</button>`;
}

function renderMarketRows(card) {
  const w = card.winner || {};
  const s = card.spread || {};
  const t = card.total || {};
  return `
    <div class="markets">
      <div class="mkt">
        <div class="mkt-k">Winner</div>
        <div class="mkt-v">
          <strong>${clean(w.pick)}</strong>
          <span class="mkt-meta">${pct(w.prob)} · ${clean(w.stake)}</span>
        </div>
        ${priceCell(w.fair_american, w.dk_price)}
        ${logBtn("winner", card.game_id)}
      </div>
      <div class="mkt ${unproven(s)}">
        <div class="mkt-k">Spread</div>
        <div class="mkt-v">
          <strong>${clean(s.label || `${s.pick} ${s.line > 0 ? "+" : ""}${s.line}`)}</strong>
          <span class="mkt-meta">${s.prob != null ? `${pct(s.prob)} · ` : ""}model line ${s.model_line > 0 ? "+" : ""}${s.model_line} · ${clean(s.stake)}</span>
        </div>
        ${priceCell(s.fair_american, s.dk_price)}
        ${logBtn("spread", card.game_id)}
      </div>
      <div class="mkt ${unproven(t)}">
        <div class="mkt-k">Total</div>
        <div class="mkt-v">
          <strong>${t.side ? `${clean(t.side)} ${t.line}` : `Model ${t.model_total}`}</strong>
          <span class="mkt-meta">${t.prob != null ? `${pct(t.prob)} · ` : ""}model ${t.model_total} · ${clean(t.stake)}</span>
        </div>
        ${priceCell(t.fair_american, t.dk_price)}
        ${logBtn("total", card.game_id)}
      </div>
    </div>
    ${bestPlayLine(card.best_play)}`;
}

function bestPlayLine(b) {
  if (!b) {
    return `<p class="best-play none">No market has an edge at the current book prices.</p>`;
  }
  return `<p class="best-play">Best priced play: <strong>${clean(b.label)}</strong> at ${odds(b.dk_price)} · model ${pct(b.prob)}</p>`;
}

function renderProps(props, gameId) {
  if (!props?.length) {
    return `<p class="empty-inline">No player props listed for this game.</p>`;
  }
  return `
    <table class="prop-table">
      <thead>
        <tr><th>Player</th><th>Market</th><th>Model</th><th>Play vs DK</th><th></th></tr>
      </thead>
      <tbody>
        ${props
          .map((p, i) => {
            const over = p.over_if_line_at_or_below ?? "";
            const under = p.under_if_line_at_or_above ?? "";
            const team = p.team ? `<span class="muted"> ${clean(p.team)}</span>` : "";
            return `<tr>
              <td>${clean(p.player)}${team}</td>
              <td>${clean(p.market_label || p.market)}</td>
              <td class="num">${p.mu ?? "—"}</td>
              <td class="tip-cell">Over ≤${over} · Under ≥${under}</td>
              <td>${logBtn("prop", gameId, i)}</td>
            </tr>`;
          })
          .join("")}
      </tbody>
    </table>`;
}

function renderCardParlays(parlays, idxBase, gameId) {
  if (!parlays?.length) {
    return `<p class="empty-inline">No same-game parlay has a proven edge for this game.</p>`;
  }
  return `
    <div class="card-parlays">
      <div class="mini-label">Same-game parlays</div>
      ${parlays
        .map((p, i) => {
          const legs = (p.legs || []).map((l) => clean(l.label)).join(" · ");
          return `
          <div class="mini-ticket">
            <div>
              <strong>${p.n_legs}-leg SGP</strong>
              <span class="muted">fair ${odds(p.fair_american)} · ${clean(p.grade || "")} · ${confPct(p.confidence_pct)}</span>
              <div class="legs-inline">${legs}</div>
            </div>
            <button type="button" class="copy-btn" data-card-parlay="${idxBase}:${i}">Copy</button>
            ${logBtn("sgp", gameId, i)}
          </div>`;
        })
        .join("")}
    </div>`;
}

function renderBoard(board) {
  const filter = state.filter;
  const list = (board || []).filter((c) => filter === "all" || c.action === filter);
  if (!list.length) {
    boardEl.innerHTML = `<div class="empty">No games in this filter.</div>`;
    return;
  }

  boardEl.innerHTML = list
    .map((c, idx) => {
      const warns = (c.warnings || []).map((w) => `<li>${clean(w)}</li>`).join("");
      const reasons = (c.reasons || []).map((r) => `<li>${clean(r)}</li>`).join("");
      return `
      <article class="game-card ${actionClass(c.action)}" id="game-${c.game_id}">
        <header class="gc-head">
          <div>
            <div class="gc-match">${clean(c.away_team)} <span class="at">at</span> ${clean(c.home_team)}</div>
            <div class="gc-day muted">${clean(c.gameday || "")}</div>
          </div>
          <div class="gc-actions">
            <span class="action-tag ${actionClass(c.action)}">${clean(c.action)}</span>
            <span class="conf-pill">${confPct(c.confidence_score)}</span>
          </div>
        </header>
        <p class="gc-instruction">${clean(c.instruction || c.detail || "No clear side.")}</p>
        ${renderMarketRows(c)}
        <div class="gc-block">
          <div class="mini-label">Player props</div>
          ${renderProps(c.props, c.game_id)}
        </div>
        ${renderCardParlays(c.parlays, idx, c.game_id)}
        ${(reasons || warns) ? `<ul class="gc-why">${reasons}${warns}</ul>` : ""}
      </article>`;
    })
    .join("");

  boardEl.querySelectorAll("[data-card-parlay]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const [ci, pi] = btn.dataset.cardParlay.split(":").map(Number);
      const card = list[ci];
      const p = card?.parlays?.[pi];
      if (!p) return;
      const text = [
        `${card.matchup} ${p.n_legs}-leg fair ${odds(p.fair_american)}`,
        ...(p.legs || []).map((l, i) => `${i + 1}. ${l.label}`),
        p.advice || "",
      ].join("\n");
      copyText("Copied game ticket", text);
    });
  });
}

function renderStats(data) {
  const board = data.board || [];
  const count = (a) => board.filter((x) => x.action === a).length;
  statsEl.innerHTML = `
    <div class="stat"><b>${count("BET")}</b><span>bet</span></div>
    <div class="stat"><b>${count("LEAN")}</b><span>lean</span></div>
    <div class="stat"><b>${count("WAIT")}</b><span>wait</span></div>
    <div class="stat"><b>${board.length}</b><span>games</span></div>
  `;
}

// ------------------------------------------------------------------ bet log

const logDialog = $("#log-dialog");
const logForm = $("#log-form");
const betsView = $("#bets-view");
const boardViewEls = [".band", "#focus-strip", ".board-wrap"].map((s) => $(s));

function stakeNum(txt) {
  const m = String(txt || "").match(/(\d+(?:\.\d+)?)u/);
  return m ? Number(m[1]) : 0.25;
}

function draftFor(card, kind, idx) {
  const base = {
    sport: state.sport,
    season: state.data?.season ?? (card.gameday ? Number(String(card.gameday).slice(0, 4)) : null),
    week: state.data?.week ?? null,
    gameday: card.gameday,
    game_id: card.game_id,
    matchup: card.matchup,
    card_action: card.action,
  };
  const w = card.winner || {};
  const s = card.spread || {};
  const t = card.total || {};
  if (kind === "winner") {
    return { ...base, market: "h2h", pick: `${w.pick} to win`, team: w.pick, is_home: w.is_home,
      price: w.dk_price ?? w.fair_american, model_prob: w.prob, stake: stakeNum(w.stake) };
  }
  if (kind === "spread") {
    return { ...base, market: "spreads", pick: s.pick, team: s.pick, is_home: s.is_home, line: s.line,
      price: s.dk_price ?? -110, model_prob: s.prob, stake: stakeNum(s.stake) };
  }
  if (kind === "total") {
    return { ...base, market: "totals", pick: "Game total", side: t.side || "Over", card_side: t.side,
      line: t.line ?? t.model_total, price: t.dk_price ?? -110, model_prob: t.prob, stake: stakeNum(t.stake) };
  }
  if (kind === "prop") {
    const p = card.props[idx];
    return { ...base, market: p.market, pick: `${p.player} ${p.market_label || ""}`.trim(), player: p.player,
      side: "Over", line: "", price: -115, model_prob: null, stake: 0.25 };
  }
  const p = card.parlays[idx];
  return { ...base, market: "sgp", pick: `${p.n_legs}-leg SGP`, legs: (p.legs || []).map((l) => l.label),
    price: p.fair_american, model_prob: p.fair_prob ?? null, stake: 0.25 };
}

function openLogDialog(draft) {
  state.draft = draft;
  const needsLine = ["spreads", "totals"].includes(draft.market) || draft.market.startsWith("player_");
  const needsSide = draft.market === "totals" || draft.market.startsWith("player_");
  $("#log-title").textContent = `${draft.matchup}`;
  $("#log-pick").textContent =
    draft.market === "sgp" ? draft.legs.join(" · ") : draft.pick;
  $("#log-side-row").hidden = !needsSide;
  $("#log-line-row").hidden = !needsLine;
  logForm.side.value = draft.side || "Over";
  logForm.line.value = draft.line ?? "";
  logForm.price.value = draft.price != null ? odds(draft.price) : "";
  logForm.stake.value = draft.stake;
  $("#log-error").textContent = "";
  logDialog.showModal();
}

async function submitLog(ev) {
  ev.preventDefault();
  const d = { ...state.draft };
  d.price = logForm.price.value.trim();
  d.stake = Number(logForm.stake.value);
  if (!$("#log-line-row").hidden) d.line = logForm.line.value;
  if (!$("#log-side-row").hidden) {
    d.side = logForm.side.value;
    if (d.market === "totals" && d.model_prob != null && d.card_side && d.side !== d.card_side) {
      d.model_prob = 1 - d.model_prob;
    }
    d.pick = d.market === "totals" ? `${d.side} ${d.line}` : `${d.player} ${d.side} ${d.line} ${d.pick.replace(d.player, "").trim()}`;
  } else if (d.market === "spreads") {
    const n = Number(d.line);
    d.pick = `${d.team} ${n > 0 ? "+" : ""}${n}`;
  }
  const res = await fetch("/api/bets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(d),
  });
  const body = await res.json();
  if (!res.ok) {
    $("#log-error").textContent = clean(body.detail || `HTTP ${res.status}`);
    return;
  }
  logDialog.close();
  state.bets = body;
  showToast(`Logged: ${body.bet.pick}`);
}

function money(x) {
  const n = Number(x);
  if (Number.isNaN(n)) return "—";
  return `${n > 0 ? "+" : ""}${n.toFixed(2)}u`;
}

function renderBets() {
  const { bets = [], summary = {} } = state.bets || {};
  const o = summary.overall || {};
  const decided = (o.won || 0) + (o.lost || 0);
  const calib =
    o.model_expected_wins != null
      ? `Model expected ${o.model_expected_wins} wins on these, actual ${o.model_actual_wins}.`
      : "";
  const breakeven =
    o.win_rate != null
      ? `You're winning ${pct(o.win_rate)} of decided bets. The prices you paid needed ${pct(o.avg_price_prob)} to break even.`
      : "No settled bets yet. Log bets from the game cards, then grade them after the games finish.";
  const markets = Object.entries(summary.by_market || {})
    .map(
      ([k, b]) => `<tr><td>${clean({ h2h: "Winner", spreads: "Spread", totals: "Total", props: "Props", sgp: "SGP" }[k] || k)}</td>
        <td>${b.won}-${b.lost}${b.push ? `-${b.push}` : ""}</td><td class="num">${money(b.profit)}</td>
        <td class="num">${b.roi != null ? pct(b.roi) : "—"}</td><td>${b.pending}</td></tr>`
    )
    .join("");
  const rows = bets
    .map((b) => {
      const status = b.status || "pending";
      const acts =
        status === "pending"
          ? `<button data-res="won" data-id="${b.id}">Won</button><button data-res="lost" data-id="${b.id}">Lost</button><button data-res="push" data-id="${b.id}">Push</button>`
          : `<button data-res="pending" data-id="${b.id}">Undo</button>`;
      return `<tr>
        <td class="muted">${clean(b.gameday || b.created_at?.slice(0, 10) || "")}</td>
        <td>${clean(b.matchup || "")}<div class="muted small">${clean(b.sport)}</div></td>
        <td><strong>${clean(b.pick)}</strong>${b.result_note ? `<div class="muted small">${clean(b.result_note)}</div>` : ""}</td>
        <td class="num">${clean(b.price)}</td>
        <td class="num">${b.stake}u</td>
        <td class="num">${b.model_prob != null ? pct(b.model_prob) : "—"}</td>
        <td><span class="status status-${status}">${status}</span></td>
        <td class="num">${b.profit != null ? money(b.profit) : ""}</td>
        <td class="row-acts">${acts}<button data-del="${b.id}" title="Delete">✕</button></td>
      </tr>`;
    })
    .join("");

  betsView.innerHTML = `
    <div class="bets-head">
      <div>
        <h2>My bets</h2>
        <p class="muted">${breakeven} ${calib}</p>
      </div>
      <button id="grade-btn" class="ghost" type="button">Grade finished games</button>
    </div>
    <div class="stats bets-stats">
      <div class="stat"><b>${o.won || 0}-${o.lost || 0}${o.push ? `-${o.push}` : ""}</b><span>record</span></div>
      <div class="stat"><b>${money(o.profit || 0)}</b><span>units</span></div>
      <div class="stat"><b>${o.roi != null ? pct(o.roi) : "—"}</b><span>ROI</span></div>
      <div class="stat"><b>${o.pending || 0}</b><span>pending</span></div>
    </div>
    ${decided ? `<table class="prop-table by-market"><thead><tr><th>Market</th><th>Record</th><th>Units</th><th>ROI</th><th>Pending</th></tr></thead><tbody>${markets}</tbody></table>` : ""}
    ${bets.length
      ? `<table class="prop-table bets-table"><thead><tr><th>Date</th><th>Game</th><th>Pick</th><th>Price</th><th>Stake</th><th>Model</th><th>Status</th><th>P/L</th><th></th></tr></thead><tbody>${rows}</tbody></table>`
      : `<div class="empty">Nothing logged yet. Hit "Log bet" on any market, prop, or parlay after you place it on Robinhood.</div>`}
  `;

  $("#grade-btn").addEventListener("click", gradeBets);
  betsView.querySelectorAll("[data-res]").forEach((btn) =>
    btn.addEventListener("click", () =>
      betsRequest(`/api/bets/${btn.dataset.id}/result`, "POST", { result: btn.dataset.res })
    )
  );
  betsView.querySelectorAll("[data-del]").forEach((btn) =>
    btn.addEventListener("click", () => {
      if (confirm("Delete this bet from the log?")) betsRequest(`/api/bets/${btn.dataset.del}`, "DELETE");
    })
  );
}

async function betsRequest(url, method = "GET", body) {
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json();
  if (!res.ok) {
    showToast(data.detail || `HTTP ${res.status}`);
    return null;
  }
  state.bets = data;
  renderBets();
  return data;
}

async function gradeBets() {
  const btn = $("#grade-btn");
  btn.disabled = true;
  btn.textContent = "Checking scores...";
  const data = await betsRequest("/api/bets/grade", "POST");
  if (data) {
    const errs = data.errors?.length ? ` (${data.errors.length} could not be checked)` : "";
    showToast(data.graded ? `Graded ${data.graded} bet(s)${errs}` : `Nothing final yet${errs}`);
  }
}

function setView(view) {
  state.view = view;
  const bets = view === "bets";
  betsView.hidden = !bets;
  boardViewEls.forEach((el) => {
    if (!el) return;
    if (el.id === "focus-strip") {
      el.hidden = bets || !el.innerHTML.trim();
    } else {
      el.hidden = bets;
    }
  });
  $("#view-bets").classList.toggle("active", bets);
  if (bets) betsRequest("/api/bets");
}

boardEl.addEventListener("click", (ev) => {
  const btn = ev.target.closest("[data-log]");
  if (!btn) return;
  const card = (state.data?.board || []).find((c) => String(c.game_id) === btn.dataset.game);
  if (!card) return;
  openLogDialog(draftFor(card, btn.dataset.log, btn.dataset.idx === "" ? null : Number(btn.dataset.idx)));
});
logForm.addEventListener("submit", submitLog);
$("#log-cancel").addEventListener("click", () => logDialog.close());
$("#view-bets").addEventListener("click", () => setView(state.view === "bets" ? "board" : "bets"));

async function loadSport(sport) {
  document.body.classList.add("loading");
  titleEl.textContent = `Loading ${sport}...`;
  try {
    const url =
      sport === "NFL" ? "/api/nfl" : sport === "CFB" ? "/api/cfb" : "/api/mlb?days=2";
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    state.data = data;
    if (sport === "NFL") {
      titleEl.textContent = `NFL ${data.season} · Week ${data.week}`;
      subEl.innerHTML = `BET when <strong>|margin| ≥ 7</strong>. One card = one game. Same-game only.`;
    } else if (sport === "CFB") {
      titleEl.textContent = `CFB ${data.season} · Week ${data.week}`;
      subEl.innerHTML = `BET when <strong>|margin| ≥ 10</strong>. Same-game parlays only.`;
    } else {
      titleEl.textContent = `MLB · upcoming slate`;
      subEl.innerHTML = `SP gap + margin gate. Same-game parlays only.`;
    }
    renderStats(data);
    renderFocus(data.board);
    renderBoard(data.board);
  } catch (err) {
    titleEl.textContent = "Could not load slate";
    boardEl.innerHTML = `<div class="empty">${clean(err.message)}</div>`;
  } finally {
    document.body.classList.remove("loading");
  }
}

document.querySelectorAll(".seg-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".seg-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.sport = btn.dataset.sport;
    setView("board");
    loadSport(state.sport);
  });
});

document.querySelectorAll(".filter-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".filter-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.filter = btn.dataset.filter;
    if (state.data) renderBoard(state.data.board);
  });
});

$("#refresh").addEventListener("click", () => loadSport(state.sport));

loadSport("NFL");
