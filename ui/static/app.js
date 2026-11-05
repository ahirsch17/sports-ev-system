const state = { sport: "NFL", data: null, gameId: null, request: 0 };

const app = document.querySelector("#app");

const MARKETS = [
  ["player_pass_yds", "Passing"],
  ["player_reception_yds", "Receiving"],
  ["player_rush_yds", "Rushing"],
];

function clean(value) {
  if (value == null) return "";
  return String(value);
}

function odds(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "";
  return n > 0 ? `+${n}` : `${n}`;
}

function whole(value) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.round(n) : "—";
}

function one(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return Math.abs(n - Math.round(n)) < 0.05 ? String(Math.round(n)) : n.toFixed(1);
}

function implied(american) {
  const price = Number(american);
  if (!Number.isFinite(price)) return null;
  const chance = price >= 0 ? 100 / (price + 100) : Math.abs(price) / (Math.abs(price) + 100);
  return Math.round(chance * 100);
}

function num(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(1) : "—";
}

function signed(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  const text = n.toFixed(1);
  return n > 0 ? `+${text}` : text;
}

function dayLabel(gameday) {
  if (!gameday) return "Date not set";
  const date = new Date(`${String(gameday).slice(0, 10)}T12:00:00`);
  if (Number.isNaN(date.getTime())) return String(gameday);
  return date.toLocaleDateString("en-US", { weekday: "long", month: "short", day: "numeric" });
}

function clock(value) {
  if (!value) return "";
  const [hourText, minuteText] = String(value).split(":");
  const hour = Number(hourText);
  if (!Number.isFinite(hour)) return "";
  const suffix = hour >= 12 ? "PM" : "AM";
  const hour12 = hour % 12 || 12;
  return `${hour12}:${String(minuteText || "00").padStart(2, "0")} ${suffix}`;
}

function neutralSite(game) {
  return String(game.location || "").toLowerCase() === "neutral";
}

function siteLabel(game) {
  return neutralSite(game) ? "Neutral site" : `${game.home_team} home`;
}

function placeLine(game) {
  return [game.stadium, siteLabel(game), clock(game.gametime)].filter(Boolean).join(" · ");
}

function hexRgb(hex) {
  const text = String(hex || "").replace("#", "");
  return [0, 2, 4].map((index) => parseInt(text.slice(index, index + 2), 16));
}

function luminance(hex) {
  const channels = hexRgb(hex).map((value) => {
    const channel = value / 255;
    return channel <= 0.03928 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2];
}

function mix(start, end, amount) {
  const left = hexRgb(start);
  const right = hexRgb(end);
  const mixed = left.map((value, index) => Math.round(value + (right[index] - value) * amount));
  return `#${mixed.map((value) => value.toString(16).padStart(2, "0")).join("")}`;
}

function contrast(a, b) {
  const left = luminance(a);
  const right = luminance(b);
  return (Math.max(left, right) + 0.05) / (Math.min(left, right) + 0.05);
}

function paint(abbr) {
  const pair =
    (typeof CFB_COLORS !== "undefined" && CFB_COLORS[abbr]) ||
    (typeof TEAM_COLORS !== "undefined" && TEAM_COLORS[abbr]) ||
    ["#1d4e89", "#c5d4ea"];
  const [primary, secondary] = pair;
  const block = luminance(primary) > 0.62 ? secondary : primary;
  const stripe = block === primary ? secondary : primary;
  const board = "#161411";
  let signal = block;
  if (contrast(block, board) < 2.05) {
    signal = contrast(stripe, board) >= 2.05 ? stripe : mix(block, "#ffffff", 0.5);
  }
  return {
    block,
    stripe,
    signal,
    fg: luminance(block) > 0.42 ? "#1c1916" : "#f6f1e7",
  };
}

function teamStyle(game) {
  const away = paint(game.away_team);
  const home = paint(game.home_team);
  return [
    `--away:${away.block}`,
    `--away-2:${away.stripe}`,
    `--home:${home.block}`,
    `--home-2:${home.stripe}`,
    `--away-on:${away.signal}`,
    `--home-on:${home.signal}`,
    `--away-fg:${away.fg}`,
    `--home-fg:${home.fg}`,
  ].join(";");
}

function games() {
  return (state.data && state.data.games) || [];
}

function cardFor(gameId) {
  const cards = (state.data && state.data.matchup_parlays) || [];
  return cards.find((card) => card.game_id === gameId) || null;
}

function bookSide(game) {
  if (game.spread_line == null || game.spread_line === "") return "";
  const line = Number(game.spread_line);
  if (!Number.isFinite(line) || line === 0) return "Pick";
  if (line > 0) return `${game.home_team} -${line}`;
  return `${game.away_team} -${Math.abs(line)}`;
}

function scoreBits(game, card) {
  const score = card && card.score;
  if (score && score.our_home != null) {
    return {
      ours: `${score.away} ${whole(score.our_away)} – ${score.home} ${whole(score.our_home)}`,
      book: bookSide(game),
      bookTotal: score.book_total != null ? String(score.book_total) : "",
      ourTotal: score.our_total != null ? String(score.our_total) : "",
    };
  }
  if (game.pred_home_points != null) {
    return {
      ours: `${game.away_team} ${whole(game.pred_away_points)} – ${game.home_team} ${whole(game.pred_home_points)}`,
      book: bookSide(game),
      bookTotal: game.total_line != null ? String(game.total_line) : "",
      ourTotal: game.pred_total != null ? String(whole(game.pred_total)) : "",
    };
  }
  return { ours: "—", book: bookSide(game), bookTotal: "", ourTotal: "" };
}

function propsTicket(card) {
  if (!card) return null;
  return (card.tickets || []).find((ticket) => ticket.team === "Props") || null;
}

function scoresTicket(card) {
  if (!card) return null;
  return (card.tickets || []).find((ticket) => ticket.team === "Scores") || null;
}

function legName(leg) {
  return leg && leg.market === "total" ? "Game total" : leg.player;
}

function totalTicket(card) {
  if (!card) return null;
  return (card.tickets || []).find((ticket) => ticket.team === "Total") || null;
}

function pairLine(card) {
  const ticket = propsTicket(card) || totalTicket(card);
  if (!ticket) return "None";
  const legs = (ticket.legs || []).map((leg) => `${legName(leg)} ${leg.side}`).join(", ");
  return `${legs} · ${odds(ticket.odds)}`;
}

function statsHold() {
  const box = state.data && state.data.data && state.data.data.player_stats;
  if (!box || box.ready !== false) return "";
  return `<section class="panel hold"><h3>Player stats are not in</h3><p>${clean(box.reason)}</p></section>`;
}

function renderList() {
  const data = state.data;
  if (!data) {
    app.innerHTML = `<p class="quiet">Loading ${state.sport}...</p>`;
    return;
  }
  const hold = statsHold();
  if (hold) {
    app.innerHTML = hold;
    return;
  }
  const title = state.sport === "MLB"
    ? "MLB"
    : `${state.sport} ${data.season || ""} · Week ${data.week || ""}`.trim();
  const list = games().slice().sort((a, b) => String(a.gameday).localeCompare(String(b.gameday)) || String(a.game_id).localeCompare(String(b.game_id)));
  if (!list.length) {
    app.innerHTML = `<h2 class="week">${clean(title)}</h2><p class="quiet">No games on this slate.</p>`;
    return;
  }
  let html = `<h2 class="week">${clean(title)}</h2><div class="slate">`;
  let lastDay = "";
  for (const game of list) {
    const day = dayLabel(game.gameday);
    if (day !== lastDay) {
      html += `<h3 class="day">${clean(day)}</h3>`;
      lastDay = day;
    }
    const card = cardFor(game.game_id);
    const bits = scoreBits(game, card);
    const college = state.sport === "CFB";
    const book = bits.bookTotal ? `${bits.book}, total ${bits.bookTotal}` : bits.book;
    const homeMark = neutralSite(game) ? "" : `<em>Home</em>`;
    const scoreText = college
      ? `${whole(game.pred_away_points)} – ${whole(game.pred_home_points)}`
      : bits.ours;
    const bookRow = book ? `<div><dt>Book</dt><dd>${clean(book)}</dd></div>` : "";
    const extraRow = college
      ? `<div><dt>Total</dt><dd>${clean(bits.ourTotal)} points</dd></div>`
      : `<div><dt>Parlay</dt><dd>${clean(pairLine(card))}</dd></div>`;
    html += `<button class="game${college ? " college" : ""}" type="button" data-game="${clean(game.game_id)}" style="${teamStyle(game)}">
      <span class="bands">
        <span class="away">${clean(game.away_team)}</span>
        <span class="home">${clean(game.home_team)}${homeMark}</span>
      </span>
      <p class="where">${clean(placeLine(game))}</p>
      <dl>
        <div><dt>Score</dt><dd>${clean(scoreText)}</dd></div>
        ${bookRow}
        ${extraRow}
      </dl>
    </button>`;
  }
  html += `</div>`;
  app.innerHTML = html;
}

function bar(leg, name) {
  return ((leg && leg.bars) || []).find((item) => item.name === name) || null;
}

function pickedNames(card) {
  const names = new Set();
  for (const ticket of (card && card.tickets) || []) {
    for (const leg of ticket.legs || []) names.add(leg.player);
  }
  return names;
}

function yardRows(game) {
  return game.yard_sheet && game.yard_sheet.length ? game.yard_sheet : (game.player_props || []);
}

function deltaOf(row) {
  const delta = row.edge_vs_roll != null ? Number(row.edge_vs_roll) : Number(row.mu) - Number(row.roll);
  return Number.isFinite(delta) ? delta : null;
}

function deltaCell(delta, maxAbs) {
  if (delta == null) return "<td>—</td>";
  const tone = delta < 0 ? "down" : delta > 0 ? "up" : "";
  const width = maxAbs > 0 ? Math.min(50, (Math.abs(delta) / maxAbs) * 50) : 0;
  const side = delta < 0 ? "neg" : "pos";
  const mark = width > 0 ? `<i class="${side}" style="width:${width.toFixed(0)}%"></i>` : "";
  return `<td class="delta ${tone}"><span>${signed(delta)}</span><span class="diverge">${mark}</span></td>`;
}

function teamTable(game, team, picked) {
  const rows = yardRows(game).filter((row) => row.team === team);
  const pass = team === game.home_team ? game.pred_home_pass_yds : game.pred_away_pass_yds;
  const rush = team === game.home_team ? game.pred_home_rush_yds : game.pred_away_rush_yds;
  const groups = [];
  const gaps = [];
  for (const [market, label] of MARKETS) {
    const group = rows.filter((row) => row.market === market).sort((a, b) => Number(b.mu) - Number(a.mu));
    if (!group.length) continue;
    groups.push([label, group]);
    for (const row of group) {
      const delta = deltaOf(row);
      if (delta != null) gaps.push(Math.abs(delta));
    }
  }
  const maxAbs = Math.max(1, ...gaps);
  let body = "";
  for (const [label, group] of groups) {
    body += `<tr class="group"><td colspan="4">${label}</td></tr>`;
    for (const row of group) {
      const on = picked.has(row.player) ? "picked" : "";
      body += `<tr class="${on}">
        <td>${clean(row.player)}</td>
        <td>${num(row.mu)}</td>
        <td>${num(row.roll)}</td>
        ${deltaCell(deltaOf(row), maxAbs)}
      </tr>`;
    }
  }
  if (!body) body = `<tr><td colspan="4">No yard projections.</td></tr>`;
  const side = team === game.away_team ? "away" : "home";
  const tag = side === "home" && !neutralSite(game) ? `<em>Home</em>` : "";
  return `<div class="sheet ${side}">
    <header>
      <h3>${clean(team)}${tag}</h3>
      <p><span>Pass</span> ${whole(pass)} <span>Rush</span> ${whole(rush)}</p>
    </header>
    <table class="yards">
      <thead><tr><th>Player</th><th>Ours</th><th>Avg</th><th>vs avg</th></tr></thead>
      <tbody>${body}</tbody>
    </table>
  </div>`;
}

function shownTotal(card) {
  const score = (card && card.score) || {};
  const away = Number(score.our_away);
  const home = Number(score.our_home);
  if (Number.isFinite(away) && Number.isFinite(home)) return Math.round(away) + Math.round(home);
  return null;
}

function legLine(leg, totalShown) {
  if (leg.market === "anytime_td") {
    return `<li><b>${clean(leg.player)}</b><span>${clean(leg.scored)} of last ${clean(leg.games)}<i>Anytime TD</i></span></li>`;
  }
  const ours = bar(leg, "Ours");
  const side = leg.side === "under" ? "Under" : "Over";
  if (leg.market === "total") {
    const book = bar(leg, "Book");
    const posted = book ? `<i>Book ${one(book.value)}</i>` : "";
    const points = totalShown != null ? totalShown : (ours ? whole(ours.value) : "—");
    return `<li><b>Game total</b><span>${side} ${points} points${posted}</span></li>`;
  }
  const avg = bar(leg, "Avg");
  const average = avg ? `<i>Avg ${num(avg.value)}</i>` : "";
  return `<li><b>${clean(leg.player)}</b><span>${side} ${ours ? num(ours.value) : "—"}${average}</span></li>`;
}

function priceCompare(ticket) {
  if (!ticket || ticket.odds == null) return "";
  const ours = implied(ticket.odds);
  const tax = implied(224);
  return `<div class="compare">
    <div><span>Our price</span><b>${ours}%</b><em>${odds(ticket.odds)}</em></div>
    <div class="tax"><span>FanDuel tax</span><b>${tax}%</b><em>+224</em></div>
  </div>`;
}

function ticketCard(title, ticket, empty, note, extra, kind, totalShown) {
  const tone = kind ? ` ${kind}` : "";
  if (!ticket) return `<article class="ticket${tone}"><div class="ticket-top"><b>${clean(title)}</b></div><p class="note">${clean(empty)}</p></article>`;
  const rows = (ticket.legs || []).map((leg) => legLine(leg, totalShown)).join("");
  const blurb = ticket.blurb ? `<p class="why">${clean(ticket.blurb)}</p>` : "";
  return `<article class="ticket${tone}">
    <div class="ticket-top"><b>${clean(title)}</b><span class="price">${odds(ticket.odds)}</span></div>
    ${blurb}
    <ul class="legs">${rows}</ul>
    ${extra || ""}
    <p class="note">${clean(note)}</p>
  </article>`;
}

function yardCards(card) {
  const tickets = ((card && card.tickets) || []).filter((ticket) => ticket.team === "Props" || ticket.team === "Total");
  if (!tickets.length) return ticketCard("Pair", null, "No parlay.", "", "", "yards");
  return tickets.map((ticket) => {
    const hasTotal = (ticket.legs || []).some((leg) => leg.market === "total");
    const title = ticket.team === "Total" ? "Total" : (hasTotal ? "Yards and total" : "Pair");
    const note = ticket.team === "Total"
      ? "Not in the parlay. The total moves with the passing game."
      : "Two separate bets. Take it when FanDuel offers longer than +224.";
    const extra = ticket.team === "Total" ? "" : priceCompare(ticket);
    return ticketCard(title, ticket, "", note, extra, ticket.team === "Total" ? "scores" : "yards", shownTotal(card));
  }).join("");
}

function scoreboard(game, card, bits) {
  const score = (card && card.score) || {};
  const awayNum = Number(score.our_away != null ? score.our_away : game.pred_away_points);
  const homeNum = Number(score.our_home != null ? score.our_home : game.pred_home_points);
  const awayPts = whole(awayNum);
  const homePts = whole(homeNum);
  const total = awayNum + homeNum;
  const share = Number.isFinite(total) && total > 0 ? (awayNum / total) * 100 : 50;
  const ourTotal = Number.isFinite(awayNum) && Number.isFinite(homeNum)
    ? String(Math.round(awayNum) + Math.round(homeNum))
    : whole(bits.ourTotal);
  const bookAway = score.book_away != null ? `<em><i>Book</i>${whole(score.book_away)}</em>` : "";
  const bookHome = score.book_home != null ? `<em><i>Book</i>${whole(score.book_home)}</em>` : "";
  const facts = [
    ["Line", bits.book],
    ["Book total", bits.bookTotal ? one(bits.bookTotal) : ""],
    ["Our total", ourTotal === "—" ? "" : ourTotal],
  ].filter((pair) => pair[1]);
  const neutral = neutralSite(game);
  const when = [dayLabel(game.gameday), clock(game.gametime)].filter(Boolean).join(" · ");
  const where = neutral ? [game.stadium, "Neutral site"].filter(Boolean).join(" · ") : "";
  const place = !neutral && game.stadium ? `<p class="place">${clean(game.stadium)}</p>` : "";
  return `<section class="board">
    <p class="when${where ? "" : " solo"}">${clean(when)}</p>
    ${where ? `<p class="where">${clean(where)}</p>` : ""}
    <div class="scoreline">
      <div class="club away"><span>${clean(game.away_team)}</span><strong>${clean(awayPts)}</strong>${bookAway}</div>
      <div class="club home"><span>${clean(game.home_team)}</span><strong>${clean(homePts)}</strong>${bookHome}${place}</div>
    </div>
    <div class="split" role="img" aria-label="Share of the points"><i style="width:${share.toFixed(1)}%"></i></div>
    <ul class="facts">${facts.map(([label, value]) => `<li><span>${clean(label)}</span><b>${clean(value)}</b></li>`).join("")}</ul>
  </section>`;
}

function rateLine(name, scored, allowed) {
  if (scored == null || allowed == null) return "";
  return `${clean(name)} has scored ${one(scored)} and allowed ${one(allowed)}.`;
}

function cfbWhy(game) {
  const lines = [
    rateLine(game.away_team, game.away_scored, game.away_allowed),
    rateLine(game.home_team, game.home_scored, game.home_allowed),
  ].filter(Boolean);
  const league = one(game.league_total || 53);
  lines.push(`Each side of the total is half of what that team has scored and half of what the other has allowed, then pulled toward ${league}. Four games is a short sample.`);
  lines.push("The clock stops on a first down, so those extra plays are already in the points. They are not added again.");
  if (neutralSite(game)) lines.push("Neutral site, so neither side gets the home bump.");
  return lines.join(" ");
}

function recentSheet(game, team, scores, side) {
  const nums = (scores || []).map(Number).filter((value) => Number.isFinite(value));
  if (!nums.length) {
    return `<div class="sheet ${side}"><header><h3>${clean(team)}</h3><p>No games yet</p></header></div>`;
  }
  const max = Math.max(45, ...nums);
  const bars = nums.map((value) => {
    const height = Math.max(8, (value / max) * 100);
    return `<div><i style="height:${height.toFixed(0)}%"></i><span>${Math.round(value)}</span></div>`;
  }).join("");
  return `<div class="sheet ${side}">
    <header><h3>${clean(team)}</h3><p>Oldest on the left</p></header>
    <div class="bars">${bars}</div>
  </div>`;
}

function renderCfb(game) {
  const bits = scoreBits(game, null);
  const shown = Number.isFinite(Number(game.pred_away_points)) && Number.isFinite(Number(game.pred_home_points))
    ? String(Math.round(Number(game.pred_away_points)) + Math.round(Number(game.pred_home_points)))
    : bits.ourTotal;
  app.innerHTML = `
    <div class="match college" style="${teamStyle(game)}">
    <button class="back" type="button" id="back">All games</button>
    ${scoreboard(game, null, bits)}
    <h3 class="section-label">Total</h3>
    <article class="ticket scores">
      <div class="ticket-top"><b>Our total</b><span class="price">${clean(shown)}</span></div>
      <p class="why">${cfbWhy(game)}</p>
      <p class="note">On its own. This feed has no posted spread or total, and the total is not paired with a side. No player props.</p>
    </article>
    <h3 class="section-label">Recent points</h3>
    <div class="team-grid">
      ${recentSheet(game, game.away_team, game.away_recent, "away")}
      ${recentSheet(game, game.home_team, game.home_recent, "home")}
    </div>
    </div>`;
  window.scrollTo(0, 0);
  document.querySelector("#back").addEventListener("click", () => {
    state.gameId = null;
    renderList();
    window.scrollTo(0, 0);
  });
}

function renderDetail(game) {
  const card = cardFor(game.game_id);
  const ticket = propsTicket(card);
  const scores = scoresTicket(card);
  const bits = scoreBits(game, card);
  const picked = pickedNames(card);
  app.innerHTML = `
    <div class="match" style="${teamStyle(game)}">
    <button class="back" type="button" id="back">All games</button>
    ${scoreboard(game, card, bits)}
    <h3 class="section-label">Parlays</h3>
    <div class="parlay-grid">
      ${yardCards(card)}
      ${ticketCard("Touchdowns", scores, "No touchdown parlay.", "Separate from the yard parlay.", "", "scores")}
    </div>
    <h3 class="section-label">Predicted yards</h3>
    <div class="team-grid">
      ${teamTable(game, game.away_team, picked)}
      ${teamTable(game, game.home_team, picked)}
    </div>
    <h3 class="section-label">Recent points</h3>
    <div class="panel" id="chart-trend">Drawing…</div>
    </div>`;
  window.scrollTo(0, 0);
  document.querySelector("#back").addEventListener("click", () => {
    state.gameId = null;
    renderList();
    window.scrollTo(0, 0);
  });
  loadCharts(game, card, ticket);
}

async function loadCharts(game, card, ticket) {
  const score = (card && card.score) || {};
  const params = new URLSearchParams({
    our_away: score.our_away != null ? score.our_away : game.pred_away_points,
    our_home: score.our_home != null ? score.our_home : game.pred_home_points,
  });
  if (score.book_away != null) params.set("book_away", score.book_away);
  if (score.book_home != null) params.set("book_home", score.book_home);
  params.set("away_color", paint(game.away_team).block);
  params.set("home_color", paint(game.home_team).block);
  try {
    const res = await fetch(`/api/nfl/charts/${game.game_id}?${params}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const charts = await res.json();
    const trendBox = document.querySelector("#chart-trend");
    if (trendBox) trendBox.innerHTML = charts.trend || "";
  } catch (err) {
    const trendBox = document.querySelector("#chart-trend");
    if (trendBox) trendBox.innerHTML = `<p class="error">Recent points did not load. ${clean(err.message)}</p>`;
  }
}

function render() {
  document.querySelectorAll(".sport").forEach((button) => {
    button.classList.toggle("on", button.dataset.sport === state.sport);
  });
  if (!state.data) {
    app.innerHTML = `<p class="quiet">Loading ${state.sport}...</p>`;
    return;
  }
  const game = games().find((item) => String(item.game_id) === String(state.gameId));
  if (game && state.sport === "NFL") renderDetail(game);
  else if (game && state.sport === "CFB") renderCfb(game);
  else renderList();
}

async function loadSport(sport) {
  const request = ++state.request;
  state.sport = sport;
  state.gameId = null;
  state.data = null;
  render();
  const url = sport === "NFL" ? "/api/nfl" : sport === "CFB" ? "/api/cfb" : "/api/mlb?days=2";
  try {
    const res = await fetch(url);
    if (request !== state.request) return;
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (request !== state.request || (data.sport && data.sport !== sport)) return;
    state.data = data;
  } catch (err) {
    if (request !== state.request) return;
    app.innerHTML = `<p class="error">Could not load ${clean(sport)}. ${clean(err.message)}</p>`;
    return;
  }
  render();
}

document.querySelector(".sports").addEventListener("click", (event) => {
  const button = event.target.closest("[data-sport]");
  if (!button) return;
  if (button.dataset.sport === state.sport) {
    state.gameId = null;
    render();
    return;
  }
  loadSport(button.dataset.sport);
});

document.querySelector(".brand").addEventListener("click", (event) => {
  event.preventDefault();
  if (state.sport !== "NFL") loadSport("NFL");
  else {
    state.gameId = null;
    render();
  }
});

app.addEventListener("click", (event) => {
  const button = event.target.closest("[data-game]");
  if (!button || (state.sport !== "NFL" && state.sport !== "CFB")) return;
  state.gameId = button.dataset.game;
  render();
});

document.querySelector("#refresh").addEventListener("click", async () => {
  document.querySelector("#refresh").disabled = true;
  try {
    await fetch(`/api/refresh?sport=${state.sport}`, { method: "POST" });
  } catch (_err) {
    /* The next load still shows what we have. */
  }
  document.querySelector("#refresh").disabled = false;
  loadSport(state.sport);
});

loadSport("NFL");
