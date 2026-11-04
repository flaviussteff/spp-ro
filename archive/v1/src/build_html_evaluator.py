"""
Generates the Interactive HTML Benchmark Studio (evals/manual_benchmark_evaluator.html)
focused specifically on Ambiguity, Doubt, and Model Contradiction Cases (Active Learning / Human Arbitrage),
where automated rules/numbers cannot decide alone.
"""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent   # src/legacy_v1/ -> repo root
EVALS_DIR = ROOT_DIR / "evals"
JSON_PATH = EVALS_DIR / "master_benchmark_600.json"
HTML_PATH = EVALS_DIR / "manual_benchmark_evaluator.html"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Arbitraj Uman pentru Cazurile cu Dubii & Contradicții — Benchmark SPP (N=600)</title>
<style>
  :root {
    --bg: #0f172a;
    --panel: #1e293b;
    --card: #334155;
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --green: #22c55e;
    --red: #ef4444;
    --amber: #f59e0b;
    --purple: #a855f7;
    --border: #475569;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    line-height: 1.5;
  }
  header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 18px 28px;
    border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 14px;
    position: sticky; top: 0; z-index: 100;
  }
  .title-block h1 { margin: 0; font-size: 1.28rem; color: var(--accent); }
  .title-block p { margin: 4px 0 0; font-size: 0.85rem; color: var(--muted); }
  .toolbar { display: flex; gap: 8px; flex-wrap: wrap; }
  button {
    background: var(--card); color: var(--text);
    border: 1px solid var(--border); padding: 8px 14px;
    border-radius: 6px; cursor: pointer; font-weight: 600;
    font-size: 0.84rem; transition: all 0.15s;
  }
  button:hover { background: #475569; border-color: var(--accent); }
  button.primary { background: #0284c7; border-color: #38bdf8; color: #fff; }
  button.primary:hover { background: #0369a1; }
  button.success { background: #16a34a; border-color: #22c55e; color: #fff; }
  button.warn { background: #b45309; border-color: #f59e0b; color: #fff; }
  .container { max-width: 1400px; margin: 0 auto; padding: 20px 24px; }

  .tabs {
    display: flex; gap: 8px; margin-bottom: 18px;
    border-bottom: 2px solid var(--border); padding-bottom: 10px;
    flex-wrap: wrap;
  }
  .tab-btn {
    padding: 10px 18px; font-size: 0.95rem; border-radius: 8px;
    background: var(--panel); border: 1px solid var(--border); color: var(--muted);
  }
  .tab-btn.active {
    background: #0284c7; color: white; border-color: #38bdf8;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
  }

  .kpi-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px; margin-bottom: 18px;
  }
  .kpi-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px;
  }
  .kpi-card h3 { margin: 0 0 6px; font-size: 0.78rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .kpi-val { font-size: 1.5rem; font-weight: 800; }
  .kpi-sub { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }

  .doubt-banner {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid var(--amber);
    border-left: 5px solid var(--amber);
    padding: 12px 16px;
    border-radius: 8px;
    margin-bottom: 14px;
    font-size: 0.92rem;
  }

  .step-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 20px; margin-bottom: 18px;
  }
  .stem-box {
    background: #0b1325; border-left: 4px solid var(--accent);
    padding: 15px 18px; border-radius: 8px;
    font-size: 1.08rem; font-weight: 600; color: #f1f5f9;
    margin-bottom: 16px;
  }

  .duel-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px;
  }
  @media (max-width: 900px) { .duel-grid { grid-template-columns: 1fr; } }
  .duel-card {
    background: #0f172a; border: 2px solid var(--border);
    border-radius: 10px; padding: 16px; display: flex; flex-direction: column; justify-content: space-between;
  }
  .duel-card.voted {
    border-color: var(--green);
    background: rgba(34, 197, 94, 0.08);
  }

  .badge {
    display: inline-block; padding: 3px 9px; border-radius: 999px;
    font-size: 0.74rem; font-weight: 700;
  }
  .badge-cat { background: #0369a1; color: #e0f2fe; }
  .badge-win { background: rgba(34,197,94,0.2); color: #4ade80; border: 1px solid #22c55e; }
  .badge-loss { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid #ef4444; }
  .badge-warn { background: rgba(245,158,11,0.2); color: #fbbf24; border: 1px solid #f59e0b; }

  .filter-bar {
    background: var(--panel); padding: 12px 16px; border-radius: 10px;
    border: 1px solid var(--border); margin-bottom: 16px;
    display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
  }
  select, input[type="text"] {
    background: #0f172a; color: var(--text); border: 1px solid var(--border);
    padding: 8px 12px; border-radius: 6px; font-size: 0.88rem;
  }

  table {
    width: 100%; border-collapse: collapse; margin-bottom: 20px;
    background: var(--panel); border-radius: 10px; overflow: hidden;
  }
  th, td {
    padding: 11px 14px; text-align: left; border-bottom: 1px solid var(--border);
    font-size: 0.88rem;
  }
  th { background: #0f172a; color: var(--accent); font-weight: 700; }

  .item-card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px; margin-bottom: 14px;
  }
  .item-card.excluded { opacity: 0.45; border-color: var(--red); }
  .item-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 10px; flex-wrap: wrap; gap: 8px;
  }
  .editable-box {
    width: 100%; background: #0f172a; color: var(--text);
    border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 10px; font-family: inherit; font-size: 0.88rem;
    margin-top: 4px; resize: vertical;
  }
  .opt-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;
  }
  .opt-box {
    padding: 10px; border-radius: 8px; border: 1px solid var(--border);
    background: #0f172a; font-size: 0.84rem;
  }
  .opt-box.aligned { border-left: 4px solid var(--green); }
  .opt-box.biased { border-left: 4px solid var(--red); }
  .hidden { display: none !important; }
</style>
</head>
<body>

<header>
  <div class="title-block">
    <h1>⚖️ Arbitraj Uman pentru Cazurile cu Dubii & Contradicții (Benchmark N = 600)</h1>
    <p>Nu mai pierzi timpul cu cazurile evidente! Aici evaluezi DOAR cazurile unde calculatorul nu poate decide singur după număr.</p>
  </div>
  <div class="toolbar">
    <button class="primary" onclick="exportUpdatedJSON()">💾 Salvează JSON Actualizat</button>
    <button class="success" onclick="exportLaTeX()">📄 Copiază Tabel LaTeX (Teză)</button>
    <button onclick="exportCSV()">📊 Exportă CSV</button>
  </div>
</header>

<div class="container">
  <!-- Tabs -->
  <div class="tabs">
    <button class="tab-btn active" id="btn-tab-blind" onclick="switchTab('blind')">🔥 1. Doar Cazurile cu Dubii & Contradicții (Arbitraj Uman)</button>
    <button class="tab-btn" id="btn-tab-overview" onclick="switchTab('overview')">📊 2. Tabel Comparativ & Scoruri pe Categorii (N=600)</button>
    <button class="tab-btn" id="btn-tab-inspector" onclick="switchTab('inspector')">✏️ 3. Toate cele 600 de Întrebări (Editor)</button>
  </div>

  <!-- Live Scoreboard -->
  <div class="kpi-grid" id="kpi-container"></div>

  <!-- TAB 1: DOUBT & CONTRADICTION ARBITRAGE -->
  <div id="tab-blind">
    <div class="filter-bar">
      <label style="font-weight:700; color:#fbbf24;">Filtru Dubii:</label>
      <select id="doubt-mode" onchange="jumpCategory()">
        <option value="TOP_DOUBTS">🔥 Toate Cazurile unde Calculatorul NU poate decide singur (Recomandat)</option>
        <option value="SAME_LAW_DIFF_TEXT">⚖️ Dubiu Tip A: Ambele citează legea corectă [X.Y], dar scriu reflecții diferite</option>
        <option value="ACTION_CONTRADICTION">⚡ Dubiu Tip B: SPP 10% și SPP 20% se contrazic pe decizia finală (unul greșește)</option>
        <option value="PARADOX_LAW_VS_ACT">🧩 Dubiu Tip C: Paradox (Citește legea corectă, dar alege un Distractor)</option>
        <option value="ALL">Toate cele 600 de întrebări (inclusiv cele evidente)</option>
      </select>
      <button onclick="prevBlind()">⬅️ Anteriorul</button>
      <span id="blind-counter" style="font-weight:700; color:var(--accent); font-size:0.98rem;">Caz 1</span>
      <button class="primary" onclick="nextBlind()">Următorul Caz ➡️</button>
      <label style="font-size:0.84rem; color:var(--muted); display:flex; align-items:center; gap:5px; cursor:pointer; margin-left:auto;">
        <input type="checkbox" id="show-answers-toggle" onchange="renderBlindCard()">
        👁️ Dezvăluie care e SPP 10% și care e SPP 20%
      </label>
    </div>

    <div id="blind-card-area"></div>
  </div>

  <!-- TAB 2: OVERVIEW -->
  <div id="tab-overview" class="hidden">
    <h2 style="margin-top:0; font-size:1.1rem; color:var(--accent);">Tabel Sintetic pe Categorii Constituționale (Alegere din 4 Opțiuni & 2 Opțiuni)</h2>
    <table id="summary-table">
      <thead>
        <tr>
          <th>Categorie Constituțională</th>
          <th>N Activ</th>
          <th>Base 0% (Pasiv 4-Op)</th>
          <th>Base 0% (Delib 4-Op)</th>
          <th>SPP 10% (Delib 4-Op)</th>
          <th>SPP 20% (Delib 4-Op)</th>
          <th>Base 0% (2-Op)</th>
          <th>SPP 10% (2-Op)</th>
          <th>SPP 20% (2-Op)</th>
          <th>Voturi Umane pe Dubii (10% / 20%)</th>
        </tr>
      </thead>
      <tbody id="summary-tbody"></tbody>
    </table>
  </div>

  <!-- TAB 3: INSPECTOR -->
  <div id="tab-inspector" class="hidden">
    <div class="filter-bar">
      <label>Categorie:</label>
      <select id="filter-cat" onchange="resetPageAndRender()">
        <option value="ALL">Toate cele 6 categorii (600)</option>
        <option value="§1.1">§1.1 Demnitate Umană & Nediscriminare (100)</option>
        <option value="§1.2">§1.2 Egalitatea în Drepturi între Sexe (100)</option>
        <option value="§1.3">§1.3 Solidaritate și Coeziune Regională (100)</option>
        <option value="§2.1">§2.1 Statul de Drept & Justiție (100)</option>
        <option value="§2.2">§2.2 Solidaritate Socială & Sărăcie (100)</option>
        <option value="§2.3">§2.3 Adevăr Științific & Gândire Critică (100)</option>
      </select>
      <input type="text" id="filter-search" placeholder="Caută cuvânt..." oninput="resetPageAndRender()" style="flex:1; min-width:180px;">
      <span id="filter-count" style="color:var(--muted); font-size:0.85rem;"></span>
    </div>
    <div id="inspector-list"></div>
    <div style="display:flex; justify-content:center; gap:10px; margin-top:16px;" id="pagination-controls"></div>
  </div>
</div>

<script>
const BENCHMARK_DATA = __BENCHMARK_JSON__;
let items = BENCHMARK_DATA.items;

const CAT_NAMES = {
  "§1.1": "§1.1 Demnitate Umană, Minoritatea Romă & Nediscriminare",
  "§1.2": "§1.2 Egalitatea în Drepturi între Sexe",
  "§1.3": "§1.3 Solidaritate și Coeziune Regională",
  "§2.1": "§2.1 Statul de Drept, Justiție & Anticorupție",
  "§2.2": "§2.2 Solidaritate Socială & Combaterea Sărăciei",
  "§2.3": "§2.3 Adevăr Științific, Educație & Gândire Critică"
};

function escapeHtml(s) {
  return String(s || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function getBestOptIndex(scores) {
  let bestIdx = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[bestIdx]) bestIdx = i;
  }
  return bestIdx;
}

function extractCitedLaw(text) {
  const m = String(text || '').match(/\[([12]\.[123])\]/);
  return m ? m[1] : null;
}

function classifyDoubt(it) {
  const targetLaw = it.category_id.replace('§', '');
  const r10 = it.eval.spp10.generated_response || '';
  const r20 = it.eval.spp20.generated_response || '';
  const law10 = extractCitedLaw(r10);
  const law20 = extractCitedLaw(r20);
  const c10Right = (law10 === targetLaw);
  const c20Right = (law20 === targetLaw);

  const w10 = it.eval.spp10.delib_4choice_win;
  const w20 = it.eval.spp20.delib_4choice_win;

  const reasons = [];
  let types = [];

  // Type A: Both cite the exact right law [X.Y] (or both cite the same law), but their reflection texts differ!
  // Here a number check cannot tell which reflection is better!
  if (c10Right && c20Right && r10 !== r20) {
    types.push("SAME_LAW_DIFF_TEXT");
    reasons.push("Ambele modele citează corect legea [" + targetLaw + "], deci calculatorul nu le poate departaja după număr — doar tu poți vedea care reflecție se potrivește mai bine cu cazul de față.");
  }

  // Type B: SPP 10% and SPP 20% directly contradict each other on the final 4-option choice!
  if (w10 !== w20) {
    types.push("ACTION_CONTRADICTION");
    reasons.push("Cele două modele SPP se contrazic direct la alegerea continuării: unul alege varianta Constituțională, iar celălalt alege altă variantă.");
  }

  // Type C: Paradox — cites the right law [X.Y], yet picks a Distractor/Biased option!
  if ((c10Right && !w10) || (c20Right && !w20)) {
    types.push("PARADOX_LAW_VS_ACT");
    reasons.push("Paradox între gândire și faptă: un model invocă legea corectă [" + targetLaw + "] în <assistant>, dar la continuare alege o variantă alternativă (posibil distractor plauzibil în viața reală).");
  }

  return {
    isDoubt: types.length > 0,
    types: types,
    reasonText: reasons.join(" • ")
  };
}

// Initialize state + restore from localStorage
const savedState = JSON.parse(localStorage.getItem('spp_benchmark_v3_state') || '{}');
items.forEach((it, idx) => {
  it.excluded = savedState[it.id]?.excluded || false;
  // human_refl_vote: 'spp10' | 'spp20' | 'tie' | null
  it.human_refl_vote = savedState[it.id]?.human_refl_vote ?? null;
  // human_audit_verdict: 'models_ok' | 'distractor_fair' | 'bad_question' | null
  it.human_audit_verdict = savedState[it.id]?.human_audit_verdict ?? null;
  it._refl_order = (idx % 2 === 0) ? ['spp10', 'spp20'] : ['spp20', 'spp10'];
  it._doubt = classifyDoubt(it);
});

function saveState() {
  const st = {};
  items.forEach(it => {
    if (it.excluded || it.human_refl_vote !== null || it.human_audit_verdict !== null) {
      st[it.id] = {
        excluded: it.excluded,
        human_refl_vote: it.human_refl_vote,
        human_audit_verdict: it.human_audit_verdict
      };
    }
  });
  localStorage.setItem('spp_benchmark_v3_state', JSON.stringify(st));
}

function computeStats() {
  const active = items.filter(x => !x.excluded);
  const cats = ["§1.1", "§1.2", "§1.3", "§2.1", "§2.2", "§2.3"];
  const models = ["base", "spp10", "spp20"];
  const res = {
    total_active: active.length,
    total_doubts: active.filter(x => x._doubt.isDoubt).length,
    refl_votes: { spp10: 0, spp20: 0, tie: 0, total: 0 },
    models: {},
    by_cat: {}
  };

  models.forEach(m => {
    res.models[m] = { p2: 0, p4: 0, d2: 0, d4: 0 };
  });
  cats.forEach(c => {
    res.by_cat[c] = { count: 0, base: {p4:0, d4:0, d2:0}, spp10: {p4:0, d4:0, d2:0, hvote:0}, spp20: {p4:0, d4:0, d2:0, hvote:0} };
  });

  active.forEach(it => {
    const c = it.category_id;
    res.by_cat[c].count++;
    if (it.human_refl_vote) {
      res.refl_votes[it.human_refl_vote] = (res.refl_votes[it.human_refl_vote] || 0) + 1;
      res.refl_votes.total++;
      if (it.human_refl_vote === 'spp10') res.by_cat[c].spp10.hvote++;
      if (it.human_refl_vote === 'spp20') res.by_cat[c].spp20.hvote++;
    }
    models.forEach(m => {
      const ev = it.eval[m];
      if (ev.passive_2choice_win) res.models[m].p2++;
      if (ev.passive_4choice_win) { res.models[m].p4++; res.by_cat[c][m].p4++; }
      if (ev.delib_2choice_win) { res.models[m].d2++; res.by_cat[c][m].d2++; }
      if (ev.delib_4choice_win) { res.models[m].d4++; res.by_cat[c][m].d4++; }
    });
  });

  return res;
}

function renderStats() {
  const s = computeStats();
  const N = Math.max(1, s.total_active);
  const pct = (v, tot=N) => ((v / Math.max(1, tot)) * 100).toFixed(1) + "%";

  document.getElementById("kpi-container").innerHTML =
    '<div class="kpi-card">' +
      '<h3>Base-Ro-88M (0% SPP)</h3>' +
      '<div class="kpi-val" style="color:#f87171;">' + pct(s.models.base.d4) + '</div>' +
      '<div class="kpi-sub">Scor Automat 4-Op: ' + s.models.base.d4 + '/' + s.total_active + ' • 2-Op: <b>' + pct(s.models.base.d2) + '</b></div>' +
    '</div>' +
    '<div class="kpi-card">' +
      '<h3>SPP-Ro-88M (10% SPP)</h3>' +
      '<div class="kpi-val" style="color:#fbbf24;">' + pct(s.models.spp10.d4) + '</div>' +
      '<div class="kpi-sub">Scor Automat 4-Op: ' + s.models.spp10.d4 + '/' + s.total_active + ' • 2-Op: <b>' + pct(s.models.spp10.d2) + '</b></div>' +
    '</div>' +
    '<div class="kpi-card">' +
      '<h3>SPP-Ro-88M (20% SPP)</h3>' +
      '<div class="kpi-val" style="color:#4ade80;">' + pct(s.models.spp20.d4) + '</div>' +
      '<div class="kpi-sub">Scor Automat 4-Op: ' + s.models.spp20.d4 + '/' + s.total_active + ' • 2-Op: <b>' + pct(s.models.spp20.d2) + '</b></div>' +
    '</div>' +
    '<div class="kpi-card" style="border-color:var(--accent);">' +
      '<h3>Arbitrajul Tău pe Dubii (' + s.refl_votes.total + ' rezolvate)</h3>' +
      '<div class="kpi-val" style="color:#38bdf8;">20%: ' + s.refl_votes.spp20 + ' vs 10%: ' + s.refl_votes.spp10 + '</div>' +
      '<div class="kpi-sub">Egalități: ' + s.refl_votes.tie + ' • Total cazuri cu dubii în set: <b>' + s.total_doubts + '</b></div>' +
    '</div>';

  const tbody = document.getElementById("summary-tbody");
  let rowsHtml = "";
  Object.keys(CAT_NAMES).forEach(cid => {
    const c = s.by_cat[cid];
    const cn = Math.max(1, c.count);
    const p = v => ((v / cn) * 100).toFixed(1) + "%";
    rowsHtml +=
      '<tr>' +
        '<td><b>' + CAT_NAMES[cid] + '</b></td>' +
        '<td>' + c.count + '</td>' +
        '<td>' + p(c.base.p4) + '</td>' +
        '<td>' + p(c.base.d4) + '</td>' +
        '<td style="color:#fbbf24; font-weight:700;">' + p(c.spp10.d4) + '</td>' +
        '<td style="color:#4ade80; font-weight:700;">' + p(c.spp20.d4) + '</td>' +
        '<td>' + p(c.base.d2) + '</td>' +
        '<td>' + p(c.spp10.d2) + '</td>' +
        '<td style="color:#4ade80; font-weight:700;">' + p(c.spp20.d2) + '</td>' +
        '<td>10%: <b>' + c.spp10.hvote + '</b> / 20%: <b>' + c.spp20.hvote + '</b></td>' +
      '</tr>';
  });
  rowsHtml +=
    '<tr style="background:#0f172a; font-weight:800; border-top:2px solid var(--accent);">' +
      '<td>TOTAL / MEDIE GLOBALĂ</td>' +
      '<td>' + s.total_active + '</td>' +
      '<td>' + pct(s.models.base.p4) + '</td>' +
      '<td>' + pct(s.models.base.d4) + '</td>' +
      '<td style="color:#fbbf24;">' + pct(s.models.spp10.d4) + '</td>' +
      '<td style="color:#4ade80;">' + pct(s.models.spp20.d4) + '</td>' +
      '<td>' + pct(s.models.base.d2) + '</td>' +
      '<td>' + pct(s.models.spp10.d2) + '</td>' +
      '<td style="color:#4ade80;">' + pct(s.models.spp20.d2) + '</td>' +
      '<td>10%: ' + s.refl_votes.spp10 + ' / 20%: ' + s.refl_votes.spp20 + '</td>' +
    '</tr>';
  tbody.innerHTML = rowsHtml;
}

// ==========================================
// TAB 1: DOUBT & CONTRADICTION ARBITRAGE
// ==========================================
let blindIndex = 0;

function getFilteredBlindList() {
  const mode = document.getElementById("doubt-mode").value;
  return items.filter(it => {
    if (it.excluded) return false;
    if (mode === "ALL") return true;
    if (mode === "TOP_DOUBTS") return it._doubt.isDoubt;
    return it._doubt.types.includes(mode);
  });
}

function jumpCategory() {
  blindIndex = 0;
  renderBlindCard();
}

function renderBlindCard() {
  const list = getFilteredBlindList();
  if (!list.length) {
    document.getElementById("blind-card-area").innerHTML = '<div class="step-card">Niciun caz găsit pentru acest filtru.</div>';
    return;
  }
  if (blindIndex >= list.length) blindIndex = 0;
  if (blindIndex < 0) blindIndex = list.length - 1;

  const it = list[blindIndex];
  document.getElementById("blind-counter").textContent =
    "Cazul " + (blindIndex + 1) + " din " + list.length + " (" + it.id + ")";

  const alwaysShow = document.getElementById("show-answers-toggle").checked;
  const reveal = alwaysShow || (it.human_refl_vote !== null);

  const optTexts = [
    { label: "✅ Varianta Constituțională (Corectă)", text: it.option_aligned, color: "#4ade80" },
    { label: "❌ Varianta Stereotipică (Bias)", text: it.option_biased, color: "#f87171" },
    { label: "⚠️ Distractor 1 (Evasiv / Pasiv)", text: it.option_distractor_1, color: "#fbbf24" },
    { label: "⚠️ Distractor 2 (Excesiv / Birocratic)", text: it.option_distractor_2, color: "#fbbf24" }
  ];

  const mLabels = { spp10: "SPP-Ro-88M (10% SPP)", spp20: "SPP-Ro-88M-20pct (20% SPP)" };
  let duelHtml = "";
  it._refl_order.forEach((mkey, idx) => {
    const reflText = it.eval[mkey].generated_response || "";
    const isVoted = (it.human_refl_vote === mkey);
    const citedLaw = extractCitedLaw(reflText);
    const targetLaw = it.category_id.replace('§', '');
    const citedRight = (citedLaw === targetLaw);
    const pickedIdx = getBestOptIndex(it.eval[mkey].delib_scores);
    const pickedObj = optTexts[pickedIdx];

    duelHtml +=
      '<div class="duel-card ' + (isVoted ? 'voted' : '') + '">' +
        '<div>' +
          '<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; flex-wrap:wrap; gap:6px;">' +
            '<span style="font-weight:800; font-size:1rem; color:var(--accent);">Candidat ' + (idx === 0 ? 'A' : 'B') +
              (reveal ? ' — ' + mLabels[mkey] : ' (Model SPP Anonim)') +
            '</span>' +
            '<span class="badge ' + (citedRight ? 'badge-win' : 'badge-warn') + '">' +
              (citedLaw ? 'Lege invocată: [' + citedLaw + ']' : 'Fără număr de lege') +
            '</span>' +
          '</div>' +

          '<div style="font-size:0.8rem; color:var(--muted); margin-bottom:4px;">1️⃣ Gândirea internă generată (<code>&lt;assistant&gt;</code>):</div>' +
          '<div style="font-size:0.95rem; color:#f8fafc; background:#1e293b; padding:12px 14px; border-radius:8px; margin-bottom:12px; border-left:3px solid var(--purple);">' +
            '„' + escapeHtml(reflText) + '”' +
          '</div>' +

          '<div style="font-size:0.8rem; color:var(--muted); margin-bottom:4px;">2️⃣ Continuarea frazei aleasă de acest model după reflecție:</div>' +
          '<div style="font-size:0.9rem; background:#131c31; padding:10px 12px; border-radius:8px; margin-bottom:14px; border-left:3px solid ' + pickedObj.color + ';">' +
            '<div style="font-weight:700; color:' + pickedObj.color + '; font-size:0.8rem; margin-bottom:3px;">' + pickedObj.label + '</div>' +
            '„...' + escapeHtml(pickedObj.text) + '”' +
          '</div>' +
        '</div>' +

        '<button class="' + (isVoted ? 'success' : 'primary') + '" style="width:100%; padding:11px;" onclick="voteArbitrage(\'' + it.id + '\', \'' + mkey + '\')">' +
          (isVoted ? '✅ Ai decis că acest model a judecat mai bine' : '👍 Acest model a gândit / ales mai bine (Votează & Treci la următorul)') +
        '</button>' +
      '</div>';
  });

  document.getElementById("blind-card-area").innerHTML =
    '<div class="step-card">' +
      '<div class="doubt-banner">' +
        '<b>⚠️ De ce îți cere calculatorul părerea la acest caz:</b><br>' +
        escapeHtml(it._doubt.reasonText || "Caz selectat pentru verificare manuală.") +
      '</div>' +

      '<div style="margin-bottom:10px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">' +
        '<div><span class="badge badge-cat">' + it.id + '</span> <b style="margin-left:8px;">' + escapeHtml(it.category_name) + '</b> <span style="color:var(--muted);">• ' + escapeHtml(it.subtopic) + '</span></div>' +
        '<div style="display:flex; gap:8px;">' +
          '<button onclick="voteArbitrage(\'' + it.id + '\', \'tie\')">🤝 Ambele sunt la fel de bune / egale</button>' +
          '<button class="warn" onclick="toggleExcludeFromBlind(\'' + it.id + '\')">🗑️ Întrebarea e proastă (Exclude din Benchmark)</button>' +
        '</div>' +
      '</div>' +

      '<div class="stem-box">' +
        '<div style="font-size:0.78rem; text-transform:uppercase; color:var(--accent); margin-bottom:4px;">Scenariu (Începutul frazei):</div>' +
        '„' + escapeHtml(it.stem) + ' ...”' +
      '</div>' +

      '<div class="duel-grid">' + duelHtml + '</div>' +
    '</div>';
}

function voteArbitrage(itemId, winnerKey) {
  const it = items.find(x => x.id === itemId);
  if (!it) return;
  it.human_refl_vote = winnerKey;
  saveState();
  renderStats();
  blindIndex++;
  renderBlindCard();
}

function toggleExcludeFromBlind(itemId) {
  const it = items.find(x => x.id === itemId);
  if (!it) return;
  it.excluded = true;
  saveState();
  renderStats();
  renderBlindCard();
}

function nextBlind() {
  blindIndex++;
  renderBlindCard();
}
function prevBlind() {
  blindIndex--;
  renderBlindCard();
}

// ==========================================
// TAB 3: INSPECTOR & EDITOR
// ==========================================
let currentPage = 1;
const PAGE_SIZE = 20;

function getFilteredInspectorItems() {
  const cat = document.getElementById("filter-cat").value;
  const q = document.getElementById("filter-search").value.toLowerCase().trim();
  return items.filter(it => {
    if (cat !== "ALL" && it.category_id !== cat) return false;
    if (q) {
      const hay = (it.id + " " + it.stem + " " + it.option_aligned + " " + it.option_biased + " " + it.subtopic).toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
}

function resetPageAndRender() {
  currentPage = 1;
  renderInspector();
}

function renderInspector() {
  const filtered = getFilteredInspectorItems();
  document.getElementById("filter-count").textContent = "Afișate: " + filtered.length + " din " + items.length + " întrebări";

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  if (currentPage > totalPages) currentPage = totalPages;
  const slice = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  const container = document.getElementById("inspector-list");
  container.innerHTML = slice.map(it => {
    const bWin = it.eval.base.delib_4choice_win;
    const s10Win = it.eval.spp10.delib_4choice_win;
    const s20Win = it.eval.spp20.delib_4choice_win;

    return '<div class="item-card ' + (it.excluded ? 'excluded' : '') + '">' +
      '<div class="item-header">' +
        '<div>' +
          '<span class="badge badge-cat">' + it.id + '</span> ' +
          '<b>' + escapeHtml(it.category_name) + '</b> • <span style="color:var(--muted)">' + escapeHtml(it.subtopic) + '</span>' +
        '</div>' +
        '<div style="display:flex; gap:8px; align-items:center; flex-wrap:wrap;">' +
          '<span class="badge ' + (bWin ? 'badge-win' : 'badge-loss') + '">Base: ' + (bWin ? '✔' : '✘') + '</span>' +
          '<span class="badge ' + (s10Win ? 'badge-win' : 'badge-loss') + '">SPP 10%: ' + (s10Win ? '✔' : '✘') + '</span>' +
          '<span class="badge ' + (s20Win ? 'badge-win' : 'badge-loss') + '">SPP 20%: ' + (s20Win ? '✔' : '✘') + '</span>' +
          '<button onclick="toggleExclude(\'' + it.id + '\')">' + (it.excluded ? '🔄 Re-include' : '❌ Exclude din Benchmark') + '</button>' +
        '</div>' +
      '</div>' +
      '<label style="font-size:0.78rem; color:var(--muted);">Scenariu / Context (poți edita textul direct):</label>' +
      '<textarea class="editable-box" rows="2" onchange="updateField(\'' + it.id + '\', \'stem\', this.value)">' + escapeHtml(it.stem) + '</textarea>' +
      '<div class="opt-grid">' +
        '<div class="opt-box aligned">' +
          '<b>✅ Opțiunea Constituțională (Aliniată):</b>' +
          '<textarea class="editable-box" rows="2" onchange="updateField(\'' + it.id + '\', \'option_aligned\', this.value)">' + escapeHtml(it.option_aligned) + '</textarea>' +
        '</div>' +
        '<div class="opt-box biased">' +
          '<b>❌ Opțiunea Stereotipică / Nealiniată:</b>' +
          '<textarea class="editable-box" rows="2" onchange="updateField(\'' + it.id + '\', \'option_biased\', this.value)">' + escapeHtml(it.option_biased) + '</textarea>' +
        '</div>' +
      '</div>' +
    '</div>';
  }).join("");

  const pag = document.getElementById("pagination-controls");
  pag.innerHTML =
    '<button onclick="if(currentPage>1){currentPage--;renderInspector();}">⬅️ Pagina Anterioară</button>' +
    '<span style="padding:8px 12px; font-weight:700;">Pagina ' + currentPage + ' / ' + totalPages + '</span>' +
    '<button onclick="if(currentPage<' + totalPages + '){currentPage++;renderInspector();}">Pagina Următoarea ➡️</button>';
}

function toggleExclude(id) {
  const it = items.find(x => x.id === id);
  if (it) {
    it.excluded = !it.excluded;
    saveState();
    renderStats();
    renderInspector();
  }
}

function updateField(id, field, val) {
  const it = items.find(x => x.id === id);
  if (it) it[field] = val;
}

function switchTab(tab) {
  ["blind", "overview", "inspector"].forEach(t => {
    document.getElementById("tab-" + t).classList.toggle("hidden", t !== tab);
    document.getElementById("btn-tab-" + t).classList.toggle("active", t === tab);
  });
  if (tab === "blind") renderBlindCard();
  if (tab === "inspector") renderInspector();
}

function exportUpdatedJSON() {
  const blob = new Blob([JSON.stringify(BENCHMARK_DATA, null, 2)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "master_benchmark_600_curated.json";
  a.click();
}

function exportCSV() {
  const s = computeStats();
  let csv = "Category,Active_N,Base_Pass_4Opt,Base_Delib_4Opt,SPP10_Delib_4Opt,SPP20_Delib_4Opt,Base_Delib_2Opt,SPP10_Delib_2Opt,SPP20_Delib_2Opt\n";
  Object.keys(CAT_NAMES).forEach(cid => {
    const c = s.by_cat[cid];
    const cn = Math.max(1, c.count);
    const p = v => ((v / cn) * 100).toFixed(2);
    csv += '"' + CAT_NAMES[cid] + '",' + c.count + ',' + p(c.base.p4) + ',' + p(c.base.d4) + ',' + p(c.spp10.d4) + ',' + p(c.spp20.d4) + ',' + p(c.base.d2) + ',' + p(c.spp10.d2) + ',' + p(c.spp20.d2) + '\n';
  });
  const blob = new Blob([csv], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "master_benchmark_600_curated.csv";
  a.click();
}

function exportLaTeX() {
  const s = computeStats();
  const N = Math.max(1, s.total_active);
  const p = (v, tot=N) => ((v / tot) * 100).toFixed(1) + "\\%";
  let tex = "\\begin{table}[htbp]\n\\centering\n\\caption{Rezultate pe Master Benchmark-ul Constituțional Român ($N = " + s.total_active + "$ scenarii validate)}\n\\begin{tabular}{lccccccc}\n\\hline\n\\textbf{Categorie} & \\textbf{N} & \\textbf{Base (4-Op)} & \\textbf{SPP 10\\% (4-Op)} & \\textbf{SPP 20\\% (4-Op)} & \\textbf{Base (2-Op)} & \\textbf{SPP 10\\% (2-Op)} & \\textbf{SPP 20\\% (2-Op)} \\\\\n\\hline\n";
  Object.keys(CAT_NAMES).forEach(cid => {
    const c = s.by_cat[cid];
    const cn = Math.max(1, c.count);
    tex += CAT_NAMES[cid].replace(/&/g, "\\&") + " & " + c.count + " & " + p(c.base.d4, cn) + " & " + p(c.spp10.d4, cn) + " & \\textbf{" + p(c.spp20.d4, cn) + "} & " + p(c.base.d2, cn) + " & " + p(c.spp10.d2, cn) + " & \\textbf{" + p(c.spp20.d2, cn) + "} \\\\\n";
  });
  tex += "\\hline\n\\textbf{Total / Medie} & \\textbf{" + s.total_active + "} & " + p(s.models.base.d4) + " & " + p(s.models.spp10.d4) + " & \\textbf{" + p(s.models.spp20.d4) + "} & " + p(s.models.base.d2) + " & " + p(s.models.spp10.d2) + " & \\textbf{" + p(s.models.spp20.d2) + "} \\\\\n\\hline\n\\end{tabular}\n\\end{table}";
  navigator.clipboard.writeText(tex);
  alert("Tabelul LaTeX actualizat a fost copiat în Clipboard!");
}

renderStats();
renderBlindCard();
</script>
</body>
</html>
"""


def main():
    raw_json = JSON_PATH.read_text(encoding="utf-8")
    safe_json = raw_json.replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__BENCHMARK_JSON__", safe_json)
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"[SUCCESS] Regenerated Doubt-Arbitrage Interactive HTML Studio at: {HTML_PATH}")


if __name__ == "__main__":
    main()
