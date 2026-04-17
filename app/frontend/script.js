"use strict";
const API_BASE = "";

// ── Helpers ──────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ── Tab switching ─────────────────────────────────────────────────────
const tabBtns   = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");
let feedLoaded  = false;

tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.tab;
    tabBtns.forEach(b   => b.classList.toggle("active", b.dataset.tab === target));
    tabPanels.forEach(p => p.classList.toggle("active", p.id === "tab-" + target));
    if (target === "feed" && !feedLoaded) { loadFeed(); feedLoaded = true; }
  });
});

// ══════════════════════════════════════════════════════════════
// DETECTOR — TAB 1
// ══════════════════════════════════════════════════════════════
const elHeadline    = document.getElementById("headline");
const elArticle     = document.getElementById("article");
const elCharCount   = document.getElementById("char-count");
const elWordCount   = document.getElementById("word-count");
const elInputError  = document.getElementById("input-error");
const btnPredict    = document.getElementById("btn-predict");
const btnClear      = document.getElementById("btn-clear");
const btnExample    = document.getElementById("btn-load-example");
const btnRetry      = document.getElementById("btn-retry");
const stateIdle     = document.getElementById("state-idle");
const stateLoading  = document.getElementById("state-loading");
const stateResult   = document.getElementById("state-result");
const stateError    = document.getElementById("state-error");
const errorMessage  = document.getElementById("error-message");
const verdictBanner = document.getElementById("verdict-banner");
const verdictIcon   = document.getElementById("verdict-icon");
const verdictLabel  = document.getElementById("verdict-label");
const verdictSub    = document.getElementById("verdict-sub");
const verdictBadge  = document.getElementById("verdict-badge");
const metricConf    = document.getElementById("metric-confidence");
const metricFake    = document.getElementById("metric-fake");
const metricReal    = document.getElementById("metric-real");
const metricTokens  = document.getElementById("metric-tokens");
const probBarFake   = document.getElementById("prob-bar-fake");
const probBarReal   = document.getElementById("prob-bar-real");
const resultWarning = document.getElementById("result-warning");
const chipsFake     = document.getElementById("chips-fake");
const chipsReal     = document.getElementById("chips-real");
const articleHL     = document.getElementById("article-highlight");
const latencyDisp   = document.getElementById("latency-display");

function showState(name) {
  stateIdle.hidden    = name !== "idle";
  stateLoading.hidden = name !== "loading";
  stateResult.hidden  = name !== "result";
  stateError.hidden   = name !== "error";
}
showState("idle");

function updateCounter() {
  const text = (elArticle.value + " " + elHeadline.value).trim();
  elCharCount.textContent = text.length.toLocaleString();
  elWordCount.textContent = text.split(/\s+/).filter(Boolean).length.toLocaleString();
}
elArticle.addEventListener("input", updateCounter);
elHeadline.addEventListener("input", updateCounter);

function showInputError(msg) { elInputError.textContent = msg; elInputError.classList.add("visible"); }
function clearInputError()   { elInputError.textContent = ""; elInputError.classList.remove("visible"); }

function validateInput() {
  const h = elHeadline.value.trim(), b = elArticle.value.trim();
  if (!h && !b) { showInputError("Please enter a headline or paste an article."); return false; }
  if ((h + " " + b).trim().length < 20) { showInputError("Text too short — provide more context."); return false; }
  clearInputError(); return true;
}

async function callPredict(headline, text) {
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ headline, text }),
  });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || `Server error (${res.status})`);
  return data;
}

function renderResult(data, rawText) {
  const isFake = data.label === "Fake";

  verdictBanner.className  = `verdict-banner ${isFake ? "is-fake" : "is-real"}`;
  verdictIcon.textContent  = isFake ? "✕" : "✓";
  verdictLabel.textContent = isFake ? "Likely Fake News" : "Likely Authentic";
  verdictSub.textContent   = isFake
    ? "Linguistic patterns suggest this may be misinformation."
    : "Linguistic patterns are consistent with authentic reporting.";
  verdictBadge.textContent = `${data.confidence.toFixed(1)}%`;

  metricConf.textContent   = `${data.confidence.toFixed(1)}%`;
  metricFake.textContent   = `${data.prob_fake.toFixed(1)}%`;
  metricReal.textContent   = `${data.prob_real.toFixed(1)}%`;
  metricTokens.textContent = data.clean_token_count;

  probBarFake.style.width = `${data.prob_fake}%`;
  probBarReal.style.width = `${data.prob_real}%`;

  if (data.warning) {
    resultWarning.textContent = data.warning;
    resultWarning.hidden = false;
  } else {
    resultWarning.hidden = true;
  }

  renderChips(chipsFake, data.top_fake_words, "fake");
  renderChips(chipsReal, data.top_real_words, "real");
  articleHL.innerHTML = highlightText(rawText, data.all_keywords);
  latencyDisp.textContent = data.latency_ms ? `Completed in ${data.latency_ms} ms` : "";

  showState("result");
}

function renderChips(container, words, direction) {
  container.innerHTML = "";
  if (!words || !words.length) {
    const s = document.createElement("span");
    s.style.cssText = "font-family:var(--mono);font-size:11px;color:var(--text-muted)";
    s.textContent = "none detected";
    container.appendChild(s);
    return;
  }
  words.forEach(({ word, score }) => {
    const chip = document.createElement("span");
    chip.className   = `chip chip-${direction}`;
    chip.textContent = word;
    chip.title       = `Score: ${score > 0 ? "+" : ""}${score.toFixed(3)}`;
    container.appendChild(chip);
  });
}

function highlightText(text, keywords) {
  if (!text || !keywords || !keywords.length) return escapeHtml(text || "");
  const lookup = new Map();
  keywords.forEach(({ word, direction }) => lookup.set(word.toLowerCase(), direction));
  return text.split(/(\s+)/).map(token => {
    const clean = token.toLowerCase().replace(/[^a-z_]/g, "");
    if (lookup.has(clean)) return `<mark class="${lookup.get(clean)}-word">${escapeHtml(token)}</mark>`;
    return escapeHtml(token);
  }).join("");
}

async function runPrediction() {
  if (!validateInput()) return;
  const headline = elHeadline.value.trim(), text = elArticle.value.trim();
  showState("loading");
  btnPredict.disabled = true;
  try {
    const data = await callPredict(headline, text);
    renderResult(data, `${headline} ${text}`.trim());
  } catch (err) {
    errorMessage.textContent = err.message || "An unexpected error occurred.";
    showState("error");
  } finally {
    btnPredict.disabled = false;
  }
}

function clearAll() {
  elHeadline.value = "";
  elArticle.value  = "";
  clearInputError();
  updateCounter();
  showState("idle");
}

// Built-in fallback examples (used if /api/examples fails)
const FALLBACK_EXAMPLES = [
  {
    headline: "DEEP STATE CAUGHT destroying evidence!!!",
    text: "BREAKING!!! The deep state was CAUGHT destroying evidence!!! Share this before it gets DELETED!! Anonymous sources confirm the mainstream media is hiding the truth from the people. They cannot silence us anymore!! WAKE UP AMERICA!!!",
  },
  {
    headline: "Senate confirms new Secretary of State",
    text: "The Senate confirmed the new Secretary of State by a bipartisan vote of 72 to 28, officials announced on Monday. The nominee received support from both Republican and Democratic lawmakers following weeks of committee hearings and background review.",
  },
];
let exampleIndex = 0;

async function loadExample() {
  try {
    const res  = await fetch(`${API_BASE}/api/examples`);
    const data = await res.json();
    if (data.examples && data.examples.length) {
      const ex = data.examples[exampleIndex % data.examples.length];
      elHeadline.value = ex.headline || "";
      elArticle.value  = ex.text || "";
      exampleIndex++;
      updateCounter(); clearInputError(); showState("idle");
      return;
    }
  } catch (_) {}
  const ex = FALLBACK_EXAMPLES[exampleIndex % FALLBACK_EXAMPLES.length];
  elHeadline.value = ex.headline;
  elArticle.value  = ex.text;
  exampleIndex++;
  updateCounter(); clearInputError(); showState("idle");
}

btnPredict.addEventListener("click", runPrediction);
btnClear.addEventListener("click", clearAll);
btnExample.addEventListener("click", loadExample);
btnRetry.addEventListener("click", () => showState("idle"));

document.addEventListener("keydown", e => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") runPrediction();
});

// Auto-predict on paste into body if headline is also filled
elArticle.addEventListener("paste", () => {
  setTimeout(() => {
    if (elHeadline.value.trim() && elArticle.value.trim().length > 50) runPrediction();
  }, 300);
});

// ══════════════════════════════════════════════════════════════
// LIVE FEED — TAB 2
// ══════════════════════════════════════════════════════════════
const feedLoading   = document.getElementById("feed-loading");
const feedGrid      = document.getElementById("feed-grid");
const feedStats     = document.getElementById("feed-stats");
const feedFilterBar = document.getElementById("feed-filter-bar");
const feedTimestamp = document.getElementById("feed-timestamp");
const fstatTotal    = document.getElementById("fstat-total");
const fstatFake     = document.getElementById("fstat-fake");
const fstatReal     = document.getElementById("fstat-real");
const fstatConf     = document.getElementById("fstat-conf");
const feedCountdown = document.getElementById("feed-countdown");
const countdownText = document.getElementById("countdown-text");

let feedFilterVerdict = "all";
let feedFilterSources = new Set(["BBC News", "Reuters", "AP News", "Al Jazeera"]);
let countdownSecs     = 300;
let countdownInterval = null;

function startCountdown() {
  if (countdownInterval) clearInterval(countdownInterval);
  countdownSecs = 300;
  countdownInterval = setInterval(() => {
    countdownSecs--;
    if (countdownSecs <= 0) {
      countdownSecs = 300;
      if (document.getElementById("tab-feed").classList.contains("active")) {
        loadFeed(true);
      }
    }
    const m = Math.floor(countdownSecs / 60);
    const s = countdownSecs % 60;
    countdownText.textContent = `${m}:${String(s).padStart(2, "0")}`;
  }, 1000);
}

function animateCount(el, targetStr) {
  const dur      = 700;
  const start    = Date.now();
  const isPercent = String(targetStr).includes("%");
  const target   = parseFloat(String(targetStr).replace("%", ""));
  function frame() {
    const t    = Math.min(1, (Date.now() - start) / dur);
    const ease = 1 - Math.pow(1 - t, 3);
    const val  = Math.round(target * ease);
    el.textContent = isPercent ? val + "%" : val;
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}

function applyFilters() {
  feedGrid.querySelectorAll(".feed-card").forEach(card => {
    const verdict = card.dataset.verdict;
    const conf    = parseFloat(card.dataset.conf);
    const source  = card.dataset.source;
    let show = true;
    if (feedFilterVerdict === "real" && verdict !== "real") show = false;
    if (feedFilterVerdict === "fake" && verdict !== "fake") show = false;
    if (feedFilterVerdict === "low"  && conf >= 65)         show = false;
    if (!feedFilterSources.has(source))                     show = false;
    card.classList.toggle("hidden-card", !show);
  });
}

function setupFilters() {
  document.querySelectorAll(".filter-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      feedFilterVerdict = btn.dataset.verdict;
      applyFilters();
    });
  });
  document.querySelectorAll(".filter-source-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const src = btn.dataset.source;
      if (feedFilterSources.has(src)) {
        if (feedFilterSources.size > 1) { feedFilterSources.delete(src); btn.classList.remove("active"); }
      } else {
        feedFilterSources.add(src); btn.classList.add("active");
      }
      applyFilters();
    });
  });
}

async function loadFeed(forceRefresh = false) {
  feedCountdown.classList.add("refreshing");
  countdownText.textContent = "Loading";
  feedLoading.style.display = "flex";
  feedGrid.style.display    = "none";
  feedStats.style.display   = "none";
  feedFilterBar.style.display = "none";

  try {
    const url  = forceRefresh ? `${API_BASE}/api/feed?refresh=1` : `${API_BASE}/api/feed`;
    const res  = await fetch(url);
    const data = await res.json();

    feedLoading.style.display = "none";
    feedCountdown.classList.remove("refreshing");
    startCountdown();

    if (!data.feedparser_available) {
      feedGrid.innerHTML = `<div class="feed-no-feedparser"><strong>feedparser not installed.</strong> Run <code>pip install feedparser</code> then restart the server.</div>`;
      feedGrid.style.display = "block";
      return;
    }
    if (data.error && !(data.articles || []).length) {
      feedGrid.innerHTML = `<div class="feed-state-empty">${escapeHtml(data.error)}</div>`;
      feedGrid.style.display = "block";
      return;
    }

    const articles = data.articles || [];
    if (!articles.length) {
      feedGrid.innerHTML = `<div class="feed-state-empty">No articles fetched. RSS feeds may be temporarily unavailable.</div>`;
      feedGrid.style.display = "block";
      return;
    }

    // Stats
    const nFake   = articles.filter(a => a.label === "Fake").length;
    const nReal   = articles.length - nFake;
    const avgConf = Math.round(articles.reduce((s, a) => s + a.confidence, 0) / articles.length);
    animateCount(fstatTotal, articles.length);
    animateCount(fstatFake,  nFake);
    animateCount(fstatReal,  nReal);
    animateCount(fstatConf,  avgConf + "%");
    feedStats.style.display = "flex";

    if (data.last_updated) {
      feedTimestamp.textContent = "Updated " + new Date(data.last_updated * 1000).toLocaleTimeString();
    }

    // Source label map (strip emoji from backend response)
    const sourceLabel = name => name.replace(/[\u{1F300}-\u{1FAFF}]/gu, "").trim();

    // Build cards
    feedGrid.innerHTML = articles.map((a, i) => {
      const isFake     = a.label === "Fake";
      const isLow      = a.confidence < 65;
      const conf       = isFake ? a.prob_fake : a.prob_real;
      const verdictCls = isFake ? "is-fake" : "is-real";
      const pubStr     = a.published ? a.published.substring(0, 16) : "";
      return `<div class="feed-card ${verdictCls}"
                   data-verdict="${isFake ? "fake" : "real"}"
                   data-conf="${a.confidence}"
                   data-source="${escapeHtml(a.source)}">
        <div class="feed-card-source-row">
          <span class="feed-card-source">${escapeHtml(a.source)}</span>
          <span>
            <span class="verdict-pill ${verdictCls}">${isFake ? "Fake" : "Real"} &middot; ${a.confidence.toFixed(1)}%</span>${isLow ? `<span class="verdict-pill-low">Low conf</span>` : ""}
          </span>
        </div>
        <div class="feed-card-title">
          <a href="${escapeHtml(a.link)}" target="_blank" rel="noopener noreferrer">${escapeHtml(a.title)}</a>
        </div>
        ${a.summary ? `<p class="feed-card-summary">${escapeHtml(a.summary)}</p>` : ""}
        <div class="feed-conf-bar-wrap">
          <div class="feed-conf-bar-label">
            <span>P(${isFake ? "Fake" : "Real"})</span>
            <span>${conf.toFixed(1)}%</span>
          </div>
          <div class="feed-conf-bar-track">
            <div class="feed-conf-bar-fill" style="width:${conf}%"></div>
          </div>
        </div>
        <div class="feed-card-footer">
          <span class="feed-card-meta-text">${pubStr}</span>
          <button class="btn-ghost btn-sm" onclick="analyseArticle(${JSON.stringify(a.title)}, ${JSON.stringify(a.summary || "")})">Analyse &rarr;</button>
        </div>
      </div>`;
    }).join("");

    feedGrid.style.display     = "grid";
    feedFilterBar.style.display = "flex";

    // Staggered entry animation
    requestAnimationFrame(() => {
      feedGrid.querySelectorAll(".feed-card").forEach((card, i) => {
        setTimeout(() => card.classList.add("animate-in"), i * 45);
      });
    });

    applyFilters();

  } catch (err) {
    feedLoading.style.display = "none";
    feedCountdown.classList.remove("refreshing");
    startCountdown();
    feedGrid.innerHTML = `<div class="feed-state-empty">Failed to load feed: ${escapeHtml(err.message)}</div>`;
    feedGrid.style.display = "block";
  }
}

function analyseArticle(title, summary) {
  tabBtns.forEach(b   => b.classList.toggle("active", b.dataset.tab === "detector"));
  tabPanels.forEach(p => p.classList.toggle("active", p.id === "tab-detector"));
  elHeadline.value = title;
  elArticle.value  = summary;
  updateCounter();
  window.scrollTo({ top: 0, behavior: "smooth" });
  setTimeout(runPrediction, 300);
}

setupFilters();