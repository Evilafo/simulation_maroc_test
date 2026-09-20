const API_URL = "http://127.0.0.1:8000";
const state = { dataset: "wide_gold_full", offset: 0, limit: 25, total: 0, columns: [], datasets: [], analysis: null };
const $ = (selector) => document.querySelector(selector);

function setStatus(online, text) {
  $("#apiStatus").innerHTML = `<span class="status-dot ${online ? "online" : ""}"></span><span>${text}</span>`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[character]));
}

function formatNumber(value, digits = 2) {
  return value === null || value === undefined || Number.isNaN(Number(value)) ? "—" : Number(value).toLocaleString("fr-FR", { maximumFractionDigits: digits });
}

function renderAnalysisKpis(scope) {
  const items = [["Lignes", scope.rows.toLocaleString("fr-FR")], ["Variables", scope.columns], ["Pays", scope.countries], ["Période", `${scope.year_min}—${scope.year_max}`]];
  $("#analysisKpis").innerHTML = items.map(([label, value]) => `<div class="analysis-kpi"><span>${label}</span><strong>${value}</strong></div>`).join("");
  $("#analysisScope").textContent = `${scope.rows.toLocaleString("fr-FR")} observations · ${scope.year_min}—${scope.year_max}`;
}

function renderSummary(summary) {
  $("#summaryChart").innerHTML = summary.map((item) => `<div class="summary-row"><div class="summary-label"><strong>${escapeHtml(item.indicator)}</strong><span>${item.observations} obs. · ${item.missing} manquante(s)</span></div><div class="summary-values"><span>Moy. <b>${formatNumber(item.mean)}</b></span><span>Méd. <b>${formatNumber(item.median)}</b></span></div></div>`).join("");
}

function renderMissingness(missingness) {
  const ordered = [...missingness].sort((a, b) => a.completeness - b.completeness);
  $("#missingChart").innerHTML = ordered.map((item) => `<div class="missing-row"><span>${escapeHtml(item.indicator)}</span><div class="bar-track"><i style="width:${item.completeness}%"></i></div><b>${formatNumber(item.completeness, 1)}%</b></div>`).join("");
}

function renderTrend(indicator) {
  const trends = state.analysis?.trends || [];
  const values = trends.map((point) => point[indicator]).filter((value) => value !== null);
  if (!values.length) { $("#trendChart").textContent = "Aucune observation disponible pour cet indicateur."; return; }
  const width = 900; const height = 230; const pad = 28;
  const min = Math.min(...values); const max = Math.max(...values); const span = max - min || 1;
  const points = trends.map((point, index) => point[indicator] === null ? null : `${pad + (index * (width - pad * 2)) / Math.max(trends.length - 1, 1)},${height - pad - ((point[indicator] - min) * (height - pad * 2)) / span}`).filter(Boolean).join(" ");
  const firstYear = trends[0]?.year; const lastYear = trends[trends.length - 1]?.year;
  $("#trendChart").innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Evolution de ${escapeHtml(indicator)}"><line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" /><polyline points="${points}" /><text x="${pad}" y="${height - 7}">${firstYear ?? ""}</text><text x="${width - pad}" y="${height - 7}" text-anchor="end">${lastYear ?? ""}</text><text x="${pad}" y="18">${formatNumber(max)}</text><text x="${width - pad}" y="18" text-anchor="end">${formatNumber(min)}</text></svg>`;
}

function renderAnalysis(data) {
  state.analysis = data;
  renderAnalysisKpis(data.scope);
  renderSummary(data.summary);
  renderMissingness(data.missingness);
  $("#trendIndicator").innerHTML = data.indicators.map((indicator) => `<option value="${indicator}">${indicator}</option>`).join("");
  renderTrend(data.indicators[0]);
}

async function loadAnalysis() {
  const params = new URLSearchParams({ dataset_id: "wide_gold_full" });
  const country = $("#countryFilter").value;
  if (country) params.set("country", country);
  if ($("#yearStart").value) params.set("year_start", $("#yearStart").value);
  if ($("#yearEnd").value) params.set("year_end", $("#yearEnd").value);
  try {
    const response = await fetch(`${API_URL}/data/analysis?${params}`);
    if (!response.ok) throw new Error("Impossible de calculer l’analyse.");
    renderAnalysis(await response.json());
    $("#analysisError").hidden = true;
  } catch (error) { $("#analysisError").textContent = error.message; $("#analysisError").hidden = false; }
}

function renderDatasetList() {
  $("#datasetList").innerHTML = state.datasets.map((dataset) => `<button class="dataset-item ${dataset.id === state.dataset ? "active" : ""}" data-dataset="${dataset.id}" type="button"><strong>${dataset.label}</strong><span>${dataset.rows.toLocaleString("fr-FR")} lignes · ${dataset.columns} colonnes</span></button>`).join("");
  document.querySelectorAll("[data-dataset]").forEach((button) => button.addEventListener("click", () => {
    state.dataset = button.dataset.dataset;
    state.offset = 0;
    renderDatasetList();
    loadRows();
  }));
}

function renderTable(rows) {
  $("#dataHead").innerHTML = `<tr>${state.columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>`;
  $("#dataBody").innerHTML = rows.map((row) => `<tr>${state.columns.map((column) => `<td>${escapeHtml(row[column])}</td>`).join("")}</tr>`).join("");
  $("#dataEmpty").hidden = rows.length > 0;
}

function populateCountries(countries) {
  if (!state.columns.includes("country_iso3")) return;
  $("#countryFilter").innerHTML = '<option value="">Tous les pays</option>';
  $("#countryFilter").innerHTML += countries.map((country) => `<option value="${country}">${country}</option>`).join("");
}

async function loadRows() {
  const params = new URLSearchParams({ offset: state.offset, limit: state.limit });
  const search = $("#searchInput").value.trim();
  const country = $("#countryFilter").value;
  if (search) params.set("search", search);
  if (country) params.set("country", country);
  if ($("#yearStart").value) params.set("year_start", $("#yearStart").value);
  if ($("#yearEnd").value) params.set("year_end", $("#yearEnd").value);
  try {
    const response = await fetch(`${API_URL}/data/${state.dataset}?${params}`);
    if (!response.ok) throw new Error("Impossible de charger cette base.");
    const data = await response.json();
    const metadata = state.datasets.find((dataset) => dataset.id === state.dataset);
    state.total = data.total;
    state.columns = data.columns;
    $("#datasetEyebrow").textContent = metadata.label.toUpperCase();
    $("#datasetTitle").textContent = metadata.label;
    $("#datasetDescription").textContent = metadata.description;
    $("#rowCount").textContent = data.total.toLocaleString("fr-FR");
    $("#pageInfo").textContent = `Page ${Math.floor(state.offset / state.limit) + 1} / ${Math.max(1, Math.ceil(state.total / state.limit))}`;
    $("#previousPage").disabled = state.offset === 0;
    $("#nextPage").disabled = state.offset + state.limit >= state.total;
    renderTable(data.rows);
    populateCountries(metadata.countries || []);
    if (state.dataset === "wide_gold_full") loadAnalysis();
    setStatus(true, "API connectée");
  } catch (error) {
    $("#dataStatus").textContent = error.message;
    setStatus(false, "API indisponible");
  }
}

async function loadCatalog() {
  const response = await fetch(`${API_URL}/data/datasets`);
  const data = await response.json();
  state.datasets = data.datasets;
  renderDatasetList();
  await loadRows();
}

let filterTimer;
["#searchInput", "#countryFilter", "#yearStart", "#yearEnd"].forEach((selector) => $(selector).addEventListener("input", () => { clearTimeout(filterTimer); filterTimer = setTimeout(() => { state.offset = 0; loadRows(); }, 250); }));
$("#trendIndicator").addEventListener("change", (event) => renderTrend(event.target.value));
$("#resetFilters").addEventListener("click", () => { $("#searchInput").value = ""; $("#countryFilter").value = ""; $("#yearStart").value = ""; $("#yearEnd").value = ""; state.offset = 0; loadRows(); });
$("#previousPage").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadRows(); });
$("#nextPage").addEventListener("click", () => { if (state.offset + state.limit < state.total) { state.offset += state.limit; loadRows(); } });
loadCatalog().catch(() => setStatus(false, "API indisponible"));