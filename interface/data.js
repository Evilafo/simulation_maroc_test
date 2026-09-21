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
  $("#summaryChart").innerHTML = summary.map((item) => `<div class="summary-row"><div class="summary-label"><strong>${escapeHtml(item.indicator)}</strong><span>${item.observations} obs. · ${item.missing} manquante(s)</span></div><div class="summary-values"><span>Moy. <b>${formatNumber(item.mean)}</b></span><span>Méd. <b>${formatNumber(item.median)}</b></span><span>Q1—Q3 <b>${formatNumber(item.q1)}—${formatNumber(item.q3)}</b></span></div></div>`).join("");
}

function renderMissingness(missingness) {
  const ordered = [...missingness].sort((a, b) => a.completeness - b.completeness);
  $("#missingChart").innerHTML = ordered.map((item) => `<div class="missing-row"><span>${escapeHtml(item.indicator)}</span><div class="bar-track"><i style="width:${item.completeness}%"></i></div><b>${formatNumber(item.completeness, 1)}%</b></div>`).join("");
}

function renderCountryComparison(countrySummary, indicator) {
  const ranked = countrySummary.map((item) => ({ country: item.country, value: item.values[indicator] })).filter((item) => item.value !== null).sort((a, b) => b.value - a.value);
  if (!ranked.length) { $("#countryChart").textContent = "Aucune observation disponible pour cet indicateur."; return; }
  const max = Math.max(...ranked.map((item) => Math.abs(item.value)), 1);
  $("#countryChart").innerHTML = ranked.map((item) => `<div class="country-row"><span>${escapeHtml(item.country)}</span><div class="bar-track"><i style="width:${Math.min(Math.abs(item.value) / max * 100, 100)}%"></i></div><b>${formatNumber(item.value)}</b></div>`).join("");
}

function renderCoverage(coverage) {
  const visible = coverage.filter((item) => item.year % 5 === 0 || item.year === coverage[coverage.length - 1]?.year);
  $("#coverageChart").innerHTML = visible.map((item) => `<div class="coverage-row"><span>${item.year}</span><div class="bar-track"><i style="width:${item.completeness}%"></i></div><b>${formatNumber(item.completeness, 1)}%</b></div>`).join("");
}

function renderCorrelations(correlations, indicators) {
  const color = (value) => {
    if (value === null) return "#eef0eb";
    const intensity = Math.round(255 - Math.abs(value) * 100);
    return value >= 0 ? `rgb(${intensity}, ${Math.min(220, intensity + 35)}, ${Math.min(190, intensity + 5)})` : `rgb( ${Math.min(225, intensity + 25)}, ${intensity}, ${intensity})`;
  };
  const lookup = Object.fromEntries(correlations.map((row) => [row.row, row.values]));
  $("#correlationChart").innerHTML = `<div class="correlation-grid" style="--correlation-count:${indicators.length}"><div></div>${indicators.map((item) => `<span class="correlation-label">${escapeHtml(item)}</span>`).join("")}${indicators.map((row) => `<span class="correlation-label row-label">${escapeHtml(row)}</span>${indicators.map((column) => { const value = lookup[row]?.[column]; return `<span class="correlation-cell" title="${row} / ${column}: ${formatNumber(value)}" style="background:${color(value)}">${formatNumber(value, 1)}</span>`; }).join("")}`).join("")}</div>`;
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
  $("#countryIndicator").innerHTML = data.indicators.map((indicator) => `<option value="${indicator}">${indicator}</option>`).join("");
  renderTrend(data.indicators[0]);
  renderCountryComparison(data.country_summary, data.indicators[0]);
  renderCoverage(data.coverage);
  renderCorrelations(data.correlations, data.indicators);
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

async function downloadDataset() {
  const params = new URLSearchParams();
  const search = $("#searchInput").value.trim();
  const country = $("#countryFilter").value;
  if (search) params.set("search", search);
  if (country) params.set("country", country);
  if ($("#yearStart").value) params.set("year_start", $("#yearStart").value);
  if ($("#yearEnd").value) params.set("year_end", $("#yearEnd").value);
  try {
    const response = await fetch(`${API_URL}/data/${state.dataset}?${params}`);
    if (!response.ok) throw new Error("Impossible de télécharger cette base.");
    const data = await response.json();
    const headers = data.columns.join(",");
    const rows = data.rows.map((row) => data.columns.map((column) => `"${String(row[column] ?? "").replace(/"/g, '""')}"`).join(",")).join("\n");
    const csv = `${headers}\n${rows}`;
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${state.dataset}_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (error) {
    $("#dataStatus").textContent = error.message;
    setStatus(false, "Erreur de téléchargement");
  }
}

let filterTimer;
["#searchInput", "#countryFilter", "#yearStart", "#yearEnd"].forEach((selector) => $(selector).addEventListener("input", () => { clearTimeout(filterTimer); filterTimer = setTimeout(() => { state.offset = 0; loadRows(); }, 250); }));
$("#trendIndicator").addEventListener("change", (event) => renderTrend(event.target.value));
$("#countryIndicator").addEventListener("change", (event) => renderCountryComparison(state.analysis?.country_summary || [], event.target.value));
$("#resetFilters").addEventListener("click", () => { $("#searchInput").value = ""; $("#countryFilter").value = ""; $("#yearStart").value = ""; $("#yearEnd").value = ""; state.offset = 0; loadRows(); });
$("#previousPage").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadRows(); });
$("#nextPage").addEventListener("click", () => { if (state.offset + state.limit < state.total) { state.offset += state.limit; loadRows(); } });
$("#downloadButton").addEventListener("click", downloadDataset);
loadCatalog().catch(() => setStatus(false, "API indisponible"));