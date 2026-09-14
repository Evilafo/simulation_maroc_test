const API_URL = "http://127.0.0.1:8000";

const targetMeta = {
  CROISSANCE_PIB: { label: "Croissance du PIB", unit: "% annuel", columns: ["CROISSANCE_PIB", "FBCF", "IDEE"] },
  INFLATION_CPI: { label: "Inflation CPI", unit: "% annuel", columns: ["INFLATION_CPI", "TCER", "CROISSANCE_PIB"] },
  SOBG: { label: "Solde budgétaire global", unit: "% du PIB", columns: ["SOBG", "CROISSANCE_PIB", "DETTE_PUBLIQUE"] },
  DETTE_PUBLIQUE: { label: "Dette publique", unit: "% du PIB", columns: ["DETTE_PUBLIQUE", "SOPR", "CROISSANCE_PIB"] },
  BALANCE_COURANTE: { label: "Balance courante", unit: "% du PIB", columns: ["BALANCE_COURANTE", "TCER", "IDEE"] },
  TACH: { label: "Taux de chômage", unit: "% population active", columns: ["TACH", "CROISSANCE_PIB", "TAAC"] },
  FBCF: { label: "FBCF", unit: "% du PIB", columns: ["FBCF", "IDEE", "CROISSANCE_PIB"] },
};

const target = document.querySelector("#target");
const historyBody = document.querySelector("#historyBody");
const selectedHorizon = { value: 1 };

Object.entries(targetMeta).forEach(([value, meta]) => {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = meta.label;
  target.appendChild(option);
});

function renderHistory() {
  const meta = targetMeta[target.value];
  document.querySelector("#targetUnit").textContent = meta.unit;
  ["columnOne", "columnTwo", "columnThree"].forEach((id, index) => {
    document.querySelector(`#${id}`).textContent = meta.columns[index];
  });
  historyBody.innerHTML = "";
  for (let year = 2019; year <= 2024; year += 1) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${year}</td>${meta.columns.map((column, index) => `<td><input required type="number" step="any" name="${column}_${year}" aria-label="${column} ${year}" value="${(index + 1) * 1.5 + (year - 2019) * 0.1}"></td>`).join("")}`;
    historyBody.appendChild(row);
  }
}

function setStatus(online, text) {
  document.querySelector("#apiStatus").innerHTML = `<span class="status-dot ${online ? "online" : ""}"></span><span>${text}</span>`;
}

async function loadModels() {
  try {
    const response = await fetch(`${API_URL}/models`);
    if (!response.ok) throw new Error();
    const data = await response.json();
    document.querySelector("#modelCount").textContent = data.count;
    setStatus(true, "API connectée");
  } catch {
    setStatus(false, "API indisponible");
  }
}

document.querySelectorAll("[data-horizon]").forEach((button) => button.addEventListener("click", () => {
  document.querySelectorAll("[data-horizon]").forEach((item) => item.classList.remove("active"));
  button.classList.add("active");
  selectedHorizon.value = Number(button.dataset.horizon);
}));

target.addEventListener("change", renderHistory);
document.querySelector("#forecastForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const meta = targetMeta[target.value];
  const history = [];
  for (let year = 2019; year <= 2024; year += 1) {
    const values = {};
    meta.columns.forEach((column) => { values[column] = Number(form.get(`${column}_${year}`)); });
    history.push({ year, values });
  }
  const button = event.currentTarget.querySelector(".primary-button");
  const error = document.querySelector("#errorMessage");
  error.textContent = "";
  button.disabled = true;
  try {
    const response = await fetch(`${API_URL}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ target: target.value, horizon: selectedHorizon.value, history }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail?.message || data.detail || "La prévision n’a pas pu être calculée.");
    document.querySelector("#resultEmpty").hidden = true;
    document.querySelector("#resultContent").hidden = false;
    document.querySelector("#resultTarget").textContent = meta.label.toUpperCase();
    document.querySelector("#predictionValue").textContent = data.prediction.toFixed(2);
    document.querySelector("#predictionUnit").textContent = meta.unit;
    document.querySelector("#targetYear").textContent = data.target_year;
    document.querySelector("#originYear").textContent = data.origin_year;
  } catch (requestError) {
    error.textContent = requestError.message;
  } finally {
    button.disabled = false;
  }
});

renderHistory();
loadModels();