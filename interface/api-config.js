// Variable globale partagée pour l'URL API
window.API_URL = window.API_URL || "http://127.0.0.1:8000";

function updateApiUrl() {
  const savedApiUrl = localStorage.getItem("apiUrl");
  if (savedApiUrl) { 
    window.API_URL = savedApiUrl; 
    const input = document.querySelector("#apiUrlInput");
    if (input) input.value = savedApiUrl; 
  }
}

function setApiUrl(url) {
  window.API_URL = url;
  localStorage.setItem("apiUrl", url);
  // Mettre à jour tous les inputs sur la page
  document.querySelectorAll("#apiUrlInput").forEach(input => {
    input.value = url;
  });
  // Afficher un message de confirmation
  const status = document.querySelector("#apiStatus span:last-child");
  if (status) {
    status.textContent = "URL mise à jour !";
    setTimeout(() => {
      status.textContent = "API connectée";
    }, 2000);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  updateApiUrl();
  
  const button = document.querySelector("#apiUrlButton");
  if (button) {
    button.addEventListener("click", () => { 
      const input = document.querySelector("#apiUrlInput");
      const url = input?.value.trim(); 
      if (url) { 
        setApiUrl(url);
      }
    });
  }
});
