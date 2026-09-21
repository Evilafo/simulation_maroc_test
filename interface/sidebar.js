(() => {
  const sidebar = document.querySelector(".side-menu");
  if (!sidebar) return;

  const toggle = document.createElement("button");
  toggle.className = "side-toggle";
  toggle.type = "button";
  toggle.setAttribute("aria-label", "Réduire le menu latéral");
  toggle.setAttribute("aria-expanded", "true");
  toggle.innerHTML = '<span aria-hidden="true">‹</span>';
  sidebar.prepend(toggle);

  if (document.body.classList.contains("maquette-home")) {
    const explorer = sidebar.querySelectorAll("nav")[1];
    if (explorer) {
      [["#svar", "09", "SVAR"], ["#dashboard", "10", "Dashboard"]].forEach(([href, number, label]) => {
        const link = document.createElement("a");
        link.href = href;
        link.innerHTML = `<span>${number}</span>${label}`;
        explorer.append(link);
      });
    }
  }

  const storageKey = "datalab-sidebar-collapsed";
  const setCollapsed = (collapsed) => {
    document.body.classList.toggle("sidebar-collapsed", collapsed);
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.setAttribute("aria-label", collapsed ? "Développer le menu latéral" : "Réduire le menu latéral");
    toggle.querySelector("span").textContent = collapsed ? "›" : "‹";
  };

  setCollapsed(localStorage.getItem(storageKey) === "true");
  toggle.addEventListener("click", () => {
    const collapsed = !document.body.classList.contains("sidebar-collapsed");
    setCollapsed(collapsed);
    localStorage.setItem(storageKey, String(collapsed));
  });
})();
