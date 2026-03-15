async function loadComponent(id, url) {
  const res = await fetch(url);
  document.getElementById(id).innerHTML = await res.text();
}

document.addEventListener("DOMContentLoaded", () => {
  loadComponent("footer", "/footer.html");
});
