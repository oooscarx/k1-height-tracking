const frame = document.querySelector("#sim-frame");
const slider = document.querySelector("#height-slider");
const targetText = document.querySelector("#target-height");
const measuredText = document.querySelector("#measured-height");
const errorText = document.querySelector("#height-error");
const terrainText = document.querySelector("#terrain-level");
const liftText = document.querySelector("#lift-scale");
const connectionDot = document.querySelector("#connection-dot");
const connectionText = document.querySelector("#connection-text");
const randomizationBadge = document.querySelector("#randomization-badge");
const disturbanceBadge = document.querySelector("#disturbance-badge");

let currentTarget = Number(slider.value);
let sliderDragging = false;

async function command(payload) {
  const response = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error(`command failed: ${response.status}`);
  return response.json();
}

function setHeight(value) {
  const minimum = Number(slider.min);
  const maximum = Number(slider.max);
  currentTarget = Math.min(maximum, Math.max(minimum, Number(value)));
  slider.value = currentTarget;
  targetText.textContent = currentTarget.toFixed(3);
  command({ action: "set_height", value: currentTarget }).catch(markOffline);
}

function adjustHeight(delta) {
  setHeight(currentTarget + delta);
}

function push(direction) {
  const vectors = {
    forward: { x: 0.45, y: 0.0 },
    back: { x: -0.45, y: 0.0 },
    left: { x: 0.0, y: 0.45 },
    right: { x: 0.0, y: -0.45 },
  };
  command({ action: "push", ...vectors[direction] }).catch(markOffline);
}

function markOffline() {
  connectionDot.classList.remove("online");
  connectionText.textContent = "连接断开";
}

async function refreshState() {
  try {
    const response = await fetch("/api/state", { cache: "no-store" });
    if (!response.ok) throw new Error("state unavailable");
    const state = await response.json();
    connectionDot.classList.add("online");
    connectionText.textContent = `${state.policy_hz.toFixed(0)} Hz / ${state.frame_hz.toFixed(1)} fps`;
    currentTarget = Number(state.target_height);
    slider.min = state.minimum_height;
    slider.max = state.maximum_height;
    if (!sliderDragging) slider.value = currentTarget;
    targetText.textContent = currentTarget.toFixed(3);
    measuredText.textContent = Number(state.measured_height).toFixed(3);
    errorText.textContent = Number(state.height_error).toFixed(3);
    terrainText.textContent = state.terrain_level;
    liftText.textContent = Number(state.lift).toFixed(3);
    randomizationBadge.classList.toggle("active", state.domain_randomization);
    disturbanceBadge.classList.toggle("active", state.external_disturbances);
  } catch (_error) {
    markOffline();
  }
}

function refreshFrame() {
  const next = new Image();
  next.onload = () => {
    frame.src = next.src;
    window.setTimeout(refreshFrame, 70);
  };
  next.onerror = () => window.setTimeout(refreshFrame, 300);
  next.src = `/frame.jpg?t=${Date.now()}`;
}

slider.addEventListener("pointerdown", () => { sliderDragging = true; });
slider.addEventListener("pointerup", () => { sliderDragging = false; setHeight(slider.value); });
slider.addEventListener("change", () => setHeight(slider.value));
document.querySelector("#height-down").addEventListener("click", () => adjustHeight(-0.02));
document.querySelector("#height-up").addEventListener("click", () => adjustHeight(0.02));
document.querySelector("#reset").addEventListener("click", () => command({ action: "reset" }).catch(markOffline));
document.querySelectorAll("[data-height]").forEach((button) => {
  button.addEventListener("click", () => setHeight(button.dataset.height));
});
document.querySelectorAll("[data-push]").forEach((button) => {
  button.addEventListener("click", () => push(button.dataset.push));
});

window.addEventListener("keydown", (event) => {
  if (event.repeat || event.target instanceof HTMLInputElement) return;
  const key = event.key.toLowerCase();
  if (key === "w") adjustHeight(0.02);
  else if (key === "s") adjustHeight(-0.02);
  else if (key === "1") setHeight(-0.5);
  else if (key === "2") setHeight(0.5);
  else if (key === "3") setHeight(Number(slider.max));
  else if (key === "r") command({ action: "reset" }).catch(markOffline);
  else if (event.key === "ArrowUp") push("forward");
  else if (event.key === "ArrowDown") push("back");
  else if (event.key === "ArrowLeft") push("left");
  else if (event.key === "ArrowRight") push("right");
  else return;
  event.preventDefault();
});

refreshFrame();
refreshState();
window.setInterval(refreshState, 250);
