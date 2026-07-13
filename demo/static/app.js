const colabUrlInput = document.getElementById("colabUrl");
const fileInput = document.getElementById("fileInput");
const runBtn = document.getElementById("runBtn");
const healthBtn = document.getElementById("healthBtn");
const rerenderBtn = document.getElementById("rerenderBtn");
const statusEl = document.getElementById("status");
const dropzone = document.getElementById("dropzone");
const inputPreview = document.getElementById("inputPreview");
const inputEmpty = document.getElementById("inputEmpty");
const renderPreview = document.getElementById("renderPreview");
const renderEmpty = document.getElementById("renderEmpty");
const semanticEl = document.getElementById("semantic");

const STORAGE_KEY = "omrDemoColabUrl";
let selectedFile = null;

function setStatus(message, kind = "") {
  statusEl.textContent = message;
  statusEl.className = `status${kind ? ` ${kind}` : ""}`;
}

function syncButtons() {
  runBtn.disabled = !(selectedFile && colabUrlInput.value.trim());
  rerenderBtn.disabled = !semanticEl.value.trim();
}

function showInputPreview(file) {
  const url = URL.createObjectURL(file);
  inputPreview.src = url;
  inputPreview.hidden = false;
  inputEmpty.hidden = true;
}

function showRender(b64) {
  renderPreview.src = `data:image/png;base64,${b64}`;
  renderPreview.hidden = false;
  renderEmpty.hidden = true;
}

function clearRender() {
  renderPreview.hidden = true;
  renderPreview.removeAttribute("src");
  renderEmpty.hidden = false;
}

colabUrlInput.value = localStorage.getItem(STORAGE_KEY) || "";
colabUrlInput.addEventListener("input", () => {
  localStorage.setItem(STORAGE_KEY, colabUrlInput.value.trim());
  syncButtons();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files?.[0];
  if (!file) return;
  selectedFile = file;
  showInputPreview(file);
  clearRender();
  semanticEl.value = "";
  setStatus(`Selected ${file.name}`);
  syncButtons();
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("active");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("active");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  if (!file) return;
  selectedFile = file;
  fileInput.value = "";
  showInputPreview(file);
  clearRender();
  semanticEl.value = "";
  setStatus(`Selected ${file.name}`);
  syncButtons();
});

semanticEl.addEventListener("input", syncButtons);

healthBtn.addEventListener("click", async () => {
  const colabUrl = colabUrlInput.value.trim();
  setStatus("Checking Colab…");
  try {
    const params = new URLSearchParams();
    if (colabUrl) params.set("colab_url", colabUrl);
    const response = await fetch(`/api/health?${params.toString()}`);
    const data = await response.json();
    if (data.colab_ok) {
      setStatus(`Colab reachable at ${data.colab_url}`, "ok");
    } else if (!data.colab_url) {
      setStatus("No Colab URL set.", "error");
    } else {
      setStatus(`Colab unreachable: ${data.colab_error || "unknown error"}`, "error");
    }
  } catch (error) {
    setStatus(`Health check failed: ${error.message}`, "error");
  }
});

runBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  const colabUrl = colabUrlInput.value.trim();
  if (!colabUrl) {
    setStatus("Paste the Colab ngrok URL first.", "error");
    return;
  }

  const form = new FormData();
  form.append("image", selectedFile, selectedFile.name);
  form.append("colab_url", colabUrl);

  runBtn.disabled = true;
  setStatus("Sending image to Colab and rendering…");
  try {
    const response = await fetch("/api/transcribe", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : data.detail
            ? JSON.stringify(data.detail)
            : `HTTP ${response.status}`;
      throw new Error(detail);
    }
    semanticEl.value = data.semantic;
    showRender(data.rendered_image_b64);
    if (data.input_image_b64) {
      inputPreview.src = `data:image/png;base64,${data.input_image_b64}`;
      inputPreview.hidden = false;
      inputEmpty.hidden = true;
    }
    setStatus("Done.", "ok");
  } catch (error) {
    setStatus(`Transcription failed: ${error.message}`, "error");
  } finally {
    syncButtons();
  }
});

rerenderBtn.addEventListener("click", async () => {
  const semantic = semanticEl.value.trim();
  if (!semantic) return;
  const form = new FormData();
  form.append("semantic", semantic);
  rerenderBtn.disabled = true;
  setStatus("Re-rendering local Verovio score…");
  try {
    const response = await fetch("/api/render", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) {
      const detail =
        typeof data.detail === "string"
          ? data.detail
          : data.detail
            ? JSON.stringify(data.detail)
            : `HTTP ${response.status}`;
      throw new Error(detail);
    }
    showRender(data.rendered_image_b64);
    setStatus("Re-rendered.", "ok");
  } catch (error) {
    setStatus(`Render failed: ${error.message}`, "error");
  } finally {
    syncButtons();
  }
});

syncButtons();
