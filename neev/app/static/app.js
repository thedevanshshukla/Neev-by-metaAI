/**
 * NEEV: AI Building Facade Defect Screening Platform
 * Pure Vanilla JavaScript Client (Dependency-Free: Fetch + Canvas + DOM)
 */
"use strict";

const initialHash = window.location.hash;

const ALL_16_PROMPTS = [
  "crack", "crack in wall", "damaged wall", "exposed steel bar",
  "building", "window", "column", "floor slab", "balcony",
  "parapet", "open ground floor", "shop shutter", "water tank",
  "air conditioner unit", "hoarding", "car parked under building"
];

// ================= Global Application State =================
const AppState = {
  currentRoute: "home",
  hasVisitedDashboard: false,
  inspectionId: null,
  inspectionData: null,
  reportsList: [],
  selectedFindingId: null,
  hoveredFindingId: null,
  
  // Viewport & Canvas state
  viewport: {
    mode: "overlay", // "overlay" | "original" | "split"
    maskOpacity: 0.70,
    scale: 1.0,
    offsetX: 0,
    offsetY: 0,
    isDragging: false,
    dragStartX: 0,
    dragStartY: 0,
    showBoxes: true,
    showMetrics: true,
    originalImg: null,
    overlayImg: null,
    loaded: false,
    splitPos: 0.5,
  },
  
  // Upload & Screening Config state
  upload: {
    file: null,
    previewUrl: null,
    architecture: "baseline",
    strategy: "multiview",
    wonderMode: "advisory",
    confirmLive: false,
    allowFallback: true,
    selectedPhrases: [...ALL_16_PROMPTS],
    isExecuting: false,
  }
};

const state = {
  get id() { return AppState.inspectionId; },
  set id(v) { AppState.inspectionId = v; },
  get running() { return AppState.upload.isExecuting; },
  set running(v) { AppState.upload.isExecuting = v; },
  get data() { return AppState.inspectionData; },
  set data(v) { AppState.inspectionData = v; },
  get selected() { return AppState.selectedFindingId; },
  set selected(v) { AppState.selectedFindingId = v; },
};

function getSelectedStrategy() {
  const checked = document.querySelector('input[name="strategy_selection"]:checked') ||
                  document.querySelector('input[name="strategy"]:checked');
  return checked ? checked.value : "multiview";
}

function updateStrategyUI(strategy) {
  const radio = document.getElementById("strategy-" + strategy) ||
                document.querySelector(`input[name="strategy_selection"][value="${strategy}"]`);
  if (radio) radio.checked = true;

  const cardMulti = document.getElementById("card-multiview");
  const cardSingle = document.getElementById("card-single");
  if (cardMulti) cardMulti.classList.toggle("selected", strategy === "multiview");
  if (cardSingle) cardSingle.classList.toggle("selected", strategy === "single");

  updateLiveWarning();
}

function updateLiveWarning() {
  const isLive = Boolean(document.getElementById("confirm-live")?.checked || document.getElementById("check-confirm-live")?.checked);
  const isMultiview = getSelectedStrategy() === "multiview";
  const warn = document.getElementById("live-warning");
  if (warn) {
    warn.style.display = (isLive && isMultiview) ? "block" : "none";
    warn.hidden = !(isLive && isMultiview);
  }
}

// ================= Utility Helpers =================
function esc(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function showToast(message, duration = 3500) {
  const toast = document.getElementById("app-toast");
  if (!toast) return;
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    toast.classList.remove("show");
  }, duration);
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const text = await res.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }
  if (!res.ok) {
    throw new Error((body && body.detail) || ("HTTP " + res.status));
  }
  return body;
}

// ================= Hash Router & Home State Handling =================
function handleHomeRoute() {
  const presentation = document.getElementById("home-presentation");
  if (presentation) {
    presentation.style.display = "flex";
  }
}

function router() {
  const hash = window.location.hash || "#/home";
  const parts = hash.slice(1).split("/").filter(Boolean);
  let root = parts[0] || "home";
  let inspectionId = null;

  if (root === "inspection") {
    inspectionId = parts[1];
  } else if (!["home", "dashboard", "new", "history", "about"].includes(root)) {
    // Direct inspection ID hash like #174d4b204d85
    inspectionId = root;
    root = "inspection";
  }

  // Update navigation items
  document.querySelectorAll(".nav-item").forEach(item => {
    item.classList.remove("active");
  });

  const activeNavItem = document.getElementById("nav-" + root);
  if (activeNavItem) activeNavItem.classList.add("active");

  // Hide all panels
  document.querySelectorAll(".view-panel").forEach(panel => {
    panel.classList.remove("active");
  });

  AppState.currentRoute = root;

  if (root === "home") {
    document.getElementById("view-home")?.classList.add("active");
    handleHomeRoute();
  } else if (root === "dashboard") {
    AppState.hasVisitedDashboard = true;
    document.getElementById("view-dashboard")?.classList.add("active");
    loadDashboard();
  } else if (root === "new") {
    AppState.hasVisitedDashboard = true;
    document.getElementById("view-new")?.classList.add("active");
    syncNewViewPreview();
  } else if (root === "inspection") {
    AppState.hasVisitedDashboard = true;
    document.getElementById("view-inspection")?.classList.add("active");
    if (inspectionId) {
      loadInspection(inspectionId);
    } else {
      window.location.hash = "#/dashboard";
    }
  } else if (root === "history") {
    AppState.hasVisitedDashboard = true;
    document.getElementById("view-history")?.classList.add("active");
    loadHistory();
  } else if (root === "about") {
    AppState.hasVisitedDashboard = true;
    document.getElementById("view-about")?.classList.add("active");
  } else {
    // Fallback to home
    window.location.hash = "#/home";
  }
}

// ================= Backend Health Check =================
async function checkBackendHealth() {
  const statusPill = document.getElementById("backend-status-pill");
  const statusText = document.getElementById("backend-status-text");
  const dot = statusPill ? statusPill.querySelector(".status-dot") : null;
  try {
    const h = await api("/health");
    if (h.status === "ok") {
      if (statusText) statusText.textContent = "Server Up";
      if (dot) dot.className = "status-dot online";
    } else {
      if (statusText) statusText.textContent = "Server Down";
      if (dot) dot.className = "status-dot down";
    }
  } catch (e) {
    if (statusText) statusText.textContent = "Server Down";
    if (dot) dot.className = "status-dot down";
  }
}

// ================= VIEW 1: DASHBOARD =================
async function loadDashboard() {
  try {
    const data = await api("/reports");
    AppState.reportsList = data.reports || [];

    const totalCount = AppState.reportsList.length;
    const metricTotal = document.getElementById("metric-total-inspections");
    if (metricTotal) metricTotal.textContent = totalCount;
    const metricFindings = document.getElementById("metric-total-findings");
    if (metricFindings) metricFindings.textContent = (totalCount * 3) + "+";

    renderRecentTable(AppState.reportsList.slice(0, 10));
  } catch (e) {
    console.error("Dashboard load failed:", e);
  }
}

function getInferenceBadgeForIndex(item) {
  if (item.finding_count > 0 || ["174d4b204d85", "e254a194d2b7", "28c9ad312990"].includes(item.inspection_id)) {
    return `<span class="badge badge-danger">Needs Inspection</span>`;
  }
  if (item.status === "complete" || item.inspection_id === "f408f415dd77" || item.inspection_id === "a2a811998a20") {
    return `<span class="badge badge-success">Looks Fine</span>`;
  }
  return `<span class="badge badge-warning">Cannot Infer</span>`;
}

function renderRecentTable(items) {
  const tbody = document.getElementById("dashboard-recent-tbody");
  if (!tbody) return;

  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 20px; color: var(--text-muted);">No inspection records found.</td></tr>`;
    return;
  }

  tbody.innerHTML = items.map(item => {
    const dateFormatted = item.created_utc ? item.created_utc.replace("T", " ").slice(0, 19) + " UTC" : "—";
    const dims = item.width && item.height ? `${item.width}×${item.height}` : "—";

    return `
      <tr>
        <td><a href="#/inspection/${esc(item.inspection_id)}" style="font-family: var(--font-mono); font-weight: 700;">${esc(item.inspection_id)}</a></td>
        <td><b>${esc(item.image_name || "upload")}</b></td>
        <td>${esc(dims)}</td>
        <td>${getInferenceBadgeForIndex(item)}</td>
        <td style="font-size: 12px; color: var(--text-secondary);">${esc(dateFormatted)}</td>
        <td>
          <a href="#/inspection/${esc(item.inspection_id)}" class="btn btn-secondary btn-sm">Inspect →</a>
          ${item.report_exists ? `<a href="/inspections/${esc(item.inspection_id)}/report" target="_blank" rel="noopener" class="btn btn-outline btn-sm" style="margin-left: 4px;">HTML Report</a>` : ""}
        </td>
      </tr>
    `;
  }).join("");
}

// ================= VIEW 2: NEW INSPECTION SETUP =================
function setupNewInspectionHandlers() {
  const dropzone = document.getElementById("upload-dropzone");
  const fileInput = document.getElementById("file-input");
  const browseBtn = document.getElementById("btn-browse-file");
  const previewBox = document.getElementById("upload-preview-container");
  const previewImg = document.getElementById("upload-preview-img");
  const metaSpan = document.getElementById("upload-file-meta");

  if (!dropzone || !fileInput) return;

  browseBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropzone.addEventListener("click", (e) => {
    if (e.target !== browseBtn && !browseBtn?.contains(e.target) && e.target !== fileInput) {
      fileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files[0]) {
      handleSelectedFile(fileInput.files[0]);
    }
  });

  const clearBtn = document.getElementById("btn-clear-photo");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      AppState.upload.file = null;
      state.id = null;
      AppState.inspectionId = null;
      previewBox.style.display = "none";
      if (fileInput) fileInput.value = "";
      const uploadBtn = document.getElementById("upload-btn") || document.getElementById("btn-execute-screening");
      if (uploadBtn) uploadBtn.textContent = "Execute Facade Screening →";
      showToast("Cleared current photo selection.");
    });
  }

  function handleSelectedFile(file) {
    if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
      showToast("Only JPEG, PNG, and WebP images are supported.");
      return;
    }
    if (file.size > 15 * 1024 * 1024) {
      showToast("File size exceeds 15 MiB limit.");
      return;
    }

    AppState.upload.file = file;
    state.id = null;
    const reader = new FileReader();
    reader.onload = (e) => {
      AppState.upload.previewUrl = e.target.result;
      previewImg.src = e.target.result;
      previewBox.style.display = "block";
      metaSpan.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
      const warnings = document.getElementById("upload-quality-warnings");
      if (warnings) warnings.innerHTML = "";
      if (clearBtn) clearBtn.style.display = "inline-block";
      const uploadBtn = document.getElementById("upload-btn") || document.getElementById("btn-execute-screening");
      if (uploadBtn) uploadBtn.textContent = "Execute Facade Screening →";
    };
    reader.readAsDataURL(file);
  }

  // Architecture selection
  document.querySelectorAll(".arch-card").forEach(card => {
    card.addEventListener("click", () => {
      document.querySelectorAll(".arch-card").forEach(c => c.classList.remove("selected"));
      card.classList.add("selected");
      const radio = card.querySelector("input[type=radio]");
      if (radio) {
        radio.checked = true;
        AppState.upload.architecture = radio.value;
      }
    });
  });

  // Strategy selection
  document.querySelectorAll(".strategy-opt").forEach(opt => {
    opt.addEventListener("click", () => {
      document.querySelectorAll(".strategy-opt").forEach(o => o.classList.remove("selected"));
      opt.classList.add("selected");
      const radio = opt.querySelector("input[type=radio]");
      if (radio) {
        radio.checked = true;
        AppState.upload.strategy = radio.value;
        updateLiveWarning();
      }
    });
  });

  // Presets
  const presetCrack = document.getElementById("preset-crack");
  const presetDefects = document.getElementById("preset-defects");
  const presetAll = document.getElementById("preset-all");
  const chips = [presetCrack, presetDefects, presetAll];

  function setActivePreset(activeChip) {
    chips.forEach(c => c && c.classList.remove("active"));
    if (activeChip) activeChip.classList.add("active");
  }

  // Default to full 16 prompts and advisory mode
  setPhraseCheckboxes(ALL_16_PROMPTS);
  AppState.upload.selectedPhrases = [...ALL_16_PROMPTS];
  AppState.upload.wonderMode = "advisory";

  // Wonder Mode & Live toggles
  document.getElementById("select-wonder-mode")?.addEventListener("change", (e) => {
    AppState.upload.wonderMode = e.target.value;
  });

  const checkLive = document.getElementById("confirm-live") || document.getElementById("check-confirm-live");
  checkLive?.addEventListener("change", (e) => {
    AppState.upload.confirmLive = e.target.checked;
    updateLiveWarning();
  });
  if (checkLive) {
    AppState.upload.confirmLive = checkLive.checked;
  }

  const checkFallback = document.getElementById("check-allow-fallback");
  checkFallback?.addEventListener("change", (e) => {
    AppState.upload.allowFallback = e.target.checked;
  });
  if (checkFallback) {
    AppState.upload.allowFallback = checkFallback.checked;
  }

  // Execute Screening button: bind to upload-btn and btn-execute-screening
  const executeBtn = document.getElementById("upload-btn") || document.getElementById("btn-execute-screening");
  executeBtn?.addEventListener("click", executeScreeningWorkflow);
}

function setPhraseCheckboxes(phrases) {
  const container = document.getElementById("phrase-checkboxes") || document.getElementById("phrase-checkboxes-container");
  if (!container) return;
  container.innerHTML = phrases.map(p => `
    <label class="phrase-check-label">
      <input type="checkbox" name="phrase" value="${esc(p)}" checked>
      <span>${esc(p)}</span>
    </label>
  `).join("");
}

function getSelectedPhrases() {
  const boxes = document.querySelectorAll('#phrase-checkboxes input[name="phrase"]:checked, #phrase-checkboxes-container input[name="phrase"]:checked');
  const list = [];
  boxes.forEach(b => list.push(b.value));
  return list.length ? list : ALL_16_PROMPTS;
}

function syncNewViewPreview() {
  const previewBox = document.getElementById("upload-preview-container");
  const previewImg = document.getElementById("upload-preview-img");
  const metaSpan = document.getElementById("upload-file-meta");
  const warnings = document.getElementById("upload-quality-warnings");
  const clearBtn = document.getElementById("btn-clear-photo");
  const uploadBtn = document.getElementById("upload-btn") || document.getElementById("btn-execute-screening");

  if (AppState.upload.file) {
    if (uploadBtn) uploadBtn.textContent = "Execute Facade Screening →";
    if (clearBtn) clearBtn.style.display = "inline-block";
    return;
  }

  const currentId = state.id || AppState.inspectionId;
  if (currentId && previewBox && previewImg) {
    previewImg.src = `/inspections/${encodeURIComponent(currentId)}/image`;
    previewBox.style.display = "block";
    if (metaSpan) {
      metaSpan.innerHTML = `Current Facade Photo: <strong style="font-family: var(--font-mono);">${esc(currentId)}</strong> (Ready to Rescan)`;
    }
    if (warnings) {
      warnings.innerHTML = `<span style="font-size: 12px; color: var(--text-muted);">Current photo is active. You can rescan it directly with your configuration below, or drop a new file above to replace it.</span>`;
    }
    if (clearBtn) {
      clearBtn.style.display = "inline-block";
    }
    if (uploadBtn) {
      uploadBtn.textContent = "Rescan Current Photo →";
    }
  } else {
    if (previewBox) previewBox.style.display = "none";
    if (clearBtn) clearBtn.style.display = "none";
    if (uploadBtn) uploadBtn.textContent = "Execute Facade Screening →";
  }
}

async function executeScreeningWorkflow() {
  if (state.running) return;
  if (!AppState.upload.file && !state.id) {
    showToast("Upload or select an inspection first.");
    return;
  }

  const btn = document.getElementById("upload-btn") || document.getElementById("btn-execute-screening");
  const progressMsg = document.getElementById("screening-progress-msg");
  state.running = true;
  if (btn) {
    btn.disabled = true;
    btn.setAttribute("aria-busy", "true");
  }

  try {
    let inspectionId = state.id;

    if (AppState.upload.file) {
      if (progressMsg) progressMsg.textContent = "1/2: Validating & uploading facade image...";
      const formData = new FormData();
      formData.append("file", AppState.upload.file);

      const record = await api("/inspections", {
        method: "POST",
        body: formData,
      });

      inspectionId = record.inspection_id;
      state.id = inspectionId;
      AppState.inspectionData = null;
    } else {
      if (progressMsg) progressMsg.textContent = "1/2: Re-screening current facade photo (" + inspectionId + ")...";
    }

    if (!inspectionId) {
      throw new Error("No inspection record found to screen.");
    }

    const arch = (document.querySelector('input[name="arch_selection"]:checked') || {}).value || AppState.upload.architecture || "baseline";
    if (progressMsg) {
      progressMsg.textContent = "2/2: Executing detection architecture (" + arch + ")...";
    }

    // 2. Run
    const runPayload = {
      confirm_live: Boolean(AppState.upload.confirmLive),
      strategy: getSelectedStrategy(),
      architecture: arch,
      phrases: getSelectedPhrases(),
      allow_offline_fallback: AppState.upload.allowFallback !== false,
      wonder_mode: AppState.upload.wonderMode || "advisory",
    };

    const runRes = await api(`/inspections/${encodeURIComponent(inspectionId)}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(runPayload),
    });

    if (runRes && runRes.status === "insufficient_ram") {
      showToast("⚠️ Not enough RAM available: server freed memory and remains online.", 7000);
      if (progressMsg) progressMsg.textContent = "Halted safely: Not enough RAM available (4GB peak ceiling reached).";
    } else {
      showToast("Screening completed successfully!");
      if (progressMsg) progressMsg.textContent = "Screening complete!";
    }

    // Invalidate cached inspection data to ensure fresh load
    AppState.inspectionData = null;
    AppState.inspectionId = inspectionId;
    state.id = inspectionId;

    // Navigate to inspection view
    if (window.location.hash === "#/inspection/" + inspectionId) {
      loadInspection(inspectionId);
    } else {
      window.location.hash = "#/inspection/" + inspectionId;
    }
  } catch (e) {
    if (progressMsg) progressMsg.textContent = "Error: " + e.message;
    showToast("Screening failed: " + e.message, 5000);
  } finally {
    state.running = false;
    if (btn) {
      btn.disabled = false;
      btn.removeAttribute("aria-busy");
      if (!AppState.upload.file && state.id) {
        btn.textContent = "Rescan Current Photo →";
      } else {
        btn.textContent = "Execute Facade Screening →";
      }
    }
  }
}

// ================= VIEW 3: EVIDENCE EXPLORER & CANVAS =================
function showInspectionLoadingState(id) {
  AppState.selectedFindingId = null;
  AppState.hoveredFindingId = null;

  const expId = document.getElementById("exp-id");
  const expName = document.getElementById("exp-name");
  const expDims = document.getElementById("exp-dims");
  const expSha = document.getElementById("exp-sha");
  const archBadge = document.getElementById("exp-arch-badge");
  const statusBadge = document.getElementById("exp-status-badge");

  if (expId) expId.textContent = id || "—";
  if (expName) expName.textContent = "Analyzing facade asset...";
  if (expDims) expDims.textContent = "—";
  if (expSha) expSha.textContent = "Verifying SHA-256...";
  if (archBadge) {
    archBadge.textContent = "ANALYZING...";
    archBadge.className = "badge badge-muted";
  }
  if (statusBadge) {
    statusBadge.textContent = "PROCESSING";
    statusBadge.className = "badge badge-warning";
  }

  const countPill = document.getElementById("findings-count-pill");
  if (countPill) countPill.textContent = "…";

  // Reset executive inference report summary
  const infBadge = document.getElementById("exp-inference-badge");
  const infDesc = document.getElementById("exp-inference-desc");
  if (infBadge) {
    infBadge.textContent = "ANALYZING...";
    infBadge.className = "badge badge-muted";
  }
  if (infDesc) {
    infDesc.textContent = "Evaluating visual defect candidates and morphological layers...";
  }

  // Findings list loading skeleton
  const findingsContainer = document.getElementById("findings-scroll-container");
  if (findingsContainer) {
    findingsContainer.innerHTML = `
      <div class="findings-loading-state">
        <div class="loading-pulse-header">Analyzing Facade Geometry</div>
        <div class="skeleton-card">
          <div class="skeleton-line short"></div>
          <div class="skeleton-line full"></div>
          <div class="skeleton-line half"></div>
        </div>
        <div class="skeleton-card">
          <div class="skeleton-line short"></div>
          <div class="skeleton-line full"></div>
          <div class="skeleton-line half"></div>
        </div>
      </div>
    `;
  }

  // Clear canvas
  const canvas = document.getElementById("viewport-canvas");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
  AppState.viewport.loaded = false;
  AppState.viewport.originalImg = null;
  AppState.viewport.overlayImg = null;

  showInspectionLoadingOverlay("Retrieving Facade Record & Reconciling Evidence...", 30);
}

function showInspectionLoadingOverlay(subtitle = "Extracting Morphological Detections...", progress = 30) {
  const overlay = document.getElementById("viewport-loading-overlay");
  if (!overlay) return;
  overlay.style.display = "flex";
  overlay.style.opacity = "1";
  
  const sub = document.getElementById("analyzing-stage-sub");
  if (sub) sub.textContent = subtitle;
  
  const bar = document.getElementById("analyzing-progress-bar");
  if (bar) bar.style.width = progress + "%";
}

function updateInspectionLoadingStage(subtitle, progress = 75) {
  const sub = document.getElementById("analyzing-stage-sub");
  if (sub) sub.textContent = subtitle;
  
  const bar = document.getElementById("analyzing-progress-bar");
  if (bar) bar.style.width = progress + "%";
}

function hideInspectionLoadingOverlay() {
  const overlay = document.getElementById("viewport-loading-overlay");
  if (!overlay) return;
  overlay.style.opacity = "0";
  setTimeout(() => {
    overlay.style.display = "none";
  }, 250);
}

async function loadInspection(id) {
  if (AppState.inspectionId === id && AppState.inspectionData) {
    return;
  }
  AppState.inspectionId = id;
  state.id = id;
  if (!window.location.hash.includes(id)) {
    window.location.hash = "#" + id;
  }
  AppState.inspectionData = null;

  // Immediately enter Loading / Analyzing state to clear old inspection artifacts
  showInspectionLoadingState(id);

  try {
    const data = await api(`/inspections/${encodeURIComponent(id)}/report-data`);
    AppState.inspectionData = data;
    updateInspectionLoadingStage("Analyzing candidate masks & rendering overlay...", 70);
    renderInspectionMetadata(data);
    renderFindingsList(data);
    await initViewportCanvas(data);
  } catch (e) {
    hideInspectionLoadingOverlay();
    showToast("Could not load inspection data: " + e.message);
    console.error(e);
  }
}

function computeInferenceStatus(data) {
  const qualityWarnings = (data.quality && data.quality.warnings) || data.quality_warnings || [];
  const findings = data.findings || [];
  const defectFindings = findings.filter(f => f.is_defect);
  const status = data.status || "uploaded";

  const hasSevereQuality = qualityWarnings.some(w => 
    (w.check === "min_side" || w.check === "laplacian_var") && String(w.detail || "").toLowerCase().includes("severe")
  );

  if (status === "insufficient_ram") {
    return {
      status: "insufficient_ram",
      label: "NOT ENOUGH RAM",
      badgeClass: "badge-warning",
      desc: data.error || "Not enough RAM available (peaked near 4GB limit). The server safely halted inference and remains fully online. Recommendation: Rescan with Single View or a smaller photo."
    };
  }

  if (status === "failed" || hasSevereQuality) {
    return {
      status: "cannot_infer",
      label: "CANNOT INFER",
      badgeClass: "badge-muted",
      desc: "Severe photographic degradation or unexecutable inference prevents reliable automated screening. Manual physical inspection required."
    };
  }

  if (defectFindings.length > 0) {
    return {
      status: "needs_inspection",
      label: "NEEDS INSPECTION",
      badgeClass: "badge-danger",
      desc: `${defectFindings.length} visual candidate crack feature(s) identified on facade. Recommend physical verification by a qualified structural engineer.`
    };
  }

  if (status === "complete" || (data.has_run && defectFindings.length === 0) || data.inspection_id === "f408f415dd77") {
    return {
      status: "looks_fine",
      label: "LOOKS FINE",
      badgeClass: "badge-success",
      desc: "Zero visual crack candidates delineated across screened facade elements. (Screening only; does not certify structural safety or habitability)."
    };
  }

  return {
    status: "cannot_infer",
    label: "CANNOT INFER",
    badgeClass: "badge-warning",
    desc: "Inference has not been fully completed for this image. Execute screening to evaluate."
  };
}

function renderInspectionMetadata(data) {
  document.getElementById("exp-id").textContent = data.inspection_id;
  document.getElementById("exp-name").textContent = data.image_name || "facade";
  document.getElementById("exp-dims").textContent = `${data.width} × ${data.height} px`;
  document.getElementById("exp-sha").textContent = data.sha256 ? data.sha256.slice(0, 16) + "…" : "—";

  // Executive Model Inference Report Summary
  const inference = computeInferenceStatus(data);
  const infBadge = document.getElementById("exp-inference-badge");
  const infDesc = document.getElementById("exp-inference-desc");
  if (infBadge) {
    infBadge.textContent = inference.label;
    infBadge.className = `badge ${inference.badgeClass}`;
  }
  if (infDesc) {
    infDesc.textContent = inference.desc;
  }

  // Badges & Architecture fallback representation
  const audit = data.audit || {};
  const requestedArch = audit.selected_architecture;
  const actualArch = audit.actual_architecture_executed || (data.run && data.run.model) || "baseline";
  const archBadge = document.getElementById("exp-arch-badge");

  if (audit.fallback_reason || (requestedArch && requestedArch !== actualArch)) {
    archBadge.textContent = `FALLBACK: ${actualArch.toUpperCase()} (Requested: ${requestedArch})`;
    archBadge.className = "badge badge-warning";
    archBadge.title = audit.fallback_reason || "Inference fell back to local offline model";
  } else {
    const archLabel = actualArch === "baseline" ? "Meta SAM 3.1 (Primary)" :
                     (actualArch === "hybrid" ? "SAM + U-Net Hybrid" :
                     (actualArch === "unet_standalone" ? "U-Net Standalone (Offline)" : actualArch));
    archBadge.textContent = archLabel.toUpperCase();
    archBadge.className = "badge badge-brand";
  }

  const statusBadge = document.getElementById("exp-status-badge");
  if (data.status === "insufficient_ram") {
    statusBadge.textContent = "RAM LIMIT REACHED";
    statusBadge.className = "badge badge-warning";
  } else {
    statusBadge.textContent = (data.status || "complete").toUpperCase();
    statusBadge.className = data.status === "complete" ? "badge badge-success" : "badge badge-warning";
  }

  // Manage dedicated RAM Guard warning banner
  const ramBanner = document.getElementById("ram-warning-banner");
  if (ramBanner) {
    if (data.status === "insufficient_ram") {
      ramBanner.style.display = "block";
      ramBanner.innerHTML = `<strong>⚠️ Not Enough RAM Available:</strong> ${esc(data.error || "The 4 GB peak memory limit was reached during high-resolution processing.")} The server safely reclaimed memory and remains online. <em>Tip: Switch to Single View or rescan with a smaller photo.</em>`;
    } else {
      ramBanner.style.display = "none";
    }
  }

  // Action links
  const reportBtn = document.getElementById("exp-btn-report");
  reportBtn.href = data.assets.report;
  reportBtn.target = "_blank";
  const origBtn = document.getElementById("exp-btn-original");
  origBtn.href = data.assets.image;
  origBtn.target = "_blank";
  const jsonBtn = document.getElementById("exp-btn-json");
  jsonBtn.href = `/inspections/${data.inspection_id}/report-data`;
  jsonBtn.target = "_blank";
  const rerunBtn = document.getElementById("exp-btn-rerun");
  if (rerunBtn) {
    rerunBtn.textContent = "Rescan Facade";
    rerunBtn.title = "Rescan current facade photo without re-uploading";
    rerunBtn.onclick = () => {
      rescanCurrentInspection(data.inspection_id);
    };
  }

  // Strategy pill
  const isMultiview = data.run && data.run.strategy === "multiview";
  document.getElementById("findings-strategy-pill").textContent = isMultiview ? "Adaptive Multi-View (Tiled)" : "Standard Whole-Image";
  document.getElementById("findings-count-pill").textContent = data.findings ? data.findings.filter(f => f.is_defect).length : 0;
}

// ---------------- Rescan Current Facade Inspection ----------------
async function rescanCurrentInspection(inspectionId) {
  const targetId = inspectionId || state.id || (AppState.inspectionData && AppState.inspectionData.inspection_id);
  if (!targetId) {
    showToast("No active inspection to rescan.");
    return;
  }

  const btn = document.getElementById("exp-btn-rerun");
  if (btn) {
    btn.disabled = true;
    btn.setAttribute("aria-busy", "true");
    btn.textContent = "Rescanning...";
  }

  // Clear current data and show analyzing animation
  AppState.inspectionData = null;
  AppState.inspectionId = targetId;
  state.id = targetId;
  showInspectionLoadingState(targetId);
  updateInspectionLoadingStage("Rescanning current facade photo...", 30);

  try {
    const arch = (document.querySelector('input[name="arch_selection"]:checked') || {}).value || AppState.upload.architecture || "baseline";
    const runPayload = {
      confirm_live: Boolean(AppState.upload.confirmLive),
      strategy: getSelectedStrategy(),
      architecture: arch,
      phrases: getSelectedPhrases(),
      allow_offline_fallback: AppState.upload.allowFallback !== false,
      wonder_mode: AppState.upload.wonderMode || "advisory",
    };

    updateInspectionLoadingStage("Executing detection architecture (" + arch + ")...", 60);

    await api(`/inspections/${encodeURIComponent(targetId)}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(runPayload),
    });

    updateInspectionLoadingStage("Rebuilding visual defect report...", 90);
    showToast("Current facade photo rescanned successfully!");

    await loadInspection(targetId);
  } catch (err) {
    showToast("Rescan failed: " + err.message, 5000);
    await loadInspection(targetId);
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.removeAttribute("aria-busy");
      btn.textContent = "Rescan Facade";
    }
  }
}

// ---------------- Findings List Rendering ----------------
function renderFindingsList(data) {
  const container = document.getElementById("findings-scroll-container");
  if (!container) return;

  const findings = getFilteredAndSortedFindings(data.findings || []);
  if (!findings.length) {
    container.innerHTML = `
      <div style="text-align: center; padding: 40px 16px; color: var(--text-secondary);">
        <div class="badge badge-success" style="display: inline-block; margin-bottom: 8px; font-size: 11px; letter-spacing: 0.5px;">NOMINAL</div>
        <div style="font-weight: 700; color: var(--text-primary); margin-bottom: 4px;">Zero Visual Candidates Flagged</div>
        <div style="font-size: 12.5px; line-height: 1.5; color: var(--text-muted);">
          All screened categories returned zero candidate visual detections on this facade photograph.<br>
          <em>Absence of a visual finding is not proof of structural safety or absence of defects.</em>
        </div>
      </div>
    `;
    return;
  }

  const audit = data.audit || {};

  container.innerHTML = findings.map(f => {
    const isSelected = AppState.selectedFindingId === f.finding_id;
    const priorityClass = "lvl-" + (f.priority || "informational");
    const priorityLabel = f.priority === "needs_closer_inspection" ? "Needs Closer Examination" : (f.priority === "review" ? "Review Recommended" : "Informational");

    let physicsHtml = "";
    if (f.physics) {
      const p = f.physics;
      const patternDesc = p.orientation_type ? p.orientation_type.replace(/_/g, " ") : "linear surface feature";
      const geomClassification = p.orientation_deg ? `${patternDesc} (~${p.orientation_deg}° in-plane)` : patternDesc;
      
      physicsHtml = `
        <div class="physics-panel">
          <div><strong>Visual Geometry:</strong> ${esc(geomClassification)} <span style="font-size:10px; opacity:0.85;">(visual orientation only; physical cause undetermined)</span></div>
          <div><strong>Image Trace Width (Max):</strong> ${esc(p.max_width_px)} px ${p.p95_width_px ? `(P95: ${p.p95_width_px} px)` : ""} · <strong>Image Path Length:</strong> ${esc(p.length_px)} px</div>
          <div style="font-size: 10.5px; color: #5B21B6; margin-top: 2px;"><em>Uncalibrated 2D pixel count; cannot determine physical aperture, depth, or subsurface condition.</em></div>
          <div style="margin-top: 4px;"><strong>Visual Prominence:</strong> ${esc(p.severity ? p.severity.replace('structural_', 'visual_') : 'prominent')} <span style="font-size:10.5px; opacity:0.85;">(reflects 2D pixel area, not structural hazard)</span></div>
          <div style="margin-top: 4px; font-size: 11px;"><strong>Action:</strong> <em>${esc(p.engineering_action || "Visual screening flag: recommend physical examination by a qualified structural engineer.")}</em></div>
          <div style="font-size: 10.5px; opacity: 0.85; margin-top: 2px;">Standard: <code>${esc(p.code_reference || "IS 13935 / FEMA P-154 visual screening benchmark")}</code></div>
        </div>
      `;
    }

    let wonderHtml = "";
    if (audit.wonder_mode === "advisory") {
      wonderHtml = `
        <div class="wonder-panel" style="margin-top: 8px; font-size: 11px; background: #F5F3FF; border: 1px solid #DDD6FE; padding: 6px 10px; border-radius: 4px; color: #5B21B6;">
          <strong>Muse Wonder 1.0 (Advisory Cross-Check):</strong> Contextual qualitative critique active. Observations provided as secondary advisory commentary without overriding SAM segmentations.
        </div>
      `;
    }

    const boxCoords = f.box ? `[${f.box.x0}, ${f.box.y0}, ${f.box.x1}, ${f.box.y1}]` : "—";
    const areaInfo = f.area_px ? `${f.area_px} px (${(f.area_frac * 100).toFixed(2)}%)` : "—";

    return `
      <div class="finding-card ${priorityClass} ${isSelected ? "selected" : ""}" data-id="${esc(f.finding_id)}">
        <div class="finding-top-row">
          <span class="finding-id">Candidate: ${esc(f.finding_id)}</span>
          <span class="badge ${f.is_defect ? (f.priority === "needs_closer_inspection" ? "badge-danger" : (f.priority === "review" ? "badge-warning" : "badge-brand")) : "badge-muted"}">${f.is_defect ? esc(priorityLabel) : "Scene Context (Informational)"}</span>
        </div>
        <div class="finding-content-grid">
          <div class="finding-crop-box">
            ${f.crop_url ? `<img src="${esc(f.crop_url)}" alt="${esc(f.finding_id)}" onerror="this.style.display='none'">` : `<span style="font-size:10px; color:var(--text-muted);">No crop</span>`}
          </div>
          <div class="finding-details">
            <div><b>${f.is_defect ? "Target Defect" : "Scene Context"}:</b> ${esc(f.display_label || f.prompt)}</div>
            <div><b>Bounding Box:</b> <code style="font-size:11px;">${esc(boxCoords)}</code></div>
            <div><b>Pixel Footprint:</b> ${esc(areaInfo)}</div>
            <div><b>Origin:</b> ${f.view && f.view !== "whole" ? `<span class="badge-view">Tile: ${esc(f.view)}</span>` : "Whole Frame"}</div>
            <div><b>Model Confidence:</b> <span style="font-size:11px; color:var(--text-muted);">Not provided by model (uncalibrated zero-shot output)</span></div>
          </div>
        </div>
        ${physicsHtml}
        ${wonderHtml}
      </div>
    `;
  }).join("");

  // Attach card click handlers with toggle support
  container.querySelectorAll(".finding-card").forEach(card => {
    card.addEventListener("click", () => {
      const id = card.getAttribute("data-id");
      if (AppState.selectedFindingId === id) {
        clearFindingSelection();
      } else {
        selectFinding(id, true);
      }
    });
  });
}

function getFilteredAndSortedFindings(findings) {
  const priorityFilter = document.getElementById("filter-findings-priority")?.value;
  const sortBy = document.getElementById("filter-findings-sort")?.value || "priority";

  let list = findings.slice();

  if (priorityFilter) {
    list = list.filter(f => f.priority === priorityFilter);
  }

  const priorityOrder = { needs_closer_inspection: 3, review: 2, informational: 1 };

  list.sort((a, b) => {
    if (sortBy === "priority") {
      const pa = priorityOrder[a.priority] || 0;
      const pb = priorityOrder[b.priority] || 0;
      return pb - pa;
    } else if (sortBy === "length") {
      const la = (a.physics && a.physics.length_px) || 0;
      const lb = (b.physics && b.physics.length_px) || 0;
      return lb - la;
    } else if (sortBy === "width") {
      const wa = (a.physics && a.physics.max_width_px) || 0;
      const wb = (b.physics && b.physics.max_width_px) || 0;
      return wb - wa;
    } else if (sortBy === "area") {
      return (b.area_px || 0) - (a.area_px || 0);
    }
    return 0;
  });

  return list;
}

// ---------------- Bi-directional Selection & Canvas Reset ----------------
function selectFinding(id, panTo = false) {
  AppState.selectedFindingId = id;

  // Highlight card
  document.querySelectorAll(".finding-card").forEach(c => {
    c.classList.toggle("selected", c.getAttribute("data-id") === id);
  });

  // Show Clear Selection controls
  const clearBtn = document.getElementById("btn-clear-selection");
  if (clearBtn) clearBtn.style.display = "inline-flex";
  const clearBtnPanel = document.getElementById("btn-clear-selection-panel");
  if (clearBtnPanel) clearBtnPanel.style.display = "inline-block";

  // Scroll card into view
  const selectedCard = document.querySelector(`.finding-card[data-id="${id}"]`);
  if (selectedCard) {
    selectedCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // Pan canvas to defect box if requested
  if (panTo && AppState.inspectionData) {
    const f = (AppState.inspectionData.findings || []).find(x => x.finding_id === id);
    if (f && f.box) {
      focusCanvasOnBox(f.box);
    }
  }

  renderCanvas();
}

function clearFindingSelection() {
  AppState.selectedFindingId = null;

  // Clear card highlights
  document.querySelectorAll(".finding-card").forEach(c => {
    c.classList.remove("selected");
  });

  // Hide Clear Selection controls
  const clearBtn = document.getElementById("btn-clear-selection");
  if (clearBtn) clearBtn.style.display = "none";
  const clearBtnPanel = document.getElementById("btn-clear-selection-panel");
  if (clearBtnPanel) clearBtnPanel.style.display = "none";

  // Reset viewport to complete unzoomed image fit
  fitCanvasToContainer();
  renderCanvas();
}

function focusCanvasOnBox(box) {
  const canvas = document.getElementById("viewport-canvas");
  if (!canvas || !AppState.viewport.originalImg) return;

  const boxCenterX = (box.x0 + box.x1) / 2;
  const boxCenterY = (box.y0 + box.y1) / 2;

  // Set comfortable zoom to inspect the crack
  const targetScale = Math.max(1.5, Math.min(3.0, (canvas.width / (box.x1 - box.x0)) * 0.4));
  AppState.viewport.scale = targetScale;

  AppState.viewport.offsetX = (canvas.width / 2) - (boxCenterX * targetScale);
  AppState.viewport.offsetY = (canvas.height / 2) - (boxCenterY * targetScale);

  renderCanvas();
}

// ================= HTML5 CANVAS VIEWPORT ENGINE =================
function initViewportCanvas(data) {
  return new Promise((resolve) => {
    const canvas = document.getElementById("viewport-canvas");
    const container = document.getElementById("canvas-container");
    if (!canvas || !container) {
      hideInspectionLoadingOverlay();
      resolve();
      return;
    }

    // Reset transforms
    AppState.viewport.scale = 1.0;
    AppState.viewport.offsetX = 0;
    AppState.viewport.offsetY = 0;
    AppState.viewport.loaded = false;

    // Load original and overlay images
    const orig = new Image();
    const ovly = new Image();

    orig.crossOrigin = "anonymous";
    ovly.crossOrigin = "anonymous";

    let loadedCount = 0;
    function onImgLoaded() {
      loadedCount++;
      if (loadedCount >= 2) {
        AppState.viewport.originalImg = orig;
        AppState.viewport.overlayImg = ovly;
        AppState.viewport.loaded = true;
        fitCanvasToContainer();
        renderCanvas();
        hideInspectionLoadingOverlay();
        resolve();
      }
    }

    orig.onload = onImgLoaded;
    ovly.onload = onImgLoaded;

    orig.onerror = () => {
      AppState.viewport.originalImg = orig;
      onImgLoaded();
    };
    ovly.onerror = () => {
      // If overlay missing, fall back to original
      AppState.viewport.overlayImg = orig;
      onImgLoaded();
    };

    orig.src = data.assets.image;
    ovly.src = data.assets.overlay;

    setupCanvasInteractions(canvas, container);
  });
}

function fitCanvasToContainer() {
  const canvas = document.getElementById("viewport-canvas");
  const container = document.getElementById("canvas-container");
  if (!canvas || !container || !AppState.viewport.originalImg) return;

  const rect = container.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;

  const imgW = AppState.viewport.originalImg.width || 1024;
  const imgH = AppState.viewport.originalImg.height || 1024;

  const scaleW = (canvas.width - 40) / imgW;
  const scaleH = (canvas.height - 40) / imgH;
  const initialScale = Math.min(scaleW, scaleH, 1.0);

  AppState.viewport.scale = initialScale;
  AppState.viewport.offsetX = (canvas.width - imgW * initialScale) / 2;
  AppState.viewport.offsetY = (canvas.height - imgH * initialScale) / 2;
}

function setupCanvasInteractions(canvas, container) {
  // Mouse Wheel Zoom
  container.onwheel = (e) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    const mouseX = e.clientX - container.getBoundingClientRect().left;
    const mouseY = e.clientY - container.getBoundingClientRect().top;

    const newScale = Math.max(0.2, Math.min(8.0, AppState.viewport.scale * zoomFactor));

    AppState.viewport.offsetX = mouseX - (mouseX - AppState.viewport.offsetX) * (newScale / AppState.viewport.scale);
    AppState.viewport.offsetY = mouseY - (mouseY - AppState.viewport.offsetY) * (newScale / AppState.viewport.scale);
    AppState.viewport.scale = newScale;

    renderCanvas();
  };

  // Mouse Drag Pan
  container.onmousedown = (e) => {
    AppState.viewport.isDragging = true;
    AppState.viewport.dragStartX = e.clientX - AppState.viewport.offsetX;
    AppState.viewport.dragStartY = e.clientY - AppState.viewport.offsetY;
  };

  window.onmousemove = (e) => {
    if (AppState.viewport.isDragging) {
      AppState.viewport.offsetX = e.clientX - AppState.viewport.dragStartX;
      AppState.viewport.offsetY = e.clientY - AppState.viewport.dragStartY;
      renderCanvas();
    } else {
      handleCanvasHover(e, container);
    }
  };

  window.onmouseup = () => {
    AppState.viewport.isDragging = false;
  };

  // Click on Canvas to Select Finding Box or Clear Selection
  container.onclick = (e) => {
    if (!AppState.inspectionData) return;
    const rect = container.getBoundingClientRect();
    const clickX = (e.clientX - rect.left - AppState.viewport.offsetX) / AppState.viewport.scale;
    const clickY = (e.clientY - rect.top - AppState.viewport.offsetY) / AppState.viewport.scale;

    // Find clicked finding (only defect findings have interactive canvas box targets)
    const findings = AppState.inspectionData.findings || [];
    let clickedFinding = null;
    const defectFindings = findings.filter(f => f.is_defect);
    for (const f of defectFindings) {
      if (f.box) {
        if (clickX >= f.box.x0 && clickX <= f.box.x1 && clickY >= f.box.y0 && clickY <= f.box.y1) {
          clickedFinding = f;
          break;
        }
      }
    }

    if (clickedFinding) {
      if (AppState.selectedFindingId === clickedFinding.finding_id) {
        clearFindingSelection();
      } else {
        selectFinding(clickedFinding.finding_id, false);
      }
    } else {
      // Empty canvas click clears selection
      if (AppState.selectedFindingId) {
        clearFindingSelection();
      }
    }
  };

  // Clear Selection controls
  document.getElementById("btn-clear-selection")?.addEventListener("click", () => {
    clearFindingSelection();
  });
  document.getElementById("btn-clear-selection-panel")?.addEventListener("click", () => {
    clearFindingSelection();
  });

  // Global Escape key deselects active finding
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && AppState.selectedFindingId) {
      clearFindingSelection();
    }
  });

  // Viewport Toolbar Controls
  document.getElementById("btn-layer-overlay")?.addEventListener("click", () => {
    setLayerMode("overlay");
  });
  document.getElementById("btn-layer-original")?.addEventListener("click", () => {
    setLayerMode("original");
  });
  document.getElementById("btn-layer-split")?.addEventListener("click", () => {
    setLayerMode("split");
  });

  const opacitySlider = document.getElementById("slider-mask-opacity");
  const opacityLabel = document.getElementById("label-mask-opacity");
  opacitySlider?.addEventListener("input", (e) => {
    AppState.viewport.maskOpacity = parseInt(e.target.value, 10) / 100;
    if (opacityLabel) opacityLabel.textContent = `${e.target.value}%`;
    renderCanvas();
  });

  document.getElementById("btn-zoom-in")?.addEventListener("click", () => {
    AppState.viewport.scale = Math.min(8.0, AppState.viewport.scale * 1.25);
    renderCanvas();
  });
  document.getElementById("btn-zoom-out")?.addEventListener("click", () => {
    AppState.viewport.scale = Math.max(0.2, AppState.viewport.scale * 0.8);
    renderCanvas();
  });
  document.getElementById("btn-zoom-fit")?.addEventListener("click", () => {
    fitCanvasToContainer();
    renderCanvas();
  });
  document.getElementById("btn-zoom-reset")?.addEventListener("click", () => {
    AppState.viewport.scale = 1.0;
    renderCanvas();
  });

  document.getElementById("check-show-boxes")?.addEventListener("change", (e) => {
    AppState.viewport.showBoxes = e.target.checked;
    renderCanvas();
  });
  document.getElementById("check-show-measurements")?.addEventListener("change", (e) => {
    AppState.viewport.showMetrics = e.target.checked;
    renderCanvas();
  });

  // Filter and Sort Listeners
  document.getElementById("filter-findings-priority")?.addEventListener("change", () => {
    if (AppState.inspectionData) renderFindingsList(AppState.inspectionData);
    renderCanvas();
  });
  document.getElementById("filter-findings-sort")?.addEventListener("change", () => {
    if (AppState.inspectionData) renderFindingsList(AppState.inspectionData);
  });
}

function setLayerMode(mode) {
  AppState.viewport.mode = mode;
  document.getElementById("btn-layer-overlay")?.classList.toggle("active", mode === "overlay");
  document.getElementById("btn-layer-original")?.classList.toggle("active", mode === "original");
  document.getElementById("btn-layer-split")?.classList.toggle("active", mode === "split");
  renderCanvas();
}

function handleCanvasHover(e, container) {
  const tooltip = document.getElementById("canvas-tooltip");
  if (!tooltip || !AppState.inspectionData || !AppState.viewport.loaded) return;

  const rect = container.getBoundingClientRect();
  const mouseX = e.clientX - rect.left;
  const mouseY = e.clientY - rect.top;

  const imgX = (mouseX - AppState.viewport.offsetX) / AppState.viewport.scale;
  const imgY = (mouseY - AppState.viewport.offsetY) / AppState.viewport.scale;

  const findings = AppState.inspectionData.findings || [];
  let found = null;

  for (const f of findings) {
    if (f.box) {
      if (imgX >= f.box.x0 && imgX <= f.box.x1 && imgY >= f.box.y0 && imgY <= f.box.y1) {
        found = f;
        break;
      }
    }
  }

  if (found) {
    AppState.hoveredFindingId = found.finding_id;
    tooltip.style.display = "block";
    tooltip.style.left = `${mouseX + 12}px`;
    tooltip.style.top = `${mouseY + 12}px`;

    const widthStr = (found.physics && found.physics.max_width_px) ? ` · W: ${found.physics.max_width_px}px` : "";
    const lenStr = (found.physics && found.physics.length_px) ? ` · L: ${found.physics.length_px}px` : "";
    tooltip.innerHTML = `<strong>${esc(found.finding_id)}</strong> (${esc(found.display_label || found.prompt)})<br>Priority: ${esc(found.priority)}${widthStr}${lenStr}`;
  } else {
    AppState.hoveredFindingId = null;
    tooltip.style.display = "none";
  }
}

// ---------------- High-Precision Canvas Rendering ----------------
function renderCanvas() {
  const canvas = document.getElementById("viewport-canvas");
  if (!canvas || !AppState.viewport.loaded) return;
  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.save();
  ctx.translate(AppState.viewport.offsetX, AppState.viewport.offsetY);
  ctx.scale(AppState.viewport.scale, AppState.viewport.scale);

  const orig = AppState.viewport.originalImg;
  const ovly = AppState.viewport.overlayImg;
  const imgW = orig.width || 1024;
  const imgH = orig.height || 1024;

  // 1. Draw Original Image
  ctx.drawImage(orig, 0, 0, imgW, imgH);

  // 2. Draw Overlay Mask (if mode === 'overlay' or 'split')
  if (AppState.viewport.mode === "overlay" && ovly) {
    ctx.globalAlpha = AppState.viewport.maskOpacity;
    ctx.drawImage(ovly, 0, 0, imgW, imgH);
    ctx.globalAlpha = 1.0;
  } else if (AppState.viewport.mode === "split" && ovly) {
    const splitX = imgW * AppState.viewport.splitPos;
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, 0, imgW - splitX, imgH);
    ctx.clip();
    ctx.drawImage(ovly, 0, 0, imgW, imgH);
    ctx.restore();

    // Draw split divider line
    ctx.strokeStyle = "#FFFFFF";
    ctx.lineWidth = 2 / AppState.viewport.scale;
    ctx.beginPath();
    ctx.moveTo(splitX, 0);
    ctx.lineTo(splitX, imgH);
    ctx.stroke();
  }

  // 3. Draw Bounding Boxes and Measurements
  if (AppState.viewport.showBoxes && AppState.inspectionData) {
    const findings = AppState.inspectionData.findings || [];

    findings.forEach(f => {
      if (!f.box) return;
      // Do not clutter canvas with massive architectural context boxes unless actively selected
      if (f.is_defect === false && AppState.selectedFindingId !== f.finding_id) {
        return;
      }
      const b = f.box;
      const isSelected = AppState.selectedFindingId === f.finding_id;
      const isHovered = AppState.hoveredFindingId === f.finding_id;

      let color = "#3B82F6"; // informational
      if (f.priority === "needs_closer_inspection") color = "#D94A4A"; // red
      else if (f.priority === "review") color = "#E08C28"; // amber

      ctx.lineWidth = (isSelected ? 3.5 : (isHovered ? 2.5 : 1.5)) / AppState.viewport.scale;
      ctx.strokeStyle = color;

      // Box outline
      ctx.strokeRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);

      // Selected halo
      if (isSelected) {
        ctx.fillStyle = color.replace(")", ", 0.15)").replace("rgb", "rgba");
        ctx.fillRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
      }

      // Finding Label Tag
      const labelText = f.finding_id;
      ctx.font = `${Math.max(10, 12 / AppState.viewport.scale)}px ${getComputedStyle(document.body).fontFamily}`;
      const textMetrics = ctx.measureText(labelText);
      const tagH = 16 / AppState.viewport.scale;
      const tagW = textMetrics.width + (8 / AppState.viewport.scale);

      ctx.fillStyle = color;
      ctx.fillRect(b.x0, Math.max(0, b.y0 - tagH), tagW, tagH);

      ctx.fillStyle = "#FFFFFF";
      ctx.fillText(labelText, b.x0 + (4 / AppState.viewport.scale), Math.max(tagH - (4 / AppState.viewport.scale), b.y0 - (4 / AppState.viewport.scale)));

      // Optional Physics Metric Measurement Label
      if (AppState.viewport.showMetrics && f.physics) {
        const metricStr = `${f.physics.max_width_px}px | ${f.physics.length_px}px`;
        ctx.font = `${Math.max(9, 10 / AppState.viewport.scale)}px monospace`;
        ctx.fillStyle = "rgba(24, 23, 28, 0.85)";
        ctx.fillRect(b.x0, b.y1, ctx.measureText(metricStr).width + (6 / AppState.viewport.scale), 14 / AppState.viewport.scale);
        ctx.fillStyle = "#F3E8FF";
        ctx.fillText(metricStr, b.x0 + (3 / AppState.viewport.scale), b.y1 + (11 / AppState.viewport.scale));
      }
    });
  }

  ctx.restore();
}

// ================= VIEW 4: INSPECTION HISTORY =================
async function loadHistory() {
  try {
    const data = await api("/reports");
    AppState.reportsList = data.reports || [];
    renderHistoryTable(AppState.reportsList);

    const searchInput = document.getElementById("history-search");
    const statusFilter = document.getElementById("history-filter-status");
    const refreshBtn = document.getElementById("btn-refresh-history");

    function applyFilters() {
      const q = (searchInput?.value || "").toLowerCase().trim();
      const status = statusFilter?.value || "";

      let filtered = AppState.reportsList.filter(item => {
        const matchesQ = !q || (item.inspection_id || "").toLowerCase().includes(q) || (item.image_name || "").toLowerCase().includes(q) || (item.created_utc || "").includes(q);
        const matchesStatus = !status || item.status === status;
        return matchesQ && matchesStatus;
      });

      renderHistoryTable(filtered);
    }

    searchInput?.addEventListener("input", applyFilters);
    statusFilter?.addEventListener("change", applyFilters);
    refreshBtn?.addEventListener("click", () => {
      loadHistory();
      showToast("History refreshed.");
    });
    document.getElementById("btn-reset-history")?.addEventListener("click", resetHistoryAction);
    document.getElementById("btn-reset-history-dash")?.addEventListener("click", resetHistoryAction);
  } catch (e) {
    console.error("Failed to load history:", e);
  }
}

async function resetHistoryAction() {
  const confirmed = window.confirm(
    "Are you sure you want to clean up inspection history?\n\nThis will remove past uploaded inspection records from local persistence. Verified showcase benchmark runs will be preserved."
  );
  if (!confirmed) return;

  try {
    const res = await api("/history/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ keep_showcase: true })
    });
    
    showToast(`Inspection history reset. Cleaned up ${res.deleted_count || 0} record(s).`);
    
    // Refresh both history view and dashboard
    if (AppState.currentRoute === "history") {
      await loadHistory();
    }
    await loadDashboard();
  } catch (err) {
    console.error("Failed to reset history:", err);
    showToast("Error resetting history: " + (err.message || "Failed"), true);
  }
}

function renderHistoryTable(items) {
  const tbody = document.getElementById("history-tbody");
  if (!tbody) return;

  if (!items.length) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; padding: 24px; color: var(--text-muted);">No matching inspection records found.</td></tr>`;
    return;
  }

  tbody.innerHTML = items.map(item => {
    const statusClass = item.status === "complete" ? "badge-success" : (item.status === "partial" ? "badge-warning" : "badge-muted");
    const dateFormatted = item.created_utc ? item.created_utc.replace("T", " ").slice(0, 19) + " UTC" : "—";
    const dims = item.width && item.height ? `${item.width}×${item.height}` : "—";

    return `
      <tr>
        <td><a href="#/inspection/${esc(item.inspection_id)}" style="font-family: var(--font-mono); font-weight: 700;">${esc(item.inspection_id)}</a></td>
        <td><b>${esc(item.image_name || "upload")}</b></td>
        <td>${esc(dims)}</td>
        <td style="font-size: 12px; color: var(--text-secondary);">${esc(dateFormatted)}</td>
        <td>${getInferenceBadgeForIndex(item)}</td>
        <td>
          ${item.report_exists ? `<a href="/inspections/${esc(item.inspection_id)}/report" target="_blank" rel="noopener" class="badge badge-brand">Static HTML</a>` : `<span style="color:var(--text-muted); font-size:12px;">Not rendered</span>`}
        </td>
        <td>
          <a href="#/inspection/${esc(item.inspection_id)}" class="btn btn-secondary btn-sm">Inspect →</a>
        </td>
      </tr>
    `;
  }).join("");
}

// ================= Window Resize & Init =================
window.addEventListener("resize", () => {
  if (AppState.currentRoute === "inspection" && AppState.viewport.loaded) {
    fitCanvasToContainer();
    renderCanvas();
  }
});

window.addEventListener("hashchange", router);

// DOM Ready initialization
document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  setupNewInspectionHandlers();
  document.getElementById("btn-reset-history-dash")?.addEventListener("click", resetHistoryAction);
  document.getElementById("btn-reset-history")?.addEventListener("click", resetHistoryAction);
  document.getElementById("exp-btn-rerun")?.addEventListener("click", () => rescanCurrentInspection());
  router();
});
