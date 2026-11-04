# P0 Selection Fix Report: Dual-Panel Reset, Keyboard Deselection & Canvas State Restoration

## 1. Objectives of the Selection & Reset Fix

To provide a seamless, standard-compliant inspection experience for structural engineers:
1. **Toggle on Re-Click:** Clicking an already selected finding card must instantly deselect it.
2. **Canvas Background Deselection:** Clicking on empty canvas space outside any defect box must deselect and reset the view.
3. **Keyboard Accessibility:** Pressing `Esc` must immediately clear selection and restore the whole-facade framing.
4. **Explicit Visual Controls:** Prominent "Clear Selection" buttons must appear in both the viewport toolbar and the findings list when a finding is active, disappearing when deselected.
5. **Lossless Framing Reset:** Deselection must restore the complete full-facade framing (`fitCanvasToContainer()`) while strictly preserving user view modes (`overlay`, `original`, `split`), mask opacity slider values, and toggle checkbox preferences (`showBoxes`, `showMetrics`).

---

## 2. Implementation Details

### 2.1 The Central Reset Function: `clearFindingSelection()` (`neev/app/static/app.js`)

A dedicated reset routine was implemented to handle all selection teardown and viewport restoration:

```javascript
function clearFindingSelection() {
  AppState.selectedFindingId = null;

  // 1. Hide clear selection UI controls
  const clearBtn = document.getElementById("btn-clear-selection");
  if (clearBtn) clearBtn.style.display = "none";
  const clearBtnNav = document.getElementById("btn-clear-selection-panel");
  if (clearBtnNav) clearBtnNav.style.display = "none";

  // 2. Remove active visual styling from all cards
  document.querySelectorAll(".finding-card").forEach(c => c.classList.remove("selected"));

  // 3. Restore complete whole-facade canvas framing
  fitCanvasToContainer();

  // 4. Redraw canvas with full facade view
  renderCanvas();
}
```

### 2.2 Finding Card Toggle Behavior (`neev/app/static/app.js`)

The card click handler was updated with toggle logic:

```javascript
card.addEventListener("click", () => {
  // If already selected, clicking again deselects and resets view
  if (AppState.selectedFindingId === f.id) {
    clearFindingSelection();
    return;
  }

  // Otherwise, select the finding and frame it
  AppState.selectedFindingId = f.id;
  document.querySelectorAll(".finding-card").forEach(c => c.classList.remove("selected"));
  card.classList.add("selected");

  // Show clear buttons
  const clearBtn = document.getElementById("btn-clear-selection");
  if (clearBtn) clearBtn.style.display = "inline-flex";
  const clearBtnNav = document.getElementById("btn-clear-selection-panel");
  if (clearBtnNav) clearBtnNav.style.display = "inline-block";

  fitFindingToCanvas(f.id);
  renderCanvas();
});
```

### 2.3 Canvas Background Click Listener (`neev/app/static/app.js`)

The canvas container click handler was updated:
- It transforms click coordinates into original image coordinate space.
- It tests if any finding's bounding box encompasses the click point.
- If a finding is clicked, it selects that finding.
- If no finding is hit and a finding is currently selected, it immediately triggers `clearFindingSelection()`.

### 2.4 Global Keyboard Shortcut (`neev/app/static/app.js`)

A global listener was bound to the `Escape` key:

```javascript
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" || e.key === "Esc") {
    if (AppState.selectedFindingId !== null) {
      clearFindingSelection();
    }
  }
});
```

### 2.5 Dual-Panel UI Reset Buttons (`neev/app/static/index.html`)

Two accessible clear controls were added:
1. **Viewport Toolbar Button (`#btn-clear-selection`):** Positioned next to the `Fit` and `1:1` zoom buttons in the canvas header. Displays a clear icon and text: `Clear Selection (Reset View)`.
2. **Findings Panel Header Button (`#btn-clear-selection-panel`):** Positioned in the header of the screened findings panel. Displays `Clear Selection`.

Both buttons are bound to `clearFindingSelection()`.

---

## 3. Preservation of Display Settings

When `clearFindingSelection()` is invoked:
- `AppState.viewport.mode` remains intact (e.g. if the user was in Split Compare or Original Only mode, the mode does not change).
- `AppState.viewport.maskOpacity` remains intact.
- `AppState.viewport.showBoxes` and `AppState.viewport.showMetrics` remain intact.
- Only the spatial zoom transform (`scale`, `offsetX`, `offsetY`) is recalculated using `fitCanvasToContainer()` to guarantee that 100% of the building facade is visible inside the container with proper aspect ratio preservation.

