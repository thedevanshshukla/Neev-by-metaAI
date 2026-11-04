# 3-Minute Video Production Guide & Script: On-Camera Intro + Screen Demo + On-Camera Outro

**Project:** Neev — AI Building Facade Defect Screening Platform  
**Presenter:** Devansh  
**Format:** 
1. **Act 1:** Full-Face Video Intro (Camera) ~25s
2. **Act 2:** System Product Video (Screen Recording + Voiceover) ~2m 05s
3. **Act 3:** Full-Face Video Conclusion (Camera) ~25s  
**Total Runtime:** ~2 minutes 50 seconds to 2 minutes 55 seconds (Under 3-minute limit)

---

## Video Architecture & Transition Flow

```
[0:00 - 0:25]  ACT 1: FULL-FACE VIDEO (Camera)
               Devansh on camera introducing the engineering challenge & Neev.
                     │
                     ▼ (Smooth Cross-Dissolve / Cut)
[0:25 - 2:30]  ACT 2: SYSTEM PRODUCT DEMO (Screen Recording + Voiceover)
               ├── 0:25 - 0:55: Baseline SAM 3.1 Control (High Precision, No False Alarms)
               ├── 0:55 - 1:20: Architecture & Multi-View Tiling Strategy Explanation
               ├── 1:20 - 1:55: Executing Hybrid Model + Voiceover during Analysis Phase
               └── 1:55 - 2:30: Exploring 14 Screened Cracks, Physics Diagnostics & Canvas Reset
                     │
                     ▼ (Smooth Cross-Dissolve / Cut)
[2:30 - 2:55]  ACT 3: FULL-FACE VIDEO (Camera)
               Devansh on camera delivering final engineering takeaways, safety disclaimers & sign-off.
```

---

## Camera & Audio Setup Tips (For the Talking Head Shots)

1. **Camera Framing:** Chest-up framing (medium close-up). Camera lens at eye level.
2. **Lighting:** Light your face from the front (window light or a desk lamp behind your camera/monitor). Avoid harsh backlighting.
3. **Eye Contact:** Look straight into the camera lens, not at your own preview on the screen.
4. **Tone:** Warm, confident, professional. Speak like an engineer explaining a breakthrough project to a respected colleague.

---

# THE SCRIPT

---

## ACT 1: On-Camera Introduction (Full-Face Video)
**Time:** `0:00 – 0:25` (25 seconds)  
**Visual:** Devansh on camera, medium close-up, looking directly into the lens.

### Presenter Cues:
- *Smile, natural hand gestures, steady posture.*
- *Deliver with energy and confidence.*

### Spoken Dialogue:
> *"Hi everyone, I’m Devansh.*
> 
> *Inspecting high-rise building facades today is one of the most hazardous and expensive challenges in civil engineering. Traditional manual inspections require scaffolding, rope access, and hundreds of human hours — yet subtle hairline fractures and joint separations still get missed until catastrophic spalling occurs.*
> 
> *To solve this, I built **Neev** — an AI-powered facade screening platform developed with Meta AI. Neev pairs Meta’s SAM 3.1 foundation model with specialized edge-recovery networks to triage building facades from ordinary photos in seconds.*
> 
> *Let’s dive into the workstation and see how it works."*

*(At "Let's dive into the workstation...", smooth transition / cut to full-screen browser recording).*

---

## ACT 2: System Product Demo (Screen Recording + Voiceover)
**Time:** `0:25 – 2:30` (125 seconds)  
**Visual:** Clean 1080p browser window at `http://127.0.0.1:8000`.

---

### Part A: Baseline SAM 3.1 Control & Semantic Disambiguation (`0:25 – 0:55`)
**Target Tab:** `#/new` $\rightarrow$ Screened View

#### On-Screen Cursor Actions:
1. Start on **Initiate New Facade Screening** (`#/new`).
2. Drop `photo_Sample/crack_wall.jpeg` into the dropzone.
3. Under Architecture, keep **`Meta SAM 3.1 (Primary Production)`**.
4. Under Strategy, click **`Standard Whole-Image`**.
5. Click **`Execute Facade Screening →`**.
6. The view navigates to inspection results:
   - Hover cursor over the summary banner: **`[LOOKS FINE] 0 visual candidate crack feature(s) identified on facade.`**
   - Point to the right sidebar: Highlight `Scene Context: building` and `Scene Context: window` with neutral `Informational` badges.

#### Voiceover Dialogue:
> *"Before an AI model can reliably find structural defects, it must first prove it won't cry wolf on ordinary architectural objects.*
> 
> *Here on the New Inspection tab, we test Meta’s Segment Anything Model 3.1 using a single whole-image pass.*
> 
> *Look at the result: Neev accurately segments the windows and structural building contours as scene context, but reports zero defect candidates. It doesn't mistake a window sill or architectural shadow for a concrete crack. This high specificity eliminates false alarms right from the baseline."*

---

### Part B: Multi-View Tiling & Hybrid Architecture Explanation (`0:55 – 1:20`)
**Target Action:** Click **`New Inspection`** in top navbar.

#### On-Screen Cursor Actions:
1. Click **`New Inspection`** in top bar.
2. The card shows `Current Facade Photo: Ready to Rescan`.
3. Move cursor to **2. Screening Configuration**:
   - Point at **Meta SAM 3.1 (Primary Production)**, then hover over **SAM + U-Net Hybrid (High-Recall)**.
   - Point at **Standard Whole-Image**, then click **Adaptive Multi-View (Tiled 2×2)**.

#### Voiceover Dialogue:
> *"Now, why do we have two execution strategies?*
> 
> *When an entire building facade is compressed into a single standard pass, subtle hairline fractures lose pixel resolution and get washed out.*
> 
> *Neev solves this with Adaptive Multi-View Tiling. The platform slices the facade into overlapping high-resolution tiles, effectively quadrupling the effective pixel density.*
> 
> *And for weathered masonry where micro-cracks blend into stucco textures, we switch from pure SAM 3.1 to our SAM plus U-Net Hybrid architecture."*

---

### Part C: Running the Distressed Facade & Behind-the-Scenes Analysis (`1:20 – 1:55`)
**Target Action:** Select Hybrid + Multi-View $\rightarrow$ Click Execute $\rightarrow$ Pulse Radar Screen.

#### On-Screen Cursor Actions:
1. Ensure **`SAM + U-Net Hybrid (High-Recall)`** is selected (blue border active).
2. Ensure **`Adaptive Multi-View (Tiled 2×2)`** is selected.
3. Click **`Rescan Current Photo →`**.
4. The progress bar updates: `1/2: Validating...` $\rightarrow$ `2/2: Executing detection architecture (hybrid)...`
5. The screen enters the loading animation:  
   *`Processing Morphological Layers... Retrieving Facade Record & Reconciling Evidence`*.
6. Hold on this loading screen while speaking.

#### Voiceover Dialogue:
> *"Let's execute the hybrid screening.*
> 
> *While the engine processes, here is what’s happening behind the scenes: First, every photo receives an immutable SHA-256 cryptographic hash to guarantee legal auditability for engineering reports.*
> 
> *Next, our hybrid engine combines Meta SAM 3.1’s zero-shot foundation understanding with a local high-recall U-Net model. SAM provides macro structural scene awareness, while U-Net isolates micro-fractures along mortar joints.*
> 
> *Finally, our spatial fusion engine resolves overlapping tile detections using spatial intersection-over-union, delivering a single unified, deduplicated facade map."*

---

### Part D: Screened Crack Findings, Physics Diagnostics & Canvas Reset (`1:55 – 2:30`)
**Target Action:** Active Inspection View (`#/inspection/<id>`).

#### On-Screen Cursor Actions:
1. The canvas loads! Point cursor to the top banner:  
   **`[NEEDS INSPECTION] 14 visual candidate crack feature(s) identified on facade.`**
2. Point cursor at the canvas showing 14 red highlight crack masks.
3. In the right sidebar, click **`Candidate: crack#1`** (the prominent vertical masonry fracture).
4. Watch the canvas smoothly zoom in to 134% and center on `crack#1`.
5. Point out the toolbar button: **`Clear Selection (Reset View)`**.
6. Click **`Clear Selection`** (or press the `Escape` key) $\rightarrow$ The canvas smoothly resets to the full-facade scale.
7. Drag the **`Mask Opacity`** slider from 70% to 0% and back to show the bare concrete beneath the mask.

#### Voiceover Dialogue:
> *"And here are the results: 14 distinct visual crack candidates delineated with pinpoint precision.*
> 
> *The system immediately flags the facade as 'Needs Inspection'.*
> 
> *Clicking on this major vertical fracture on the left masonry wall instantly zooms in on the distress zone. Behind every finding, Neev computes deterministic 2D physical morphology: Zhang-Suen skeleton path length, orthogonal distance width profiling, and in-plane shear orientation.*
> 
> *Inspectors can adjust mask opacity to visually verify the bare masonry, and simply hit 'Clear Selection' or tap Escape to restore the full facade view instantly."*

*(Smooth cut / transition back to full-face camera).*

---

## ACT 3: On-Camera Conclusion (Full-Face Video)
**Time:** `2:30 – 2:55` (25 seconds)  
**Visual:** Devansh on camera, medium close-up, looking directly into the lens.

### Presenter Cues:
- *Warm, professional, grounded conclusion.*
- *Deliver with conviction.*

### Spoken Dialogue:
> *"Crucially, Neev is designed as an intelligent triage tool — not a replacement for certified structural engineers. As highlighted in our platform disclaimers, visual 2D screening directs licensed inspectors to the exact zones requiring physical, non-destructive testing.*
> 
> *By combining Meta SAM 3.1’s vision intelligence with local high-recall edge recovery, Neev transforms hours of manual inspection into minutes of objective, auditable evidence.*
> 
> *Thank you for watching Neev — built by Devansh with Meta AI."*

*(Smile, pause 1 second, end recording).*

---

## Time & Word Count Audit

| Segment | Visual Source | Duration | Word Count |
|---|---|---|---|
| **Act 1: Intro** | Full-Face Camera | ~25 sec | ~72 words |
| **Act 2: Demo Part A** | Screen Recording (SAM Baseline) | ~30 sec | ~75 words |
| **Act 2: Demo Part B** | Screen Recording (Strategy & Arch) | ~25 sec | ~65 words |
| **Act 2: Demo Part C** | Screen Recording (Analysis Loader) | ~35 sec | ~85 words |
| **Act 2: Demo Part D** | Screen Recording (Findings & Canvas) | ~35 sec | ~85 words |
| **Act 3: Outro** | Full-Face Camera | ~25 sec | ~68 words |
| **TOTAL** | **Full Video** | **~2m 55s** | **~450 words** |

---

## How to Record This in 3 Simple Steps

1. **Step 1 (Screen Record First):**  
   Open OBS Studio, start screen recording `http://127.0.0.1:8000`, follow the Act 2 mouse cursor instructions step-by-step. Keep it around 2 minutes.
2. **Step 2 (Record Talking Head Shots):**  
   Set up your phone or webcam, record yourself speaking Act 1 (Intro) and Act 3 (Outro). Do 2 or 3 takes until you feel relaxed and natural.
3. **Step 3 (Assemble & Voice Over):**  
   Drop the screen video into your video editor (CapCut, Premiere, DaVinci, or iMovie). Put Act 1 at the start, Act 3 at the end, and record your voiceover over the screen capture in the middle.
