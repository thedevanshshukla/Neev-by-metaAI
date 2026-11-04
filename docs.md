Contents

1. [The idea](#idea)
2. [Why it matters](#why)
3. [Who it is for](#users)
4. [Scope](#scope)
5. [How it works](#flow)
6. [What the AI sees](#sees)
7. [Screening checklist](#checklist)
8. [Scoring](#score)
9. [Architecture](#arch)
10. [Prompts and data](#prompts)
11. [Validation](#validate)
12. [App screens](#screens)
13. [Build plan](#plan)
14. [Demo script](#demo)
15. [Rubric fit](#rubric)
16. [Risks and ethics](#risks)
17. [Cost and stack](#cost)
18. [Submission](#submit)
19. [Sources](#sources)

Scope v1 · 31 Oct 2026 · Meta Global AI Developer Hackathon · Indie track

# Neev

Earthquake risk for every building on a street, from one phone video.

- Owner **Sanchit**
- Models **SAM 3.1, Muse Wonder**
- Submit by **Thu 5 Nov, 7:30 AM IST**

The building Neev looks for. Walls make the upper floors stiff, so an open parking floor takes most of the sway and can fail first.

Contents

1. [The idea](#idea)
2. [Why it matters](#why)
3. [Who it is for](#users)
4. [Scope](#scope)
5. [How it works](#flow)
6. [What the AI sees](#sees)
7. [Screening checklist](#checklist)
8. [Scoring](#score)
9. [Architecture](#arch)
10. [Prompts and data](#prompts)
11. [Validation](#validate)
12. [App screens](#screens)
13. [Build plan](#plan)
14. [Demo script](#demo)
15. [Rubric fit](#rubric)
16. [Risks and ethics](#risks)
17. [Cost and stack](#cost)
18. [Submission](#submit)
19. [Sources](#sources)

01

## The idea in simple words

**Neev** (Hindi for foundation) checks how risky a building is in an earthquake, using only a video shot from the street. Someone walks down a lane and films the buildings. Neev finds every building in the video, looks for the same warning signs an engineer looks for during a street survey, and gives each building a risk colour. Every answer comes with the video frame that proves it.

The owner or the RWA gets a simple report in Hindi or English: what looks risky, why it matters, and what to do next.

1. **Film the street**A 1 to 3 minute walk with the phone at chest height. Or pick an area that already has Mapillary street photos.
2. **Neev screens each building**SAM 3.1 finds and tracks buildings and their parts. Muse Wonder fills the screening checklist with photo proof.
3. **Get a ranked list**Red, amber or green per building, owner reports in Hindi and English, and a list engineers can visit in order.

**What Neev is not**Neev is not a safety certificate. It only sees the outside of a building. Its job is to tell an engineer which buildings to visit first.

Main user

RWA committee screening its colony

Input

Street video, or Mapillary photos of an area

Meta models

SAM 3.1 (track + segment), Muse Wonder (judge + report)

Output

Risk colour, checklist with evidence, owner report

Proof

San Francisco soft-story list + hand-labelled Delhi lanes

Physical work

None, or one 15-minute walk in your lane

02

## Why it matters

Delhi sits in a high seismic zone, and most of its buildings were never designed for it. Finding the risky ones today needs trained professionals standing in front of each building. Neev makes that first screening cheap enough for every colony to do.

Zone IVDelhi's seismic zone, with a design ground acceleration of about 0.24g. Home to 3.35 crore people.[IAS Gyan](https://www.iasgyan.in/daily-editorials/a-tectonic-shift-in-thinking-to-build-seismic-resilience)

\~10%Share of buildings in MCD areas that meet earthquake norms, as observed by the Delhi High Court in a PIL.[Legally India](https://www.legallyindia.com/Bar-Bench-Litigation/9-of-10-delhi-buildings-not-earthquake-proof-observes-delhi-hc-in-pil)

88%Share of 100 multi-storey buildings in Ahmedabad's Maninagar with soft storeys, in a 2019 street survey. 51% had heavy overhangs, 29% pounding risk.[PreventionWeb](https://preventionweb.net/quick/19637)

- In the 2001 Bhuj earthquake, buildings with parking on the ground floor in Ahmedabad were damaged much more than those without. Many Delhi buildings use the same stilt design. [RMSI study](https://egov.eletsonline.com/2016/01/delhis-vulnerability-to-high-intensity-earthquakes-analysis-of-a-6-5-magnitude-event/)
- East Delhi and the Yamuna floodplain sit on soft alluvial soil that amplifies shaking. [IAS Gyan](https://www.iasgyan.in/daily-editorials/a-tectonic-shift-in-thinking-to-build-seismic-resilience)
- The standard street screening method (FEMA P-154 style rapid visual screening) works well but must be done by experienced professionals, one building at a time. [Wang et al.](https://web.eecs.umich.edu/~stellayu/publication/doc/2022softstoryEESD.pdf)
- For balance: a 2019 government study based on a sample of 84 buildings classed Delhi's overall earthquake risk as medium. [MoES, Rajya Sabha answer, 2025](https://www.moes.gov.in/sites/default/files/3243English.pdf)

### What a soft storey is

Under IS 1893 (Part 1): 2016, Cl. 7.10.1, a storey is soft when its lateral stiffness is less than 70% of the storey above, or less than 80% of the average of the three storeys above. Stilt parking floors are the common case: infill walls stiffen the upper floors, so the open ground floor takes most of the sideways movement, as the drawing at the top shows. [InfraLens](https://infralens.in/term/soft-storey), [IRJET](https://issuu.com/irjet/docs/analysis_of_soft_storey_mechanism_with_advanced_st)

03

## Who it is for

The demo is built for the RWA. Owners get the plain report; engineers get the ranked list.

| User | Their problem | What Neev gives them | In the hackathon build |
| --- | --- | --- | --- |
| **RWA or society committee** | Dozens of buildings in the colony, no budget to audit all of them | Ranked list of every building, map, shareable report | Main demo |
| **Building owner or resident** | Does not know if the building is risky; an engineer visit costs money | Plain report in Hindi or English: warning signs seen, what they mean, next steps | Yes |
| **Municipal or disaster engineer** | Manual surveys are slow; no building-level data | Priority list with evidence frames, exportable as CSV | CSV export only |
| **Tenant or home buyer** | No way to judge a building before renting or buying | Quick check of one building | Later |
| **Researchers and NGOs** | Missing building inventories for risk models | Anonymised dataset of screened buildings | Later |

04

## Scope

Ship the must-haves by Tuesday night. Start a stretch item only when every must-have works end to end.

### Must ship

- Upload a street video, or fetch Mapillary photos for a map box
- Frame sampling with blur filter
- SAM 3.1 building tracking with stable IDs
- SAM 3.1 facade-part masks and measured features
- Muse Wonder checklist with an evidence frame per answer
- Score, colour band and Re-film rule
- Results page: video with outlines, ranked list, building card
- Owner report in Hindi and English
- Validation numbers from Test A and Test B
- Demo video, write-up, public repo

### Stretch

- Map view using Google Open Buildings footprints
- CSV export for engineers
- Re-screening memory: compare the same building across visits
- Offline mode with Meta open-weights models
- Face and number-plate blur on uploaded videos

### Out of scope

- Structural safety certificates
- Anything inside the building: column sizes, steel, concrete grade
- Soil and liquefaction analysis
- A public map of risky buildings
- Retrofit design

### Definition of done

- A judge can open the live link, pick the sample Laxmi Nagar video, and see ranked buildings within about two minutes.
- Every yes or no answer links to a frame. Every hidden item says unclear.
- The write-up reports precision and recall from Test A, even if they miss the target.

05

## How it works

Six stages turn one video into a ranked list. The highlight moves through the pipeline in order. Tap any stage to see its inputs, the model call, its output, and the trap to avoid.

1. 
2. 
3. 
4. 
5. 
6. 

06

## What the AI sees in one frame

A walkthrough of one frame from a lane with three buildings. It plays step by step; use the buttons to pause or jump.

07

## The screening checklist

These are the warning signs that Indian street surveys record, taken from published rapid visual screening (RVS) forms. Neev only answers what a camera can see from the street. Everything else is marked unknown instead of guessed. [Gandhidham RVS checklist](https://repository.nirmauni.ac.in/jspui/bitstream/123456789/2188/1/ITFCL002-6.pdf), [Agartala RVS survey](https://www.slideshare.net/slideshow/rapid-visual-screening-of-existing-residential-building-of-agartala-city/121055754), [FEMA P-154](https://fema.gov/media-library/assets/documents/15212)

| Warning sign | Why it matters | How Neev checks it | Visible from street |
| --- | --- | --- | --- |
| **Open ground storey (soft storey)** | Stilt parking or shop floors without walls take most of the sway | Prompts: car parked under building, open ground floor, shop shutter. Compare wall area on the ground floor against upper floors. | High |
| **Number of storeys** | Taller buildings face larger earthquake forces | Count window rows and slab lines; Muse Wonder cross-checks | High |
| **Heavy overhangs** | Rooms or balconies sticking out add weight high up, away from the columns | Upper-floor width compared with ground-floor width; balcony masks | High |
| **Pounding** | Buildings touching with floors at different heights hit each other's columns | Gap between neighbouring masks; slab lines at different heights | High |
| **Falling hazards** | Parapets, roof water tanks, hoardings and AC units can fall on the street | Prompts: water tank, hoarding, parapet, air conditioner unit | High |
| **Short columns** | Columns held by half-height walls or ribbon windows are short and stiff, and fail in shear | Column masks partly covered by low walls or high window strips | Medium |
| **Vertical irregularity, floors added later** | Setbacks or extra floors change stiffness and weight suddenly | Muse Wonder compares finish, material and alignment floor by floor | Medium |
| **Apparent quality** | Cracks, exposed steel and damp point to poor construction or upkeep | Prompts: crack in wall, exposed steel bar; needs sharp frames | Medium |
| **Construction type** | RC frame and load-bearing brick behave differently | Visible columns and beams against brick walls | Medium |
| **Plan shape, soil** | Irregular plans twist; soft soil amplifies shaking | Not judged. Shown as unknown. Soil zone can come from maps later. | Low |

08

## Scoring

Neev adds points for each warning sign it can see. FEMA P-154 uses the same idea in reverse: a basic score minus penalties, where a low final score means the building needs a detailed check. [FEMA P-155](https://www.atcouncil.org/docman/fema/17-fema-p-155/file)

**These weights are ours**They are starting values for the demo, not taken from any code or standard. Tune them on the validation set, and ask a structural engineer to review them before showing them as advice.

#### Example: building B1 (G+3, stilt parking, overhangs)

55High

0Medium from 25High from 50100

- Open ground storey +40
- G+3 storeys +5
- Heavy overhangs +10

| Warning sign seen | Points |
| --- | --- |
| Open ground storey | 40 |
| Storeys: G+2 or G+3 | 5 |
| Storeys: G+4 or more | 10 |
| Heavy overhangs | 10 |
| Short columns | 10 |
| Vertical irregularity or floors added later | 10 |
| Pounding with a neighbour | 10 |
| Poor apparent quality | 5 |
| Falling hazards | 5 |

| Band | Score | What the report says |
| --- | --- | --- |
| Low | 0 to 24 | No major warning signs visible from the street |
| Medium | 25 to 49 | Some warning signs; an engineer check is advised |
| High | 50 to 100 | Several warning signs; put this building first for an engineer visit |
| Re-film | Any | Open ground storey is unclear, or 3 or more items are unclear. No colour is given; Neev asks for a better view. |

Unclear answers add zero points and are always listed in the report, so a missing view never looks like a safe building.

09

## System architecture

A small Python backend does the heavy work in a worker; the browser only uploads and shows results. The moving dot follows one job from upload to result.

On a phone, scroll the diagram sideways.

| Step | What happens | Detail |
| --- | --- | --- |
| 1 | User uploads a street video or picks a map box | Web app sends the file or bounding box |
| 2 | Web app creates a job | `POST /jobs` returns a job ID; the page polls `GET /jobs/{id}` for progress |
| 3 | API puts the job on a queue | A background thread is enough for the demo; RQ or Celery later |
| 4 | Worker fetches street images if needed | Mapillary `/images` with a bbox; keep `thumb_2048_url`, `computed_geometry`, `compass_angle` |
| 5 | Worker calls Meta Model API | SAM 3.1 for tracking and parts; Muse Wonder for checklist and reports |
| 6 | Worker saves results | Buildings, checklist JSON, crops and overlays |
| 7 | API reads results | Ranked by score, Re-film buildings listed separately |
| 8 | Web app shows results | Video with outlines, ranked list, building card, report |

10

## Prompts, model calls and data

### SAM 3.1 text prompts

Treat this list as a starting guess. On day one, run each prompt on 10 street photos and keep only the ones SAM handles well.

| Pass | Text prompts | Feeds |
| --- | --- | --- |
| **Buildings** (video) | `building` | Track IDs, best frames per building |
| **Ground floor** | `car parked under building`, `open ground floor`, `shop shutter`, `column` | Open ground storey, short columns |
| **Storeys** | `window`, `floor slab`, `balcony` | Storey count, overhangs |
| **Hazards** | `water tank`, `hoarding`, `parapet`, `air conditioner unit` | Falling hazards |
| **Condition** | `crack in wall`, `exposed steel bar` | Apparent quality |
| **Privacy** | `face`, `license plate` | Blur before storing your own videos |

### Measured features (plain Python on SAM masks)

- **Storeys:** cluster window-mask centres by height; each row is a storey. If this disagrees with Muse Wonder, the answer becomes unclear.
- **Ground-floor openness:** share of the ground-floor band not covered by wall, compared with the floors above.
- **Overhang:** widest upper-floor width divided by ground-floor width. Above 1.1 is a flag.
- **Pounding:** gap between neighbouring building masks under 1% of frame width, plus slab rows that do not line up.

### Muse Wonder judge prompt (draft)

prompts/judge_system.md

```
You are a seismic screening assistant. You get 2 or 3 street photos of ONE
building, the same photos with SAM 3.1 mask overlays, and measured features.

For each checklist item, answer "yes", "no" or "unclear".
- Use only what is visible in the photos.
- If the view is blocked, too far, blurred or at a bad angle, answer "unclear".
- For every yes or no, name the photo that shows it and give one short reason.
- If your storey count differs from the measured one, answer "unclear".
- Never say a building is safe or unsafe. You only record visible warning signs.

Return JSON that matches the schema exactly. No other text.
```

### Output for one building

building.json (example values)

```
{
  "building_id": "LN-B1",
  "best_frames": ["f_0132.jpg", "f_0140.jpg"],
  "storeys": {"label": "G+3", "measured": 4, "judge": 4},
  "checklist": [
    {"id": "open_ground_storey", "answer": "yes", "confidence": 0.92,
     "evidence": "f_0132.jpg",
     "reason": "Two cars parked under the building; no walls between ground-floor columns."},
    {"id": "heavy_overhang", "answer": "yes", "confidence": 0.81,
     "evidence": "f_0140.jpg",
     "reason": "Upper floors extend past the left ground-floor column."},
    {"id": "pounding", "answer": "no", "confidence": 0.74,
     "evidence": "f_0132.jpg", "reason": "Clear gap to B2."},
    {"id": "short_column", "answer": "unclear", "confidence": null,
     "evidence": null, "reason": "Rear columns hidden by parked cars."}
  ],
  "score": 55,
  "band": "high",
  "models": {"segment": "sam-3.1", "judge": "muse-wonder-1.0"}
}
```

### Calling Muse Wonder

Meta Model API takes the OpenAI SDK with base URL `https://api.meta.ai/v1`, a bearer key, and the Responses API. [Meta Model API docs](https://ai.developer.meta.com/docs/), [promptfoo guide](https://www.promptfoo.dev/docs/providers/meta/). The image field names below follow the OpenAI Responses format; confirm them and the JSON-output option on day one.

pipeline/judge.py

```
import base64, json, os
from openai import OpenAI

client = OpenAI(base_url="https://api.meta.ai/v1",
                api_key=os.environ["MODEL_API_KEY"])

def image(path):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}"}

def judge(building):
    content = [{"type": "input_text", "text": json.dumps(building["features"])}]
    content += [image(p) for p in building["crops"] + building["overlays"]]
    resp = client.responses.create(
        model="muse-wonder-1.0",
        instructions=open("prompts/judge_system.md").read(),
        input=[{"role": "user", "content": content}],
    )
    return json.loads(resp.output_text)
```

SAM 3.1 runs through the same API under the model name `sam-3.1`. One call returns detections, masks and video tracks. [Mindverse](https://www.mind-verse.de/news/meta-sam-3-1-bild-video-analyse-api), [Meta announcement](https://digg.com/ai/icmesfai). Take the exact request shape from the docs; the open-source [open-redactor](https://pypi.org/project/open-redactor/) project has a working SAM 3.1 API client and an IoU tracking fallback you can copy.

### Repo layout

```
neev/
  app/          main.py (FastAPI routes), templates/, static/
  pipeline/     frames.py  track.py  parts.py  features.py
                judge.py   score.py  report.py
  data/         mapillary.py  sf_softstory.py
  eval/         test_a_sf.py  test_b_delhi.py  labels_delhi.csv
  prompts/      judge_system.md  report_system.md  checklist_schema.json
  samples/      laxmi_nagar_lane.mp4
  README.md
```

11

## Validation plan

Judges trust a real number more than a polished claim. Run two tests and report both, good or bad.

### Test A: soft storey, San Francisco

1. **Soft-Story list**San Francisco's open dataset of soft-story properties (PDDL licence)
2. **Sample**40 listed addresses and 40 nearby addresses not on the list
3. **Street photo**Nearest Mapillary image facing each building, using compass angle
4. **Neev**Open ground storey: yes, no or unclear
5. **Compare**Precision, recall and unclear rate

- Data: [Soft-Story Properties, DataSF](https://data.sfgov.org/d/beah-shgi). Imagery: [Mapillary API v4](https://www.mapillary.com/developer/api-documentation), CC BY-SA, faces and plates already blurred. [Atlas](https://atlas.co/data-sources/mapillary)
- Bar to beat: trained deep-learning classifiers reached about 87% accuracy on soft-story detection from street view, but needed labelled training data. Neev uses none. [Wang et al.](https://par.nsf.gov/servlets/purl/10472357)
- Caveat to state in the write-up: San Francisco's soft-story buildings are wood frame, unlike Delhi's RC buildings, and some on the list are already retrofitted.

On the list

Not on the list

Neev says yes

True positive

False positive

Neev says no

False negative

True negative

Count unclear answers separately. Report recall with and without them.

### Test B: full checklist, Delhi

- Film 2 or 3 lanes near home, or use Mapillary if coverage is good.
- Hand-label 30 buildings for the items any person can see: open ground floor, storey count, overhangs, pounding, roof tank.
- Report agreement per item and exact-match rate for storey count.

### Test C: consistency

Screen 10 buildings from two different videos. Report how often the answers match. Big differences mean the prompt or the best-frame picker needs work.

| Item | n | Precision | Recall | Unclear |
| --- | --- | --- | --- | --- |
| Open ground storey (Test A) | 80 | to fill | to fill | to fill |
| Open ground storey (Test B) | 30 | to fill | to fill | to fill |
| Storey count, exact (Test B) | 30 | to fill | to fill |  |
| Overhangs, pounding, tank (Test B) | 30 | to fill | to fill | to fill |

12

## App screens

Four screens are enough for the demo. Build them mobile-first; RWA members will open the link on a phone.

Screen 1 · Start

NeevEN · हिं

**Screen a street**

Upload street video

Pick an area on the map

Film in daylight. Walk slowly. One side of the street at a time.

Try the sample lane

Screen 2 · Processing

Lane 4, Laxmi NagarJob 0412

Frames sampled240

Buildings found12

Checking facades7 of 12

Writing reportswaiting

Screen 3 · Street results

B1 · G+3High 55

B3 · G+5Medium 30

B2 · G+2Re-film

Screen 4 · Building card

B1 · G+3High 55

Open ground storeyYes · f132

Heavy overhangsYes · f140

PoundingNo · f132

Short columnsUnclear

Report: हिंदी · English

### Owner report (sample)

#### Building B1 (G+3), Lane 4, Laxmi Nagar · Risk: High

**What we saw:** The ground floor is open for parking, with no walls between the columns. The upper floors stick out past the ground-floor columns on one side.

**What this means:** In a strong earthquake, the open ground floor can sway the most and may fail first.

**What to do:** Ask a structural engineer to check the ground-floor columns, and ask about strengthening (retrofitting) the parking floor.

This is a screening from street photos, not a safety certificate. Address is an example.

#### इमारत B1 (G+3), गली 4, लक्ष्मी नगर · जोखिम: ज़्यादा

**हमने क्या देखा:** नीचे की मंज़िल पार्किंग के लिए खुली है, खंभों के बीच कोई दीवार नहीं है। ऊपर की मंज़िलें एक तरफ़ नीचे के खंभों से बाहर निकली हुई हैं।

**इसका मतलब:** तेज़ भूकंप में खुली निचली मंज़िल सबसे ज़्यादा हिल सकती है और सबसे पहले टूट सकती है।

**क्या करें:** किसी स्ट्रक्चरल इंजीनियर से नीचे के खंभों की जाँच कराएँ और पार्किंग वाली मंज़िल को मज़बूत (रेट्रोफ़िट) कराने के बारे में पूछें।

यह सड़क से ली गई तस्वीरों पर आधारित जाँच है, सुरक्षा प्रमाणपत्र नहीं।

13

## Build plan

Five working days. Tick tasks as you finish them; ticks stay in this browser only.

Sat 31 Oct · evening*Today*

### Setup and smoke tests

- [ ] Meta Model API key works; Muse Wonder answers one image prompt
- [ ] SAM 3.1 on 5 street photos with every prompt in section 10
- [ ] SAM 3.1 video call: note max objects and clip length
- [ ] Mapillary token; check coverage near home and in San Francisco
- [ ] Download the SF Soft-Story CSV
- [ ] Create repo with the layout from section 10

Done when: one photo goes in and masks plus a Wonder answer come out.

Sun 1 Nov*Today*

### Core pipeline

- [ ] Frame sampler with blur filter
- [ ] Building tracking with stable IDs
- [ ] Best-frame picker per building
- [ ] Part masks and measured features
- [ ] If Mapillary is thin: film 2 or 3 lanes (15 minutes)

Done when: a lane video gives B1 to Bn with crops and features JSON.

Mon 2 Nov*Today*

### Judge, score, Test A

- [ ] Judge prompt and JSON schema
- [ ] Scoring and Re-film rule
- [ ] Report writer in Hindi and English
- [ ] Run Test A on 80 SF buildings
- [ ] Start Test B labels

Done when: Test A precision and recall are written down.

Tue 3 Nov*Today*

### App and deploy

- [ ] Upload, processing and results screens
- [ ] Building card and report screen
- [ ] Deploy with the sample lane preloaded
- [ ] Finish Tests B and C; freeze numbers
- [ ] Stretch items only if all of the above work

Done when: a fresh phone can run the sample lane end to end.

Wed 4 Nov*Today*

### Demo and submit

- [ ] Record demo video by 3 PM
- [ ] Write-up and README
- [ ] Check links, repo and live demo in a private window
- [ ] Submit by 11 PM IST

Done when: the submission page shows your entry.

**Hard deadline**Thu 5 Nov, 7:30 AM IST (Wed 4 Nov, 9:00 PM ET)Aim for Wednesday night and keep the morning as buffer.

14

## Demo video script

About 2 minutes 45 seconds. The demo, video included, is 25% of the score, so show real input and real output, not slides.

1. 0:00 to 0:15

   **Hook**Delhi street at eye level. "The Delhi High Court observed that only about one in ten buildings in MCD areas meets earthquake norms. Nobody knows which ones."
2. 0:15 to 0:35

   **Why stilt buildings fail**The soft-storey animation from this page. One line on how screening needs an engineer per building today.
3. 0:35 to 1:25

   **Live run**Upload the Laxmi Nagar lane. Building outlines appear with IDs, open ground floors light up, the checklist fills in with evidence frames.
4. 1:25 to 1:50

   **Building card and report**Open B1, tap a "yes" to jump to its frame, switch the report to Hindi.
5. 1:50 to 2:15

   **Proof**Test A precision and recall, Test B agreement. Say the caveats out loud.
6. 2:15 to 2:35

   **How Meta models do it**SAM 3.1 tracks buildings and parts; Muse Wonder judges with proof; no training data needed.
7. 2:35 to 2:45

   **Vision**"Every RWA screens its colony in an afternoon. Engineers visit the red ones first."

15

## Rubric fit

| Criterion | Weight | How Neev earns it | Proof to show |
| --- | --- | --- | --- |
| **Impact** | 30% | Lives in every seismic city; first screening becomes cheap enough for any colony | Delhi stats, Ahmedabad survey, RWA user story |
| **Creative use of Meta AI tools** | 25% | SAM 3.1 used as a surveying instrument: tracking buildings and measuring facades. Muse Wonder fills an engineering checklist with evidence, zero training | Walkthrough in section 6, evidence frames in the card |
| **Demo** | 25% | Real lane, real outlines, real Hindi report, live link judges can try | Live URL with sample lane preloaded |
| **Technical depth** | 20% | Video tracking, geometric cross-checks, unclear handling, measured validation | Test A, B and C tables |

Weights from the official rules as reported by [MadRobot](https://madrobot.blog/2026/10/03/meta-global-ai-developer-hackathon-1-million-prizes-muse-model-api/). The Indie track is judged on creativity, technical ambition and whether it works. [Event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)

### How Neev differs from earlier work

| Earlier work | What it does | What Neev adds |
| --- | --- | --- |
| [Yu et al., 2020](https://web.eecs.umich.edu/~stellayu/publication/doc/2020softStoryEEEV.pdf); [Wang et al.](https://web.eecs.umich.edu/~stellayu/publication/doc/2022softstoryEESD.pdf) | Trained CNNs that classify or segment soft-story buildings in street view images | No training data; full checklist, not one attribute; evidence per answer |
| [BRAILS](https://github.com/NHERI-SimCenter/BRAILS) and [BRAILS++](https://simcenter.designsafe-ci.org/products/backend-components/brails/) | City-scale building inventories for US hazard models; BRAILS++ includes vision-language model tutorials | Video walk input, Indian RC building signs, owner and RWA reports in Hindi |
| Manual RVS surveys | Trained professionals fill a form per building | Same checklist in minutes per lane; engineers spend time only on red buildings |

16

## Risks and ethics

| Risk | What to do |
| --- | --- |
| **Only the outside is visible** | Call it screening everywhere; unclear answers; never a safety verdict |
| **Muse Wonder states something not in the photo** | Evidence frame required for every yes or no; geometry cross-check; disagreement becomes unclear |
| **SAM misses small parts** | 2048 px images, best-frame picker, crop per building, ask for a re-film |
| **SAM 3.1 video limits unknown** | Test on day one; fall back to per-frame calls with IoU tracking |
| **Thin Mapillary coverage in Delhi** | Own 15-minute lane video; keep Mapillary for Test A |
| **SF buildings differ from Delhi's** | Say so in the write-up; Test B covers Delhi |
| **Muse Image is blocked in India** | Not needed; Neev uses SAM 3.1 and Muse Wonder only. [ReconScribe](https://reconscribe.com/meta-global-ai-developer-hackathon-2026/) |
| **Solo, 4.5 days** | Strict must-have list; stretch items only after Tuesday's end-to-end run |

### Ethics and privacy

- **No public shaming map.** Reports go to the owner or the RWA that ran the screening. Labelling buildings in public can hurt residents and property values.
- **Careful words.** Reports say "warning signs seen", never "unsafe" or "will collapse".
- **Faces and plates.** Mapillary blurs them already. For uploaded videos, blur with SAM 3.1 before storing.
- **Data for Meta training.** Use Standard mode on the API, not Contributor mode, for any real addresses. [ReconScribe](https://reconscribe.com/meta-global-ai-developer-hackathon-2026/)
- **Your rights.** You keep ownership, but organisers get a perpetual licence to use the submission. Do not use Buzzboard code or data. [MadRobot](https://madrobot.blog/2026/10/03/meta-global-ai-developer-hackathon-1-million-prizes-muse-model-api/)
- **Attribution.** Mapillary images are CC BY-SA; credit them on the results page.

17

## Cost, limits and tech stack

| Item | Price or limit |
| --- | --- |
| SAM 3.1 images | $2.50 per 1,000 images |
| SAM 3.1 video | $0.20 per 1,000 frames |
| SAM 3.1 rate limit | 3,000 requests per minute |
| Muse Wonder | $4 per million input tokens, $20 per million output tokens |
| Hackathon credits | $150 of Meta Model API credits per participant |

SAM 3.1 prices, rate limit and credits: [ReconScribe](https://reconscribe.com/meta-global-ai-developer-hackathon-2026/). Muse Wonder prices: [YourStory](https://yourstory.com/ai-story/muse-wonder-meta-ai-race-frontier-models).

**Rough cost of one lane** (estimate; measure real token counts on day one): a 2-minute video at 2 frames a second is 240 frames, about $0.05 for tracking. 30 buildings × 3 crops × 6 part prompts is 540 image calls, about $1.35. Judging 30 buildings at about 5,000 input and 800 output tokens each is about $1.10. So one lane costs around $2.50, and the whole build should stay well under $50.

| Layer | Pick | Why |
| --- | --- | --- |
| Backend | Python 3.11, FastAPI | Fast to build, async uploads, easy job endpoints |
| Meta calls | OpenAI Python SDK, base URL `https://api.meta.ai/v1` | Officially supported by Meta Model API |
| Video and images | ffmpeg, OpenCV, Pillow | Frame sampling, blur score, crops, overlays |
| Street photos | Mapillary API v4, or [ZenSVI](https://pypi.org/project/zensvi/) | Area download in a few lines |
| Storage | SQLite and a local folder | Zero setup for a demo |
| Front end | Server-rendered HTML with HTMX, Leaflet for the map | Mobile-friendly without a build step; use Next.js if you are faster in it |
| Hosting | Render, Railway or Modal | Modal if you received partner credits. [Event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon) |

18

## Submission checklist

The submission needs a demo video and a project write-up; check the form for any extra fields on day one. [Participant notes](https://github.com/yokoten-labs/yokoten/issues/386)

- Demo video, about 2 to 3 minutes, public link
- Public repo with README: setup, models used, what was built during the hackathon
- Write-up: problem, users, how SAM 3.1 and Muse Wonder are used, validation numbers, limits, next steps
- Live demo link with the sample lane preloaded
- Screenshots of the results page, building card and Hindi report
- Indie track selected; one submission only
- Mapillary attribution and data licences listed in the README

19

## Sources for building

### Meta models and API

- [Meta Model API docs](https://ai.developer.meta.com/docs/)Base URL, auth, Responses API, SDK setup
- [promptfoo: Meta provider](https://www.promptfoo.dev/docs/providers/meta/)Confirms base URL and the Responses endpoint
- [SAM 3.1 on Model API (German)](https://www.mind-verse.de/news/meta-sam-3-1-bild-video-analyse-api)Model name sam-3.1, one call for detect, mask and track
- [SAM 3.1 launch summary](https://digg.com/ai/icmesfai)Detections, masks and identity-preserving video tracks
- [SAM 3.1 model card (mirror)](https://huggingface.co/unsloth/sam3.1)Text prompts, Object Multiplex tracking; points to facebook/sam3.1
- [open-redactor](https://pypi.org/project/open-redactor/)Working SAM 3.1 API client, caching, IoU tracking fallback
- [Muse Wonder overview](https://yourstory.com/ai-story/muse-wonder-meta-ai-race-frontier-models)Inputs and API pricing
- [Muse Wonder review (Portuguese)](https://www.eesel.ai/pt/blog/meta-muse-wonder-analise)Model ID muse-wonder-1.0, 1,048,576-token context

### Screening method and codes

- [FEMA P-154 handbook, 3rd edition](https://fema.gov/media-library/assets/documents/15212)The rapid visual screening procedure
- [FEMA P-155](https://www.atcouncil.org/docman/fema/17-fema-p-155/file)Technical basis for the scoring
- [RVS in Gandhidham (Nirma University)](https://repository.nirmauni.ac.in/jspui/bitstream/123456789/2188/1/ITFCL002-6.pdf)Indian checklist: soft storey observables, overhangs, falling hazards
- [RVS survey of Agartala](https://www.slideshare.net/slideshow/rapid-visual-screening-of-existing-residential-building-of-agartala-city/121055754)Parameters and how often each was found
- [IS 1893 soft storey (InfraLens)](https://infralens.in/term/soft-storey)Cl. 7.10 definition and design rule
- [Soft storey mechanism (IRJET)](https://issuu.com/irjet/docs/analysis_of_soft_storey_mechanism_with_advanced_st)Why stilt floors fail

### Imagery and datasets

- [Mapillary API v4](https://www.mapillary.com/developer/api-documentation)Images, fields like thumb_2048_url and compass_angle
- [Mapillary API FAQ](https://www.mapillary.com/developer/api-documentation/faq)Python examples for bbox and image queries
- [Mapillary download script](https://gist.github.com/cbeddow/93cd66a9122bcecc32ac1913369511d8)Fetch images for an area through vector tiles
- [Mapillary licence notes](https://atlas.co/data-sources/mapillary)CC BY-SA 4.0, faces and plates blurred
- [ZenSVI](https://pypi.org/project/zensvi/)Python package to download and process street view images
- [SF Soft-Story Properties](https://data.sfgov.org/d/beah-shgi)Ground truth for Test A, PDDL licence
- [Google Open Buildings](https://atlas.co/data-sources/google-open-buildings/)Building footprints across India for the map
- [Google + Microsoft buildings, combined](https://gee-community-catalog.org/projects/global_buildings/)India layer as GeoParquet or in Earth Engine

### Earlier research and code

- [Yu et al. 2020](https://web.eecs.umich.edu/~stellayu/publication/doc/2020softStoryEEEV.pdf)Soft-story classification from street view
- [Wang et al.](https://web.eecs.umich.edu/~stellayu/publication/doc/2022softstoryEESD.pdf)Soft-story instance segmentation with semi-automatic labels
- [NSF repository copy](https://par.nsf.gov/servlets/purl/10472357)Summary of earlier accuracy figures
- [BRAILS on GitHub](https://github.com/NHERI-SimCenter/BRAILS)Building inventories from images, soft-story module
- [BRAILS++](https://simcenter.designsafe-ci.org/products/backend-components/brails/)Latest version, vision-language model tutorial

### Hackathon

- [Event page](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)Themes, tracks, dates, partner credits
- [Official rules summary (MadRobot)](https://madrobot.blog/2026/10/03/meta-global-ai-developer-hackathon-1-million-prizes-muse-model-api/)Judging weights, prizes, licence terms
- [ReconScribe guide](https://reconscribe.com/meta-global-ai-developer-hackathon-2026/)Pricing, rate limits, India API notes, submission notes

### Context and statistics

- [Delhi seismic risk (IAS Gyan)](https://www.iasgyan.in/daily-editorials/a-tectonic-shift-in-thinking-to-build-seismic-resilience)Zone IV, 0.24g, soft soil, population
- [Delhi HC observation](https://www.legallyindia.com/Bar-Bench-Litigation/9-of-10-delhi-buildings-not-earthquake-proof-observes-delhi-hc-in-pil)About 10% compliance in MCD areas
- [RMSI Delhi scenario](https://egov.eletsonline.com/2016/01/delhis-vulnerability-to-high-intensity-earthquakes-analysis-of-a-6-5-magnitude-event/)Stilt parking damage seen in Bhuj
- [Ahmedabad RVS findings](https://preventionweb.net/quick/19637)88% soft storeys in a 100-building survey
- [MoES Rajya Sabha answer, 2025](https://www.moes.gov.in/sites/default/files/3243English.pdf)Government view: Delhi classed medium risk

Neev scope, version 1, 31 October 2026. Prepared for Sanchit's Indie track entry. Figures marked as examples or estimates are not measurements.