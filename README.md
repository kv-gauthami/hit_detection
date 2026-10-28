# Shuttle hit detection in badminton rally clips

Given rally clips (`.mp4`), output the frame number of every shuttle hit
(serve included) as a CSV.

```
video ──► TrackNetV3 (shuttle x,y per frame) ──► track cleaning ──► hit detection ──► hits.csv
          third-party, pretrained                   this repo          this repo
```

## Results on the three sample clips

44 hand-labelled hits (11 + 14 + 19), match tolerance ±1 frame (±33 ms):

| clip | hits | TP | FP | FN | precision | recall | mean frame error |
|---|---|---|---|---|---|---|---|
| amogh_far_23 | 11 | 11 | 0 | 0 | 1.00 | 1.00 | 0.00 |
| video3_03 | 14 | 14 | 0 | 0 | 1.00 | 1.00 | 0.00 |
| video3_27 | 19 | 19 | 0 | 0 | 1.00 | 1.00 | 0.11 |
| **all** | **44** | **44** | **0** | **0** | **1.00** | **1.00** | **0.05** |

Hit detection itself takes ~20 ms per clip; the run time is dominated by
TrackNet. Read these numbers with the caveats under
[Limitations](#limitations): three clips, one camera setup, and
thresholds chosen on the same clips. The sensitivity study below is
the evidence that they are not overfitted.

## Method

### 1. Shuttle tracking (TrackNetV3, unchanged)
[TrackNetV3](https://github.com/qaz812345/TrackNetV3) gives the shuttle
position in each frame. Clips are first scaled to 720p: TrackNet works at
512×288 internally, so 1080p input only costs memory (the 720p and 1080p
tracks were identical on our clips). It runs in streaming mode
(`--large_video`), so memory use does not grow with clip length.

### 2. Track cleaning (`hitdet/track.py`)
Raw TrackNet output had three error types on these clips:

| error | cause seen in the clips | removed by |
|---|---|---|
| single-frame spikes | a second shuttle on the adjacent court; bright objects | distance from the rolling median, relative to the shuttle's local speed |
| long static detections | the far player's white shoes; the shuttle lying on the floor | barely moving for 0.5 s (a flying shuttle never does) |
| short isolated fragments | spurious detections between gaps | runs shorter than 0.25 s |

Gaps up to 0.1 s are bridged only when both ends are consistent with
how fast the shuttle can move. Out-of-frame periods are never bridged.

### 3. Hit detection (`hitdet/hits.py`)
**Definition.** A hit is the frame where a racket changes the shuttle's
motion. Between hits the shuttle flies a smooth, decelerating arc. At a
hit two things happen together:

1. **direction changes sharply**: turn angle between incoming and
   outgoing motion (±2 frames);
2. **energy is injected**: outgoing speed ≥ the local flight speed.

The decision combines these with the rally structure:

- *sharp hit*: turn ≥ 60° and outgoing speed ≥ 1.0 × local speed;
- *driven hit*: turn ≥ 35° and outgoing speed ≥ 2.0 × local speed (drives and kills that keep roughly the same direction);
- *follow-through*: the shuttle keeps flying afterwards (rejects landings);
- *serve gate*: the rally starts at the first strong launch (ignores handling before the serve);
- *contact slack*: at 30 fps contact usually falls between two frames, so the speed-up may appear one frame after the sharpest turn;
- *temporal NMS*: at most one hit per 0.3 s.

Turns that fail these rules are, in practice, the apex of high shots, the
shuttle landing, handling before the serve, and tracking drift.

### Robustness: no pixel or frame constants
Every threshold is an angle, a ratio to the shuttle's own speed, or a
duration in seconds converted with each clip's fps (`hitdet/config.py`).
Nothing depends on resolution, zoom, court size or frame rate. The
near-court and far-court players appear at very different scales here,
and the same thresholds work for both.

**Sensitivity** (F1 with one threshold changed, default in brackets):
```
min_turn_deg           40=0.99  50=1.00  [60]=1.00  70=0.99  80=0.99  90=0.99
min_out_speed_rel      0.7=1.00  0.8=1.00  0.9=1.00  [1.0]=1.00  1.2=0.96  1.4=0.94
strong_out_speed_rel   1.5=1.00  [2.0]=1.00  2.5=0.99  3.0=0.98
min_follow_speed_rel   0.3=1.00  0.4=1.00  [0.5]=1.00  0.6=1.00  0.7=0.99
outlier_k              4=0.99  5=1.00  [6]=1.00  8=0.99  10=0.99
static_range_rel       1.2=1.00  1.5=1.00  [1.8]=1.00  2.2=1.00  2.6=1.00
min_hit_gap_s          0.2=1.00  0.25=1.00  [0.3]=1.00  0.35=1.00
```

**Ablation** (one component removed):

| variant | TP | FP | FN | F1 |
|---|---|---|---|---|
| full method | 44 | 0 | 0 | 1.00 |
| no track cleaning | 42 | 6 | 2 | 0.91 |
| turn angle only (no speed rules) | 41 | 14 | 3 | 0.83 |
| no weak-turn branch | 41 | 0 | 3 | 0.97 |
| no follow-through rule | 44 | 1 | 0 | 0.99 |
| no contact slack | 43 | 0 | 1 | 0.99 |
| no serve gate | 44 | 1 | 0 | 0.99 |

## Usage

### Setup
```bash
conda create -n badminton python=3.10 -y && conda activate badminton
pip install torch torchvision          # CUDA, Apple MPS or CPU build
pip install -r requirements.txt

git clone https://github.com/qaz812345/TrackNetV3.git third_party/TrackNetV3
# download the TrackNetV3 checkpoints (link in its README) into
#   third_party/TrackNetV3/ckpts/TrackNet_best.pt
#   third_party/TrackNetV3/ckpts/InpaintNet_best.pt
# then install any of its missing imports (e.g. pip install pycocotools)
```
`ffmpeg` is used for resizing if installed (`conda install -c conda-forge ffmpeg`);
otherwise OpenCV is used.

### Detect hits
```bash
python detect_hits.py data/videos/                    # all clips in a folder
python detect_hits.py clip.mp4 -o outputs/hits.csv    # specific clips
python detect_hits.py data/videos/ --track-dir outputs/tracknet   # reuse cached tracks
```
Output `outputs/hits.csv`:

| column | meaning |
|---|---|
| clip | video file name without extension |
| hit_no | 1-based index within the rally (1 = serve) |
| frame | **0-based** frame index in the original video |
| time_s | frame / fps |
| turn_deg, out_rel, score | evidence for the hit |

### Evaluate and visualise
```bash
python -m tools.evaluate --tol 1 2 --robustness     # tables above
python -m tools.plot_hits                           # outputs/plots/<clip>_hits.png
python -m tools.label_hits data/videos/clip.mp4 --track outputs/tracknet/clip_shuttle.csv
```

## Repository layout
```
detect_hits.py        end-to-end CLI: videos -> hits.csv
hitdet/
  config.py           all thresholds (relative units / seconds)
  tracknet.py         resize + run TrackNetV3
  track.py            load and clean shuttle tracks
  hits.py             trajectory features and hit decision
  evaluate.py         event matching, precision/recall
  video.py            frame-accurate video reader
  labels.py           label file I/O
tools/
  label_hits.py       keyboard tool used to label hit frames
  evaluate.py         evaluation, sensitivity sweep, ablation
  plot_hits.py        diagnostic plots
labels/
  hits_gt.csv         ground truth: clip, frame, hitter (near/far)
  notes.md            labelling notes and observed failure cases
```

## Assumptions
- One fixed camera per clip. Each clip is one rally, starting before the serve.
- Singles play on the main court. Other courts may be visible.
- A hit is labelled at the frame closest to racket–shuttle contact. When contact is hidden (near player's body), it is the frame where the trajectory visibly changes.
- The serve counts as a hit.
- No hit happens while the shuttle is out of frame (players are always in frame).
- Frame numbers are 0-based, in decode order.

## Limitations
- **Small evaluation set.** 44 hits from 3 clips with the same camera
  setup, and thresholds chosen on these clips. The wide sensitivity
  plateaus suggest no overfitting, but other camera angles and
  doubles play are untested.
- **Label bias.** Labels were made by reviewing candidate frames from an
  early version of the detector and then scanning each clip for misses.
  This can bias labels towards the detector's notion of the contact frame.
- **Frame-rate limit.** At 30 fps contact often falls between frames, so
  ±1 frame (±33 ms) is the realistic precision.
- **Depends on the tracker.** A hit is missed if TrackNet loses the
  shuttle around the contact (heavy blur, occlusion by the near player).
  Hits whose surrounding flight is out of frame cannot be detected.
- **No hitter attribution yet.** The labels record near/far, and hits
  alternate perfectly in all three clips, but the detector does not
  output who hit.

## Future work
- Attribute each hit to the near or far player with YOLO-pose
  (shuttle–wrist distance normalised by player height) plus the
  alternation constraint, which would also flag missed hits.
- Fine-tune TrackNet on this camera angle to remove the shoe and
  adjacent-court confusions at the source.
- Replace the rules with a small temporal classifier (1D-CNN/GRU) on the
  same features once more labelled rallies exist.
- Higher frame rate capture (60–120 fps) for sub-frame contact timing.

## Credits
- TrackNetV3: Chen & Wang, *TrackNetV3: Enhancing ShuttleCock Tracking
  with Augmentations and Trajectory Rectification*, MMAsia 2023.
  Code and weights: https://github.com/qaz812345/TrackNetV3 (MIT).
