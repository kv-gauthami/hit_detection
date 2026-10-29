# Shuttle hit detection in badminton rally videos

Input: rally video(s) (`.mp4`). Output: a CSV with the frame number of every
shuttle hit, serve included.

```
video ──► TrackNetV3 ──► track cleaning ──► hit detection ──► hits.csv
          shuttle x,y      (this repo)        (this repo)
          per frame
          (pretrained)
```

## Quick start
```bash
conda create -n badminton python=3.10 -y && conda activate badminton
bash setup.sh
python detect_hits.py path/to/clip.mp4 -o outputs/hits.csv
```
`setup.sh` installs the Python requirements, clones TrackNetV3 and downloads
its pretrained weights. It installs the default PyTorch build if none is
present; for a specific CUDA version, install torch from pytorch.org first and
the script will keep it. Manual steps are under [Setup](#setup).

## Results

Match tolerance ±1 frame. All numbers are from the full pipeline, run end to
end on the original videos.

The thresholds were set while looking at the sample clips and the broadcast
clip, so those results are **development scores** and optimistic. The doubles
and overhead clips were never used to set anything; they are the fair test.

**Sample clips (development set)**: the 3 provided clips, fixed phone camera behind the court, 30 fps:

| clip | hits | correct | false alarms | missed | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| amogh_far_23 | 11 | 10 | 0 | 1 | 1.00 | 0.91 | 0.95 |
| video3_03 | 14 | 14 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| video3_27 | 19 | 19 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| **all** | **44** | **43** | **0** | **1** | **1.00** | **0.98** | **0.99** |

Mean timing error on correct hits: 0.23 frames (~8 ms). Every correct hit is
within ±1 frame of the label.

**Held-out test: doubles**: an unseen doubles rally, 30 fps, labelled blind
before viewing the detections; no thresholds changed afterwards:

| clip | hits | correct | false alarms | missed | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| doubles | 18 | 15 | 0 | 3 | 1.00 | 0.83 | 0.91 |

Mean timing error: 0.53 frames. The shuttle track was continuous throughout,
so all three misses come from the hit rules, not the tracker
Results are identical at ±1 to ±4 frames, so the misses are genuine,
not timing offsets.

**Other cameras** (rallies downloaded from YouTube; different venue, 25 fps):

| clip | camera | hits | correct | false alarms | missed | F1 |
|---|---|---|---|---|---|---|
| Best_Badminton_Rallies_1 | broadcast, high behind court (development) | 32 | 26 | 0 | 6 | 0.90 |
| top_view | overhead (held out) | 6 | 5 | 3 | 1 | 0.71 |
| stock clip | net height, zoomed, watermarked | n/a | n/a | n/a | n/a | tracker fails (shuttle found in 1% of frames) |

The hit rules carry over to new cameras. Accuracy follows the quality of the
shuttle track: clean on the broadcast view, noisy on the overhead view (which
TrackNet was not trained on), and absent on the zoomed stock clip.

**Efficiency.** TrackNet is the bottleneck; track cleaning plus hit detection
takes ~20 ms per clip.

| machine | clip | end-to-end time |
|---|---|---|
| MacBook M1 Pro (Apple MPS) | 15–18 s clips | ~3–3.5 min (~2.5 frames/s) |
| NVIDIA H100 | 16.5 s doubles clip (494 frames) | ~105 s (~6× real time) |

On the H100, raising TrackNet's batch size did not help (107 s at batch 4,
103 s at batch 16): the time goes into TrackNetV3's CPU-side work in streaming
mode (frame decoding, median-background estimation, preprocessing), not the
network. Memory use stays flat whatever the clip length, because TrackNet runs
in streaming mode on a 720p copy; this is what lets it run on a 16 GB laptop.

## Method

### 1. Shuttle tracking: TrackNetV3, unchanged
[TrackNetV3](https://github.com/qaz812345/TrackNetV3) gives the shuttle
position in each frame. Videos taller than 720p are first scaled to 720p. TrackNet works at
512×288 internally, so full HD only costs memory, and every frame is kept so
frame numbers match the original. It runs with `--large_video` (streaming).

### 2. Track cleaning (`hitdet/track.py`)
Raw TrackNet output had these error types on our clips:

| error | cause seen | removed by |
|---|---|---|
| start-up artifact | in 3 of 7 unrelated videos, the first 1–3 output frames are the same points, starting at (601, 203) | dropping the first 0.1 s |
| single-frame spikes | a second shuttle on the adjacent court | distance from the rolling median, relative to the shuttle's local speed |
| long static detections | the far player's white shoes; the shuttle on the floor | barely moving for 0.5 s (a flying shuttle never does) |
| short isolated fragments | spurious detections between gaps | runs shorter than 0.25 s |



### 3. Hit detection (`hitdet/hits.py`)
**Definition.** A hit is the frame where a racket changes the shuttle's motion.
Between hits the shuttle flies a smooth, decelerating arc. A racket injects
energy, so the shuttle leaves **faster** than it was flying, and usually in a
**new direction**. A frame is a hit if either:

- **Turn hit:** it is the sharpest turn nearby, and
  - turn ≥ 60° with outgoing speed ≥ 1× the local flight speed, or
  - turn ≥ 35° with outgoing speed ≥ 2× (drives and kills that keep roughly the same direction); or
- **Speed-up hit:** it is the sharpest speed-up nearby, with outgoing speed ≥ 2.5× incoming and ≥ 2× local flight speed. A hit seen along the camera's line of sight (common from a high broadcast camera) barely changes direction on screen, only speed.

Both kinds must also show **follow-through**: the shuttle keeps flying
afterwards, which rejects landings.

Rally structure:
- **Rally start:** candidates more than 0.2 s before the shuttle first reaches its typical speed are handling before the serve, and are ignored.
- **Contact slack:** at 25–30 fps contact usually falls between two frames, so the speed-up may show one frame after the sharpest turn.
- **Temporal NMS:** at most one hit per 0.3 s; the candidate with the highest score (turn angle × speed-up) is kept.

Turns rejected by these rules were, in practice, the apex of high shots,
landings, handling before the serve, and tracker drift.

### No pixel or frame constants
Every threshold is an angle, a ratio to the shuttle's own speed, or a duration
in seconds converted with each video's fps (`hitdet/config.py`). Nothing
depends on resolution, zoom, court size or frame rate. The same settings ran
unchanged on 30 fps phone footage and 25 fps broadcast footage, on singles and
doubles, and on near-court and far-court players, which appear at very
different scales.

### Why these values
Each threshold encodes a physical expectation. Values were set by inspecting
the turn and speed signals on the sample clips (`tools.plot_hits`), then
checked with the sweep below: each default sits inside a flat region, not at
a single best point.

Speeds are measured on the shuttle itself: the **typical speed** is its median
per-frame step over the clip, the **local speed** the median step in a 1 s
window around the frame. The turn angle compares the direction over the 2
frames before and the 2 frames after (±0.067 s).

| parameter | value | reason |
|---|---|---|
| `min_turn_deg` | 60° | A return reverses or strongly redirects the shuttle; drag and gravity bend the path only slightly over ±2 frames. |
| `min_out_speed_rel` | 1.0× local | The racket adds energy, so the shuttle leaves at least as fast as it was flying. At an apex or landing it is at its slowest. |
| `weak_turn_deg`, `strong_out_speed_rel` | 35°, 2.0× local | Drives and flat returns change direction little on screen but leave much faster. |
| `accel_ratio` | 2.5× incoming | A hit along the camera's line of sight shows as a speed jump, not a turn. Also requires 2.0× local speed. |
| `contact_slack_s` | 1 frame | Contact falls between frames, so the speed-up can appear one frame after the turn. |
| `follow_s`, `min_follow_speed_rel` | 0.3 s, 0.5× typical | After a hit the shuttle keeps flying; after a landing it stops. |
| `min_hit_gap_s` | 0.3 s | The fastest exchanges are about 0.4 s apart, so closer candidates are the same hit. |
| `rally_start_window_s` | 0.2 s | Window used to find when the shuttle first flies at its typical speed (the serve). |
| `outlier_k`, `outlier_floor_rel` | 6, 1.2 | A point further from the 0.25 s rolling median than 6× the shuttle's speed (plus 1.2× the typical step) is another object. |
| `static_window_s`, `static_range_rel` | 0.5 s, 1.8× typical step | A flying shuttle is never that still for 0.5 s; shoes and a grounded shuttle are. |
| `min_fragment_s` | 0.25 s | Shorter isolated runs are spurious detections. |
| `max_gap_fill_s` | 0.1 s | Short enough to interpolate safely; bridged only if the jump is physically possible. |
| `startup_trim_s` | 0.1 s | TrackNetV3's first 1–3 output frames repeat the same points in several unrelated videos. No hit happens in the first 0.1 s, so dropping them is safe. |

### Robustness evidence
**Sensitivity** (sample clips; F1 with one threshold changed, default in
brackets). A flat row means the exact value does not matter.
```
min_turn_deg           40=0.99  50=0.99  [60]=0.99  70=0.99  80=0.96  90=0.96
min_out_speed_rel      0.7=0.98  0.8=0.98  0.9=0.99  [1.0]=0.99  1.2=0.94  1.4=0.93
strong_out_speed_rel   1.5=0.99  [2.0]=0.99  2.5=0.98  3.0=0.96
min_follow_speed_rel   0.3=0.98  0.4=0.98  [0.5]=0.99  0.6=0.99  0.7=0.98
outlier_k              4=0.97  5=0.99  [6]=0.99  8=0.99  10=0.99
static_range_rel       1.2=0.99  1.5=0.99  [1.8]=0.99  2.2=0.99  2.6=0.99
min_hit_gap_s          0.2=0.99  0.25=0.99  [0.3]=0.99  0.35=0.99
accel_ratio            2.0=0.99  [2.5]=0.99  3.0=0.99  4.0=0.99
```
On the two other-camera clips, F1 stays between 0.81 and 0.89 over the same ranges.

**Ablation** (F1 with one component removed):

| variant | sample clips | other cameras |
|---|---|---|
| full method | **0.99** | **0.86** |
| no track cleaning | 0.94 | 0.82 |
| turn angle only (no speed rules) | 0.78 (18 false alarms) | 0.72 (15 false alarms) |
| no follow-through rule | 0.95 (+4 false alarms) | 0.89 |
| no speed-up branch | 0.99 | 0.82 |
| no contact slack | 0.99 | 0.85 |
| no weak-turn branch | 0.96 | 0.86 |
| no rally-start rule | 0.99 | 0.86 |

The speed rules are the core of the method. Without them, every apex and
landing becomes a false alarm. The speed-up branch does nothing on the sample
clips but recovers broadcast-view hits (and one doubles hit). The
follow-through rule is a real trade-off: it prevents landing false alarms on
the sample clips but rejects some soft shots on the broadcast and doubles
clips.

## Usage

### Setup
Recommended: `bash setup.sh` inside an activated Python 3.10 environment (see
[Quick start](#quick-start)). The script refuses to run if `python` is not
inside an activated conda/venv environment. Manual equivalent:
```bash
conda create -n badminton python=3.10 -y && conda activate badminton
pip install torch torchvision          # CUDA, Apple MPS or CPU build
pip install -r requirements.txt        # includes TrackNetV3's parse, pycocotools
conda install -c conda-forge ffmpeg    # optional; OpenCV is used if missing

git clone https://github.com/qaz812345/TrackNetV3.git third_party/TrackNetV3
# Download the TrackNetV3 checkpoints (Google Drive link in its README) into
#   third_party/TrackNetV3/ckpts/TrackNet_best.pt
#   third_party/TrackNetV3/ckpts/InpaintNet_best.pt
```
Verified from a fresh clone in a new environment on a Linux H100 server, and
on a MacBook M1 Pro.

### Detect hits
```bash
python detect_hits.py data/videos/                    # every video in a folder
python detect_hits.py a.mp4 b.mp4 -o outputs/hits.csv
python detect_hits.py data/videos/ --track-dir outputs/tracknet   # reuse cached tracks
python detect_hits.py a.mp4 --config my_config.json             # override thresholds (JSON of config.py fields)
```
Output (`outputs/hits.csv` by default):

| column | meaning |
|---|---|
| clip | video file name without extension |
| hit_no | 1-based index within the clip (1 = first hit, usually the serve) |
| frame | **0-based** frame index in the original video |
| time_s | frame / fps |
| turn_deg, out_rel, score | evidence for the hit |

Saved tracks in `outputs/tracknet/` are reused automatically; delete them to
force TrackNet to run again.

### Evaluate, visualise, label
```bash
python -m tools.evaluate --tol 1 2 --robustness       # scores clips in outputs/hits.csv
python -m tools.evaluate --pred outputs/hits_test.csv --tol 1
python -m tools.plot_hits --track-dir outputs/tracknet --out outputs/plots   # <clip>_hits.png
python -m tools.label_hits outputs/tracknet/_resized/<clip>.mp4 --track outputs/tracknet/<clip>_shuttle.csv
python -m tools.clean_labels                          # drop non-hit ("unknown") label rows
```
`label_hits` opens an OpenCV window, so it needs a display. On a headless
server, label from a copy of the video with frame numbers burned in instead
(this is how the doubles clip was labelled).

## Repository layout
```
setup.sh              one-command setup (requirements, TrackNetV3, weights)
detect_hits.py        end-to-end CLI: videos -> hits.csv
hitdet/
  config.py           all thresholds (angles, speed ratios, seconds)
  tracknet.py         resize + run TrackNetV3, store <clip>_shuttle.csv
  track.py            load and clean shuttle tracks
  hits.py             trajectory features and hit decision
  evaluate.py         one-to-one event matching, precision/recall
  video.py            frame-accurate video reader
  labels.py           label file I/O
tools/
  label_hits.py       keyboard tool used to label hit frames
  clean_labels.py     turns raw labelling output into ground truth
  evaluate.py         evaluation, sensitivity sweep, ablation
  plot_hits.py        diagnostic plots
labels/
  hits_gt.csv         ground truth: clip, frame, hitter (near/far)
  notes.md            labelling procedure and observed tracker failures
```

## Assumptions
**Video.** One fixed camera (no pan or zoom) with the main court in view;
other courts may be visible. One rally per video, starting before the serve.
At least about 25 fps. A full or near-full court view that TrackNetV3 can
track.

**Shuttle motion.** The hit rules assume:
1. Between hits the shuttle flies a smooth arc and slows down; only a racket changes its motion abruptly.
2. A hit adds energy: the shuttle leaves faster than it was flying. This separates a hit from an apex (slowest point) and a landing (it stops).
3. After a hit the shuttle keeps flying.
4. Hits are at least 0.3 s apart.
5. The serve is the first sustained flight; earlier movement is handling.
6. Hits along the camera's line of sight show mainly as a speed change, hence the speed-up rule.
7. The shuttle is tracked around the contact.

A net-cord touch or the shuttle hitting a player would also be reported as a hit.

**Labels.** The hit frame is the frame closest to racket–shuttle contact, or,
if contact is hidden, where the trajectory visibly changes. The serve counts
as a hit. Frames are 0-based in decode order. A detection matches a label
within ±1 frame.

## Limitations
- **Small evaluation set.** 100 labelled hits over 6 clips. Thresholds were set
  on the sample clips. The speed-up branch and the rally-start rule were added
  after studying misses on the broadcast clip, so that clip is a development
  clip, not an unseen test (before the change: 23/32 correct, 0 false alarms).
  The overhead and doubles clips were never used to design rules; the doubles
  clip is the cleanest held-out test.
- **Accuracy is bounded by the tracker.** A hit is missed if TrackNet loses
  the shuttle around contact (amogh_far_23 frame 386: the shuttle re-entered
  from the top of the frame untracked; broadcast serve, frame 6). Noisy tracks
  cause false alarms (overhead view). Footage unlike TrackNet's training data
  (the zoomed stock clip) yields no track at all.
- **Soft shots** can fall under the speed and follow-through thresholds 
- **Rally-ending shots** with little flight afterwards can fail the follow-through check (doubles frame 404).
- **Frame-rate limit.** At 25–30 fps, contact often falls between frames, so ±1 frame is the realistic precision.
- **Label bias.** The sample clips were labelled by reviewing candidate frames from an early detector and then scanning each clip for misses. The doubles clip was labelled blind.
- **Untested:** one long video containing many rallies.
- 

## Future work
- Attribute each hit to a player with YOLO-pose This would also flag likely missed hits.
- Fine-tune TrackNet on the target camera angles to remove shoe and
  adjacent-court confusion at the source.
- Speed up tracking: faster frame decoding, or TrackNetV3's non-streaming mode
  on machines with enough RAM.
- Replace the rules with a small temporal classifier (1D-CNN/GRU) on the same
  features once more labelled rallies exist; soft serves and soft shots are
  where the rules lose most.
- Higher frame-rate capture (60–120 fps) for sub-frame contact timing.

## Credits
TrackNetV3: Chen & Wang, *TrackNetV3: Enhancing ShuttleCock Tracking with
Augmentations and Trajectory Rectification*, MMAsia 2023.
Code and weights: https://github.com/qaz812345/TrackNetV3 (MIT licence).