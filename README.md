# Shuttle hit detection in badminton rally videos

Input: rally video(s) (`.mp4`). Output: a CSV with the frame number of every
shuttle hit, serve included.

```
video ──► TrackNetV3 ──► track cleaning ──► hit detection ──► hits.csv
          shuttle x,y      (this repo)        (this repo)
          per frame
          (pretrained)
```

## Results

Match tolerance ±1 frame. All numbers are from the full pipeline, run end to
end on the original videos.

**Sample clips** (the 3 provided clips: fixed phone camera behind the court, 30 fps):

| clip | hits | correct | false alarms | missed | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| amogh_far_23 | 11 | 10 | 0 | 1 | 1.00 | 0.91 | 0.95 |
| video3_03 | 14 | 14 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| video3_27 | 19 | 19 | 0 | 0 | 1.00 | 1.00 | 1.00 |
| **all** | **44** | **43** | **0** | **1** | **1.00** | **0.98** | **0.99** |

Mean timing error on correct hits: 0.23 frames (~8 ms). Every correct hit is
within ±1 frame of the label.

**Other cameras** (rallies downloaded from YouTube; different venue, 25 fps):

| clip | camera | hits | correct | false alarms | missed | F1 |
|---|---|---|---|---|---|---|
| Best_Badminton_Rallies_1 | broadcast, high behind court | 32 | 26 | 0 | 6 | 0.90 |
| top_view | overhead | 6 | 5 | 3 | 1 | 0.71 |
| stock clip | net height, zoomed, watermarked | n/a | n/a | n/a | n/a | tracker fails (shuttle found in 1% of frames) |

The hit rules carry over to new cameras. Accuracy follows the quality of the
shuttle track: clean on the broadcast view, noisy on the overhead view (which
TrackNet was not trained on), and absent on the zoomed stock clip.

**Efficiency** (MacBook M1 Pro, Apple MPS): TrackNet takes ~3–3.5 min per
15–18 s clip (~2.5 frames/s) and is the bottleneck. Track cleaning plus hit
detection takes ~15 ms per clip. Memory use stays flat whatever the clip
length, because TrackNet runs in streaming mode on a 720p copy.

## Method

### 1. Shuttle tracking: TrackNetV3, unchanged
[TrackNetV3](https://github.com/qaz812345/TrackNetV3) gives the shuttle
position in each frame. Each video is first scaled to 720p. TrackNet works at
512×288 internally, so full HD only costs memory, and every frame is kept so
frame numbers match the original. It runs with `--large_video` (streaming).

### 2. Track cleaning (`hitdet/track.py`)
Raw TrackNet output had these error types on our clips:

| error | cause seen | removed by |
|---|---|---|
| start-up artifact | the first ~3 output frames have identical coordinates in every video | dropping the first 0.1 s |
| single-frame spikes | a second shuttle on the adjacent court | distance from the rolling median, relative to the shuttle's local speed |
| long static detections | the far player's white shoes; the shuttle on the floor | barely moving for 0.5 s (a flying shuttle never does) |
| short isolated fragments | spurious detections between gaps | runs shorter than 0.25 s |

Gaps up to 0.1 s are bridged only when both ends are physically consistent.
Out-of-frame periods are never bridged.

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
- **Rally start:** turns before the shuttle first flies at its typical speed are pre-serve handling and are ignored.
- **Contact slack:** at 25–30 fps contact usually falls between two frames, so the speed-up may show one frame after the sharpest turn.
- **Temporal NMS:** at most one hit per 0.3 s.

Turns rejected by these rules were, in practice, the apex of high shots,
landings, handling before the serve, and tracker drift.

### No pixel or frame constants
Every threshold is an angle, a ratio to the shuttle's own speed, or a duration
in seconds converted with each video's fps (`hitdet/config.py`). Nothing
depends on resolution, zoom, court size or frame rate. The same settings ran
unchanged on 30 fps phone footage and 25 fps broadcast footage, and on both
near-court and far-court players, which appear at very different scales.

### Robustness evidence
**Sensitivity** (sample clips; F1 with one threshold changed, default in brackets):
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
| no weak-turn branch | 0.97 | 0.86 |
| no rally-start rule | 0.99 | 0.86 |

The speed rules are the core of the method. Without them, every apex and
landing becomes a false alarm. The speed-up branch does nothing on the sample
clips but recovers broadcast-view hits. The follow-through rule is a real
trade-off: it prevents landing false alarms on the sample clips but rejects
some soft net shots on the broadcast clip.

## Usage

### Setup
```bash
conda create -n badminton python=3.10 -y && conda activate badminton
pip install torch torchvision          # CUDA, Apple MPS or CPU build
pip install -r requirements.txt
conda install -c conda-forge ffmpeg    # optional; OpenCV is used if missing

git clone https://github.com/qaz812345/TrackNetV3.git third_party/TrackNetV3
# Download the TrackNetV3 checkpoints (link in its README) into
#   third_party/TrackNetV3/ckpts/TrackNet_best.pt
#   third_party/TrackNetV3/ckpts/InpaintNet_best.pt
pip install pycocotools                # needed by TrackNetV3's predict.py
```

### Detect hits
```bash
python detect_hits.py data/videos/                    # every video in a folder
python detect_hits.py a.mp4 b.mp4 -o outputs/hits.csv
python detect_hits.py data/videos/ --track-dir outputs/tracknet   # reuse cached tracks
```
Output (`outputs/hits.csv` by default):

| column | meaning |
|---|---|
| clip | video file name without extension |
| hit_no | 1-based index within the clip (1 = first hit, usually the serve) |
| frame | **0-based** frame index in the original video |
| time_s | frame / fps |
| turn_deg, out_rel, score | evidence for the hit |

### Evaluate, visualise, label
```bash
python -m tools.evaluate --tol 1 2 --robustness       # scores clips in outputs/hits.csv
python -m tools.evaluate --pred outputs/hits_test.csv --tol 1
python -m tools.plot_hits                             # outputs/plots/<clip>_hits.png
python -m tools.label_hits outputs/tracknet/_resized/<clip>.mp4 --track outputs/tracknet/<clip>_shuttle.csv
python -m tools.clean_labels                          # drop non-hit ("unknown") label rows
```

## Repository layout
```
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
- One fixed camera per video, with the main court in view. Other courts may be visible.
- Each video contains one rally, as in the provided samples.
- A hit is labelled at the frame closest to racket–shuttle contact. When contact is hidden by the player's body, it is the frame where the trajectory visibly changes.
- The serve counts as a hit.
- No hit can be detected while the shuttle is out of frame.
- Frame numbers are 0-based, in decode order.

## Limitations
- **Small evaluation set.** 82 labelled hits over 5 clips. Thresholds were set
  on the sample clips. The speed-up branch and the rally-start rule were added
  after studying misses on the broadcast clip, so that clip is a development
  clip, not an unseen test (before the change: 23/32 correct, 0 false alarms).
  The overhead clip was never used to design rules.
- **Accuracy is bounded by the tracker.** A hit is missed if TrackNet loses
  the shuttle around contact (amogh_far_23 frame 386: the shuttle re-entered
  from the top of the frame untracked; broadcast serve, frame 6). Noisy tracks
  cause false alarms (overhead view). Footage unlike TrackNet's training data
  (the zoomed stock clip) yields no track at all.
- **Soft shots near the net** can fall under the follow-through threshold (broadcast frames 276, 454).
- **Frame-rate limit.** At 25–30 fps, contact often falls between frames, so ±1 frame (33–40 ms) is the realistic precision.
- **Label bias.** The sample clips were labelled by reviewing candidate frames from an early detector and then scanning each clip for misses.
- **Untested:** one long video containing many rallies, and doubles play.
- **No hitter attribution.** The labels record near/far, and hits alternate perfectly in every clip, but the detector does not output who hit.

## Future work
- Attribute each hit to a player with YOLO-pose (shuttle–wrist distance
  normalised by player height) plus the near/far alternation constraint. This
  would also flag likely missed hits, and could detect hits with an untracked
  incoming flight from the player's swing.
- Fine-tune TrackNet on the target camera angles to remove shoe and
  adjacent-court confusion at the source.
- Segment long videos into rallies by long pauses with no flying shuttle, and
  apply the rally-start rule per rally.
- Replace the rules with a small temporal classifier (1D-CNN/GRU) on the same
  features once more labelled rallies exist.
- Higher frame-rate capture (60–120 fps) for sub-frame contact timing.

## Credits
TrackNetV3: Chen & Wang, *TrackNetV3: Enhancing ShuttleCock Tracking with
Augmentations and Trajectory Rectification*, MMAsia 2023.
Code and weights: https://github.com/qaz812345/TrackNetV3 (MIT licence).