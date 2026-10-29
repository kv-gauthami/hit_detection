# Report: shuttle hit detection in badminton rally videos

## Problem
Given a rally video, output the frame of every shuttle hit, serve included.
A hit is an event in time, not an object in a frame, so object detection
alone cannot solve it. I treated it as finding the moments where the
shuttle's flight is changed by a racket.

## Approach
1. **Track the shuttle** with pretrained TrackNetV3 (unchanged), which gives
   the shuttle position in each frame.
2. **Clean the track.** On these clips TrackNet's errors were a second
   shuttle on the next court, the far player's white shoes, a grounded
   shuttle, short spurious fragments, and repeated points in the first
   1–3 frames. Each is removed with a simple rule (`hitdet/track.py`).
3. **Detect hits from the trajectory.** Between hits the shuttle flies a
   smooth arc and slows down. A racket redirects it and adds energy, so at a
   hit the shuttle turns sharply *and* leaves faster than its local flight
   speed. The speed condition is what separates hits from the two
   look-alikes: the apex of a high shot (direction changes, but the shuttle
   is slowest) and a landing (it stops). A speed-up rule catches hits along
   the camera's line of sight, which barely turn on screen.

All thresholds are angles, ratios to the shuttle's own speed, or seconds,
so nothing depends on resolution, zoom or frame rate.

I chose rules over a learned model because only three labelled clips were
available: too few to train on, but enough to check physically motivated
rules.

## Evaluation
I labelled every hit frame myself and matched detections within ±1 frame.

| set | clips | F1 |
|---|---|---|
| Development (thresholds set while looking at these) | 3 sample clips | 0.99 |
| Development | broadcast clip | 0.90 |
| **Held out, labelled blind, no retuning** | **doubles rally** | **0.91** (15/18 hits, 0 false alarms) |
| Held out | overhead view | 0.71 |

The development scores are optimistic by construction. The doubles result is
the fair one: different play (doubles), never used for tuning. Its three
misses were a short soft serve, a soft shot and the rally's final shot;
in all three the track was fine and the rules rejected them.

A sensitivity sweep shows F1 flat over wide ranges around each threshold
(e.g. turn angle 40–70°), so the result does not hinge on exact values. An
ablation shows the speed conditions are the core: turn angle alone gives
F1 0.78 with 18 false alarms.

**Runtime:** about 105 s for a 16.5 s clip on an H100; almost all of it is
TrackNet, and hit detection takes about 20 ms. Memory stays constant
because TrackNet runs in streaming mode.

## Limitations
- Small evaluation set (100 hits, 6 clips); only two clips are truly held out.
- Accuracy is bounded by the tracker: a shuttle lost at contact means a missed hit, and zoomed or unusual views give no track at all.
- Soft shots and rally-ending shots fall under the speed and follow-through thresholds.
- Assumes one fixed camera and one rally per video; no hitter attribution.

## Next steps
- Attribute each hit to a player with pose estimation; the near/far alternation would also flag missed hits. Also, this would help in downstream application of player evaluation.
- Fine-tune TrackNet on the target camera views.
- With more labelled rallies, replace the fixed thresholds with a small temporal classifier on the same features, mainly to recover soft shots.

Code, setup and full results: see `README.md`.