# Labelling notes

Labels: `hits_gt.csv` (clip, frame, hitter). Frames are 0-based. They
were labelled on the 720p copies, which keep every frame, so they apply
to the original videos unchanged.

Procedure: jump through candidate frames (`candidates.csv`), step ±2
frames to find the contact frame, mark near/far; then play the whole
clip to find any hits not in the candidate list.

Totals: amogh_far_23 11, video3_03 14, video3_27 19 = 44 hits.
Test clips: Best_Badminton_Rallies_1 32, top_view 6.
Near/far alternate in every clip.

## Observed tracker failures (TrackNetV3, pretrained)
- amogh_far_23 370–375, 402–423: locks onto the far player's white shoes.
- amogh_far_23 ~368–377: shuttle leaves the top of the frame; returns and is hit at 386.
- video3_03 243–266: alternates every few frames with a second shuttle at x≈1100 (adjacent court).
- video3_03 461: single-frame jump to x≈1220.
- All clips: high serves leave the top of the frame briefly
- All clips: after the rally the shuttle lies on the floor and is still "detected".
