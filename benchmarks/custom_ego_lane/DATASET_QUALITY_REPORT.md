# Custom ego-lane dataset quality report

## Gate result

The validation diversity gate passes. Training contains 455 reviewed records
(211 valid, 244 invalid). Validation contains 90 reviewed records (70 valid, 20
invalid), so false-lane behavior and correct abstention are measurable.

The new independent validation source is `validatoion_video.mp4`: 99.032 s,
640x360, 29.970 FPS. It contains urban residential and light-commercial roads,
unmarked sections, intersections, parked vehicles, curbs, turns, occlusion,
and some one-sided centerline scenes. It is useful for negative validation and
abstention testing; centerline frames without reviewed polylines were rejected.

## Sampling audit

| Video | Split | First | Last | Samples | Approximate distribution | Near-duplicates rejected |
|---|---|---:|---:|---:|---|---:|
| `driving.mp4` | Train | 0.000 s | 59.393 s | 179 | 45 / 45 / 44 / 45 quarters | 0 |
| `project_video.mp4` | Train | 0.000 s | 50.120 s | 180 | 45 / 45 / 45 / 45 quarters | 0 |
| `sample_video.mp4` | Train | 0.000 s | 108.933 s | 167 | 38 / 43 / 43 / 43 quarters | 6 |
| `challenge_video.mp4` | Validation | 0.000 s | 16.016 s | 121 | 31 / 30 / 30 / 30 quarters | 0 |
| `validatoion_video.mp4` | Validation | 0.000 s | 97.597 s | 48 | 16 / 12 / 10 / 10 quarters | 0 |

The new clip includes 40 uniform samples plus eight deduplicated review
candidates from unmarked intervals. All sources cover their complete duration.
Train and validation remain separated by source video.

## Reviewed distribution

| Split | Sampled | Reviewed | Valid | Invalid | Ambiguous | Rejected |
|---|---:|---:|---:|---:|---:|---:|
| Train | 526 | 455 | 211 | 244 | 0 | 71 |
| Validation | 169 | 90 | 70 | 20 | 0 | 79 |

| Split | Left only | Right only | Both visible | Neither visible |
|---|---:|---:|---:|---:|
| Train | 51 | 59 | 101 | 244 |
| Validation | 2 | 5 | 63 | 20 |

| Source | Reviewed | Valid | Invalid | Rejected |
|---|---:|---:|---:|---:|
| `driving.mp4` | 143 | 66 | 77 | 36 |
| `project_video.mp4` | 145 | 145 | 0 | 35 |
| `sample_video.mp4` | 167 | 0 | 167 | 0 |
| `challenge_video.mp4` | 70 | 70 | 0 | 51 |
| `validatoion_video.mp4` | 20 | 0 | 20 | 28 |

Reviewed scene tags overlap. Train: clear 171, intersection 197,
parked/curb 203, crosswalk/stop 58, unmarked 167, curve/turn 232, occlusion
162. Validation: clear 70, intersection 17, parked/curb 20, unmarked 20,
curve/turn 73, occlusion 13.

## Review policy

Automatic proposals remained pending until contact sheets and selected
full-resolution frames were inspected. The 20 clearly unmarked/junction frames
were recorded as `HUMAN_REVIEWED` invalid examples. Twenty-eight frames with a
visible centerline or insufficient support were marked `REJECTED`. Their
`proposal_origin` remains `AUTO_HIGH_CONFIDENCE`; nothing was silently promoted.
Existing training labels were not rebuilt or relabeled.
