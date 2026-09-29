# Limitations

This project is a feature-frozen engineering system for CPU road-scene
perception. It is not a safety-certified or production autonomous-driving stack.

## Hardware and deployment

Development and validation focused on one Intel i5-6300U Linux laptop. Performance
can change with CPU frequency, thermal state, runtime versions, video decoding,
and display environment. Other operating systems and accelerators have not been
validated. The normal application requires an OpenCV GUI and a desktop session.

## Detection

The detector can miss, duplicate, or misclassify objects under blur, occlusion,
small scale, unusual viewpoints, adverse lighting, and domain shift. Tracking
cannot recover an object the detector never observes. Stable identity does not
guarantee a correct semantic subtype.

## Tracking

Association is image-space and heuristic. Long occlusion, crowded crossings,
abrupt scale changes, scene cuts, and similar nearby objects can cause ID
switches or expiry. Coasting tracks are predictions with explicit limits, not
new observations.

## Motion compensation

Sparse optical flow estimates image motion. It is sensitive to low texture,
motion blur, parallax, independently moving foreground features, abrupt camera
motion, and discontinuities. Image-space velocity is not ego-motion, world
velocity, or physical object speed.

## Pose

Pose runs only when a fresh tracked person is available. Small, occluded, or
edge-cropped people may have missing or incorrect joints. Pose freshness is
independent of tracking freshness.

## Drivable-area segmentation

The optional road mask is coarse semantic evidence. It can spill across curb
margins or crossing approaches and contain holes around paint, vehicles, or
occlusion. Its roughly 1 Hz cadence means it is older than the current image
between updates. It is not a lane, vehicle path, or guaranteed free-space map.

## Unsupported capabilities

The system has no validated:

- ego-lane estimate
- stable forward corridor
- camera calibration or metric distance
- depth or 3D localization
- physical velocity or acceleration
- time-to-collision
- collision prediction or warning
- intent prediction
- steering, planning, or vehicle control

The Perception Priority Map is a relative image-space visualization. It must not
be interpreted as calibrated collision risk, human attention, or a control
signal.

## Evidence limits

The included replay establishes a regression baseline on a specific video. It
does not cover the environmental diversity required for a safety or production
autonomy claim. Reported percentages and percentiles retain their original
sample sizes and benchmark conditions; rejected research is not promoted on the
basis of plausible-looking examples.
