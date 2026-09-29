# Architecture

## Design objective

The pipeline is designed for useful, recent perception on a constrained CPU.
It intentionally optimizes **freshness rather than processed-frame count**.
Capture and display can continue near source cadence while neural inference runs
at lower task-specific rates.

## Data flow

```text
VideoCapture
    |
    | timestamped FramePacket
    v
LatestFrameBuffer  <--- newest publish replaces the prior reference
    |
    v
scheduler_worker
    |
    | one newest pending task per type
    v
ComputeScheduler
    |
    | priority + deadline admission
    v
compute_worker (one serial heavy-compute lane)
    |-- detection -> MultiObjectTracker
    |-- pose, only when a fresh tracked person exists
    +-- optional road observation, demo only
    |
    v
PerceptionStore
    |
    | independently timestamped immutable observations
    v
freshness checks -> image-motion projection -> display smoothing -> renderer
```

## LatestFrameBuffer

Capture publishes a `FramePacket` containing a monotonically increasing frame ID,
a monotonic capture timestamp, the image, and an optional immutable motion
snapshot. The buffer stores one reference. Publishing never creates a FIFO
backlog; a slow consumer reads the newest available packet.

Published image arrays are treated as read-only. Renderers copy before drawing.
A consumer retaining an old packet keeps only that array alive until its current
work completes.

## Scheduler and newest-pending replacement

`scheduler_worker` offers detection for each new packet, pose only when fresh
tracking contains a person, and road only when the optional task is configured.
`ComputeScheduler` holds at most one pending task per task type. A newer frame
replaces older waiting work.

Detection has the highest priority. A fairness rule prevents one ready type from
starving all others, but optional work must also fit before the next detection
freshness deadline. Costs are learned from completed work; road warm-up seeds its
initial cost so a long optional task is not probed blindly.

Rates are upper bounds, not promises. Under overload, optional work yields to
fresh tracking.

## Serial heavy-compute worker

Detection, pose, and optional road inference execute on one worker. This avoids
competing neural runtimes saturating the two-core CPU and makes deadline
admission observable. The detector adapter converts backend-specific output to
plain immutable detections before tracking.

Detection updates the tracker with the source frame's monotonic timestamp.
Tracking snapshots are immutable. Pose is matched to fresh tracked people and
has independent metadata. Road segmentation produces an immutable native-size
binary mask; it is disabled by default.

## Perception store

The store atomically merges three independently timed observations:

- tracking entities and detector metrics
- per-person pose
- optional drivable-road mask

A newer tracking update can preserve pose without pretending pose was refreshed.
Likewise, attaching a road observation does not change tracking timestamps.
Out-of-order observations cannot overwrite newer ones.

## Time semantics and freshness

Every expensive result retains:

- `source_frame_id`: the input frame
- `source_timestamp`: when that frame was captured
- `produced_timestamp`: when processing completed

Processing latency is produced time minus source time. Display age is current
packet time minus source time. A renderer accepts a result only when it is not
from a future frame and its age is within the configured limit. Missing, future,
or stale state clears display smoothing.

Pose and road have their own freshness limits. Reading, projecting, or rendering
an observation never refreshes its source timestamp.

## Tracking and motion history

The tracker uses confidence-aware association and a constant-velocity Kalman
state. Weak detections may maintain a confirmed track but cannot create or
confirm one. Missed tracks coast only within explicit count and time limits.

Sparse optical flow is computed opportunistically during capture on reduced
images. `MotionSnapshot` stores a short chain of robust affine steps plus local
motion evidence. Rendering can carry delayed boxes toward the current image for
a bounded time. If history is missing, discontinuous, or unreliable, projection
fails closed and bounded Kalman prediction is used instead.

All velocities and projections are image-space quantities, not physical motion.

## Rendering and shutdown

GUI calls remain on the main thread. The renderer reads the newest frame and
perception state independently, checks freshness, projects and smooths display
geometry, and draws on a copy.

Worker exceptions are logged with a traceback, set the shared stop event, wake
the scheduler, and are re-raised on the main thread after cleanup. Shutdown wakes
the scheduler, joins every started worker, releases recording output, and then
destroys OpenCV windows. In-flight native inference may finish; Python threads
cannot forcibly cancel a hung backend call.
