# Local replay input

The default configuration expects:

```text
test_data/
+-- test_video.mp4
```

`test_video.mp4` is the approximately 24 FPS road-scene replay used by the
documented local benchmarks. Video files are excluded from Git and must be
copied separately.

Run with the default replay:

```bash
python src/main.py
```

Or choose another local video without editing configuration:

```bash
python src/main.py --video /path/to/video.mp4
```

The input must be readable by the installed OpenCV build. A missing or unreadable
path fails with the exact requested location. Synthetic unit tests and the
committed tracking fixture do not require this video.
