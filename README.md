# Assignment 3: 3D Motion Data Post-Processing

AI & Machine Learning (2026-2), Junha Yoo (20242134)

BlazePose 3D keypoints from six construction-work clips (hammering, shoveling) are cleaned up
and visualized in Viser and Unity.

## Files

| File | Role |
|---|---|
| `scripts/p0_download_videos.py` | Download the reference videos and extract keypoints |
| `scripts/p1_create_raw_keypoints.py` | Provided: keypoint extraction |
| `scripts/p1_2_post_processing.py` | Provided: height correction (baseline) |
| `scripts/p1_2_post_processing_enhanced.py` | Modified post-processing, metrics and comparison plots |
| `scripts/animator_cleanup.py` | Impact keys, key reduction, PCHIP in-betweening, foot anchoring, foot IK |
| `scripts/rigid_tool.py` | Rigid-tool constraint for the two hands and virtual tool joints |
| `scripts/p1_5_make_work_clips.py` | Glitch-frame repair and work-segment clips for the Unity demo |
| `scripts/p1_1_view_motion.py`, `scripts/p1_4_motion_studio.py` | Viser viewers |
| `scripts/analyze_failure_case.py` | Limitation analysis on walking segments |
| `utils/viser_test.py` | Provided: Viser helper |
| `unity/*.cs` | Unity demo scripts (motion player, crew placement, free camera) |

## Run

```
conda env create -f environment.yml
conda activate ai_ml_final
python scripts/p1_2_post_processing_enhanced.py      # --method animator | savgol | oneeuro
python scripts/p1_5_make_work_clips.py
python scripts/p1_4_motion_studio.py
python scripts/analyze_failure_case.py shovel/HmyvWOXlSt0
```

Run from the project root. Raw keypoints are read from `data/raw_keypoints/<class>/<video_id>.npz`
and results are written to `data/processed/`. Videos and data files are not included.
