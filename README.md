# Cycling Biomechanics Analysis Tool

This repository contains tools for analyzing cycling biomechanics from video data using 3D pose estimation and motion analysis techniques.

## Overview

This project processes cycling videos to extract 3D pose data and provide comprehensive biomechanical analysis for cycling technique evaluation. The workflow consists of three main steps:

1. 3D pose estimation from video
2. Biomechanical analysis of individual cycling sessions
3. Comparative analysis between different cycling sessions

## Directory Structure

```
├── infer_wild_2.py                    # Main script for 3D pose estimation and smoothing
├── extract_biomechanics_single.py     # Analysis of single cycling session
├── extract_biomechanics_compare.py    # Comparison between two cycling sessions
├── biomech_outputs_*                  # Output directories for each analysis
```

## Output Data

For each analysis, the following outputs are generated in the `biomech_outputs_{file_name}` directory:

- **CSV Files**:

  - `joint_angles.csv`: Hip, knee, and ankle angles over time
  - `angular_velocities.csv`: Rate of angular change for each joint
  - `range_of_motion.csv`: Min, max, and total ROM for each joint
  - `trajectory_smoothness.csv`: Smoothness metrics for joint trajectories
  - `summary_biomechanics.csv`: Consolidated biomechanical metrics

- **Visualizations**:

  - `joint_angles_plot.png`: Plots of joint angles over time
  - `angular_velocities_plot.png`: Plots of angular velocities
  - `pedaling_cycles.png`: Detected pedaling cycles with cadence analysis

- **Reports**:
  - `comprehensive_report.json`: Complete analysis in JSON format
  - `pedaling_cycles.json`: Detailed pedaling metrics

## Dataset

The cycling videos used for testing and demonstration are sourced from YouTube. You can access the dataset via the following Google Drive link:

[Cycling Video Dataset](https://drive.google.com/drive/folders/1TTb8qRPu0ksYjk9waqLGMUaMSMBdGDsN?usp=sharing)
