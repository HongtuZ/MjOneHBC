# OneHBC: A Unified Framework for Humanoid Body Control

# MjOneHBC

> A Unified Framework for Humanoid Body Control Training
> Based on Mjlab
> 
> 

OneHBC is a dedicated research framework for training humanoid robot locomotion and whole\-body control policies using Mjlab with the mujoco physics engine\. It supports high\-performance end\-to\-end reinforcement learning and motion imitation for humanoid robots, with a focus on speed tracking, AMP\-based motion imitation, and whole\-body trajectory tracking\. Future extensions will support general whole\-body VLA \(Vision\-Language\-Action\) control\.

---

## Features

- **Velocity Tracking Control**: Omnidirectional speed command tracking for robust locomotion

- **AMP \(Adversarial Motion Priors\)**: High\-quality natural motion imitation learning

- **Whole\-Body Trajectory Tracking**: Accurate task\-space and joint\-space trajectory following

- **Under Development**: General whole\-body VLA \(Vision\-Language\-Action\) control pipeline

---

## Environment Requirements

- OS: Ubuntu 22\.04 / 24\.04

- Mjlab: `>=1.3.0`

- Python: 3\.12

- CUDA: 12\.8 or higher

---

## Installation

### 1\. Set Up Conda Environment and Dependencies

```bash
conda create -n onehbc python=3.12
conda activate onehbc
```

### 2\. Install mjlab and rsl-rl Library

```bash
pip install mjlab
pip install rsl-rl
```

### 3\. Clone MjOneHBC

```bash
git clone https://github.com/HongtuZ/MjOneHBC.git
cd MjOneHBC
pip install -e source/OneHBC
```

---

## Training Commands

### Velocity Tracking Control

```bash
# Train
python scripts/rsl_rl/train.py Velocity-Flat-THS23DOF --num_envs 4096

# Train with video recording
python scripts/rsl_rl/train.py Velocity-Flat-THS23DOF --num_envs 4096 --video

# Play
python scripts/rsl_rl/play.py Velocity-Flat-THS23DOF --num_envs 16

### AMP Imitation Learning
TODO: Add AMP training and evaluation commands

### Whole\-Body Trajectory Tracking
TODO: Add whole-body trajectory tracking training and evaluation commands
```