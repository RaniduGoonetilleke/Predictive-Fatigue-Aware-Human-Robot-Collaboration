# Predictive Fatigue-Aware Human-Robot Collaboration

A ROS 2 / Gazebo implementation of a Human-Centric Digital Twin (HCDT) framework that adapts a UR3 cobot's motion to the operator's physiological state in real time. The robot fuses hand-gesture recognition with eye-tracking-derived fatigue signals to scale its velocity, gate confirmations, and trigger acute-stress soft e-stops based on the operator's current state.

Final-year project (BEng Robotics, Nottingham Trent University) by **Ranidu Goonetilleke** · Supervisor: Dr Azfar Khalid.

## Highlights

- **5-class hand-gesture classifier** (Random Forest on MediaPipe landmarks) running at 20 Hz on CPU; **99.62%** test accuracy on a leak-free split-then-augment pipeline.
- **Composite fatigue score** fusing blink rate, blink duration, PERCLOS, and hand-jerk (weighted 0.30 / 0.25 / 0.30 / 0.15), banded into FRESH / MILD / MODERATE / SEVERE.
- **Graded velocity scaling** (100% → 70% → 40% → 15% → 0%) and OKAY-confirmation gating at MODERATE+.
- **TEPR-triggered soft e-stop**: robot freezes within ≤ 200 ms when pupil diameter exceeds the 3-second rolling baseline by ≥ 15%.
- **Analytical inverse kinematics** for the UR3 (closed-form, law-of-cosines elbow solution) — no MoveIt planning overhead.
- **Ghost-arm intention overlay** in RViz so the operator sees the planned trajectory before motion.
- Five-layer noise-suppression chain for gesture stability (MediaPipe confidence floor → visibility gate → velocity gate → classifier confidence threshold → consecutive-frame debounce).

## Repository structure

```
ros2_ws/
├── src/
│   ├── intention_gazebo/   # UR3 + workstation Gazebo sim, URDFs, launch files
│   └── vision_input/       # Perception, fatigue, robot control, training scripts
├── data/
│   ├── grasp.csv, point.csv, stop.csv, okay.csv, neutral.csv   # Training data
│   ├── gesture_brain.joblib                                    # Deployed RF model
│   ├── table_calibration.yaml                                  # Homography config
│   └── tripod_view.jpg                                         # Calibration ref
├── src/commands.txt        # Quick command reference (build, sim, real)
└── README.md
```

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3.10+ with: `scikit-learn`, `mediapipe`, `opencv-python`, `numpy`, `pandas`, `joblib`
- Gazebo Classic 11 (for simulation)
- A UR3 with Robotiq 85 gripper (for real-robot mode)
- (Optional) Tobii Pro Glasses 3 for live eye-tracking; simulator runs without

## Installation

### 1. Clone this repo

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
git clone <repo-url> .
```

### 2. Clone the third-party dependencies

These are gitignored from this repo and must be cloned separately into `src/`:

```bash
cd ~/ros2_ws/src
git clone https://github.com/ros-controls/gazebo_ros2_control.git
git clone https://github.com/PickNikRobotics/ros2_robotiq_gripper.git
git clone https://github.com/wjwwood/serial.git
git clone https://github.com/IFRA-Cranfield/IFRA_LinkAttacher.git
```

### 3. Install Python dependencies

```bash
pip3 install -r requirements.txt
```

The versions are pinned to the ones used to produce the accuracy and inference figures in the paper, so the comparison table reproduces exactly. If you only want to run the live system without retraining, the strict pinning is not required, but it is the safest default.

### 4. Build

```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

## Quick start

### Simulation (Gazebo, default webcam input)

```bash
ros2 launch intention_gazebo gazebo_sim.launch.py
```

Variants:
- Gamepad teleoperation: `... use_gamepad:=true`
- Virtual operator (keyboard/mouse): `... use_virtual_operator:=true`

### Real UR3 (lab)

Terminal 1 — UR driver:

```bash
ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur3 robot_ip:=192.168.1.10 \
  launch_rviz:=false use_fake_hardware:=false \
  description_package:=intention_gazebo \
  description_file:=ur3_robotiq_real.urdf.xacro
```

Terminal 2 — application layer:

```bash
ros2 launch intention_gazebo real_ur3.launch.py
```

Then start the External Control program on the UR3 pendant (speed ~75%).

See [`src/commands.txt`](src/commands.txt) for the full reference (network setup, build shortcuts, mode flags).

## Retraining the gesture classifier

The shipped [`data/gesture_brain.joblib`](data/gesture_brain.joblib) is the deployed model. To retrain on the bundled CSVs:

```bash
cd ~/ros2_ws/src/vision_input/vision_input
python3 train_model.py
```

To reproduce the classifier-family comparison table from the paper:

```bash
python3 compare_classifiers.py
```

Both scripts use a deterministic leak-free pipeline (raw → stratified 80/20 split → mirror-augment training fold only). Output:

| Classifier | Accuracy | Inference (µs/sample) |
|---|---:|---:|
| Random Forest (deployed) | 99.62% | 10.7 |
| MLP (64,32) | 100.00% | 1.0 |
| KNN (k=5) | 100.00% | ~90 |

Random Forest is retained for the deployed system because its feature importances directly informed the geometric override layer in `camera_node.py`, which catches the two RF errors (Okay↔Neutral confusions) before the gesture is committed.

## Attribution

This project uses the following third-party packages (each clone-separately and licensed by its respective authors):

- [`gazebo_ros2_control`](https://github.com/ros-controls/gazebo_ros2_control) — Gazebo integration for ros2_control
- [`ros2_robotiq_gripper`](https://github.com/PickNikRobotics/ros2_robotiq_gripper) — Robotiq 85 gripper driver
- [`serial`](https://github.com/wjwwood/serial) — cross-platform serial communication
- [`IFRA_LinkAttacher`](https://github.com/IFRA-Cranfield/IFRA_LinkAttacher) — Gazebo plugin for grasp simulation
- [`Universal_Robots_ROS2_Driver`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver) — UR3 hardware driver
- [`MediaPipe`](https://github.com/google/mediapipe) — hand-landmark detection (Lugaresi et al., 2019)

## Citation

If you use this work, please cite:

> Goonetilleke, R. (2026). *Predictive Fatigue-Aware Human-Robot Collaboration: A Human-Centric Digital Twin for Industry 5.0 Cobotics.* Final-year project, Nottingham Trent University.

## Licence

MIT — see [LICENSE](LICENSE).
