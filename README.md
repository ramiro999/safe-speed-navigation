<div align="center">
<img src="https://i.ibb.co/GPtTvqy/Captura-desde-2025-01-14-13-24-06.png" alt="fondo">
</div>
<div align="center">
  <h2> Estimation of safe navigation speed for autonomous vehicles </h2>
</div>

This project addresses the estimation of safe navigation speed for autonomous vehicles using recent computer vision techniques. Images captured with a stereo vision system estimate the depth of objects in the vehicle environment. The proposal includes mathematically modeling stopping and avoidance maneuvers, and determining braking distances and curvature limits to avoid obstacles safely. During the development of the simulator, a method is implemented that combines visual perception with deep learning algorithms, such as the DETR model for object detection and the Neural Markov Random Field (NMRF) algorithm for depth estimation. Critical factors such as system latency, reaction time, and vehicle geometric characteristics are considered to adjust the navigation speed. Thus, the proposed system is validated as a useful tool to evaluate navigation decisions in controlled scenarios through simulations analyzing angle of view (AOV), instantaneous field of view (IFOV), and braking and avoidance distances The results show that the approach improves the ability of autonomous vehicles to operate safely in dynamic environments by optimizing navigation decisions and avoiding collisions. 🚗


## 📝 How to Build

The code is developed on Ubuntu 22.04 using Python 3.11 and torch 2.5.1. To build the packages, follow these steps:

```shell
# Open a terminal (Command Prompt or PowerShell for Windows, Terminal for macOS or Linux)

# Ensure Git is installed
# Visit https://git-scm.com to download and install console Git if not already installed

# Clone the repository
git clone https://github.com/ramiro999/safe-speed-navigation.git

# Navigate to the project directory
cd safe-speed-navigation

# Set up a Python environment
python3.11 -m venv venv
source venv/bin/activate

# Build MultiScaleDeformableAttention:
# Navigate to the stereo/NMRF/ops directory and run the following command:

cd stereo/NMRF/ops
sh make.sh
cd ../../..

# Restore dependencies
pip install -r requirements.txt

# Run the project
python app.py

```

## 🤝 Feedback and Contributions
All contributions are welcome. 

