# Installation
## Prebuild environment
* OS  (Ubuntu 22.04 LTS)  
* CPU  (aarch64 and x86_64)   
* Compiler  (gcc version 11.4.0) 

## Create a conda env
```
conda create -n fc_booster python=3.10
conda activate fc_booster
```
## Install Pinocchio for Inverse Kinematics
```
conda install pinocchio=3.2.0 -c conda-forge
```
## Install booster_robotics_sdk for Booster T1 deployment
Note that the official [booster_robotics_sdk](https://github.com/BoosterRobotics/booster_robotics_sdk) does not provide state publisher and command receiver, so I improve the repo a bit and add them in my [forked repo](https://github.com/hang0610/booster_robotics_sdk). Also booster sdk is NOT supported on Mac OS yet.
```bash
git clone https://github.com/hang0610/booster_robotics_sdk
# Install python package for building python binding locally
pip3 install pybind11
pip3 install pybind11-stubgen
# Build & Install
mkdir build
cd build
cmake .. -DBUILD_PYTHON_BINDING=on
make
sudo make install
```
## Install others
```bash
cd sim2real
pip install -r requirements.txt
```

# Deployment
> [!IMPORTANT]
> For sim2sim, you need to start Mujoco and then launch the policy, but for sim2real, you **only** need to launch the policy.
> Make sure you read the keyboard and joystick control protocol in `sim2real/rl_policy/base_policy.py`.
> All the deployment scripts are running under `sim2real`, so do `cd sim2real` first.

Here are some **keyboard shortcuts**:

<details>
<summary>Keyboard Shortcuts in Mujoco</summary>

- `7`: raise elastic band height
- `8`: lower elastic band height
- `9`: toggle elastic band
- `backspace`: reset simulation

</details>

<details>
<summary>Keyboard Shortcuts in Policy Terminal</summary>

- `]`: start using policy actions
- `o`: stop using policy action and set actions to zero
- `=`: switch between standing and stepping
- `w`: increase linear velocity in `x` direction
- `s`: decrease linear velocity in `x` direction
- `a`: increase linear velocity in `y` direction
- `d`: decrease linear velocity in `y` direction
- `q`: decrease angular velocity in `z` direction
- `e`: increase angular velocity in `z` direction
- `z`: set velocity to zero
- `1`: increase base height (if the policy allows)
- `2`: decrease base height (if the policy allows)
- `5`: decrease kp scale by 0.01
- `6`: increase kp scale by 0.01
- `4`: decrease kp scale by 0.1
- `7`: increase kp scale by 0.1
- `0`: reset kp scale to 1.0
  
</details>

## T1 29DoF

Launch the Policy
```bash
python rl_policy/loco_manip/loco_manip.py \
--config=config/t1_29dof.yaml \
--model_path=models/t1_29dof.onnx
```

https://github.com/user-attachments/assets/e35ff90e-428b-41ea-8cac-64d9906c78e8

## Sim2Real Tips
> [!CAUTION]
> **FALCON is a strong policy trained for robust locomotion and manipulation.** Before deploying to real robots, ensure:

### Network Configuration
- Set correct `INTERFACE` in config file (e.g., 'en0', 'eth0')
- Verify network connectivity between computer and robot
- Check firewall settings if using specific ports

### Testing Protocol
1. Always do sim2sim before real-robot deployment
2. Start with small kp, kd gains
3. Ensure robot feet touch the ground before running falcon policies

### Emergency Control
- **Keyboard**: Press 'o' to stop policy actions
- **Joystick**: Press 'B+Y' to stop policy actions

### Real-time Inference
- `booster_robotics_sdk` works fine as its backend is written in cpp with a pybinding wrapper.

# Acknowledgement
We thank the following open-sourced repos that we build upon:
- [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate)
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)
- [booster_robotics_python](https://github.com/BoosterRobotics/booster_robotics_sdk)
