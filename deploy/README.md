# Deploy on Booster Robot

## Installation

Follow these steps to set up your environment:

1. Install Python dependencies:

    ```sh
    $ pip install -r requirements.txt
    ```

2. Install Booster Robotic SDK:

    Refer to the [Booster Robotics SDK Guide](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-WDzedC8AiovU8gxSjeGcQ5CInSf) and ensure you complete the section on [Compile Sample Programs and Install Python SDK](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-EI5fdtSucoJWO4xd49QcE5JxnCf).

## Usage

1. Prepare the robot:

    - **Simulation:** Set up the simulation by referring to [Development Based on Webots Simulation](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-IsE9d2DrIow8tpxCBUUcogdwn5d) or [Development Based on Isaac Simulation Link](https://booster.feishu.cn/wiki/DtFgwVXYxiBT8BksUPjcOwG4n4f#share-Jczjd4UKMou7QlxjvJ4c9NNfnwb).

    - **Real World:** Power on the robot and switch it to PREP Mode. Place the robot to a stable standing position on the ground.

2. Run the deployment script:

    ```sh
    $ python deploy.py --config=T1.yaml
    ```

    - `--config`: Name of the configuration file, located in the `configs/` folder.
    - `--net`: Network interface for SDK communication. Default is `127.0.0.1`.

3. Exit Safely:

    Switch back to PREP Mode before terminating the program to safely release control.

---

# Joint IDs in Real Robot

| Joint ID | Joint Name | Limit (Degrees) |  |
|---------|-----------|-----------------|-----------------|
|  |  | Max | Min |
| 0 | Head Yaw Joint | 58 | -58 |
| 1 | Head Pitch Joint | 47 | -18 |
| 2 | Left Shoulder Pitch Joint | 68 | -188 |
| 3 | Left Shoulder Roll Joint | 88 | -94 |
| 4 | Left Shoulder Yaw Joint | 128 | -128 |
| 5 | Left Elbow Joint | 2 | -120 |
| 6 | Right Shoulder Pitch Joint | 68 | -188 |
| 7 | Right Shoulder Roll Joint | 94 | -88 |
| 8 | Right Shoulder Yaw Joint | 128 | -128 |
| 9 | Right Elbow Joint | 120 | -2 |
| 10 | Waist Yaw Joint | 58 | -58 |
| 11 | Left Hip Pitch Joint | 118 | -118 |
| 12 | Left Hip Roll Joint | 88 | -21 |
| 13 | Left Hip Yaw Joint | 58 | -58 |
| 14 | Left Knee Joint | 123 | -11 |
| 15 | Left Ankle Up Joint | 49 | -23 |
| 16 | Left Ankle Down Joint | 45 | -24 |
| 17 | Right Hip Pitch Joint | 118 | -118 |
| 18 | Right Hip Roll Joint | 21 | -88 |
| 19 | Right Hip Yaw Joint | 58 | -58 |
| 20 | Right Knee Joint | 123 | -11 |
| 21 | Right Ankle Up Joint | 49 | -23 |
| 22 | Right Ankle Down Joint | 45 | -24 |


# Joint IDs for Mujoco

| Relative Index (effective_id) | Joint Name | Absolute DoF |
|---------------------------|-------------------|--------------| 
| 0 | Left_Shoulder_Pitch | 6 |
| 1 | Left_Shoulder_Roll | 7 |
| 2 | Left_Elbow_Pitch | 8 |
| 3 | Left_Elbow_Yaw | 9 |
| 4 | Right_Shoulder_Pitch | 10 |
| 5 | Right_Shoulder_Roll | 11 |
| 6 | Right_Elbow_Pitch | 12 |
| 7 | Right_Elbow_Yaw | 13 |
| 8 | Waist | 14 |
| 9 | Left_Hip_Pitch | 15 |
| 10 | Left_Hip_Roll | 16 |
| 11 | Left_Hip_Yaw | 17 |
| 12 | Left_Knee_Pitch | 18 |
| 13 | Left_Ankle_Pitch | 19 |
| 14 | Left_Ankle_Roll | 20 |
| 15 | Right_Hip_Pitch | 21 |
| 16 | Right_Hip_Roll | 22 |
| 17 | Right_Hip_Yaw | 23 |
| 18 | Right_Knee_Pitch | 24 |
| 19 | Right_Ankle_Pitch | 25 |
| 20 | Right_Ankle_Roll | 26 |

---

# Combined Table (Real Joint IDs mapped to Mujoco T1_serial_collision Indices)

| Joint ID | Joint Name | Relative Mujoco Index (effective_id) | Absolute DoF | Limit (Degrees) in Real | |
|---------|-----------|---------------------------|--------------|-----------------|-----------------|
| | | | | Max | Min |
| 0 | Head_Yaw | - | - | 58 | -58 |
| 1 | Head_Pitch | - | - | 47 | -18 |
| 2 | Left_Shoulder_Pitch | 0 | 6 | 68 | -188 |
| 3 | Left_Shoulder_Roll | 1 | 7 | 88 | -94 |
| 4 | Left_Elbow_Pitch | 2 | 8 | 128 | -128 |
| 5 | Left_Elbow_Yaw | 3 | 9 | 2 | -120 |
| 6 | Right_Shoulder_Pitch | 4 | 10 | 68 | -188 |
| 7 | Right_Shoulder_Roll | 5 | 11 | 94 | -88 |
| 8 | Right_Elbow_Pitch | 6 | 12 | 128 | -128 |
| 9 | Right_Elbow_Yaw | 7 | 13 | 120 | -2 |
| 10 | Waist | 8 | 14 | 58 | -58 |
| 11 | Left_Hip_Pitch | 9 | 15 | 118 | -118 |
| 12 | Left_Hip_Roll | 10 | 16 | 88 | -21 |
| 13 | Left_Hip_Yaw | 11 | 17 | 58 | -58 |
| 14 | Left_Knee_Pitch | 12 | 18 | 123 | -11 |
| 15 | Left_Ankle_Pitch | 13 | 19 | 49 | -23 |
| 16 | Left_Ankle_Roll | 14 | 20 | 45 | -24 |
| 17 | Right_Hip_Pitch | 15 | 21 | 118 | -118 |
| 18 | Right_Hip_Roll | 16 | 22 | 21 | -88 |
| 19 | Right_Hip_Yaw | 17 | 23 | 58 | -58 |
| 20 | Right_Knee_Pitch | 18 | 24 | 123 | -11 |
| 21 | Right_Ankle_Pitch | 19 | 25 | 49 | -23 |
| 22 | Right_Ankle_Roll | 20 | 26 | 45 | -24 |