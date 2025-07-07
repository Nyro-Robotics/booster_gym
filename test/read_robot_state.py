from booster_robotics_sdk_python import ChannelFactory, B1LowStateSubscriber
import time
import sys


def handler(low_state_msg):
    print("=" * 60)
    print("ROBOT STATE READINGS:")
    print("=" * 60)
    
    # IMU data
    imu_state = low_state_msg.imu_state
    print(f"IMU State:")
    print(f"  Roll/Pitch/Yaw: {imu_state.rpy[0]:.3f}, {imu_state.rpy[1]:.3f}, {imu_state.rpy[2]:.3f} rad")
    print(f"  Gyro (x/y/z):   {imu_state.gyro[0]:.3f}, {imu_state.gyro[1]:.3f}, {imu_state.gyro[2]:.3f} rad/s")
    print(f"  Accel (x/y/z):  {imu_state.acc[0]:.3f}, {imu_state.acc[1]:.3f}, {imu_state.acc[2]:.3f} m/s²")
    print()
    
    # Joint names for reference
    joint_names = [
        "Head_Yaw", "Head_Pitch",
        "L_Shoulder_Pitch", "L_Shoulder_Roll", "L_Elbow_Pitch", "L_Elbow_Yaw", "L_Wrist_Pitch", "L_Wrist_Yaw", "L_Hand_Roll",
        "R_Shoulder_Pitch", "R_Shoulder_Roll", "R_Elbow_Pitch", "R_Elbow_Yaw", "R_Wrist_Pitch", "R_Wrist_Yaw", "R_Hand_Roll",
        "Waist",
        "L_Hip_Pitch", "L_Hip_Roll", "L_Hip_Yaw", "L_Knee_Pitch", "L_Crank_Up", "L_Crank_Down",
        "R_Hip_Pitch", "R_Hip_Roll", "R_Hip_Yaw", "R_Knee_Pitch", "R_Crank_Up", "R_Crank_Down"
    ]
    
    # Serial motor states (position control)
    print(f"Serial Motors ({len(low_state_msg.motor_state_serial)} joints):")
    print("  Joint                 | Position (rad) | Velocity (rad/s) | Torque (Nm)")
    print("  " + "-" * 70)
    
    for i, motor in enumerate(low_state_msg.motor_state_serial):
        if i < len(joint_names):
            name = joint_names[i]
        else:
            name = f"Joint_{i}"
        
        print(f"  {name:<20} | {motor.q:>11.3f} | {motor.dq:>13.3f} | {motor.tau_est:>9.3f}")
    
    print()
    
    # Parallel motor states (if any)
    if len(low_state_msg.motor_state_parallel) > 0:
        print(f"Parallel Motors ({len(low_state_msg.motor_state_parallel)} joints):")
        print("  Joint                 | Position (rad) | Velocity (rad/s) | Torque (Nm)")
        print("  " + "-" * 70)
        
        for i, motor in enumerate(low_state_msg.motor_state_parallel):
            print(f"  Parallel_{i:<12} | {motor.q:>11.3f} | {motor.dq:>13.3f} | {motor.tau_est:>9.3f}")
        print()
    
    # Summary of key positions
    print("KEY JOINT POSITIONS SUMMARY:")
    key_joints = [
        (2, "L_Shoulder_Pitch"), (3, "L_Shoulder_Roll"), (4, "L_Elbow_Pitch"),
        (9, "R_Shoulder_Pitch"), (10, "R_Shoulder_Roll"), (11, "R_Elbow_Pitch"),
        (16, "Waist"),
        (17, "L_Hip_Pitch"), (20, "L_Knee_Pitch"),
        (23, "R_Hip_Pitch"), (26, "R_Knee_Pitch")
    ]
    
    for joint_idx, name in key_joints:
        if joint_idx < len(low_state_msg.motor_state_serial):
            pos = low_state_msg.motor_state_serial[joint_idx].q
            print(f"  {name}: {pos:.3f} rad ({pos * 180 / 3.14159:.1f}°)")
    
    print("=" * 60)
    print()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} networkInterface")
        print("Example: python3 read_robot_state.py eth0")
        sys.exit(-1)
    
    print("Initializing robot state subscriber...")
    ChannelFactory.Instance().Init(0, sys.argv[1])
    
    channel_subscriber = B1LowStateSubscriber(handler)
    channel_subscriber.InitChannel()
    
    print("Robot state subscriber initialized!")
    print("Reading robot state... (Press Ctrl+C to stop)")
    print()
    
    try:
        while True:
            time.sleep(2)  # Update every 2 seconds
    except KeyboardInterrupt:
        print("\nShutting down robot state reader...")
        channel_subscriber.CloseChannel()
        print("Done!")


if __name__ == "__main__":
    main() 