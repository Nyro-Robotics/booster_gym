import asyncio
import websockets
import json
import time
from typing import Dict, Any, Optional
from collections import deque
import numpy as np


class WebSocketClient:
    """WebSocket client for teleoperation communication"""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        
    async def handle_connection(self, teleop_active_ref, latest_joint_data_ref, latest_hand_data_ref, 
                               message_latencies_ref, joint_position_latencies_ref, 
                               total_latency_sum_ref, total_latency_count_ref, 
                               min_latency_ref, max_latency_ref):
        """Handle WebSocket connection and message processing"""
        
        try:
            print(f"🔌 Connecting to WebSocket server at {self.uri}")
            
            async with websockets.connect(self.uri) as websocket:
                print(f"✅ Connected to WebSocket server successfully!")
                print(f"🎯 Listening for teleoperation commands...")
                
                async for message in websocket:
                    if not teleop_active_ref.value:
                        break
                        
                    try:
                        # Capture receive time immediately for latency calculation
                        receive_time = time.time()
                        
                        # Parse JSON message
                        data = json.loads(message)
                        message_type = data.get('type', 'unknown')
                        
                        # Calculate latency if message has timestamp
                        latency_ms = None
                        message_timestamp = data.get('timestamp')
                        if message_timestamp is not None:
                            latency_s = receive_time - message_timestamp
                            latency_ms = latency_s * 1000  # Convert to milliseconds
                            
                            # Update global latency statistics
                            message_latencies_ref.append(latency_ms)
                            total_latency_sum_ref.value += latency_ms
                            total_latency_count_ref.value += 1
                            min_latency_ref.value = min(min_latency_ref.value, latency_ms)
                            max_latency_ref.value = max(max_latency_ref.value, latency_ms)
                            
                            # Track joint position specific latency
                            if message_type == 'joint_positions':
                                joint_position_latencies_ref.append(latency_ms)
                        
                        if message_type == 'joint_positions':
                            # Extract organized joint data (no fallback to all_joints)
                            message_data = data.get('data', {})
                            organized = message_data.get('organized', {})
                            
                            # Validate that we have the expected organized structure
                            if not organized:
                                print(f"⚠️ Warning: Received joint_positions message without organized data structure")
                                continue
                            
                            if 'upper_body' not in organized:
                                print(f"⚠️ Warning: Missing 'upper_body' in organized data")
                                continue
                            
                            # Update latest joint data with organized structure
                            latest_joint_data_ref.clear()
                            latest_joint_data_ref.update(organized.copy())
                            
                            # Extract hand data from organized data
                            latest_hand_data_ref.clear()
                            latest_hand_data_ref.update({
                                'left_hand': organized.get('left_hand', {}),
                                'right_hand': organized.get('right_hand', {})
                            })
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error: {e}")
                    except Exception as e:
                        print(f"❌ Error processing message: {e}")
                        
        except ConnectionRefusedError:
            print(f"❌ Connection refused to {self.uri}")
            print(f"   Make sure the WebSocket relay server is running")
        except Exception as e:
            print(f"❌ WebSocket error: {e}")
        finally:
            print(f"🔌 Disconnected from WebSocket server")


def send_initialization_message_websocket(message: Dict[str, Any], ws_host: str = "localhost", ws_port: int = 8765):
    """Send initialization message via WebSocket"""
    try:
        async def send_message():
            # Use the specified WebSocket host/port
            uri = f"ws://{ws_host}:{ws_port}"
            try:
                async with websockets.connect(uri, ping_timeout=2, close_timeout=2) as websocket:
                    await websocket.send(json.dumps(message))
                    print(f"✅ Robot joint positions sent via WebSocket to {uri}")
            except Exception as e:
                print(f"⚠️ WebSocket initialization send failed: {e}")
        
        # Run the async function
        asyncio.run(send_message())
        
    except ImportError:
        print("⚠️ WebSocket library not available for initialization message")
    except Exception as e:
        print(f"⚠️ WebSocket initialization failed: {e}")


def create_robot_joint_initialization_message(joint_positions) -> Dict[str, Any]:
    """Create initialization message with robot joint positions"""
    # Map joint positions to organized structure for teleoperator
    joint_mapping = {
        # Head joints
        "AAHead_yaw": 0, "Head_pitch": 1,
        # Left arm joints  
        "Left_Shoulder_Pitch": 2, "Left_Shoulder_Roll": 3, "Left_Elbow_Pitch": 4, 
        "Left_Elbow_Yaw": 5, "Left_Wrist_Pitch": 6, "Left_Wrist_Yaw": 7, "Left_Hand_Roll": 8,
        # Right arm joints
        "Right_Shoulder_Pitch": 9, "Right_Shoulder_Roll": 10, "Right_Elbow_Pitch": 11, 
        "Right_Elbow_Yaw": 12, "Right_Wrist_Pitch": 13, "Right_Wrist_Yaw": 14, "Right_Hand_Roll": 15,
        # Waist joint
        "Waist": 16
    }
    
    # Create organized joint data structure
    organized_joints = {}
    for joint_name, joint_index in joint_mapping.items():
        if joint_index < len(joint_positions):
            organized_joints[joint_name] = joint_positions[joint_index]
    
    # Create initialization message
    init_message = {
        "type": "robot_joint_initialization",
        "timestamp": time.time(),
        "data": {
            "organized": {
                "upper_body": organized_joints,
                "left_hand": {},  # No hand joints in this initialization
                "right_hand": {}  # No hand joints in this initialization
            }
        },
        "metadata": {
            "source": "deploy_robot_hardware",
            "robot_mode": "mc_initialization",
            "total_joints": len(organized_joints),
            "description": "Current robot joint positions for teleoperator initialization"
        }
    }
    
    return init_message 