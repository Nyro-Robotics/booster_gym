import zmq
import json
import time
import re
from typing import Dict, Any
from collections import deque
import numpy as np


class ZeroMQClient:
    """ZeroMQ client for teleoperation communication"""
    
    def __init__(self, zmq_address: str = "tcp://localhost:5555"):
        self.zmq_address = zmq_address
        
    def handle_subscriber(self, teleop_active_ref, latest_joint_data_ref, latest_hand_data_ref,
                         message_latencies_ref, joint_position_latencies_ref,
                         total_latency_sum_ref, total_latency_count_ref,
                         min_latency_ref, max_latency_ref):
        """Handle ZeroMQ subscriber connection and message processing - optimized for low latency"""
        
        # Initialize ZeroMQ context and socket
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        
        # Optimized socket options for high-frequency, low-latency operation
        socket.setsockopt(zmq.RCVHWM, 10)      # Very low high water mark to minimize buffering
        socket.setsockopt(zmq.SNDHWM, 10)      # Low send high water mark
        socket.setsockopt(zmq.RCVTIMEO, 1)     # 1ms timeout instead of 1000ms for quick polling
        socket.setsockopt(zmq.LINGER, 0)       # Don't linger on close
        socket.setsockopt(zmq.TCP_KEEPALIVE, 1) # Enable TCP keepalive
        socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 1)  # 1 second keepalive idle
        socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 1) # 1 second keepalive interval
        socket.setsockopt(zmq.SUBSCRIBE, b"joint_positions")  # Subscribe to joint_positions topic
        
        # Create a poller for efficient message checking
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        
        try:
            print(f"🔌 Connecting to ZeroMQ publisher at {self.zmq_address}")
            socket.connect(self.zmq_address)
            print(f"✅ Connected to ZeroMQ publisher successfully!")
            print(f"🎯 Listening for teleoperation commands via ZeroMQ (optimized for low latency)...")
            
            message_count = 0
            last_stats_time = time.time()
            
            while teleop_active_ref.value:
                try:
                    # Use poller for efficient checking (1ms timeout for responsive checking)
                    socks = dict(poller.poll(1))  # 1ms poll timeout
                    
                    if socket in socks and socks[socket] == zmq.POLLIN:
                        # Message available - receive immediately
                        topic, message = socket.recv_multipart(zmq.NOBLOCK)
                        receive_time = time.time()
                        message_count += 1
                        
                        # Parse JSON message
                        data = json.loads(message.decode('utf-8'))
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
                            
                        # Print periodic statistics (every 5 seconds)
                        if time.time() - last_stats_time > 5.0:
                            msg_rate = message_count / (time.time() - last_stats_time) if (time.time() - last_stats_time) > 0 else 0
                            print(f"⚡ ZeroMQ: {message_count} messages in {time.time() - last_stats_time:.1f}s ({msg_rate:.1f} Hz)")
                            if latency_ms:
                                print(f"   Current latency: {latency_ms:.1f}ms")
                            message_count = 0
                            last_stats_time = time.time()
                            
                    else:
                        # No message available - brief sleep to prevent CPU spinning
                        time.sleep(0.0001)  # 0.1ms sleep to be CPU-friendly while staying responsive
                        
                except zmq.Again:
                    # No message available (shouldn't happen with poller, but just in case)
                    time.sleep(0.0001)  # 0.1ms sleep
                    continue
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                except Exception as e:
                    print(f"❌ Error processing ZeroMQ message: {e}")
                    
        except Exception as e:
            print(f"❌ ZeroMQ error: {e}")
        finally:
            print(f"🔌 Disconnected from ZeroMQ publisher")
            poller.unregister(socket)
            socket.close()
            context.term()


def send_initialization_message_zeromq(message: Dict[str, Any], zmq_address: str = "tcp://localhost:5555"):
    """Send initialization message via ZeroMQ"""
    try:
        # Calculate the initialization address (port + 1)
        init_address = zmq_address
        if zmq_address.startswith('tcp://'):
            match = re.search(r'tcp://([^:]+):(\d+)', zmq_address)
            if match:
                host = match.group(1)
                port = int(match.group(2)) + 1  # Use port + 1 for initialization
                init_address = f"tcp://{host}:{port}"
        
        # Create ZeroMQ context and socket
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        
        # Configure socket for quick send
        socket.setsockopt(zmq.LINGER, 0)  # Don't linger on close
        socket.setsockopt(zmq.SNDHWM, 1)  # Low high water mark
        
        # Connect to the initialization address
        socket.connect(init_address)
        
        # Give ZeroMQ a moment to establish connection
        time.sleep(0.5)  # Longer delay for PUB/SUB pattern to ensure subscriber is ready
        
        # Send message with topic multiple times to ensure delivery
        topic = b"robot_joint_initialization"
        message_bytes = json.dumps(message).encode('utf-8')
        
        print(f"📤 Sending robot joint initialization message to {init_address}")
        print(f"   Topic: {topic}")
        print(f"   Message size: {len(message_bytes)} bytes")
        
        # Send multiple times to ensure delivery (PUB/SUB can miss initial messages)
        for i in range(3):
            socket.send_multipart([topic, message_bytes])
            time.sleep(0.1)
        
        print(f"✅ Robot joint positions sent via ZeroMQ to {init_address} (sent 3 times)")
        
        # Give a moment for the message to be sent
        time.sleep(0.2)
        
        # Clean up
        socket.close()
        context.term()
        
    except ImportError:
        print("⚠️ ZeroMQ library not available for initialization message")
    except Exception as e:
        print(f"⚠️ ZeroMQ initialization failed: {e}") 