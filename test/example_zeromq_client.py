#!/usr/bin/env python3
"""
Example ZeroMQ Client for MuJoCo Hardware Interface
=================================================

This script demonstrates how to connect to the MuJoCo ZeroMQ hardware interface
and receive real-time joint position data. It includes comprehensive latency tracking
and message analysis capabilities for benchmarking performance.

Usage:
    python example_zeromq_client.py --address tcp://localhost:5555
"""

import asyncio
import json
import signal
import time
import zmq
import argparse
import sys
import threading
from datetime import datetime
from collections import deque
from typing import Dict, List, Any, Union, Optional
import numpy as np

# Global flag for graceful shutdown
running = True

class JointMinMaxTracker:
    """Track minimum and maximum values for each joint over time"""
    
    def __init__(self):
        self.joint_stats = {}  # joint_name -> {'min': float, 'max': float, 'count': int}
        self.last_update_time = time.time()
        
    def update_joint(self, joint_name: str, value: Union[float, int], joint_type: str = "unknown"):
        """Update joint statistics with new value"""
        if joint_name not in self.joint_stats:
            self.joint_stats[joint_name] = {
                'min': value,
                'max': value,
                'count': 1,
                'type': joint_type
            }
        else:
            stats = self.joint_stats[joint_name]
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['count'] += 1
            stats['type'] = joint_type
            
        self.last_update_time = time.time()
    
    def update_from_organized_data(self, organized_data: Dict[str, Dict[str, Union[float, int]]]):
        """Update all joint statistics from organized data structure"""
        for category, joints in organized_data.items():
            joint_type = "arm" if category == "upper_body" else "hand"
            if isinstance(joints, dict):
                for joint_name, value in joints.items():
                    self.update_joint(joint_name, value, joint_type)
    
    def get_stats_by_category(self) -> Dict[str, Dict]:
        """Get statistics organized by joint category"""
        categories = {
            'upper_body': {},
            'left_hand': {},
            'right_hand': {},
            'unknown': {}
        }
        
        for joint_name, stats in self.joint_stats.items():
            # Categorize joints based on name patterns
            if any(x in joint_name.lower() for x in ['left_hand', 'left_finger', 'left_thumb', 'left_index', 'left_middle', 'left_ring', 'left_pinky']):
                categories['left_hand'][joint_name] = stats
            elif any(x in joint_name.lower() for x in ['right_hand', 'right_finger', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky']):
                categories['right_hand'][joint_name] = stats
            elif any(x in joint_name.lower() for x in ['thumb', 'index', 'middle', 'ring', 'pinky']):
                # Generic hand joints (determine by value range)
                if stats['max'] <= 1000:
                    categories['right_hand'][joint_name] = stats  # Assume right hand if ambiguous
                else:
                    categories['upper_body'][joint_name] = stats
            else:
                categories['upper_body'][joint_name] = stats
        
        return categories
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all joints"""
        if not self.joint_stats:
            return {"total_joints": 0, "total_updates": 0}
        
        total_updates = sum(stats['count'] for stats in self.joint_stats.values())
        categories = self.get_stats_by_category()
        
        return {
            "total_joints": len(self.joint_stats),
            "total_updates": total_updates,
            "categories": {
                "upper_body": len(categories['upper_body']),
                "left_hand": len(categories['left_hand']),
                "right_hand": len(categories['right_hand']),
                "unknown": len(categories['unknown'])
            },
            "last_update": self.last_update_time
        }
    
    def format_range_summary(self, max_joints_per_category: int = 3) -> str:
        """Format a summary of joint ranges organized by category"""
        categories = self.get_stats_by_category()
        lines = []
        
        for category, joints in categories.items():
            if not joints:
                continue
                
            lines.append(f"📊 {category.upper()} JOINTS:")
            
            # Sort joints by name for consistent output
            sorted_joints = sorted(joints.items())
            
            if len(sorted_joints) <= max_joints_per_category:
                # Show all joints
                for joint_name, stats in sorted_joints:
                    range_val = stats['max'] - stats['min']
                    if category in ['left_hand', 'right_hand']:
                        lines.append(f"   {joint_name}: {stats['min']:4.0f} - {stats['max']:4.0f} (range: {range_val:4.0f}, count: {stats['count']})")
                    else:
                        lines.append(f"   {joint_name}: {stats['min']:6.3f} - {stats['max']:6.3f} (range: {range_val:6.3f}, count: {stats['count']})")
            else:
                # Show first few joints and summary
                for joint_name, stats in sorted_joints[:max_joints_per_category]:
                    range_val = stats['max'] - stats['min']
                    if category in ['left_hand', 'right_hand']:
                        lines.append(f"   {joint_name}: {stats['min']:4.0f} - {stats['max']:4.0f} (range: {range_val:4.0f}, count: {stats['count']})")
                    else:
                        lines.append(f"   {joint_name}: {stats['min']:6.3f} - {stats['max']:6.3f} (range: {range_val:6.3f}, count: {stats['count']})")
                
                remaining = len(sorted_joints) - max_joints_per_category
                total_updates = sum(stats['count'] for stats in joints.values())
                lines.append(f"   ... and {remaining} more joints (total updates: {total_updates})")
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all tracking statistics"""
        self.joint_stats.clear()
        self.last_update_time = time.time()

class ZeroMQClientFrequencyMonitor:
    """Monitor frequency of ZeroMQ messages and track statistics"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.message_times = deque(maxlen=window_size)
        self.message_counts = {}  # message_type -> count
        self.last_report_time = time.time()
        
        # Latency tracking
        self.latency_values = deque(maxlen=window_size)
        self.total_latency_sum = 0.0
        self.total_latency_count = 0
        self.min_latency = float('inf')
        self.max_latency = 0.0
        
        # Message type tracking
        self.joint_position_times = deque(maxlen=window_size)
        self.joint_position_latencies = deque(maxlen=window_size)
        
        # Overall statistics
        self.start_time = time.time()
        self.total_messages = 0
        self.total_bytes = 0
        
    def record_message(self, message_type: str = "unknown", message_size: int = 0, latency_ms: Optional[float] = None):
        """Record a message reception event"""
        current_time = time.time()
        self.message_times.append(current_time)
        self.total_messages += 1
        self.total_bytes += message_size
        
        # Track message type
        if message_type not in self.message_counts:
            self.message_counts[message_type] = 0
        self.message_counts[message_type] += 1
        
        # Track joint position messages specifically
        if message_type == 'joint_positions':
            self.joint_position_times.append(current_time)
            if latency_ms is not None:
                self.joint_position_latencies.append(latency_ms)
        
        # Track latency
        if latency_ms is not None:
            self.latency_values.append(latency_ms)
            self.total_latency_sum += latency_ms
            self.total_latency_count += 1
            self.min_latency = min(self.min_latency, latency_ms)
            self.max_latency = max(self.max_latency, latency_ms)
    
    def get_current_frequencies(self) -> Dict:
        """Calculate current message frequencies"""
        current_time = time.time()
        
        # Overall frequency
        overall_freq = 0.0
        if len(self.message_times) >= 2:
            time_span = self.message_times[-1] - self.message_times[0]
            if time_span > 0:
                overall_freq = (len(self.message_times) - 1) / time_span
        
        # Joint position frequency
        joint_freq = 0.0
        if len(self.joint_position_times) >= 2:
            time_span = self.joint_position_times[-1] - self.joint_position_times[0]
            if time_span > 0:
                joint_freq = (len(self.joint_position_times) - 1) / time_span
        
        # Latency statistics
        latency_stats = {}
        if len(self.latency_values) > 0:
            latencies = list(self.latency_values)
            latency_stats = {
                'current_avg': np.mean(latencies),
                'current_std': np.std(latencies),
                'current_min': np.min(latencies),
                'current_max': np.max(latencies),
                'current_median': np.median(latencies),
                'window_size': len(latencies)
            }
        
        # Joint position latency
        joint_latency_stats = {}
        if len(self.joint_position_latencies) > 0:
            joint_latencies = list(self.joint_position_latencies)
            joint_latency_stats = {
                'avg': np.mean(joint_latencies),
                'std': np.std(joint_latencies),
                'min': np.min(joint_latencies),
                'max': np.max(joint_latencies),
                'count': len(joint_latencies)
            }
        
        # Overall statistics
        elapsed_time = current_time - self.start_time
        avg_message_rate = self.total_messages / elapsed_time if elapsed_time > 0 else 0
        avg_data_rate = self.total_bytes / elapsed_time if elapsed_time > 0 else 0
        
        return {
            'overall_freq_hz': overall_freq,
            'joint_freq_hz': joint_freq,
            'latency': latency_stats,
            'joint_latency': joint_latency_stats,
            'message_counts': self.message_counts.copy(),
            'total_messages': self.total_messages,
            'total_bytes': self.total_bytes,
            'avg_message_rate_hz': avg_message_rate,
            'avg_data_rate_bps': avg_data_rate,
            'elapsed_time_s': elapsed_time
        }
    
    def should_print_report(self, interval_seconds: float = 5.0) -> bool:
        """Check if it's time to print a frequency report"""
        current_time = time.time()
        if current_time - self.last_report_time >= interval_seconds:
            self.last_report_time = current_time
            return True
        return False
    
    def print_frequency_summary(self):
        """Print a comprehensive frequency and latency summary"""
        stats = self.get_current_frequencies()
        
        print("\n" + "="*60)
        print("📡 ZEROMQ CLIENT PERFORMANCE REPORT")
        print("="*60)
        print(f"Overall Message Rate: {stats['overall_freq_hz']:.1f} Hz")
        print(f"Joint Position Rate: {stats['joint_freq_hz']:.1f} Hz")
        print(f"Total Messages: {stats['total_messages']}")
        print(f"Total Data: {stats['total_bytes']/1024:.1f} KB")
        print(f"Average Data Rate: {stats['avg_data_rate_bps']/1024:.1f} KB/s")
        print(f"Elapsed Time: {stats['elapsed_time_s']:.1f}s")
        
        # Latency report
        if stats['latency']:
            lat = stats['latency']
            quality = "✅ Excellent" if lat['current_avg'] < 10 else "✅ Good" if lat['current_avg'] < 20 else "⚠️ Fair" if lat['current_avg'] < 50 else "❌ High"
            print(f"\n🕐 LATENCY ANALYSIS:")
            print(f"Current Average: {lat['current_avg']:.1f}ms (±{lat['current_std']:.1f}ms) - {quality}")
            print(f"Current Range: {lat['current_min']:.1f} - {lat['current_max']:.1f}ms")
            print(f"Current Median: {lat['current_median']:.1f}ms")
            print(f"Window Size: {lat['window_size']} messages")
            
            if self.total_latency_count > 0:
                lifetime_avg = self.total_latency_sum / self.total_latency_count
                lifetime_min = self.min_latency if self.min_latency != float('inf') else 0
                print(f"Lifetime Average: {lifetime_avg:.1f}ms")
                print(f"Lifetime Range: {lifetime_min:.1f} - {self.max_latency:.1f}ms")
                print(f"Lifetime Count: {self.total_latency_count}")
        
        # Joint position latency
        if stats['joint_latency']:
            jlat = stats['joint_latency']
            print(f"\n🎯 JOINT POSITION LATENCY:")
            print(f"Average: {jlat['avg']:.1f}ms (±{jlat['std']:.1f}ms)")
            print(f"Range: {jlat['min']:.1f} - {jlat['max']:.1f}ms")
            print(f"Count: {jlat['count']}")
        
        # Message type breakdown
        if stats['message_counts']:
            print(f"\n📊 MESSAGE TYPES:")
            for msg_type, count in stats['message_counts'].items():
                percentage = (count / stats['total_messages']) * 100
                print(f"  {msg_type}: {count} ({percentage:.1f}%)")
        
        print("="*60)

class MuJoCoZeroMQClient:
    """ZeroMQ client for receiving MuJoCo hardware interface data."""
    
    def __init__(self, address: str = "tcp://localhost:5555"):
        """Initialize the ZeroMQ client."""
        if not address or not address.startswith(('tcp://', 'ipc://', 'inproc://')):
            raise ValueError(f"Invalid ZeroMQ address: {address}")
        
        self.address = address
        self.context = zmq.Context()
        self.socket = None
        self.is_connected = False
        
        # Initialize frequency monitor
        self.frequency_monitor = ZeroMQClientFrequencyMonitor(window_size=50)
        
        # Initialize min/max tracker
        self.joint_tracker = JointMinMaxTracker()
        
        # Legacy statistics (kept for backward compatibility)
        self.total_messages_received = 0
        self.message_types_count = {}
        self.start_time = time.time()
        self.last_fps_time = time.time()
        self.fps_message_count = 0
        
        # Rate limiting for display
        self.last_print_time = 0.0
        self.print_interval = 1.0  # Print every 1 second
        self.buffered_messages = []
        
        # Message tracking
        self.last_joint_data = {}
        self.last_ping_time = None
        
        # Display options
        self.show_full_joint_data = True
        self.show_organized_data = True
        self.show_metadata = True
        self.show_ping_messages = True
        self.show_raw_structure = True  # New option to show raw message structure
        self.show_minmax_summary = True  # Show min/max tracking
        self.fps_display_interval = 5.0  # Show FPS every 5 seconds
        
        # Frequency reporting
        self.last_frequency_report = time.time()
        self.frequency_report_interval = 10.0  # Report every 10 seconds
        
        # Min/max reporting
        self.last_minmax_report = time.time()
        self.minmax_report_interval = 15.0  # Report every 15 seconds
        
        # Statistics
        self.connection_attempts = 0
        self.message_count = 0
        self.message_history = deque(maxlen=100)
        self.latest_message = None
        
        print(f"MuJoCo ZeroMQ Client initialized")
        print(f"Target address: {address}")
        print("Ready to connect and receive joint position data")

    def analyze_message_structure(self, data: Dict[str, Any]) -> str:
        """Analyze and format the raw message structure."""
        lines = []
        lines.append("   🔍 RAW MESSAGE STRUCTURE:")
        
        def analyze_value(key: str, value: Any, indent: str = "     ") -> None:
            if isinstance(value, dict):
                lines.append(f"{indent}{key}: dict ({len(value)} keys)")
                if len(value) <= 10:  # Show keys for small dicts
                    for sub_key in sorted(value.keys()):
                        lines.append(f"{indent}  └─ {sub_key}")
                else:
                    # Show first few keys for large dicts
                    sorted_keys = sorted(value.keys())
                    for sub_key in sorted_keys[:5]:
                        lines.append(f"{indent}  ├─ {sub_key}")
                    lines.append(f"{indent}  └─ ... and {len(value) - 5} more keys")
                    
            elif isinstance(value, list):
                lines.append(f"{indent}{key}: list (length {len(value)})")
                if len(value) > 0:
                    first_item = value[0]
                    item_type = type(first_item).__name__
                    lines.append(f"{indent}  └─ items: {item_type}")
                    
            elif isinstance(value, (int, float)):
                lines.append(f"{indent}{key}: {type(value).__name__} = {value}")
                
            elif isinstance(value, str):
                truncated = value[:30] + "..." if len(value) > 30 else value
                lines.append(f"{indent}{key}: str = \"{truncated}\"")
                
            else:
                lines.append(f"{indent}{key}: {type(value).__name__}")
        
        for key, value in data.items():
            analyze_value(key, value)
            
        return "\n".join(lines)
        
    def format_joint_positions(self, joint_data: Dict[str, Union[float, int]], max_joints: int = 8, 
                              data_type: str = "mixed") -> str:
        """Format joint positions for display."""
        if not joint_data:
            return "No joint data"
            
        # Sort joints by name for consistent display
        sorted_joints = sorted(joint_data.items())
        
        if len(sorted_joints) <= max_joints:
            # Show all joints if count is reasonable
            if data_type == "hand":
                joint_strs = [f"{name}: {pos:4d}" for name, pos in sorted_joints]
            else:
                joint_strs = [f"{name}: {pos:.3f}" for name, pos in sorted_joints]
            return "\n      ".join(joint_strs)
        else:
            # Show first few joints and summary for large datasets
            shown_joints = sorted_joints[:max_joints]
            if data_type == "hand":
                joint_strs = [f"{name}: {pos:4d}" for name, pos in shown_joints]
            else:
                joint_strs = [f"{name}: {pos:.3f}" for name, pos in shown_joints]
            remaining = len(sorted_joints) - max_joints
            joint_strs.append(f"... and {remaining} more joints")
            return "\n      ".join(joint_strs)
    
    def format_organized_data(self, organized: Dict[str, Dict[str, Union[float, int]]]) -> str:
        """Format organized joint data for display."""
        if not organized:
            return "No organized data"
            
        lines = []
        for category, joints in organized.items():
            if joints:
                lines.append(f"     {category.replace('_', ' ').title()} ({len(joints)} joints):")
                
                # Determine data type for formatting
                data_type = "hand" if "hand" in category else "upper_body"
                
                if len(joints) <= 3:
                    for name, pos in sorted(joints.items()):
                        if data_type == "hand":
                            lines.append(f"       {name}: {pos:4d}")
                        else:
                            lines.append(f"       {name}: {pos:.3f}")
                else:
                    # Show first 2 joints for large categories
                    sorted_joints = sorted(joints.items())
                    for name, pos in sorted_joints[:2]:
                        if data_type == "hand":
                            lines.append(f"       {name}: {pos:4d}")
                        else:
                            lines.append(f"       {name}: {pos:.3f}")
                    remaining = len(joints) - 2
                    lines.append(f"       ... and {remaining} more")
        
        return "\n".join(lines)
    
    def format_latest_message_summary(self, messages: list) -> str:
        """Format a summary of the latest messages received."""
        if not messages:
            return "No messages in buffer"
            
        latest_msg = messages[-1]
        timestamp = latest_msg.get('timestamp', time.time())
        dt = datetime.fromtimestamp(timestamp)
        msg_type = latest_msg.get('type', 'unknown')
        
        lines = []
        lines.append(f"🤖 LATEST MESSAGE SUMMARY (from {len(messages)} recent messages)")
        lines.append(f"   Timestamp: {dt.strftime('%H:%M:%S.%f')[:-3]}")
        lines.append(f"   Type: {msg_type}")
        
        # Show raw structure
        if self.show_raw_structure:
            lines.append(self.analyze_message_structure(latest_msg))
        
        # Handle joint positions specifically
        if msg_type == 'joint_positions':
            message_data = latest_msg.get('data', {})
            all_joints = message_data.get('all_joints', {})
            organized = message_data.get('organized', {})
            metadata = latest_msg.get('metadata', {})
            
            # Update min/max tracking from organized data
            if organized:
                self.joint_tracker.update_from_organized_data(organized)
            
            # Show metadata with VR tracking info
            if self.show_metadata and metadata:
                lines.append(f"\n   📊 METADATA:")
                lines.append(f"     Source: {metadata.get('source', 'unknown')}")
                lines.append(f"     Total joints: {metadata.get('total_joint_count', 0)}")
                lines.append(f"     Update rate: {metadata.get('update_hz', 0):.1f} Hz")
                
                # Show VR tracking specific info
                hand_data_source = metadata.get('hand_data_source', 'unknown')
                hand_data_range = metadata.get('hand_data_range', 'unknown')
                if hand_data_source != 'unknown':
                    lines.append(f"     Hand data: {hand_data_source} ({hand_data_range})")
                
                # Show joint breakdown
                upper_count = metadata.get('upper_body_joints', 0)
                left_hand_count = metadata.get('left_hand_joints', 0)
                right_hand_count = metadata.get('right_hand_joints', 0)
                if upper_count > 0 or left_hand_count > 0 or right_hand_count > 0:
                    lines.append(f"     Breakdown: {upper_count} upper body, {left_hand_count} left hand, {right_hand_count} right hand")
            
            # Show organized data (more readable with proper formatting)
            if self.show_organized_data and organized:
                lines.append(f"\n   📋 ORGANIZED JOINT DATA:")
                lines.append(self.format_organized_data(organized))
            
            # Show sample of all joints (backward compatibility)
            if self.show_full_joint_data and all_joints:
                joint_count = len(all_joints)
                lines.append(f"\n   🎯 SAMPLE JOINTS ({joint_count} total):")
                lines.append(f"      {self.format_joint_positions(all_joints, max_joints=6)}")
            
            # Store for comparison
            self.last_joint_data = all_joints.copy()
            
        elif msg_type == 'ping':
            if self.show_ping_messages:
                source = latest_msg.get('source', 'unknown')
                lines.append(f"   🏓 Ping from: {source}")
                self.last_ping_time = timestamp
        
        return "\n".join(lines)
    
    def connect(self) -> bool:
        """Connect to ZeroMQ publisher"""
        self.connection_attempts += 1
        
        print(f"🔌 Connecting to ZeroMQ publisher at {self.address} (attempt {self.connection_attempts})")
        
        try:
            # Create SUB socket
            self.socket = self.context.socket(zmq.SUB)
            
            # Set socket options
            self.socket.setsockopt(zmq.RCVHWM, 1000)  # High water mark
            self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout
            self.socket.setsockopt(zmq.SUBSCRIBE, b"joint_positions")  # Subscribe to joint_positions topic
            
            # Connect to publisher
            self.socket.connect(self.address)
            
            # Brief pause to allow connection to establish
            time.sleep(0.1)
            
            self.is_connected = True
            print(f"✅ Connected to ZeroMQ publisher successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to ZeroMQ publisher: {e}")
            self.is_connected = False
            if self.socket:
                self.socket.close()
                self.socket = None
            return False
    
    def disconnect(self):
        """Disconnect from ZeroMQ publisher"""
        print("🔌 Disconnecting from ZeroMQ publisher...")
        self.is_connected = False
        if self.socket:
            self.socket.close()
            self.socket = None
        if self.context:
            self.context.term()
    
    def update_statistics(self, message_type: str, message_size: int = 0, latency_ms: Optional[float] = None):
        """Update message statistics."""
        # Use frequency monitor for accurate tracking
        self.frequency_monitor.record_message(message_type, message_size, latency_ms)
        
        # Legacy statistics (kept for backward compatibility)
        self.total_messages_received += 1
        self.fps_message_count += 1
        self.message_count += 1
        
        if message_type not in self.message_types_count:
            self.message_types_count[message_type] = 0
        self.message_types_count[message_type] += 1
        
        # Display FPS periodically using legacy method
        current_time = time.time()
        if current_time - self.last_fps_time >= self.fps_display_interval:
            elapsed = current_time - self.last_fps_time
            fps = self.fps_message_count / elapsed
            
            print(f"\n📈 LEGACY STATISTICS (last {elapsed:.1f}s)")
            print(f"   Message rate: {fps:.1f} messages/sec")
            print(f"   Total received: {self.total_messages_received}")
            print(f"   By type: {self.message_types_count}")
            
            self.last_fps_time = current_time
            self.fps_message_count = 0
    
    def check_and_print_frequency_report(self):
        """Check if it's time to print a detailed frequency report."""
        current_time = time.time()
        if current_time - self.last_frequency_report >= self.frequency_report_interval:
            self.frequency_monitor.print_frequency_summary()
            self.last_frequency_report = current_time
    
    def check_and_print_minmax_report(self):
        """Check if it's time to print a min/max tracking report."""
        current_time = time.time()
        if (current_time - self.last_minmax_report >= self.minmax_report_interval and 
            self.show_minmax_summary):
            print(self.joint_tracker.format_range_summary())
            self.last_minmax_report = current_time
    
    def handle_message(self, topic: bytes, message: bytes):
        """Handle received ZeroMQ message with rate limiting."""
        try:
            # Capture receive time immediately for latency calculation
            receive_time = time.time()
            
            # Parse JSON message
            message_str = message.decode('utf-8')
            data = json.loads(message_str)
            message_type = data.get('type', 'unknown')
            
            # Calculate latency if message has timestamp
            latency_ms = None
            message_timestamp = data.get('timestamp')
            if message_timestamp is not None:
                latency_s = receive_time - message_timestamp
                latency_ms = latency_s * 1000  # Convert to milliseconds
            
            # Update statistics with latency
            message_size = len(message_str)
            self.update_statistics(message_type, message_size, latency_ms)
            
            # Store message
            self.latest_message = data
            self.message_history.append(data)
            
            # Add to buffer for rate-limited display
            self.buffered_messages.append(data)
            
            # Update joint tracking if this is a joint_positions message
            if message_type == 'joint_positions':
                message_data = data.get('data', {})
                if 'organized' in message_data:
                    self.joint_tracker.update_from_organized_data(message_data['organized'])
            
            # Check if it's time to print
            current_time = time.time()
            if current_time - self.last_print_time >= self.print_interval:
                # Print summary of buffered messages
                print(self.format_latest_message_summary(self.buffered_messages))
                
                # Clear buffer and update print time
                self.buffered_messages = []
                self.last_print_time = current_time
                
            # Check and print frequency report
            self.check_and_print_frequency_report()
            
            # Check and print min/max report
            self.check_and_print_minmax_report()
            
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON DECODE ERROR: {e}")
            print(f"   Raw message: {message_str[:200]}...")
        except Exception as e:
            print(f"\n❌ MESSAGE HANDLING ERROR: {e}")
            print(f"   Raw message: {message_str[:200]}...")
    
    def listen_for_messages(self):
        """Listen for messages from ZeroMQ publisher."""
        global running
        
        print(f"🎯 Listening for messages from ZeroMQ publisher...")
        print(f"   Displaying updates every {self.print_interval:.1f} seconds")
        print(f"   Frequency reports every {self.frequency_report_interval:.1f} seconds")
        print(f"   Min/Max reports every {self.minmax_report_interval:.1f} seconds")
        print(f"   Press Ctrl+C to stop")
        print(f"   While running, press 'f' + Enter to show frequency report")
        print(f"   While running, press 'm' + Enter to show min/max report")
        print()
        
        while running and self.is_connected:
            try:
                # Receive message with timeout
                topic, message = self.socket.recv_multipart(zmq.NOBLOCK)
                self.handle_message(topic, message)
                
                # Handle user input
                if not self.handle_user_input():
                    break
                
            except zmq.Again:
                # No message available, this is expected with NOBLOCK
                continue
            except Exception as e:
                print(f"❌ Error receiving message: {e}")
                break
    
    def print_final_statistics(self):
        """Print final statistics."""
        total_time = time.time() - self.start_time
        avg_rate = self.total_messages_received / total_time if total_time > 0 else 0
        
        print(f"\n📊 FINAL STATISTICS")
        print(f"   Total runtime: {total_time:.1f} seconds")
        print(f"   Total messages: {self.total_messages_received}")
        print(f"   Average rate: {avg_rate:.1f} messages/sec")
        print(f"   Connection attempts: {self.connection_attempts}")
        print(f"   Message types: {self.message_types_count}")
        
        # Print final frequency monitor report
        print(f"\n📊 FINAL FREQUENCY ANALYSIS:")
        self.frequency_monitor.print_frequency_summary()
        
        # Print final min/max tracking report
        print(f"\n📊 FINAL MIN/MAX TRACKING:")
        print(self.joint_tracker.format_range_summary(max_joints_per_category=10))
        
        # Get detailed frequency stats
        stats = self.frequency_monitor.get_current_frequencies()
        if 'joint_freq_hz' in stats:
            expected_hz = 150.0  # Expected from teleoperation system
            actual_hz = stats['joint_freq_hz']
            efficiency = (actual_hz / expected_hz) * 100 if expected_hz > 0 else 0
            print(f"\n🎯 JOINT POSITION FREQUENCY ANALYSIS:")
            print(f"   Expected: {expected_hz:.1f}Hz")
            print(f"   Actual: {actual_hz:.1f}Hz")
            print(f"   Efficiency: {efficiency:.1f}%")
            if efficiency < 90:
                print(f"   ⚠️  Low efficiency detected - possible throttling or network issues")
            elif efficiency > 95:
                print(f"   ✅ Good efficiency - receiving data at expected rate")
        
        # Print latest message summary
        if self.message_history:
            print(f"\n📖 LATEST MESSAGE SUMMARY:")
            print(self.format_latest_message_summary(list(self.message_history)))
    
    def handle_user_input(self):
        """Handle user keyboard input for manual commands."""
        try:
            # Check if input is available
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.readline().strip().lower()
                if user_input == 'f':
                    print("\n📊 MANUAL FREQUENCY REPORT:")
                    self.frequency_monitor.print_frequency_summary()
                    print("   (Type 'f' + Enter anytime for frequency report)")
                elif user_input == 'm':
                    print("\n📊 MANUAL MIN/MAX REPORT:")
                    print(self.joint_tracker.format_range_summary(max_joints_per_category=10))
                    print("   (Type 'm' + Enter anytime for min/max report)")
                elif user_input == 'r':
                    print("\n🔄 RESETTING MIN/MAX TRACKING:")
                    self.joint_tracker.reset()
                    print("   Min/max tracking data has been reset")
                elif user_input == 'raw':
                    self.show_raw_structure = not self.show_raw_structure
                    print(f"\n🔍 Raw structure display: {'ON' if self.show_raw_structure else 'OFF'}")
                elif user_input == 'l' or user_input == 'latest':
                    if self.message_history:
                        print("\n📖 LATEST MESSAGE:")
                        print(self.format_latest_message_summary(list(self.message_history)))
                    else:
                        print("\n📖 No messages received yet")
                elif user_input == 'q':
                    print("👋 Quit command received")
                    return False
                elif user_input in ['j', 'joints']:
                    print("\n🎯 JOINT STATISTICS:")
                    range_summary = self.joint_tracker.format_range_summary()
                    if range_summary:
                        print(f"{range_summary}")
                    else:
                        print("No joint data available yet")
                elif user_input:
                    print(f"❓ Unknown command: '{user_input}'")
                    print("Available commands: 'f'=frequency, 'm'=min/max, 'r'=reset, 'raw'=toggle raw, 'l'=latest, 'j'=joints, 'q'=quit")
        except Exception:
            pass  # Ignore input errors
        return True

def run_client(address: str, test_mode: bool = False, 
              frequency_report_interval: float = 10.0, frequency_window_size: int = 50,
              enable_minmax_tracking: bool = True, minmax_report_interval: float = 15.0):
    """Run the ZeroMQ client."""
    global running
    
    # Create client
    client = MuJoCoZeroMQClient(address)
    
    # Configure frequency monitoring
    client.frequency_monitor = ZeroMQClientFrequencyMonitor(window_size=frequency_window_size)
    client.frequency_report_interval = frequency_report_interval
    
    # Configure min/max tracking
    client.show_minmax_summary = enable_minmax_tracking
    client.minmax_report_interval = minmax_report_interval
    
    # Configure display options
    if test_mode:
        print("🧪 Test mode: Limited output with faster updates")
        client.print_interval = 0.5  # Faster updates in test mode
        client.show_full_joint_data = False  # Reduce output in test mode
        client.fps_display_interval = 2.0  # More frequent stats in test mode
        client.frequency_report_interval = 5.0  # More frequent frequency reports in test mode
        client.minmax_report_interval = 8.0  # More frequent min/max reports in test mode
    else:
        client.print_interval = 1.0  # Normal 1-second updates
        client.show_full_joint_data = True
        client.fps_display_interval = 5.0
    
    client.show_organized_data = True
    client.show_metadata = True
    client.show_ping_messages = True
    client.show_raw_structure = True
    
    print(f"📊 Frequency monitoring enabled:")
    print(f"   Report interval: {client.frequency_report_interval:.1f}s")
    print(f"   Window size: {client.frequency_monitor.window_size} messages")
    
    if enable_minmax_tracking:
        print(f"📊 Min/Max tracking enabled:")
        print(f"   Report interval: {client.minmax_report_interval:.1f}s")
    
    # Connect to publisher
    if not client.connect():
        print("❌ Failed to connect to ZeroMQ publisher")
        return False
    
    if test_mode:
        print("🧪 Test mode: Will listen for a few messages then exit")
        test_start_time = time.time()
        test_duration = 5.0  # 5 seconds
        
        while running and client.is_connected and (time.time() - test_start_time) < test_duration:
            try:
                topic, message = client.socket.recv_multipart(zmq.NOBLOCK)
                client.handle_message(topic, message)
                
                # Check for periodic reports
                client.check_and_print_frequency_report()
                
                if enable_minmax_tracking:
                    client.check_and_print_minmax_report()
                
            except zmq.Again:
                continue
            except Exception as e:
                print(f"❌ Error in test mode: {e}")
                break
        
        print("🧪 Test mode completed")
    else:
        # Start message listening in a separate thread
        listen_thread = threading.Thread(target=client.listen_for_messages, daemon=True)
        listen_thread.start()
        
        try:
            while running and client.is_connected:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        except Exception as e:
            print(f"❌ Error in main loop: {e}")
    
    # Cleanup
    running = False
    client.disconnect()
    
    # Print final statistics
    client.print_final_statistics()
    
    return True

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    global running
    print("\n🛑 Received interrupt signal, shutting down...")
    running = False

def main():
    """Main function."""
    global running
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='ZeroMQ Client for MuJoCo Hardware Interface')
    parser.add_argument('--address', type=str, default='tcp://localhost:5555',
                       help='ZeroMQ publisher address (default: tcp://localhost:5555)')
    parser.add_argument('--test', action='store_true',
                       help='Run in test mode (brief connection test)')
    parser.add_argument('--frequency_report_interval', type=float, default=10.0,
                       help='Frequency report interval in seconds (default: 10.0)')
    parser.add_argument('--frequency_window_size', type=int, default=50,
                       help='Frequency calculation window size (default: 50)')
    parser.add_argument('--no_frequency_reports', action='store_true',
                       help='Disable automatic frequency reports')
    parser.add_argument('--minmax_report_interval', type=float, default=15.0,
                       help='Min/max report interval in seconds (default: 15.0)')
    parser.add_argument('--no_minmax_tracking', action='store_true',
                       help='Disable min/max tracking and reports')
    
    args = parser.parse_args()
    
    # Validate address
    if not args.address.startswith(('tcp://', 'ipc://', 'inproc://')):
        print(f"❌ Invalid ZeroMQ address: {args.address}")
        print("   Address should start with tcp://, ipc://, or inproc://")
        return 1
    
    print("🚀 MuJoCo ZeroMQ Client")
    print("=" * 50)
    print(f"Connecting to: {args.address}")
    print(f"Test mode: {args.test}")
    if args.test:
        print("Display updates: Every 0.5 seconds")
    else:
        print("Display updates: Every 1.0 seconds")
    
    # Frequency monitoring configuration
    if args.no_frequency_reports:
        print("Frequency reports: Disabled")
        frequency_interval = float('inf')  # Effectively disable by setting very large interval
    else:
        print(f"Frequency reports: Every {args.frequency_report_interval:.1f}s")
        frequency_interval = args.frequency_report_interval
    
    print(f"Frequency window: {args.frequency_window_size} messages")
    
    # Min/Max tracking configuration  
    if args.no_minmax_tracking:
        print("Min/Max tracking: Disabled")
        enable_minmax = False
        minmax_interval = float('inf')
    else:
        print(f"Min/Max reports: Every {args.minmax_report_interval:.1f}s")
        enable_minmax = True
        minmax_interval = args.minmax_report_interval
        
    print("=" * 50)
    
    # Run the client
    try:
        success = run_client(
            address=args.address,
            test_mode=args.test,
            frequency_report_interval=frequency_interval,
            frequency_window_size=args.frequency_window_size,
            enable_minmax_tracking=enable_minmax,
            minmax_report_interval=minmax_interval
        )
        
        if success:
            print("✅ ZeroMQ client completed successfully")
            return 0
        else:
            print("❌ ZeroMQ client failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 