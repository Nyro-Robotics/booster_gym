#!/usr/bin/env python3
"""
Example WebSocket Client for MuJoCo Hardware Interface
====================================================

This client connects to the WebSocket relay server and receives all messages
from the MuJoCo hardware interface, including joint positions and other data.

Latency Measurement and Clock Synchronization:
-----------------------------------------------
This client measures latency by comparing message timestamps (from sender) with
receive times (from receiver). Due to clock synchronization differences between
systems, you may see negative latency values when the sender's clock is ahead
of the receiver's clock.

This is a normal behavior in distributed systems. By default, negative latencies
are clamped to 0ms for statistical purposes. Use --raw-latency to see actual
raw values for debugging clock synchronization issues.

Solutions for accurate latency measurement:
- Synchronize clocks using NTP (Network Time Protocol)
- Use high-resolution monotonic clocks for timestamps
- Implement dedicated latency measurement protocols

Usage:
    python example_client.py [--host HOST] [--port PORT]
    python example_client.py --raw-latency  # Show negative latencies
"""

import asyncio
import websockets
import json
import time
import argparse
import sys
import signal
import logging
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebSocketClientFrequencyMonitor:
    """Frequency monitor for WebSocket message reception."""
    
    def __init__(self, window_size: int = 50):
        if window_size <= 0:
            raise ValueError("Window size must be positive")
        
        self.window_size = window_size
        self.message_times = deque(maxlen=window_size)
        self.joint_position_times = deque(maxlen=window_size)
        self.ping_times = deque(maxlen=window_size)
        
        # Latency tracking
        self.latencies = deque(maxlen=window_size)
        self.joint_position_latencies = deque(maxlen=window_size)
        self.ping_latencies = deque(maxlen=window_size)
        
        self.last_message_time = None
        self.last_joint_position_time = None
        self.last_ping_time = None
        self.last_report_time = time.time()
        
        # Counters
        self.total_messages = 0
        self.total_joint_positions = 0
        self.total_pings = 0
        self.start_time = time.time()
        
        # Message type tracking
        self.message_types_count = {}
        
        # Latency statistics
        self.total_latency_sum = 0.0
        self.min_latency = float('inf')
        self.max_latency = 0.0
        
    def record_message(self, message_type: str = "unknown", message_timestamp: Optional[float] = None, raw_latency: bool = False):
        """Record a message received with optional latency calculation."""
        current_time = time.perf_counter()
        receive_time = time.time()  # Wall clock time for latency calculation
        
        # Calculate latency if message timestamp is provided
        latency_ms = None
        if message_timestamp is not None:
            latency_s = receive_time - message_timestamp
            latency_ms = latency_s * 1000  # Convert to milliseconds
            
            # Validate latency - negative values indicate clock synchronization issues
            # Skip validation if raw_latency mode is enabled (for debugging)
            if not raw_latency:
                if latency_ms < 0:
                    # Log warning for significant negative latencies (more than -100ms suggests real clock skew)
                    if latency_ms < -100:
                        logger.warning(f"⚠️  Large negative latency detected: {latency_ms:.1f}ms - "
                                     f"sender clock may be ahead of receiver clock")
                    # Clamp negative latencies to 0 for statistics
                    latency_ms = 0.0
                elif latency_ms > 10000:  # More than 10 seconds suggests timestamp error
                    logger.warning(f"⚠️  Extremely high latency detected: {latency_ms:.1f}ms - "
                                 f"possible timestamp format mismatch")
                    # Don't record obviously invalid latencies
                    latency_ms = None
            
            # Update latency statistics only for valid latencies
            if latency_ms is not None:
                self.latencies.append(latency_ms)
                self.total_latency_sum += latency_ms
                self.min_latency = min(self.min_latency, latency_ms)
                self.max_latency = max(self.max_latency, latency_ms)
        
        # Record general message timing
        if self.last_message_time is not None:
            dt = current_time - self.last_message_time
            self.message_times.append(dt)
        self.last_message_time = current_time
        self.total_messages += 1
        
        # Track message types
        if message_type not in self.message_types_count:
            self.message_types_count[message_type] = 0
        self.message_types_count[message_type] += 1
        
        # Record specific message type timing and latency
        if message_type == "joint_positions":
            if self.last_joint_position_time is not None:
                dt = current_time - self.last_joint_position_time
                self.joint_position_times.append(dt)
            self.last_joint_position_time = current_time
            self.total_joint_positions += 1
            
            # Record joint position latency
            if latency_ms is not None:
                self.joint_position_latencies.append(latency_ms)
            
        elif message_type == "ping":
            if self.last_ping_time is not None:
                dt = current_time - self.last_ping_time
                self.ping_times.append(dt)
            self.last_ping_time = current_time
            self.total_pings += 1
            
            # Record ping latency
            if latency_ms is not None:
                self.ping_latencies.append(latency_ms)
    
    def get_current_frequencies(self) -> Dict:
        """Get current frequency measurements including latency."""
        stats = {}
        
        # Overall message frequency
        if len(self.message_times) > 5:
            msg_freq = 1.0 / np.mean(self.message_times)
            msg_jitter = np.std(self.message_times) * 1000  # ms
            stats['message_hz'] = msg_freq
            stats['message_jitter_ms'] = msg_jitter
        
        # Joint position frequency
        if len(self.joint_position_times) > 5:
            joint_freq = 1.0 / np.mean(self.joint_position_times)
            joint_jitter = np.std(self.joint_position_times) * 1000  # ms
            stats['joint_position_hz'] = joint_freq
            stats['joint_position_jitter_ms'] = joint_jitter
        
        # Ping frequency
        if len(self.ping_times) > 5:
            ping_freq = 1.0 / np.mean(self.ping_times)
            ping_jitter = np.std(self.ping_times) * 1000  # ms
            stats['ping_hz'] = ping_freq
            stats['ping_jitter_ms'] = ping_jitter
        
        # Overall latency statistics
        if len(self.latencies) > 0:
            stats['avg_latency_ms'] = np.mean(self.latencies)
            stats['latency_jitter_ms'] = np.std(self.latencies)
            stats['min_latency_ms'] = np.min(self.latencies)
            stats['max_latency_ms'] = np.max(self.latencies)
            stats['median_latency_ms'] = np.median(self.latencies)
        
        # Joint position latency statistics
        if len(self.joint_position_latencies) > 0:
            stats['joint_avg_latency_ms'] = np.mean(self.joint_position_latencies)
            stats['joint_latency_jitter_ms'] = np.std(self.joint_position_latencies)
            stats['joint_min_latency_ms'] = np.min(self.joint_position_latencies)
            stats['joint_max_latency_ms'] = np.max(self.joint_position_latencies)
        
        # Ping latency statistics
        if len(self.ping_latencies) > 0:
            stats['ping_avg_latency_ms'] = np.mean(self.ping_latencies)
            stats['ping_latency_jitter_ms'] = np.std(self.ping_latencies)
            stats['ping_min_latency_ms'] = np.min(self.ping_latencies)
            stats['ping_max_latency_ms'] = np.max(self.ping_latencies)
        
        # Average frequencies
        runtime = time.time() - self.start_time
        if runtime > 0:
            stats['avg_message_hz'] = self.total_messages / runtime
            stats['avg_joint_position_hz'] = self.total_joint_positions / runtime
            stats['avg_ping_hz'] = self.total_pings / runtime
            stats['runtime_s'] = runtime
        
        # Overall latency statistics (lifetime)
        if self.total_messages > 0:
            stats['lifetime_avg_latency_ms'] = self.total_latency_sum / self.total_messages
            stats['lifetime_min_latency_ms'] = self.min_latency if self.min_latency != float('inf') else 0
            stats['lifetime_max_latency_ms'] = self.max_latency
        
        # Message type breakdown
        stats['message_types'] = self.message_types_count.copy()
        
        return stats
    
    def should_print_report(self, interval_seconds: float = 5.0) -> bool:
        """Check if it's time to print a frequency report."""
        current_time = time.time()
        if current_time - self.last_report_time >= interval_seconds:
            self.last_report_time = current_time
            return True
        return False
    
    def print_frequency_summary(self):
        """Print a detailed frequency summary."""
        stats = self.get_current_frequencies()
        
        if 'message_hz' in stats:
            logger.info(f"📊 WEBSOCKET CLIENT FREQUENCY REPORT:")
            logger.info(f"   Overall: {stats['message_hz']:.1f}Hz (±{stats['message_jitter_ms']:.1f}ms)")
            
            if 'joint_position_hz' in stats:
                logger.info(f"   Joint Positions: {stats['joint_position_hz']:.1f}Hz (±{stats['joint_position_jitter_ms']:.1f}ms)")
            
            if 'ping_hz' in stats:
                logger.info(f"   Pings: {stats['ping_hz']:.1f}Hz (±{stats['ping_jitter_ms']:.1f}ms)")
            
            logger.info(f"   Average Rates: Msg={stats.get('avg_message_hz', 0):.1f}Hz, "
                       f"Joints={stats.get('avg_joint_position_hz', 0):.1f}Hz, "
                       f"Runtime={stats.get('runtime_s', 0):.1f}s")
            
            # Latency information (only if enabled and available)
            if hasattr(self, 'show_latency') and self.show_latency and 'avg_latency_ms' in stats:
                logger.info(f"🕐 LATENCY REPORT:")
                logger.info(f"   Overall: {stats['avg_latency_ms']:.1f}ms avg (±{stats['latency_jitter_ms']:.1f}ms)")
                logger.info(f"   Range: {stats['min_latency_ms']:.1f}ms min, {stats['max_latency_ms']:.1f}ms max, {stats['median_latency_ms']:.1f}ms median")
                
                if 'joint_avg_latency_ms' in stats:
                    logger.info(f"   Joint Positions: {stats['joint_avg_latency_ms']:.1f}ms avg (±{stats['joint_latency_jitter_ms']:.1f}ms)")
                    logger.info(f"     Range: {stats['joint_min_latency_ms']:.1f}-{stats['joint_max_latency_ms']:.1f}ms")
                
                if 'ping_avg_latency_ms' in stats:
                    logger.info(f"   Pings: {stats['ping_avg_latency_ms']:.1f}ms avg (±{stats['ping_latency_jitter_ms']:.1f}ms)")
                    logger.info(f"     Range: {stats['ping_min_latency_ms']:.1f}-{stats['ping_max_latency_ms']:.1f}ms")
                
                # Latency quality assessment
                avg_latency = stats['avg_latency_ms']
                if avg_latency < 10:
                    logger.info(f"   ✅ Excellent latency (<10ms)")
                elif avg_latency < 50:
                    logger.info(f"   ✅ Good latency (<50ms)")
                elif avg_latency < 100:
                    logger.info(f"   ⚠️  Fair latency (50-100ms)")
                else:
                    logger.info(f"   ❌ High latency (>100ms) - possible network issues")
            
            # Message type breakdown
            if 'message_types' in stats and stats['message_types']:
                type_summary = ", ".join([f"{k}={v}" for k, v in stats['message_types'].items()])
                logger.info(f"   Message Types: {type_summary}")
        else:
            logger.info("📊 WEBSOCKET CLIENT: Not enough data for frequency analysis")


class MuJoCoWebSocketClient:
    """WebSocket client for receiving MuJoCo hardware interface data."""
    
    def __init__(self, host: str = "localhost", port: int = 8765, show_latency: bool = True, raw_latency: bool = False):
        """Initialize the WebSocket client."""
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self.websocket = None
        
        # Latency tracking configuration
        self.show_latency = show_latency
        self.raw_latency = raw_latency
        
        # Initialize frequency monitor
        self.frequency_monitor = WebSocketClientFrequencyMonitor(window_size=50)
        
        # Clock synchronization warning flag
        self._clock_sync_warning_shown = False
        
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
        self.fps_display_interval = 5.0  # Show FPS every 5 seconds
        
        # Frequency reporting
        self.last_frequency_report = time.time()
        self.frequency_report_interval = 10.0  # Report every 10 seconds
    
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
        
    def format_joint_positions(self, joint_data: Dict[str, float], max_joints: int = 8) -> str:
        """Format joint positions for display."""
        if not joint_data:
            return "No joint data"
            
        # Sort joints by name for consistent display
        sorted_joints = sorted(joint_data.items())
        
        if len(sorted_joints) <= max_joints:
            # Show all joints if count is reasonable
            joint_strs = [f"{name}: {pos:.3f}" for name, pos in sorted_joints]
            return "\n      ".join(joint_strs)
        else:
            # Show first few joints and summary for large datasets
            shown_joints = sorted_joints[:max_joints]
            joint_strs = [f"{name}: {pos:.3f}" for name, pos in shown_joints]
            remaining = len(sorted_joints) - max_joints
            joint_strs.append(f"... and {remaining} more joints")
            return "\n      ".join(joint_strs)
    
    def format_organized_data(self, organized: Dict[str, Dict[str, float]]) -> str:
        """Format organized joint data for display."""
        if not organized:
            return "No organized data"
            
        lines = []
        for category, joints in organized.items():
            if joints:
                lines.append(f"     {category.replace('_', ' ').title()} ({len(joints)} joints):")
                if len(joints) <= 3:
                    for name, pos in sorted(joints.items()):
                        lines.append(f"       {name}: {pos:.3f}")
                else:
                    # Show first 2 joints for large categories
                    sorted_joints = sorted(joints.items())
                    for name, pos in sorted_joints[:2]:
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
        receive_time = latest_msg.get('receive_time', time.time())
        latency_ms = latest_msg.get('latency_ms')
        dt = datetime.fromtimestamp(timestamp)
        msg_type = latest_msg.get('type', 'unknown')
        
        lines = []
        lines.append(f"🤖 LATEST MESSAGE SUMMARY (from {len(messages)} recent messages)")
        lines.append(f"   Timestamp: {dt.strftime('%H:%M:%S.%f')[:-3]}")
        lines.append(f"   Type: {msg_type}")
        
        # Show latency information if available and enabled
        if self.show_latency:
            if latency_ms is not None:
                if latency_ms < 10:
                    status = "✅ Excellent"
                elif latency_ms < 50:
                    status = "✅ Good"
                elif latency_ms < 100:
                    status = "⚠️ Fair"
                else:
                    status = "❌ High"
                lines.append(f"   🕐 Latency: {latency_ms:.1f}ms ({status})")
            else:
                lines.append(f"   🕐 Latency: No timestamp available")
            
            # Show buffer latency statistics if we have multiple messages
            if len(messages) > 1:
                buffer_latencies = [msg.get('latency_ms') for msg in messages if msg.get('latency_ms') is not None]
                if buffer_latencies:
                    avg_latency = sum(buffer_latencies) / len(buffer_latencies)
                    min_latency = min(buffer_latencies)
                    max_latency = max(buffer_latencies)
                    lines.append(f"   📊 Buffer stats: {avg_latency:.1f}ms avg ({min_latency:.1f}-{max_latency:.1f}ms range)")
        
        # Show raw structure
        if self.show_raw_structure:
            lines.append(self.analyze_message_structure(latest_msg))
        
        # Handle joint positions specifically
        if msg_type == 'joint_positions':
            message_data = latest_msg.get('data', {})
            all_joints = message_data.get('all_joints', {})
            organized = message_data.get('organized', {})
            metadata = latest_msg.get('metadata', {})
            
            # Show metadata
            if self.show_metadata and metadata:
                lines.append(f"\n   📊 METADATA:")
                lines.append(f"     Source: {metadata.get('source', 'unknown')}")
                lines.append(f"     Total joints: {metadata.get('total_joint_count', 0)}")
                lines.append(f"     Update rate: {metadata.get('update_hz', 0):.1f} Hz")
                
                # Show joint breakdown
                upper_count = metadata.get('upper_body_joints', 0)
                left_hand_count = metadata.get('left_hand_joints', 0)
                right_hand_count = metadata.get('right_hand_joints', 0)
                if upper_count > 0 or left_hand_count > 0 or right_hand_count > 0:
                    lines.append(f"     Breakdown: {upper_count} upper body, {left_hand_count} left hand, {right_hand_count} right hand")
            
            # Show organized data (more readable)
            if self.show_organized_data and organized:
                lines.append(f"\n   📋 ORGANIZED JOINT DATA:")
                lines.append(self.format_organized_data(organized))
            
            # Show sample of all joints
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
    
    def update_statistics(self, message_type: str, message_timestamp: Optional[float] = None):
        """Update message statistics."""
        # Use frequency monitor for accurate tracking
        # Pass raw_latency setting to control validation
        if hasattr(self, 'raw_latency'):
            self.frequency_monitor.record_message(message_type, message_timestamp, self.raw_latency)
        else:
            self.frequency_monitor.record_message(message_type, message_timestamp)
        
        # Legacy statistics (kept for backward compatibility)
        self.total_messages_received += 1
        self.fps_message_count += 1
        
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
    
    async def handle_message(self, message_str: str):
        """Handle incoming WebSocket message with rate limiting."""
        receive_time = time.time()  # Capture receive time immediately
        
        try:
            # Parse JSON
            data = json.loads(message_str)
            message_type = data.get('type', 'unknown')
            
            # Extract message timestamp for latency calculation (if enabled)
            message_timestamp = None
            latency_ms = None
            
            if self.show_latency:
                message_timestamp = data.get('timestamp')
                
                # Calculate latency if timestamp is available
                if message_timestamp is not None:
                    latency_s = receive_time - message_timestamp
                    latency_ms = latency_s * 1000  # Convert to milliseconds
                    
                    # Validate latency - negative values indicate clock synchronization issues
                    # Skip validation if raw_latency mode is enabled (for debugging)
                    if not self.raw_latency:
                        if latency_ms < 0:
                            # Show educational warning about clock synchronization (once per session)
                            self._show_clock_sync_warning()
                            
                            # Log warning for significant negative latencies (more than -100ms suggests real clock skew)
                            if latency_ms < -100:
                                logger.warning(f"⚠️  Large negative latency detected: {latency_ms:.1f}ms - "
                                             f"sender clock may be ahead of receiver clock")
                            # Clamp negative latencies to 0 for display
                            latency_ms = 0.0
                        elif latency_ms > 10000:  # More than 10 seconds suggests timestamp error
                            logger.warning(f"⚠️  Extremely high latency detected: {latency_ms:.1f}ms - "
                                         f"possible timestamp format mismatch")
                            # Don't record obviously invalid latencies
                            latency_ms = None
            
            # Update statistics with timestamp for latency tracking
            self.update_statistics(message_type, message_timestamp if self.show_latency else None)
            
            # Add to buffer for rate-limited display with latency info
            data['receive_time'] = receive_time
            if self.show_latency:
                data['latency_ms'] = latency_ms
            self.buffered_messages.append(data)
            
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
                
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON DECODE ERROR: {e}")
            print(f"   Raw message: {message_str[:200]}...")
        except Exception as e:
            print(f"\n❌ MESSAGE HANDLING ERROR: {e}")
            print(f"   Raw message: {message_str[:200]}...")
    
    async def connect_and_listen(self):
        """Connect to WebSocket server and listen for messages."""
        try:
            print(f"🔌 Connecting to WebSocket server at {self.uri}")
            
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                print(f"✅ Connected successfully!")
                print(f"🎯 Listening for messages from MuJoCo hardware interface...")
                print(f"   Displaying updates every {self.print_interval:.1f} seconds")
                print(f"   Frequency reports every {self.frequency_report_interval:.1f} seconds")
                print(f"   Press Ctrl+C to stop")
                print(f"   While running, press 'f' + Enter to show frequency report")
                print()
                
                # Listen for messages
                async for message in websocket:
                    await self.handle_message(message)
                    
        except websockets.exceptions.ConnectionRefusedError:
            print(f"❌ Connection refused to {self.uri}")
            print(f"   Make sure the WebSocket relay server is running:")
            print(f"   python ref/websocket_server.py --host {self.host} --port {self.port}")
        except websockets.exceptions.WebSocketException as e:
            print(f"❌ WebSocket error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        finally:
            print(f"\n🔌 Disconnected from WebSocket server")
    
    async def send_test_message(self):
        """Send a test message to the server."""
        if not self.websocket:
            print("❌ Not connected to WebSocket server")
            return
            
        test_message = {
            "type": "test_client_message",
            "timestamp": time.time(),
            "source": "example_websocket_client",
            "message": "Hello from Python client!"
        }
        
        try:
            await self.websocket.send(json.dumps(test_message))
            print(f"📤 Sent test message to server")
        except Exception as e:
            print(f"❌ Error sending test message: {e}")
    
    def print_final_statistics(self):
        """Print final statistics."""
        total_time = time.time() - self.start_time
        avg_rate = self.total_messages_received / total_time if total_time > 0 else 0
        
        print(f"\n📊 FINAL STATISTICS")
        print(f"   Total runtime: {total_time:.1f} seconds")
        print(f"   Total messages: {self.total_messages_received}")
        print(f"   Average rate: {avg_rate:.1f} messages/sec")
        print(f"   Message types: {self.message_types_count}")
        
        # Print final frequency monitor report
        print(f"\n📊 FINAL FREQUENCY ANALYSIS:")
        self.frequency_monitor.print_frequency_summary()
        
        # Get detailed frequency stats
        stats = self.frequency_monitor.get_current_frequencies()
        if 'joint_position_hz' in stats:
            expected_hz = 150.0  # Expected from teleoperation system
            actual_hz = stats['joint_position_hz']
            efficiency = (actual_hz / expected_hz) * 100 if expected_hz > 0 else 0
            print(f"\n🎯 JOINT POSITION FREQUENCY ANALYSIS:")
            print(f"   Expected: {expected_hz:.1f}Hz")
            print(f"   Actual: {actual_hz:.1f}Hz")
            print(f"   Efficiency: {efficiency:.1f}%")
            if efficiency < 90:
                print(f"   ⚠️  Low efficiency detected - possible throttling or network issues")
            elif efficiency > 95:
                print(f"   ✅ Good efficiency - receiving data at expected rate")
        
        # Print detailed latency analysis
        if hasattr(self, 'show_latency') and self.show_latency and 'avg_latency_ms' in stats:
            print(f"\n🕐 FINAL LATENCY ANALYSIS:")
            print(f"   Overall Average: {stats['avg_latency_ms']:.1f}ms (±{stats['latency_jitter_ms']:.1f}ms)")
            print(f"   Overall Range: {stats['min_latency_ms']:.1f}ms - {stats['max_latency_ms']:.1f}ms")
            print(f"   Overall Median: {stats['median_latency_ms']:.1f}ms")
            
            # Lifetime statistics
            if 'lifetime_avg_latency_ms' in stats:
                print(f"   Lifetime Average: {stats['lifetime_avg_latency_ms']:.1f}ms")
                print(f"   Lifetime Range: {stats['lifetime_min_latency_ms']:.1f}ms - {stats['lifetime_max_latency_ms']:.1f}ms")
            
            # Joint position specific latency
            if 'joint_avg_latency_ms' in stats:
                print(f"\n   Joint Position Latency:")
                print(f"     Average: {stats['joint_avg_latency_ms']:.1f}ms (±{stats['joint_latency_jitter_ms']:.1f}ms)")
                print(f"     Range: {stats['joint_min_latency_ms']:.1f}ms - {stats['joint_max_latency_ms']:.1f}ms")
                
                # Latency quality assessment for joint data
                joint_latency = stats['joint_avg_latency_ms']
                if joint_latency < 10:
                    print(f"     ✅ Excellent joint latency for real-time control")
                elif joint_latency < 20:
                    print(f"     ✅ Good joint latency for teleoperation")
                elif joint_latency < 50:
                    print(f"     ⚠️  Fair joint latency - acceptable for most tasks")
                else:
                    print(f"     ❌ High joint latency - may affect teleoperation quality")
            
            # Ping specific latency
            if 'ping_avg_latency_ms' in stats:
                print(f"\n   Ping Latency:")
                print(f"     Average: {stats['ping_avg_latency_ms']:.1f}ms (±{stats['ping_latency_jitter_ms']:.1f}ms)")
                print(f"     Range: {stats['ping_min_latency_ms']:.1f}ms - {stats['ping_max_latency_ms']:.1f}ms")
            
            # Network quality assessment
            avg_latency = stats['avg_latency_ms']
            jitter = stats['latency_jitter_ms']
            print(f"\n   🌐 NETWORK QUALITY ASSESSMENT:")
            if avg_latency < 10 and jitter < 5:
                print(f"     ✅ Excellent network quality (Low latency, low jitter)")
            elif avg_latency < 50 and jitter < 20:
                print(f"     ✅ Good network quality")
            elif avg_latency < 100 and jitter < 50:
                print(f"     ⚠️  Fair network quality")
            else:
                print(f"     ❌ Poor network quality - high latency and/or jitter")
                print(f"       Consider optimizing network connection")
        elif hasattr(self, 'show_latency') and not self.show_latency:
            print(f"\n🕐 LATENCY ANALYSIS: Disabled (use --show-latency to enable)")
        else:
            print(f"\n🕐 LATENCY ANALYSIS: No timestamp data available")
    
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
                elif user_input == 'q':
                    print("👋 Quit command received")
                    return False
                elif user_input:
                    print(f"❓ Unknown command: '{user_input}' (try 'f' for frequency, 'q' to quit)")
        except Exception:
            pass  # Ignore input errors
        return True

    def _show_clock_sync_warning(self):
        """Show a one-time warning about clock synchronization issues."""
        if not self._clock_sync_warning_shown:
            print(f"\n🕐 CLOCK SYNCHRONIZATION INFO:")
            print(f"   Negative latency values indicate that the sender's system clock")
            print(f"   is ahead of the receiver's system clock. This is normal in distributed")
            print(f"   systems where machines have slightly different time synchronization.")
            print(f"   ")
            print(f"   Solutions:")
            print(f"   - Use NTP (Network Time Protocol) to synchronize clocks")
            print(f"   - Consider using high_resolution_clock() for sender timestamps")
            print(f"   - For exact latency measurement, use synchronized hardware clocks")
            print(f"   ")
            print(f"   Note: Negative latencies are clamped to 0ms for statistics.")
            print(f"   This warning will only be shown once per session.")
            print()
            self._clock_sync_warning_shown = True


async def main_client_loop(host: str, port: int, test_mode: bool = False, frequency_report_interval: float = 10.0, frequency_window_size: int = 50, show_latency: bool = True, raw_latency: bool = False):
    """Main client loop."""
    client = MuJoCoWebSocketClient(host, port, show_latency, raw_latency)
    
    # Configure frequency monitoring
    client.frequency_monitor = WebSocketClientFrequencyMonitor(window_size=frequency_window_size)
    client.frequency_report_interval = frequency_report_interval
    
    # Configure display options
    if test_mode:
        print("🧪 Test mode: Limited output with faster updates")
        client.print_interval = 0.5  # Faster updates in test mode
        client.show_full_joint_data = False  # Reduce output in test mode
        client.fps_display_interval = 2.0  # More frequent stats in test mode
        client.frequency_report_interval = 5.0  # More frequent frequency reports in test mode
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
    
    try:
        await client.connect_and_listen()
        
    except KeyboardInterrupt:
        print(f"\n🛑 Received keyboard interrupt")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
    finally:
        client.print_final_statistics()


def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    print(f"\n🛑 Received signal {signum}. Shutting down...")
    sys.exit(0)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="WebSocket Client for MuJoCo Hardware Interface")
    parser.add_argument("--host", type=str, default="localhost",
                       help="WebSocket server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8765,
                       help="WebSocket server port (default: 8765)")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with reduced output and faster updates")
    parser.add_argument("--frequency_report_interval", type=float, default=10.0,
                       help="Frequency report interval in seconds (default: 10.0)")
    parser.add_argument("--frequency_window_size", type=int, default=50,
                       help="Frequency monitor window size for averaging (default: 50)")
    parser.add_argument("--no_frequency_reports", action="store_true",
                       help="Disable automatic frequency reports")
    parser.add_argument("--show-latency", action="store_true", default=True,
                       help="Show latency information (default: enabled)")
    parser.add_argument("--no-latency", dest="show_latency", action="store_false",
                       help="Disable latency tracking and display")
    parser.add_argument("--raw-latency", action="store_true", default=False,
                       help="Show raw latency values without validation (for debugging clock sync issues)")
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🤖 MuJoCo WebSocket Client")
    print("=" * 50)
    print(f"Connecting to: ws://{args.host}:{args.port}")
    print(f"Test mode: {args.test}")
    if args.test:
        print("Display updates: Every 0.5 seconds")
    else:
        print("Display updates: Every 1.0 seconds")
    
    # Latency tracking configuration
    print(f"Latency tracking: {'Enabled' if args.show_latency else 'Disabled'}")
    if args.show_latency and args.raw_latency:
        print(f"Raw latency mode: Enabled (negative values will be shown for debugging)")
    elif args.show_latency:
        print(f"Latency validation: Enabled (negative values clamped to 0ms)")
    
    # Frequency monitoring configuration
    if args.no_frequency_reports:
        print("Frequency reports: Disabled")
        frequency_interval = float('inf')  # Effectively disable by setting very large interval
    else:
        print(f"Frequency reports: Every {args.frequency_report_interval:.1f}s")
        frequency_interval = args.frequency_report_interval
    
    print(f"Frequency window: {args.frequency_window_size} messages")
    print("=" * 50)
    
    # Run the client
    try:
        asyncio.run(main_client_loop(args.host, args.port, args.test, frequency_interval, args.frequency_window_size, args.show_latency, args.raw_latency))
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 