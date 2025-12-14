"""
Async version of AMUZA Bluetooth communication with command queue and priority handling.

Key improvements:
- Priority queue for commands (STOP has highest priority)
- Async send with proper locking
- Better error handling and retries
- Proper resource cleanup
- Mock mode support
"""

import asyncio
import time
import logging
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Callable
import threading

# Try to import bluetooth, fall back to mock if unavailable
try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False

from config import HARDWARE
from app_state import AppState


logger = logging.getLogger(__name__)


class CommandPriority(IntEnum):
    """Priority levels for commands (lower number = higher priority)"""
    EMERGENCY_STOP = 0
    STOP = 1
    STATUS_QUERY = 5
    MOVEMENT = 10
    OTHER = 20


@dataclass
class BluetoothCommand:
    """Represents a command to send over Bluetooth"""
    command: str
    priority: CommandPriority
    response_callback: Optional[Callable[[str], None]] = None
    timeout: float = 5.0
    retry_count: int = 3
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority < other.priority


class MockBluetoothSocket:
    """Mock Bluetooth socket for testing without hardware"""
    
    def __init__(self):
        self.connected = False
        self.in_progress = False
        self.position = "Home"
        logger.info("MockBluetoothSocket initialized")
    
    def connect(self, address):
        logger.info(f"Mock connect to {address}")
        self.connected = True
        time.sleep(0.1)  # Simulate connection delay
    
    def send(self, data):
        msg = data.decode('utf-8') if isinstance(data, bytes) else data
        logger.info(f"Mock send: {msg}")
        
        # Simulate command processing
        if msg.startswith('@T'):
            self.in_progress = False
            logger.info("Mock: Stopped movement")
        elif msg.startswith('@P'):
            self.in_progress = True
            logger.info(f"Mock: Moving to position {msg[2:]}")
            self.position = msg[2:]
        elif msg.startswith('@Q'):
            # Query will be handled by recv
            pass
        
        return len(data)
    
    def recv(self, buffer_size):
        """Mock receive - simulate status responses"""
        if not self.in_progress:
            response = f"@R{self.position}\n"
        else:
            # Simulate movement completion after delay
            time.sleep(0.05)
            self.in_progress = False
            response = f"@R{self.position}\n"
        
        return response.encode('utf-8')
    
    def close(self):
        logger.info("Mock socket closed")
        self.connected = False


class Method:
    """Represents a single movement method (same as original)"""
    
    def __init__(self, pos, wait=60, eject=False, insert=False):
        self.pos = pos
        self.wait = wait
        self.eject = eject
        self.insert = insert
    
    def __str__(self):
        eject_str = " with eject" if self.eject else ""
        insert_str = " with insert" if self.insert else ""
        return f"Move to {self.pos}, wait {self.wait}s{eject_str}{insert_str}"


class Sequence:
    """Represents a sequence of methods (same as original)"""
    
    def __init__(self, name="Unnamed Sequence"):
        self.name = name
        self.methods = []
    
    def add_method(self, method: Method):
        self.methods.append(method)
    
    def get_method(self, index: int) -> Optional[Method]:
        if 0 <= index < len(self.methods):
            return self.methods[index]
        return None
    
    def __len__(self):
        return len(self.methods)
    
    def __str__(self):
        return f"Sequence '{self.name}' with {len(self.methods)} methods"


class AsyncAmuzaConnection:
    """
    Async version of AmuzaConnection with command queue and priority handling.
    
    Key improvements:
    - Commands are queued with priority
    - STOP commands jump to front of queue
    - Thread-safe command sending with asyncio.Lock
    - Better error handling and retries
    - Proper async cleanup
    - Response parsing for status updates
    """
    
    # State list from original AMUZA_Master.py
    STATE_LIST = ["Resting", "Ejected Tray", "Unknown", "Unknown", "Moving Tray",
                  "Unknown", "Unknown", "Unknown", "Unknown", "Moving",
                  "Unknown", "Unknown", "Unknown", "Unknown", "Unknown",
                  "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"]
    
    def __init__(self, device_address: str, use_mock: bool = False):
        self.device_address = device_address
        self.use_mock = use_mock
        
        # Socket and connection state
        self.socket: Optional[any] = None
        self.is_connected = False
        
        # Command queue (priority queue)
        self.command_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.send_lock = asyncio.Lock()
        
        # Status tracking
        self.current_position = "Unknown"
        self.current_state = 0
        self.current_method = 0
        self.time_remaining = 0
        self.isInProgress = False
        self.last_response = ""
        
        # Background tasks
        self.command_sender_task: Optional[asyncio.Task] = None
        self.query_task: Optional[asyncio.Task] = None
        self.receiver_task: Optional[asyncio.Task] = None
        
        # Stop event for graceful shutdown
        self.stop_event = asyncio.Event()
        
        logger.info(f"AsyncAmuzaConnection initialized (mock={use_mock})")
    
    async def connect(self) -> bool:
        """
        Connect to the AMUZA device via Bluetooth.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            # Run blocking Bluetooth connection in thread pool
            connected = await asyncio.to_thread(self._connect_blocking)
            
            if connected:
                self.is_connected = True
                
                # Start background tasks
                self.command_sender_task = asyncio.create_task(
                    self._command_sender_loop(),
                    name="command_sender"
                )
                self.query_task = asyncio.create_task(
                    self._query_status_loop(),
                    name="status_query"
                )
                self.receiver_task = asyncio.create_task(
                    self._response_receiver_loop(),
                    name="response_receiver"
                )
                
                logger.info("Connected to AMUZA device")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def _connect_blocking(self) -> bool:
        """Blocking connection (run in thread pool)"""
        try:
            if self.use_mock or not BLUETOOTH_AVAILABLE:
                self.socket = MockBluetoothSocket()
                self.socket.connect(self.device_address)
                return True
            else:
                self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                self.socket.connect((self.device_address, 1))
                return True
        except Exception as e:
            logger.error(f"Blocking connect failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from device and cleanup resources"""
        logger.info("Disconnecting from AMUZA device")
        
        # Signal stop to background tasks
        self.stop_event.set()
        
        # Wait for tasks to complete (with timeout)
        tasks = [t for t in [self.command_sender_task, self.query_task, self.receiver_task] if t]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=HARDWARE.CONNECTION_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                logger.warning("Background tasks didn't complete in time, cancelling")
                for task in tasks:
                    task.cancel()
        
        # Close socket
        if self.socket:
            try:
                await asyncio.to_thread(self.socket.close)
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
        
        self.is_connected = False
        logger.info("Disconnected from AMUZA device")
    
    async def send_command(
        self,
        command: str,
        priority: CommandPriority = CommandPriority.OTHER,
        wait_for_completion: bool = False,
        timeout: float = 5.0
    ) -> bool:
        """
        Queue a command to be sent to the device.
        
        Args:
            command: Command string to send
            priority: Priority level for the command
            wait_for_completion: If True, wait until command is sent
            timeout: Timeout for waiting
        
        Returns:
            True if command was queued (or sent if wait_for_completion)
        """
        if not self.is_connected:
            logger.warning(f"Cannot send command, not connected: {command}")
            return False
        
        # Create command object
        cmd = BluetoothCommand(
            command=command,
            priority=priority,
            timeout=timeout
        )
        
        # Add to queue
        await self.command_queue.put(cmd)
        logger.debug(f"Queued command with priority {priority}: {command}")
        
        # Optionally wait for it to be sent
        if wait_for_completion:
            # TODO: Implement completion tracking with futures
            await asyncio.sleep(0.1)
        
        return True
    
    async def _command_sender_loop(self):
        """Background task that sends queued commands"""
        logger.info("Command sender loop started")
        
        while not self.stop_event.is_set():
            try:
                # Get next command (with timeout so we can check stop_event)
                try:
                    cmd = await asyncio.wait_for(
                        self.command_queue.get(),
                        timeout=0.5
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Send the command
                success = await self._send_command_with_retry(cmd)
                
                if not success:
                    logger.error(f"Failed to send command after retries: {cmd.command}")
                
                # Mark task as done
                self.command_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info("Command sender cancelled")
                break
            except Exception as e:
                logger.error(f"Error in command sender loop: {e}")
                await asyncio.sleep(0.5)
        
        logger.info("Command sender loop stopped")
    
    async def _send_command_with_retry(self, cmd: BluetoothCommand) -> bool:
        """
        Send a command with retry logic.
        
        Args:
            cmd: Command to send
        
        Returns:
            True if sent successfully
        """
        for attempt in range(cmd.retry_count):
            try:
                async with self.send_lock:
                    # Run blocking send in thread pool
                    await asyncio.wait_for(
                        asyncio.to_thread(self._send_blocking, cmd.command),
                        timeout=cmd.timeout
                    )
                
                logger.debug(f"Sent command: {cmd.command}")
                return True
                
            except asyncio.TimeoutError:
                logger.warning(f"Command send timeout (attempt {attempt + 1}/{cmd.retry_count}): {cmd.command}")
            except Exception as e:
                logger.error(f"Error sending command (attempt {attempt + 1}/{cmd.retry_count}): {e}")
            
            if attempt < cmd.retry_count - 1:
                await asyncio.sleep(0.5)
        
        return False
    
    def _send_blocking(self, command: str):
        """Blocking send (run in thread pool)"""
        if not self.socket:
            raise RuntimeError("Socket not connected")
        
        data = command.encode('utf-8') if isinstance(command, str) else command
        self.socket.send(data)
    
    def _recv_blocking(self) -> str:
        """Blocking receive (run in thread pool)"""
        if not self.socket:
            return ""
        try:
            data = self.socket.recv(1024)
            if data:
                return data.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.debug(f"Receive error: {e}")
        return ""
    
    async def _response_receiver_loop(self):
        """Background task that receives and parses responses from device"""
        logger.info("Response receiver loop started")
        current_cmd = ""
        
        while not self.stop_event.is_set():
            try:
                # Run blocking receive in thread pool with timeout
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._recv_blocking),
                    timeout=2.0
                )
                
                if response:
                    current_cmd += response
                    # Process complete messages (ending with newline)
                    if current_cmd.endswith("\n"):
                        self._handle_response(current_cmd.strip())
                        current_cmd = ""
                
            except asyncio.TimeoutError:
                # No data received, continue
                continue
            except asyncio.CancelledError:
                logger.info("Response receiver cancelled")
                break
            except Exception as e:
                logger.error(f"Error in response receiver: {e}")
                await asyncio.sleep(0.5)
        
        logger.info("Response receiver loop stopped")
    
    def _handle_response(self, cmd: str):
        """
        Parse and handle a response from the device.
        
        Response formats (from original AMUZA_Master.py):
        - @E,X - Exit with exit code X
        - @q,state,method,well,time - Status query response
        - @R... - Position response
        """
        logger.debug(f"Handling response: {cmd}")
        self.last_response = cmd
        
        if not cmd:
            return
        
        try:
            if cmd.startswith("@E"):
                # Exit response
                exit_code = cmd[3] if len(cmd) > 3 else "?"
                logger.info(f"Device exit code: {exit_code}")
                
            elif cmd.startswith("@q"):
                # Status query response: @q,state,method,well,time
                data = cmd[3:].split(',')
                if len(data) >= 2:
                    self.current_state = int(data[0])
                    
                    if data[1] == '0':
                        self.isInProgress = False
                    else:
                        self.isInProgress = True
                        self.current_method = int(data[1])
                        
                        if len(data) >= 3:
                            self.current_position = data[2].strip('0') or "Home"
                        if len(data) >= 4:
                            self.time_remaining = int(data[3].strip('0') or 0)
                            
                        logger.debug(f"Status: method={self.current_method}, "
                                   f"pos={self.current_position}, time={self.time_remaining}s")
                
            elif cmd.startswith("@R"):
                # Position response
                self.current_position = cmd[2:].strip()
                logger.debug(f"Position updated: {self.current_position}")
                
            else:
                logger.debug(f"Unknown response: {cmd}")
                
        except Exception as e:
            logger.error(f"Error parsing response '{cmd}': {e}")
    
    def get_state_description(self) -> str:
        """Get human-readable description of current state"""
        if 0 <= self.current_state < len(self.STATE_LIST):
            return self.STATE_LIST[self.current_state]
        return "Unknown"
    
    async def _query_status_loop(self):
        """Background task that periodically queries device status"""
        logger.info("Status query loop started")
        
        while not self.stop_event.is_set():
            try:
                # Send status query
                await self.send_command("@Q\n", priority=CommandPriority.STATUS_QUERY)
                
                # Wait before next query
                await asyncio.sleep(HARDWARE.QUERY_INTERVAL_S)
                
            except asyncio.CancelledError:
                logger.info("Query loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in query loop: {e}")
                await asyncio.sleep(1.0)
        
        logger.info("Status query loop stopped")
    
    async def move_to_position(self, position: str) -> bool:
        """
        Move to a specific position.
        
        Args:
            position: Position identifier (e.g., "A1", "B2")
        
        Returns:
            True if command was queued successfully
        """
        command = f"@P{position}"
        self.isInProgress = True
        return await self.send_command(command, priority=CommandPriority.MOVEMENT)
    
    async def stop_movement(self) -> bool:
        """
        Stop any ongoing movement (highest priority).
        
        Returns:
            True if stop command was queued
        """
        self.isInProgress = False
        return await self.send_command("@T", priority=CommandPriority.STOP)
    
    async def eject(self) -> bool:
        """Eject sample"""
        return await self.send_command("@Y", priority=CommandPriority.MOVEMENT)
    
    async def insert(self) -> bool:
        """Insert sample"""
        return await self.send_command("@Z", priority=CommandPriority.MOVEMENT)
    
    async def execute_method(self, method: Method, stop_event: asyncio.Event) -> bool:
        """
        Execute a single method with interruptible wait.
        
        Args:
            method: Method to execute
            stop_event: Event to check for stop request
        
        Returns:
            True if completed, False if stopped
        """
        # Eject if needed
        if method.eject:
            await self.eject()
            await asyncio.sleep(2)  # Wait for eject to complete
        
        # Move to position
        await self.move_to_position(method.pos)
        
        # Wait for movement to complete
        await asyncio.sleep(2)  # Wait for movement to start
        
        # Insert if needed
        if method.insert:
            await self.insert()
            await asyncio.sleep(2)  # Wait for insert to complete
        
        # Interruptible wait
        for _ in range(int(method.wait / 0.5)):
            if stop_event.is_set():
                logger.info("Method execution stopped by user")
                await self.stop_movement()
                return False
            await asyncio.sleep(0.5)
        
        return True
    
    async def execute_sequence(self, sequence: Sequence, stop_event: asyncio.Event) -> bool:
        """
        Execute a sequence of methods with stop support.
        
        Args:
            sequence: Sequence to execute
            stop_event: Event to check for stop request
        
        Returns:
            True if completed, False if stopped
        """
        logger.info(f"Starting sequence: {sequence}")
        
        for i, method in enumerate(sequence.methods):
            if stop_event.is_set():
                logger.info(f"Sequence stopped at step {i + 1}/{len(sequence)}")
                return False
            
            logger.info(f"Executing step {i + 1}/{len(sequence)}: {method}")
            
            completed = await self.execute_method(method, stop_event)
            if not completed:
                return False
        
        logger.info("Sequence completed successfully")
        return True
    
    def checkProgress(self) -> bool:
        """Check if a movement is in progress"""
        return self.isInProgress
    
    def getCurrentPosition(self) -> str:
        """Get current position"""
        return self.current_position


# Example usage
async def main():
    """Example usage of AsyncAmuzaConnection"""
    
    # Create connection in mock mode
    connection = AsyncAmuzaConnection(
        device_address=HARDWARE.BLUETOOTH_DEVICE_ADDRESS,
        use_mock=True
    )
    
    # Connect
    if await connection.connect():
        print("Connected!")
        
        # Create a simple sequence
        sequence = Sequence("Test Sequence")
        sequence.add_method(Method("A1", wait=5))
        sequence.add_method(Method("B2", wait=5))
        sequence.add_method(Method("C3", wait=5))
        
        # Create stop event
        stop_event = asyncio.Event()
        
        # Execute sequence
        completed = await connection.execute_sequence(sequence, stop_event)
        print(f"Sequence completed: {completed}")
        
        # Disconnect
        await connection.disconnect()
    else:
        print("Connection failed")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
