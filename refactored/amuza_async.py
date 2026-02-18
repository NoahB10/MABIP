"""
Async version of AMUZA Bluetooth communication with command queue and priority handling.

Key improvements over original:
- Priority queue for commands (STOP has highest priority)
- Async send with proper locking
- Proper protocol timing (0.1s units for time parameter)
- Full status response parsing (7 fields)
- Completion detection by polling status
- Temperature and heater control
- Connection health monitoring
- Better error handling and retries
- Proper resource cleanup
- Mock mode support
"""

import asyncio
import time
import logging
import math
import json
from pathlib import Path
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict
from datetime import datetime

# Try to import bluetooth, fall back to mock if unavailable
try:
    import bluetooth
    BLUETOOTH_AVAILABLE = True
except ImportError:
    BLUETOOTH_AVAILABLE = False

from config import HARDWARE, FILES
from app_state import AppState


logger = logging.getLogger(__name__)


class DeviceState(IntEnum):
    """AMUZA device state codes from protocol analysis"""
    INITIAL = 0      # Just connected, not ready
    READY = 1        # Idle, ready for commands
    TRANSITIONAL = 2 # Mode changing
    MOVING = 5       # Physical motion (homing, eject, insert)
    RUNNING = 10     # Executing program/sequence


class CommandTiming:
    """Timing constants for each command type (in seconds)"""
    EJECT = 5.0
    INSERT = 7.5
    STOP = 0.5
    QUERY = 0.1
    MOVE_MIN = 7.0   # Move to A1 (closest)
    MOVE_MAX = 12.0  # Move to H12 (farthest)
    COMMAND_RESPONSE = 0.15  # Typical response time


class CommandPriority(IntEnum):
    """Priority levels for commands (lower number = higher priority)"""
    EMERGENCY_STOP = 0
    STOP = 1
    STATUS_QUERY = 5
    MOVEMENT = 10
    TEMPERATURE = 15
    OTHER = 20


@dataclass
class BluetoothCommand:
    """Represents a command to send over Bluetooth"""
    command: str
    priority: CommandPriority
    expected_duration: float = 0.0
    response_callback: Optional[Callable[[str], None]] = None
    timeout: float = 5.0
    retry_count: int = 3

    def __lt__(self, other):
        return self.priority < other.priority

    def __hash__(self):
        return hash(self.command)

    def __eq__(self, other):
        if isinstance(other, BluetoothCommand):
            return self.command == other.command
        return False


@dataclass
class DeviceStatus:
    """Parsed status from @q response"""
    state: int = 0
    is_moving: bool = False
    current_well: int = 0
    countdown: int = 0
    temperature1: float = 0.0  # In Celsius
    temperature2: float = 0.0  # In Celsius
    heater_on: bool = False
    raw_response: str = ""


class MockBluetoothSocket:
    """Mock Bluetooth socket for testing without hardware"""

    def __init__(self):
        self.connected = False
        self.state = DeviceState.INITIAL
        self.current_well = 0
        self.countdown = 0
        self.temperature1 = 25.0
        self.temperature2 = 37.0
        self.heater_on = False
        self.is_moving = False
        self._move_end_time = 0
        self._response_queue = []
        logger.info("MockBluetoothSocket initialized")

    def connect(self, address):
        logger.info(f"Mock connect to {address}")
        self.connected = True
        self.state = DeviceState.INITIAL
        time.sleep(0.1)

    def send(self, data):
        msg = data.decode('utf-8') if isinstance(data, bytes) else data
        msg = msg.strip()
        logger.debug(f"Mock send: {repr(msg)}")

        # Generate appropriate response
        if msg == '@?':
            self._response_queue.append('@?,20160407\n')
        elif msg == '@Q':
            # Generate status response
            state = self.state
            moving = 1 if self.is_moving else 0
            well = self.current_well
            countdown = max(0, int(self._move_end_time - time.time())) if self.is_moving else 0
            temp1 = int(self.temperature1 * 10)
            temp2 = int(self.temperature2 * 10)
            heater = 1 if self.heater_on else 0
            self._response_queue.append(f'@q,{state:02d},{moving},{well:03d},{countdown:04d},{temp1},{temp2},{heater}\n')
        elif msg == '@Z':
            self._response_queue.append('@E,0\n')
            self.state = DeviceState.MOVING
            self._move_end_time = time.time() + 5
            self.is_moving = True
            # Schedule state change
            asyncio.get_event_loop().call_later(5, self._complete_insert)
        elif msg == '@Y':
            self._response_queue.append('@E,0\n')
            self.state = DeviceState.MOVING
            self._move_end_time = time.time() + 3
            self.is_moving = True
            asyncio.get_event_loop().call_later(3, self._complete_eject)
        elif msg == '@T':
            self._response_queue.append('@E,0\n')
            self.is_moving = False
            self.state = DeviceState.READY
        elif msg.startswith('@V,'):
            # Temperature command
            self._response_queue.append('@E,0\n')
            try:
                temp = int(msg[3:])
                self.temperature2 = temp / 10.0
            except:
                pass
        elif msg.startswith('@K,'):
            # Heater command
            self._response_queue.append('@E,0\n')
            self.heater_on = msg[3] == '1'
        elif msg.startswith('@P,'):
            # Move command - parse and simulate
            self._response_queue.append('@E,1\n')  # Program started
            self.state = DeviceState.RUNNING
            self.is_moving = True
            # Parse time from command (format: @P,M1,TTTT,PP,)
            try:
                parts = msg.split(',')
                if len(parts) >= 4:
                    time_param = int(parts[2])  # Time in 0.1s units
                    self.current_well = int(parts[3])
                    duration = time_param / 10.0 + 2  # Add movement time
                    self._move_end_time = time.time() + duration
                    asyncio.get_event_loop().call_later(duration, self._complete_move)
            except Exception as e:
                logger.error(f"Mock parse error: {e}")
                self._move_end_time = time.time() + 10
                asyncio.get_event_loop().call_later(10, self._complete_move)

        return len(data)

    def _complete_insert(self):
        self.is_moving = False
        self.state = DeviceState.READY
        logger.debug("Mock: Insert complete")

    def _complete_eject(self):
        self.is_moving = False
        self.state = DeviceState.READY
        logger.debug("Mock: Eject complete")

    def _complete_move(self):
        self.is_moving = False
        self.state = DeviceState.READY
        logger.debug(f"Mock: Move complete at well {self.current_well}")

    def recv(self, buffer_size):
        if self._response_queue:
            response = self._response_queue.pop(0)
            return response.encode('utf-8')
        return b''

    def close(self):
        logger.info("Mock socket closed")
        self.connected = False

    def setblocking(self, blocking):
        pass


class Method:
    """Represents a single movement method"""

    def __init__(self, pos, wait=60, buffer_time=0, eject=False, insert=False):
        self.pos = pos
        self.wait = wait  # Sampling time (time at well) in seconds
        self.buffer_time = buffer_time  # Buffer time before move in seconds
        self.eject = eject
        self.insert = insert

    def __str__(self):
        return f"Move to {self.pos}, buffer {self.buffer_time}s, sample {self.wait}s"


class Sequence:
    """Represents a sequence of methods"""

    def __init__(self, name="Unnamed Sequence"):
        self.name = name
        self.methods: List[Method] = []

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
    Async AMUZA Bluetooth connection with proper protocol handling.

    Protocol details (from btsnoop analysis):
    - Commands: @Q (query), @Z (insert), @Y (eject), @T (stop), @P (move), @V (temp), @K (heater)
    - Status response: @q,state,moving,well,countdown,temp1,temp2,heater
    - Acknowledgment: @E,0 (success), @E,1 (program started)
    - Time parameter in @P command is in 0.1 second units
    - Single \\n terminator for most commands
    """

    STATE_NAMES = {
        DeviceState.INITIAL: "Initial",
        DeviceState.READY: "Ready",
        DeviceState.TRANSITIONAL: "Transitional",
        DeviceState.MOVING: "Moving",
        DeviceState.RUNNING: "Running Program"
    }

    def __init__(self, device_address: str = None, use_mock: bool = False):
        self.device_address = device_address or HARDWARE.BLUETOOTH_DEVICE_ADDRESS
        self.use_mock = use_mock

        # Socket and connection state
        self.socket: Optional[any] = None
        self.is_connected = False

        # Command queue
        self.command_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.send_lock = asyncio.Lock()
        self.pending_commands: set = set()

        # Status tracking
        self.status = DeviceStatus()
        self.last_response = ""
        self.last_ack_code = ""

        # Completion tracking
        self._completion_event = asyncio.Event()
        self._waiting_for_completion = False

        # Connection health
        self.consecutive_failures = 0
        self.max_failures = 3
        self.connection_healthy = True

        # Timeout callback for GUI notification
        self._timeout_callback: Optional[callable] = None

        # Background tasks
        self.command_sender_task: Optional[asyncio.Task] = None
        self.query_task: Optional[asyncio.Task] = None
        self.receiver_task: Optional[asyncio.Task] = None

        # Stop event for graceful shutdown
        self.stop_event = asyncio.Event()

        # Logging
        self._init_logging()

        logger.info(f"AsyncAmuzaConnection initialized (mock={use_mock}, address={self.device_address})")

    def set_timeout_callback(self, callback: callable):
        """Set callback to be called when commands timeout after all retries.
        Callback signature: callback(command: str, attempts: int)
        """
        self._timeout_callback = callback

    def _notify_timeout(self, command: str, attempts: int):
        """Notify GUI of timeout error"""
        if self._timeout_callback:
            try:
                self._timeout_callback(command, attempts)
            except Exception as e:
                logger.error(f"Timeout callback error: {e}")

    def _init_logging(self):
        """Initialize AMUZA command logging"""
        try:
            log_dir = Path(FILES.AMUZA_LOGS_FOLDER)
            log_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime(FILES.LOG_TIMESTAMP_FORMAT)
            self.log_file = log_dir / f"AMUZA-{timestamp}.log"
            logger.info(f"AMUZA commands will be logged to {self.log_file}")
        except Exception as e:
            logger.warning(f"Could not create log directory: {e}")
            self.log_file = None

    def _log_command(self, direction: str, message: str):
        """Log a command or response to file"""
        if self.log_file:
            try:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                with open(self.log_file, 'a') as f:
                    f.write(f"{timestamp} [{direction}] {message}\n")
            except Exception as e:
                logger.debug(f"Could not write to log file: {e}")

    @staticmethod
    def calculate_move_time(well_id: str) -> float:
        """Calculate move time based on well position using Euclidean distance"""
        try:
            row_letter = well_id[0].upper()
            col_number = int(well_id[1:])
            row = ord(row_letter) - ord('A')
            col = col_number - 1
            distance = math.sqrt(row**2 + col**2)
            max_distance = math.sqrt(7**2 + 11**2)
            if distance == 0:
                return CommandTiming.MOVE_MIN
            time_range = CommandTiming.MOVE_MAX - CommandTiming.MOVE_MIN
            return CommandTiming.MOVE_MIN + (distance / max_distance) * time_range
        except Exception as e:
            logger.warning(f"Could not calculate move time for {well_id}: {e}")
            return CommandTiming.MOVE_MAX

    async def connect(self) -> bool:
        """Connect to the AMUZA device via Bluetooth"""
        try:
            connected = await asyncio.to_thread(self._connect_blocking)

            if connected:
                self.is_connected = True
                self.connection_healthy = True
                self.consecutive_failures = 0

                # Start background tasks
                self.command_sender_task = asyncio.create_task(
                    self._command_sender_loop(), name="command_sender"
                )
                self.query_task = asyncio.create_task(
                    self._query_status_loop(), name="status_query"
                )
                self.receiver_task = asyncio.create_task(
                    self._response_receiver_loop(), name="response_receiver"
                )

                logger.info("Connected to AMUZA device")
                self._log_command("SYSTEM", "Connected")
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

            # Scan for devices
            logger.info("Scanning for Bluetooth devices...")
            nearby_devices = bluetooth.discover_devices(
                lookup_names=True, lookup_class=True, duration=8
            )
            logger.info(f"Found {len(nearby_devices)} devices")

            # Look for AMUZA device
            device_address = None
            device_name = HARDWARE.BT_DEVICE_NAME

            for addr, name, device_class in nearby_devices:
                logger.info(f"Found device: {name} at {addr}")
                if name == device_name:
                    device_address = addr
                    logger.info(f"Found AMUZA device at {addr}")
                    break

            if device_address is None:
                logger.error(f"AMUZA device '{device_name}' not found")
                return False

            # Connect
            logger.info(f"Connecting to AMUZA at {device_address}...")
            self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            self.socket.connect((device_address, 1))
            self.socket.setblocking(False)

            # Initial handshake
            self.socket.send(b"@?\n")
            time.sleep(0.2)
            self.socket.send(b"@Q\n")
            time.sleep(0.2)

            logger.info("Successfully connected to AMUZA device")
            return True

        except Exception as e:
            logger.error(f"Blocking connect failed: {e}")
            return False

    async def disconnect(self):
        """Disconnect from device and cleanup resources"""
        logger.info("Disconnecting from AMUZA device")
        self._log_command("SYSTEM", "Disconnecting")

        self.stop_event.set()

        tasks = [t for t in [self.command_sender_task, self.query_task, self.receiver_task] if t]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Background tasks didn't complete, cancelling")
                for task in tasks:
                    task.cancel()

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
        expected_duration: float = 0.0,
        wait_for_completion: bool = False,
        timeout: float = 5.0
    ) -> bool:
        """Queue a command to be sent to the device"""
        if not self.is_connected:
            logger.warning(f"Cannot send command, not connected: {command}")
            return False

        cmd = BluetoothCommand(
            command=command,
            priority=priority,
            expected_duration=expected_duration,
            timeout=timeout
        )

        # Deduplication (except STOP)
        if priority not in (CommandPriority.STOP, CommandPriority.EMERGENCY_STOP):
            if cmd in self.pending_commands:
                logger.debug(f"Skipping duplicate command: {command}")
                return True

        self.pending_commands.add(cmd)
        await self.command_queue.put(cmd)
        logger.debug(f"Queued command: {command.strip()}")

        return True

    async def _command_sender_loop(self):
        """Background task that sends queued commands"""
        logger.info("Command sender loop started")

        while not self.stop_event.is_set():
            try:
                try:
                    cmd = await asyncio.wait_for(self.command_queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                success = await self._send_command_with_retry(cmd)

                if success:
                    if cmd.expected_duration > 0:
                        logger.debug(f"Waiting {cmd.expected_duration}s for: {cmd.command.strip()}")
                        await asyncio.sleep(cmd.expected_duration)
                else:
                    logger.error(f"Failed to send: {cmd.command}")
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_failures:
                        self.connection_healthy = False
                        logger.error("Connection health degraded")

                self.pending_commands.discard(cmd)
                self.command_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in command sender: {e}")
                await asyncio.sleep(0.5)

        logger.info("Command sender loop stopped")

    async def _send_command_with_retry(self, cmd: BluetoothCommand) -> bool:
        """Send a command with retry logic"""
        for attempt in range(cmd.retry_count):
            try:
                async with self.send_lock:
                    await asyncio.wait_for(
                        asyncio.to_thread(self._send_blocking, cmd.command),
                        timeout=cmd.timeout
                    )

                self._log_command("TX", cmd.command.strip())
                self.consecutive_failures = 0
                return True

            except asyncio.TimeoutError:
                logger.warning(f"Timeout (attempt {attempt + 1}/{cmd.retry_count}): {cmd.command.strip()}")
                self.consecutive_failures += 1
            except Exception as e:
                logger.error(f"Error (attempt {attempt + 1}/{cmd.retry_count}): {e}")
                self.consecutive_failures += 1

            if attempt < cmd.retry_count - 1:
                await asyncio.sleep(0.5)

        # All retries failed - notify GUI
        self._log_command("ERROR", f"Command failed after {cmd.retry_count} attempts: {cmd.command.strip()}")
        self._notify_timeout(cmd.command.strip(), cmd.retry_count)

        # Mark connection as unhealthy if too many consecutive failures
        if self.consecutive_failures >= self.max_failures:
            self.connection_healthy = False
            self.is_connected = False
            logger.error(f"Connection marked unhealthy after {self.consecutive_failures} consecutive failures")

        return False

    def _send_blocking(self, command: str):
        """Blocking send"""
        if not self.socket:
            raise RuntimeError("Socket not connected")
        data = command.encode('utf-8') if isinstance(command, str) else command
        self.socket.send(data)

    def _recv_blocking(self) -> str:
        """Blocking receive"""
        if not self.socket:
            return ""
        try:
            data = self.socket.recv(1024)
            if data:
                return data.decode('utf-8', errors='ignore')
        except BlockingIOError:
            pass
        except Exception as e:
            logger.debug(f"Receive error: {e}")
        return ""

    async def _response_receiver_loop(self):
        """Background task that receives and parses responses"""
        logger.info("Response receiver loop started")
        buffer = ""

        while not self.stop_event.is_set():
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(self._recv_blocking),
                    timeout=2.0
                )

                if response:
                    buffer += response
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        if line:
                            self._handle_response(line)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in receiver: {e}")
                await asyncio.sleep(0.5)

        logger.info("Response receiver loop stopped")

    def _handle_response(self, response: str):
        """Parse and handle a response from the device"""
        self._log_command("RX", response)
        self.last_response = response

        if not response:
            return

        try:
            if response.startswith("@E"):
                # Acknowledgment: @E,0 or @E,1
                parts = response.split(',')
                self.last_ack_code = parts[1] if len(parts) > 1 else "0"
                logger.debug(f"Ack received: {self.last_ack_code}")

            elif response.startswith("@q"):
                # Status: @q,state,moving,well,countdown,temp1,temp2,heater
                self._parse_status_response(response)

            elif response.startswith("@?"):
                # Identity response
                logger.info(f"Device identity: {response}")

            else:
                logger.debug(f"Unknown response: {response}")

        except Exception as e:
            logger.error(f"Error parsing '{response}': {e}")

    def _parse_status_response(self, response: str):
        """Parse @q status response with all 7 fields"""
        try:
            # Format: @q,state,moving,well,countdown,temp1,temp2,heater
            data = response[3:].split(',')

            self.status.raw_response = response

            if len(data) >= 1:
                self.status.state = int(data[0])

            if len(data) >= 2:
                self.status.is_moving = data[1] == '1'

            if len(data) >= 3:
                self.status.current_well = int(data[2])

            if len(data) >= 4:
                self.status.countdown = int(data[3])

            if len(data) >= 5:
                # Temperature in 0.1C units
                self.status.temperature1 = int(data[4]) / 10.0

            if len(data) >= 6:
                self.status.temperature2 = int(data[5]) / 10.0

            if len(data) >= 7:
                self.status.heater_on = data[6] == '1'

            # Check for completion
            if self._waiting_for_completion:
                if self.status.state == DeviceState.READY and not self.status.is_moving:
                    self._completion_event.set()

            logger.debug(f"Status: state={self.status.state}, moving={self.status.is_moving}, "
                        f"well={self.status.current_well}, countdown={self.status.countdown}")

        except Exception as e:
            logger.error(f"Error parsing status '{response}': {e}")

    async def _query_status_loop(self):
        """Background task that periodically queries device status"""
        logger.info("Status query loop started")

        while not self.stop_event.is_set():
            try:
                await self.send_command(
                    "@Q\n",
                    priority=CommandPriority.STATUS_QUERY,
                    expected_duration=CommandTiming.QUERY
                )
                await asyncio.sleep(HARDWARE.QUERY_INTERVAL_S)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in query loop: {e}")
                await asyncio.sleep(1.0)

        logger.info("Status query loop stopped")

    async def wait_for_ready(self, timeout: float = 60.0) -> bool:
        """Wait until device returns to ready state"""
        self._waiting_for_completion = True
        self._completion_event.clear()

        try:
            await asyncio.wait_for(self._completion_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for ready state after {timeout}s")
            return False
        finally:
            self._waiting_for_completion = False

    async def wait_for_busy(self, timeout: float = 5.0) -> bool:
        """Wait until device becomes busy (started moving/executing)"""
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            if self.is_busy():
                logger.debug("Device is now busy")
                return True
            await asyncio.sleep(0.1)

        logger.warning(f"Timeout waiting for device to become busy after {timeout}s")
        return False

    async def wait_for_completion(self, timeout: float = 120.0) -> bool:
        """
        Wait for a command to complete by first detecting busy state,
        then waiting for return to ready. This prevents false early completion.
        """
        # First, wait for device to start executing (become busy)
        # Give it a short time to transition from READY to MOVING/RUNNING
        busy_detected = await self.wait_for_busy(timeout=5.0)

        if not busy_detected:
            # Device might have already completed very quickly, or command wasn't accepted
            # Check if it's in ready state
            if self.is_ready():
                logger.info("Device already ready (command may have completed quickly)")
                return True
            logger.warning("Device never became busy - command may not have been accepted")

        # Now wait for device to return to ready
        logger.debug(f"Waiting for completion (timeout={timeout}s)")
        return await self.wait_for_ready(timeout=timeout)

    def get_state_name(self) -> str:
        """Get human-readable state name"""
        return self.STATE_NAMES.get(self.status.state, f"Unknown({self.status.state})")

    def is_ready(self) -> bool:
        """Check if device is ready for commands"""
        return self.status.state == DeviceState.READY and not self.status.is_moving

    def is_busy(self) -> bool:
        """Check if device is busy"""
        return self.status.state in (DeviceState.MOVING, DeviceState.RUNNING) or self.status.is_moving

    # === Movement Commands ===

    async def insert(self) -> bool:
        """Insert sample tray (home/zero)"""
        logger.info("Inserting tray...")
        result = await self.send_command(
            "@Z\n",
            priority=CommandPriority.MOVEMENT,
            expected_duration=CommandTiming.INSERT
        )
        if result:
            # Wait for completion
            await self.wait_for_ready(timeout=15.0)
            logger.info("Tray inserted")
        return result

    async def eject(self) -> bool:
        """Eject sample tray"""
        logger.info("Ejecting tray...")
        result = await self.send_command(
            "@Y\n",
            priority=CommandPriority.MOVEMENT,
            expected_duration=CommandTiming.EJECT
        )
        if result:
            await self.wait_for_ready(timeout=10.0)
            logger.info("Tray ejected")
        return result

    async def stop_movement(self) -> bool:
        """Stop any ongoing movement (highest priority)"""
        logger.info("Sending STOP command")
        return await self.send_command(
            "@T\n",
            priority=CommandPriority.STOP,
            expected_duration=CommandTiming.STOP
        )

    # === Temperature & Heater ===

    async def set_temperature(self, temp_celsius: float) -> bool:
        """Set temperature setpoint (0-99.9 C)"""
        if temp_celsius < HARDWARE.TEMP_MIN or temp_celsius > HARDWARE.TEMP_MAX:
            logger.error(f"Temperature {temp_celsius} out of range")
            return False

        # Send temperature value directly (matches legacy code)
        logger.info(f"Setting temperature to {temp_celsius}C")
        return await self.send_command(
            f"@V,{temp_celsius}\n",
            priority=CommandPriority.TEMPERATURE
        )

    async def set_heater(self, on: bool) -> bool:
        """Turn heater on/off"""
        state = "1" if on else "0"
        logger.info(f"Setting heater {'ON' if on else 'OFF'}")
        return await self.send_command(
            f"@K,{state}\n",
            priority=CommandPriority.TEMPERATURE
        )

    def get_temperature(self) -> float:
        """Get current temperature reading"""
        return self.status.temperature1

    def is_heater_on(self) -> bool:
        """Check if heater is on"""
        return self.status.heater_on

    # === Well Mapping ===

    def well_mapping(self, locations: List[str]) -> List[int]:
        """Map well locations (e.g., 'A1', 'B2') to port numbers (1-96)"""
        well_map = {}
        rows = "ABCDEFGH"
        counter = 1

        for column in range(1, 13):
            for row in rows:
                well_map[f"{row}{column}"] = counter
                counter += 1

        return [well_map.get(loc) for loc in locations]

    def _format_method_command(self, port: int, time_seconds: int) -> str:
        """
        Format a move command in AMUZA protocol.

        Time is in seconds, zero-padded to 4 digits (e.g., 15 seconds = "0015")
        Port is 2-digit zero-padded
        """
        time_str = f"{int(time_seconds):04d}"
        port_str = f"{port:02d}"
        return f"@P,M1,{time_str},{port_str},\n"

    async def move_to_well(self, well_id: str, dwell_time: int = 0) -> bool:
        """
        Move to a specific well position.

        Args:
            well_id: Well identifier (e.g., "A1", "H12")
            dwell_time: Time to stay at well in seconds (0 for just move)
        """
        port_numbers = self.well_mapping([well_id])
        if not port_numbers or port_numbers[0] is None:
            logger.error(f"Invalid well ID: {well_id}")
            return False

        port = port_numbers[0]
        command = self._format_method_command(port, dwell_time)
        move_time = self.calculate_move_time(well_id)
        total_time = move_time + dwell_time

        logger.info(f"Moving to well {well_id} (port {port}), dwell {dwell_time}s")
        logger.info(f"Command: {repr(command)}")

        result = await self.send_command(
            command,
            priority=CommandPriority.MOVEMENT,
            expected_duration=move_time  # Just movement time, we'll wait for completion
        )

        if result:
            # Wait for the full operation to complete using improved detection
            await self.wait_for_completion(timeout=total_time + 60)

        return result

    async def execute_method(
        self,
        method: Method,
        stop_event: asyncio.Event,
        on_progress: Optional[Callable[[str, int, int], None]] = None
    ) -> bool:
        """
        Execute a single method with interruptible waits.

        Args:
            method: Method to execute
            stop_event: Event to check for stop request
            on_progress: Callback(status_message, current_second, total_seconds)
        """
        total_time = method.buffer_time + method.wait + 10  # +10 for movement
        current_second = 0

        # Buffer time wait (interruptible)
        if method.buffer_time > 0:
            logger.info(f"Buffer wait: {method.buffer_time}s")
            for i in range(int(method.buffer_time * 2)):  # Check every 0.5s
                if stop_event.is_set():
                    logger.info("Stopped during buffer")
                    return False
                await asyncio.sleep(0.5)
                current_second = (i + 1) // 2
                if on_progress:
                    on_progress(f"Buffer: {current_second}/{method.buffer_time}s", current_second, total_time)

        # Check stop before move
        if stop_event.is_set():
            return False

        # Convert well ID to port
        if isinstance(method.pos, str):
            ports = self.well_mapping([method.pos])
            if not ports or ports[0] is None:
                logger.error(f"Invalid well: {method.pos}")
                return False
            port = ports[0]
        else:
            port = method.pos

        # Send move command
        command = self._format_method_command(port, method.wait)
        move_time = self.calculate_move_time(str(method.pos))

        logger.info(f"Moving to {method.pos} (port {port}), sampling for {method.wait}s")
        logger.info(f"Command: {repr(command)}")

        await self.send_command(
            command,
            priority=CommandPriority.MOVEMENT,
            expected_duration=0.2  # Just send time
        )

        # Wait for device to start moving (prevents false early completion)
        logger.info("Waiting for device to start moving...")
        await self.wait_for_busy(timeout=5.0)

        # Now monitor device status until completion, with interruptible checks
        # Use device's actual countdown rather than software timing
        max_wait = method.wait + int(move_time) + 30  # Max timeout with buffer
        start_time = asyncio.get_event_loop().time()

        logger.info(f"Monitoring device until completion (max {max_wait}s)...")

        while (asyncio.get_event_loop().time() - start_time) < max_wait:
            # Check for stop request
            if stop_event.is_set():
                logger.info("Stopped during sampling")
                await self.stop_movement()
                return False

            # Check if device returned to ready
            if self.is_ready():
                elapsed = asyncio.get_event_loop().time() - start_time
                logger.info(f"Device ready after {elapsed:.1f}s")
                return True

            # Update progress using device's countdown if available
            elapsed = int(asyncio.get_event_loop().time() - start_time)
            current_second = method.buffer_time + elapsed
            if on_progress:
                countdown = self.status.countdown
                if countdown > 0:
                    on_progress(f"Sampling: well {method.pos} ({countdown}s left)", current_second, total_time)
                else:
                    on_progress(f"Sampling: well {method.pos}", current_second, total_time)

            await asyncio.sleep(0.5)

        # Timeout - device didn't return to ready
        logger.warning(f"Timeout waiting for well {method.pos} completion after {max_wait}s")
        return False

    async def execute_sequence(
        self,
        sequence: Sequence,
        stop_event: asyncio.Event,
        well_completed_callback: Optional[Callable[[str, List[str]], None]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
        timing_provider: Optional[Callable[[], tuple]] = None
    ) -> bool:
        """
        Execute a sequence of methods with stop support.

        Args:
            sequence: Sequence to execute
            stop_event: Event to check for stop request
            well_completed_callback: Callback(well_id, completed_wells_list)
            progress_callback: Callback(status, current_well_idx, total_wells)
            timing_provider: Optional callback returning (buffer_time, sampling_time)
                            If provided, updates method timing before each well
        """
        logger.info(f"Starting sequence: {sequence}")
        completed_wells = []

        try:
            # Auto-insert if needed
            if self.status.state != DeviceState.READY or self.is_busy():
                logger.info("Device not ready, inserting plate...")
                await self.insert()

            for i, method in enumerate(sequence.methods):
                if stop_event.is_set():
                    logger.info(f"Sequence stopped at well {i + 1}/{len(sequence)}")
                    await self.stop_movement()
                    return False

                # Get latest timing settings if provider is available
                if timing_provider:
                    try:
                        t_buffer, t_sampling = timing_provider()
                        if method.buffer_time != t_buffer or method.wait != t_sampling:
                            logger.info(f"Timing updated for {method.pos}: buffer={t_buffer}s, sampling={t_sampling}s")
                            method.buffer_time = t_buffer
                            method.wait = t_sampling
                    except Exception as e:
                        logger.warning(f"Could not get timing from provider: {e}")

                logger.info(f"Well {i + 1}/{len(sequence)}: {method}")

                if progress_callback:
                    progress_callback(f"Processing well {method.pos}", i, len(sequence))

                completed = await self.execute_method(method, stop_event)

                if not completed:
                    logger.info(f"Method not completed, stopping sequence")
                    return False

                completed_wells.append(str(method.pos))
                logger.info(f"Completed well {method.pos}")

                if well_completed_callback:
                    well_completed_callback(str(method.pos), completed_wells.copy())

            logger.info(f"Sequence complete. Wells: {', '.join(completed_wells)}")
            return True

        except Exception as e:
            logger.error(f"Sequence error: {e}")
            return False

    # === Status Methods ===

    def checkProgress(self) -> bool:
        """Check if an operation is in progress"""
        return self.is_busy()

    def getCurrentPosition(self) -> str:
        """Get current well position"""
        well_num = self.status.current_well
        if well_num == 0:
            return "Home"
        # Convert port number back to well ID
        row = (well_num - 1) % 8
        col = (well_num - 1) // 8 + 1
        return f"{chr(65 + row)}{col}"

    def getCountdown(self) -> int:
        """Get remaining time at current well"""
        return self.status.countdown


# Example usage
async def main():
    """Example usage of AsyncAmuzaConnection"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    connection = AsyncAmuzaConnection(use_mock=True)

    if await connection.connect():
        print("Connected!")
        print(f"State: {connection.get_state_name()}")

        # Test temperature
        await connection.set_temperature(37.0)
        await connection.set_heater(True)

        # Wait a bit for status updates
        await asyncio.sleep(2)
        print(f"Temperature: {connection.get_temperature()}C")
        print(f"Heater: {'ON' if connection.is_heater_on() else 'OFF'}")

        # Test sequence
        sequence = Sequence("Test")
        sequence.add_method(Method("A1", wait=5, buffer_time=2))
        sequence.add_method(Method("B2", wait=5, buffer_time=2))

        stop_event = asyncio.Event()

        def on_complete(well, completed):
            print(f"Completed: {well}, total: {completed}")

        result = await connection.execute_sequence(
            sequence, stop_event,
            well_completed_callback=on_complete
        )
        print(f"Sequence result: {result}")

        await connection.disconnect()
    else:
        print("Connection failed")


if __name__ == "__main__":
    asyncio.run(main())
