"""
Async version of potentiostat sensor reader with binary packet protocol.

Supports both:
- Binary protocol (25-byte packets with checksum) for real hardware
- CSV text format for mock mode and testing

Key improvements:
- Non-blocking serial reads with serial_asyncio
- Proper binary packet validation (header, checksum)
- Async file writing with aiofiles
- Legacy file format compatibility
- Proper cleanup with stop events
- OPTIMIZED: Batched file writes for better I/O performance
- OPTIMIZED: Data callback for direct GUI updates without file polling
"""

import asyncio
import logging
import struct
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Callable
from pathlib import Path
import aiofiles

try:
    import serial_asyncio
    SERIAL_ASYNCIO_AVAILABLE = True
except ImportError:
    SERIAL_ASYNCIO_AVAILABLE = False

try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

from config import HARDWARE, FILES, SENSOR


logger = logging.getLogger(__name__)


# Binary protocol constants
PACKET_LENGTH = 25
PACKET_HEADER = bytes([0x04, 0x68, 0x13, 0x13, 0x68])
PACKET_START_BYTE = 0x16

# Conversion factor for raw values to voltage
GAIN_CONVERSION = 50 / (2**15 - 1)  # ~0.001526

# I/O optimization settings
WRITE_BUFFER_SIZE = 10  # Write to file every N readings
WRITE_BUFFER_TIMEOUT_S = 2.0  # Or every N seconds, whichever comes first


@dataclass
class SensorReading:
    """Represents a single sensor reading with all channels"""
    timestamp: datetime
    time_minutes: float  # Time since start in minutes
    counter: int
    channels: List[float]  # Channels 1-6 in voltage/current units
    temperature: float  # Channel 7 as temperature in Celsius
    status: int = 1  # Channel 8 status flag

    def to_legacy_line(self, channel_names: List[str] = None) -> str:
        """Convert to legacy tab-separated format"""
        values = [str(self.counter), f"{self.time_minutes:.4f}"]

        # Add channel values (1-6)
        for i, ch_val in enumerate(self.channels[:6]):
            values.append(f"{ch_val:.3f}")

        # Add temperature (channel 7)
        values.append(f"{self.temperature:.3f}")

        # Add status flag (channel 8)
        values.append(str(self.status))

        # Pad with zeros for remaining channels (up to 64)
        while len(values) < 66:  # counter + time + 64 channels
            values.append("0")

        # Add fit values (all zeros)
        fit_values = ["0"] * 320  # 64 X(Fit) + 256 Fit coefficients

        return "\t".join(values) + "\t" + "\t".join(fit_values) + "\n"

    def to_csv_line(self) -> str:
        """Convert to simple CSV format"""
        values = [
            f"{self.time_minutes * 60:.2f}",  # Time in seconds
            *[f"{ch:.6f}" for ch in self.channels[:6]],
            f"{self.temperature:.2f}"
        ]
        return ",".join(values) + "\n"


class AsyncPotentiostatReader:
    """
    Async potentiostat reader supporting binary packet protocol.

    Binary packet format (25 bytes):
    - Byte 0: Start byte (0x16)
    - Byte 1: Checksum
    - Bytes 2-19: Data (9 x 16-bit signed integers, little-endian reversed)
    - Bytes 20-24: Header (0x04, 0x68, 0x13, 0x13, 0x68)

    OPTIMIZATIONS:
    - Batched file writes (every N readings or M seconds)
    - Optional data callback for direct GUI updates
    - Efficient circular buffer for packet processing
    """

    def __init__(
        self,
        port: str,
        baudrate: int = HARDWARE.SERIAL_BAUD_RATE,
        output_file: str = None,
        use_mock: bool = False,
        data_callback: Callable[[SensorReading], None] = None
    ):
        self.port = port
        self.baudrate = baudrate
        self.use_mock = use_mock
        self.data_callback = data_callback  # Optional callback for real-time GUI updates

        # Generate output file path with timestamp
        if output_file:
            self.output_file = output_file
        else:
            timestamp = datetime.now().strftime(FILES.TIMESTAMP_FORMAT)
            self.output_file = str(
                Path(FILES.SENSOR_READINGS_FOLDER) /
                FILES.SENSOR_FILENAME_FORMAT.format(timestamp=timestamp)
            )

        # Serial connection
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.serial_port = None

        # State
        self.is_running = False
        self.stop_event = asyncio.Event()
        self.readings_count = 0
        self.start_time: Optional[datetime] = None

        # Binary packet buffer
        self.packet_buffer = bytearray()

        # Write buffer for batched I/O
        self.write_buffer: List[str] = []
        self.last_flush_time: float = 0

        # Channel names for file header
        self.channel_names = [f"#1ch{i}" for i in range(1, 17)] + \
                            [f"#2ch{i}" for i in range(1, 17)] + \
                            [f"#3ch{i}" for i in range(1, 17)] + \
                            [f"#4ch{i}" for i in range(1, 17)]

        logger.info(f"AsyncPotentiostatReader initialized (port={port}, mock={use_mock})")
        logger.info(f"Output file: {self.output_file}")

    def set_data_callback(self, callback: Callable[[SensorReading], None]):
        """Set callback for real-time data updates (more efficient than file polling)"""
        self.data_callback = callback

    async def connect(self) -> bool:
        """Connect to the serial port"""
        try:
            if self.use_mock:
                logger.info("Using mock serial connection")
                return True

            # Ensure output directory exists
            output_dir = Path(self.output_file).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            if SERIAL_ASYNCIO_AVAILABLE:
                # Use serial_asyncio for async I/O
                self.reader, self.writer = await serial_asyncio.open_serial_connection(
                    url=self.port,
                    baudrate=self.baudrate
                )
                logger.info(f"Connected to serial port {self.port} (async)")
            elif SERIAL_AVAILABLE:
                # Fallback to blocking serial
                self.serial_port = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=HARDWARE.SERIAL_TIMEOUT
                )
                logger.info(f"Connected to serial port {self.port} (blocking)")
            else:
                logger.error("No serial library available")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to connect to serial port: {e}")
            return False

    async def disconnect(self):
        """Disconnect from serial port"""
        logger.info("Disconnecting from serial port")
        self.stop_event.set()

        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception as e:
                logger.error(f"Error closing serial writer: {e}")

        if self.serial_port:
            try:
                await asyncio.to_thread(self.serial_port.close)
            except Exception as e:
                logger.error(f"Error closing serial port: {e}")

        logger.info(f"Disconnected. Total readings: {self.readings_count}")

    async def start_reading(self):
        """Start continuous reading from sensor"""
        if self.is_running:
            logger.warning("Already running")
            return

        self.is_running = True
        self.stop_event.clear()
        self.start_time = datetime.now()
        self.readings_count = 0
        self.last_flush_time = asyncio.get_event_loop().time()

        logger.info("Starting sensor reading loop")

        try:
            # Write file header
            await self._write_file_header()

            if self.use_mock:
                await self._mock_reading_loop()
            elif SERIAL_ASYNCIO_AVAILABLE and self.reader:
                await self._async_binary_reading_loop()
            elif self.serial_port:
                await self._blocking_binary_reading_loop()
            else:
                logger.error("No valid serial connection")

        except Exception as e:
            logger.error(f"Error in reading loop: {e}")
        finally:
            # Flush any remaining buffered data
            await self._flush_write_buffer()
            self.is_running = False
            logger.info("Sensor reading loop stopped")

    async def _write_file_header(self):
        """Write legacy format header to output file"""
        async with aiofiles.open(self.output_file, 'w') as f:
            # Created timestamp
            created_time = self.start_time.strftime("%m/%d/%Y\t%I:%M:%S %p")
            await f.write(f"Created: {created_time}\n")

            # Column headers
            header_cols = ["counter", "t[min]"] + self.channel_names
            # Add X(Fit) columns
            header_cols += [f"X(Fit{k})" for k in range(1, 65)]
            # Add Fit coefficient columns
            header_cols += [f"Fit{k}a{i}" for k in range(1, 65) for i in range(1, 5)]
            await f.write("\t".join(header_cols) + "\n")

            # Start timestamp
            start_time = self.start_time.strftime("%m/%d/%Y\t%I:%M:%S %p")
            await f.write(f"Start: {start_time}\n")

        logger.info(f"File header written to {self.output_file}")

    async def _flush_write_buffer(self):
        """Flush buffered data to file"""
        if not self.write_buffer:
            return

        try:
            async with aiofiles.open(self.output_file, 'a') as f:
                await f.write(''.join(self.write_buffer))
            self.write_buffer.clear()
            self.last_flush_time = asyncio.get_event_loop().time()
        except Exception as e:
            logger.error(f"Error flushing write buffer: {e}")

    async def _maybe_flush_buffer(self):
        """Flush buffer if conditions are met (size or time)"""
        current_time = asyncio.get_event_loop().time()
        time_since_flush = current_time - self.last_flush_time

        if len(self.write_buffer) >= WRITE_BUFFER_SIZE or time_since_flush >= WRITE_BUFFER_TIMEOUT_S:
            await self._flush_write_buffer()

    async def _async_binary_reading_loop(self):
        """Non-blocking binary reading loop using serial_asyncio"""
        while not self.stop_event.is_set():
            try:
                # Read available bytes
                data = await asyncio.wait_for(
                    self.reader.read(PACKET_LENGTH),
                    timeout=HARDWARE.SERIAL_READ_TIMEOUT_S
                )

                if data:
                    self.packet_buffer.extend(data)

                    # Process complete packets
                    while len(self.packet_buffer) >= PACKET_LENGTH:
                        reading = self._process_packet_buffer()
                        if reading:
                            # Buffer the line for batched writing
                            self.write_buffer.append(reading.to_legacy_line())

                            # Call data callback if set (for direct GUI updates)
                            if self.data_callback:
                                try:
                                    self.data_callback(reading)
                                except Exception as e:
                                    logger.error(f"Data callback error: {e}")

                            if self.readings_count % 100 == 0:
                                logger.debug(f"Processed {self.readings_count} readings")

                    # Flush buffer if needed
                    await self._maybe_flush_buffer()

            except asyncio.TimeoutError:
                # Still flush on timeout to ensure data is saved
                await self._maybe_flush_buffer()
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error reading data: {e}")
                await asyncio.sleep(0.5)

    async def _blocking_binary_reading_loop(self):
        """Fallback blocking binary reading loop"""
        while not self.stop_event.is_set():
            try:
                # Read in thread pool
                data = await asyncio.wait_for(
                    asyncio.to_thread(self._read_bytes_blocking, PACKET_LENGTH),
                    timeout=HARDWARE.SERIAL_READ_TIMEOUT_S
                )

                if data:
                    self.packet_buffer.extend(data)

                    while len(self.packet_buffer) >= PACKET_LENGTH:
                        reading = self._process_packet_buffer()
                        if reading:
                            # Buffer the line for batched writing
                            self.write_buffer.append(reading.to_legacy_line())

                            # Call data callback if set
                            if self.data_callback:
                                try:
                                    self.data_callback(reading)
                                except Exception as e:
                                    logger.error(f"Data callback error: {e}")

                    # Flush buffer if needed
                    await self._maybe_flush_buffer()

            except asyncio.TimeoutError:
                await self._maybe_flush_buffer()
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in blocking read: {e}")
                await asyncio.sleep(0.5)

    def _read_bytes_blocking(self, count: int) -> bytes:
        """Blocking read (run in thread pool)"""
        if self.serial_port and self.serial_port.is_open:
            return self.serial_port.read(count)
        return b''

    def _process_packet_buffer(self) -> Optional[SensorReading]:
        """Process packet buffer and extract valid reading.

        Note: The wire format has header at the START and sync byte (0x16) at the END.
        The legacy code reversed bytes by inserting at front of buffer. We reverse
        the packet here before validation to match the expected format (0x16 first,
        header last).
        """
        if len(self.packet_buffer) < PACKET_LENGTH:
            return None

        # Extract potential packet and REVERSE it to match expected format
        # Wire format: [header...data...checksum...0x16]
        # Expected:    [0x16...checksum...data...header]
        raw_packet = bytes(self.packet_buffer[:PACKET_LENGTH])
        packet = bytes(reversed(raw_packet))

        # Validate packet (now in expected format)
        if self._validate_packet(packet):
            # Remove valid packet from buffer
            del self.packet_buffer[:PACKET_LENGTH]

            # Parse and return reading
            return self._parse_packet(packet)
        else:
            # Invalid packet - shift buffer by 1 byte and try again
            del self.packet_buffer[0]
            return None

    def _validate_packet(self, packet: bytes) -> bool:
        """
        Validate a 25-byte packet.

        Checks:
        - Start byte (0x16)
        - Header bytes at end
        - Checksum
        """
        if len(packet) != PACKET_LENGTH:
            return False

        # Check start byte
        if packet[0] != PACKET_START_BYTE:
            return False

        # Check header at end
        if packet[-5:] != PACKET_HEADER:
            return False

        # Validate checksum
        checksum = 0
        for byte in packet[2:-4]:
            checksum = (checksum + byte) & 0xFF

        if packet[1] != checksum:
            logger.debug(f"Checksum mismatch: expected {checksum}, got {packet[1]}")
            return False

        return True

    def _parse_packet(self, packet: bytes) -> SensorReading:
        """Parse valid packet into SensorReading"""
        # Extract data bytes (bytes 2-19, reversed for little-endian)
        data_bytes = list(packet[2:-5])
        data_bytes.reverse()

        # Convert to 16-bit signed integers
        values = []
        for i in range(0, len(data_bytes), 2):
            if i + 1 < len(data_bytes):
                # Combine bytes into 16-bit signed integer
                raw_value = struct.unpack('>h', bytes([data_bytes[i], data_bytes[i+1]]))[0]
                values.append(raw_value)

        # Convert to voltages/currents
        channels = []
        for i, val in enumerate(values[:6]):
            channels.append(round(val * GAIN_CONVERSION, 3))

        # Temperature (channel 7) - different conversion
        temperature = 0.0
        if len(values) > 6:
            temperature = round(values[6] / 16.0, 3)

        # Calculate time since start
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            time_minutes = elapsed / 60.0
        else:
            time_minutes = 0.0

        self.readings_count += 1

        return SensorReading(
            timestamp=datetime.now(),
            time_minutes=time_minutes,
            counter=self.readings_count,
            channels=channels,
            temperature=temperature,
            status=1
        )

    async def _mock_reading_loop(self):
        """Mock reading loop for testing without hardware"""
        import random
        import math

        while not self.stop_event.is_set():
            try:
                # Calculate elapsed time
                elapsed = (datetime.now() - self.start_time).total_seconds()
                time_minutes = elapsed / 60.0

                # Generate mock channel data with realistic patterns
                base_time = elapsed / 10.0
                channels = []

                # Channel 1-2: Glutamate sensor pair
                ch1 = 0.5 + 0.1 * math.sin(base_time) + random.uniform(-0.02, 0.02)
                ch2 = 0.3 + 0.05 * math.sin(base_time + 0.5) + random.uniform(-0.01, 0.01)
                channels.extend([ch1, ch2])

                # Channel 3: Glutamine
                ch3 = 0.4 + 0.08 * math.cos(base_time) + random.uniform(-0.015, 0.015)
                channels.append(ch3)

                # Channel 4-5: Glucose sensor pair
                ch4 = 0.2 + 0.03 * math.sin(base_time * 0.5) + random.uniform(-0.01, 0.01)
                ch5 = 0.6 + 0.12 * math.sin(base_time * 0.5 + 0.3) + random.uniform(-0.02, 0.02)
                channels.extend([ch4, ch5])

                # Channel 6: Lactate
                ch6 = 0.35 + 0.07 * math.cos(base_time * 0.7) + random.uniform(-0.01, 0.01)
                channels.append(ch6)

                # Temperature with slight drift
                temperature = 37.0 + 0.5 * math.sin(base_time * 0.1) + random.uniform(-0.1, 0.1)

                self.readings_count += 1

                reading = SensorReading(
                    timestamp=datetime.now(),
                    time_minutes=time_minutes,
                    counter=self.readings_count,
                    channels=channels,
                    temperature=temperature,
                    status=1
                )

                # Buffer the line
                self.write_buffer.append(reading.to_legacy_line())

                # Call data callback if set
                if self.data_callback:
                    try:
                        self.data_callback(reading)
                    except Exception as e:
                        logger.error(f"Data callback error: {e}")

                # Flush buffer if needed
                await self._maybe_flush_buffer()

                if self.readings_count % 50 == 0:
                    logger.debug(f"Mock: {self.readings_count} readings, time={time_minutes:.2f}min")

                # Simulate ~2 Hz data rate
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in mock reading: {e}")
                await asyncio.sleep(1.0)

    async def stop(self):
        """Request stop"""
        logger.info("Stop requested")
        self.stop_event.set()

    def get_readings_count(self) -> int:
        """Get total readings processed"""
        return self.readings_count

    def get_output_file(self) -> str:
        """Get output file path"""
        return self.output_file


class DataProcessor:
    """Process and convert sensor data with calibration"""

    @staticmethod
    def calculate_metabolites(
        channels: List[float],
        gains: dict = None
    ) -> dict:
        """
        Calculate metabolite concentrations from channel differences.

        Metabolite calculations (from legacy code):
        - Glutamate = Channel 1 - Channel 2
        - Glutamine = Channel 3 - Channel 1
        - Glucose = Channel 5 - Channel 4
        - Lactate = Channel 6 - Channel 4
        """
        if gains is None:
            gains = SENSOR.DEFAULT_GAINS

        metabolites = {}

        if len(channels) >= 2:
            raw = channels[0] - channels[1]
            metabolites['Glutamate'] = raw * gains.get('Glutamate', 1.0)

        if len(channels) >= 3:
            raw = channels[2] - channels[0]
            metabolites['Glutamine'] = raw * gains.get('Glutamine', 1.0)

        if len(channels) >= 5:
            raw = channels[4] - channels[3]
            metabolites['Glucose'] = raw * gains.get('Glucose', 1.0)

        if len(channels) >= 6:
            raw = channels[5] - channels[3]
            metabolites['Lactate'] = raw * gains.get('Lactate', 1.0)

        return metabolites

    @staticmethod
    async def process_file(
        input_file: str,
        output_file: str,
        gains: dict = None
    ):
        """Process a legacy format file and output metabolite data"""
        if gains is None:
            gains = SENSOR.DEFAULT_GAINS

        readings = []

        # Read and parse input file
        async with aiofiles.open(input_file, 'r') as f:
            lines = await f.readlines()

        # Skip header (first 3 lines)
        data_lines = lines[3:] if len(lines) > 3 else []

        for line in data_lines:
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue

            try:
                counter = int(parts[0])
                time_min = float(parts[1])
                channels = [float(parts[i]) for i in range(2, 8)]
                temperature = float(parts[8]) if len(parts) > 8 else 0

                metabolites = DataProcessor.calculate_metabolites(channels, gains)

                readings.append({
                    'counter': counter,
                    'time_min': time_min,
                    'channels': channels,
                    'temperature': temperature,
                    'metabolites': metabolites
                })
            except (ValueError, IndexError) as e:
                logger.debug(f"Skipping invalid line: {e}")
                continue

        # Write output file
        async with aiofiles.open(output_file, 'w') as f:
            # Header
            await f.write("Time(min),Glutamate,Glutamine,Glucose,Lactate,Temperature\n")

            for r in readings:
                m = r['metabolites']
                line = f"{r['time_min']:.4f},"
                line += f"{m.get('Glutamate', 0):.4f},"
                line += f"{m.get('Glutamine', 0):.4f},"
                line += f"{m.get('Glucose', 0):.4f},"
                line += f"{m.get('Lactate', 0):.4f},"
                line += f"{r['temperature']:.2f}\n"
                await f.write(line)

        logger.info(f"Processed {len(readings)} readings to {output_file}")


# Example usage
async def main():
    """Example usage of AsyncPotentiostatReader"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create reader in mock mode
    reader = AsyncPotentiostatReader(
        port="COM3",
        use_mock=True
    )

    if await reader.connect():
        print(f"Connected! Output file: {reader.get_output_file()}")

        # Start reading (run for 10 seconds)
        read_task = asyncio.create_task(reader.start_reading())

        await asyncio.sleep(10)

        # Stop reading
        await reader.stop()
        await read_task

        print(f"Collected {reader.get_readings_count()} readings")

        # Disconnect
        await reader.disconnect()
    else:
        print("Connection failed")


if __name__ == "__main__":
    asyncio.run(main())
