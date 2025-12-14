"""
Async version of potentiostat sensor reader using pyserial-asyncio.

Key improvements:
- Non-blocking serial reads with serial_asyncio
- Async file writing with aiofiles
- Proper cleanup with stop events
- Better error handling
- Data validation before writing
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List
import aiofiles

try:
    import serial_asyncio
    SERIAL_ASYNCIO_AVAILABLE = True
except ImportError:
    SERIAL_ASYNCIO_AVAILABLE = False
    import serial

from config import HARDWARE, FILES


logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Represents a single sensor reading"""
    timestamp: float
    channel: int
    value: float
    unit: str = "A"  # Amperes
    
    def to_csv_line(self) -> str:
        """Convert to CSV format"""
        return f"{self.timestamp},{self.channel},{self.value}\n"


class AsyncPotentiostatReader:
    """
    Async potentiostat reader with non-blocking serial communication.
    
    Features:
    - Uses serial_asyncio for non-blocking reads
    - Validates data before writing
    - Graceful shutdown with stop event
    - Async file I/O with aiofiles
    """
    
    def __init__(
        self,
        port: str,
        baudrate: int = HARDWARE.SERIAL_BAUD_RATE,
        output_file: str = FILES.OUTPUT_FILE_PATH,
        use_mock: bool = False
    ):
        self.port = port
        self.baudrate = baudrate
        self.output_file = output_file
        self.use_mock = use_mock
        
        # Serial connection
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.serial_port = None
        
        # State
        self.is_running = False
        self.stop_event = asyncio.Event()
        self.readings_count = 0
        
        # Data buffer
        self.pending_buffer = ""
        
        logger.info(f"AsyncPotentiostatReader initialized (port={port}, mock={use_mock})")
    
    async def connect(self) -> bool:
        """
        Connect to the serial port.
        
        Returns:
            True if connected successfully
        """
        try:
            if self.use_mock:
                logger.info("Using mock serial connection")
                return True
            
            if not SERIAL_ASYNCIO_AVAILABLE:
                logger.error("serial_asyncio not available, falling back to blocking serial")
                self.serial_port = serial.Serial(
                    port=self.port,
                    baudrate=self.baudrate,
                    timeout=1.0
                )
                return True
            
            # Use serial_asyncio for async I/O
            self.reader, self.writer = await serial_asyncio.open_serial_connection(
                url=self.port,
                baudrate=self.baudrate
            )
            
            logger.info(f"Connected to serial port {self.port}")
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
        
        logger.info("Disconnected from serial port")
    
    async def start_reading(self):
        """Start continuous reading from sensor"""
        if self.is_running:
            logger.warning("Already running")
            return
        
        self.is_running = True
        self.stop_event.clear()
        
        logger.info("Starting sensor reading loop")
        
        try:
            if self.use_mock:
                await self._mock_reading_loop()
            elif SERIAL_ASYNCIO_AVAILABLE and self.reader:
                await self._async_reading_loop()
            else:
                await self._blocking_reading_loop()
        except Exception as e:
            logger.error(f"Error in reading loop: {e}")
        finally:
            self.is_running = False
            logger.info("Sensor reading loop stopped")
    
    async def _async_reading_loop(self):
        """Non-blocking reading loop using serial_asyncio"""
        async with aiofiles.open(self.output_file, 'a') as f:
            while not self.stop_event.is_set():
                try:
                    # Read line with timeout
                    line_bytes = await asyncio.wait_for(
                        self.reader.readline(),
                        timeout=HARDWARE.SERIAL_READ_TIMEOUT_S
                    )
                    
                    if not line_bytes:
                        await asyncio.sleep(0.1)
                        continue
                    
                    # Decode and process
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        # Validate and write
                        if self._validate_data_line(line):
                            await f.write(line + '\n')
                            await f.flush()
                            self.readings_count += 1
                            
                            if self.readings_count % 100 == 0:
                                logger.debug(f"Processed {self.readings_count} readings")
                    
                except asyncio.TimeoutError:
                    # No data available, continue
                    continue
                except asyncio.CancelledError:
                    logger.info("Reading loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error reading data: {e}")
                    await asyncio.sleep(1.0)
    
    async def _blocking_reading_loop(self):
        """Fallback blocking reading loop"""
        async with aiofiles.open(self.output_file, 'a') as f:
            while not self.stop_event.is_set():
                try:
                    # Read in thread pool to avoid blocking
                    line = await asyncio.wait_for(
                        asyncio.to_thread(self._read_line_blocking),
                        timeout=HARDWARE.SERIAL_READ_TIMEOUT_S
                    )
                    
                    if line and self._validate_data_line(line):
                        await f.write(line + '\n')
                        await f.flush()
                        self.readings_count += 1
                    
                except asyncio.TimeoutError:
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in blocking read: {e}")
                    await asyncio.sleep(1.0)
    
    def _read_line_blocking(self) -> str:
        """Blocking read (run in thread pool)"""
        if self.serial_port and self.serial_port.is_open:
            line = self.serial_port.readline()
            return line.decode('utf-8', errors='ignore').strip()
        return ""
    
    async def _mock_reading_loop(self):
        """Mock reading loop for testing"""
        import random
        
        async with aiofiles.open(self.output_file, 'a') as f:
            while not self.stop_event.is_set():
                try:
                    # Generate mock data
                    timestamp = datetime.now().timestamp()
                    
                    # Simulate 6 channels
                    for channel in range(1, 7):
                        value = random.uniform(-1e-6, 1e-6)  # Mock current in microamps
                        reading = SensorReading(
                            timestamp=timestamp,
                            channel=channel,
                            value=value
                        )
                        
                        await f.write(reading.to_csv_line())
                    
                    await f.flush()
                    self.readings_count += 6
                    
                    # Simulate data rate (adjust as needed)
                    await asyncio.sleep(0.5)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in mock reading: {e}")
                    await asyncio.sleep(1.0)
    
    def _validate_data_line(self, line: str) -> bool:
        """
        Validate a data line before writing.
        
        Args:
            line: Data line to validate
        
        Returns:
            True if valid
        """
        if not line or line.startswith('#'):
            return False
        
        # Check for expected format (timestamp,channel,value)
        parts = line.split(',')
        if len(parts) < 3:
            return False
        
        try:
            # Validate timestamp
            float(parts[0])
            
            # Validate channel
            int(parts[1])
            
            # Validate value
            float(parts[2])
            
            return True
        except ValueError:
            logger.warning(f"Invalid data line: {line}")
            return False
    
    async def stop(self):
        """Request stop"""
        logger.info("Stop requested")
        self.stop_event.set()
    
    def get_readings_count(self) -> int:
        """Get total readings processed"""
        return self.readings_count


class DataProcessor:
    """Process and convert sensor data"""
    
    @staticmethod
    def convert_data(
        data: List[SensorReading],
        gain: float = 1e6,
        sensor_area_cm2: float = 0.0314
    ) -> List[SensorReading]:
        """
        Convert raw sensor data to calibrated values.
        
        Args:
            data: List of sensor readings
            gain: Amplifier gain (default 1e6 for microamps)
            sensor_area_cm2: Sensor area in cm²
        
        Returns:
            List of converted readings
        """
        converted = []
        
        for reading in data:
            # Convert based on gain and area
            converted_value = reading.value / (gain * sensor_area_cm2)
            
            converted_reading = SensorReading(
                timestamp=reading.timestamp,
                channel=reading.channel,
                value=converted_value,
                unit="µA/cm²"
            )
            converted.append(converted_reading)
        
        return converted
    
    @staticmethod
    async def process_file(
        input_file: str,
        output_file: str,
        gain: float = 1e6,
        sensor_area_cm2: float = 0.0314
    ):
        """
        Process an entire file with conversion.
        
        Args:
            input_file: Input file path
            output_file: Output file path
            gain: Amplifier gain
            sensor_area_cm2: Sensor area
        """
        readings = []
        
        # Read input file
        async with aiofiles.open(input_file, 'r') as f:
            async for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        reading = SensorReading(
                            timestamp=float(parts[0]),
                            channel=int(parts[1]),
                            value=float(parts[2])
                        )
                        readings.append(reading)
                    except ValueError:
                        continue
        
        # Convert data
        converted = DataProcessor.convert_data(readings, gain, sensor_area_cm2)
        
        # Write output file
        async with aiofiles.open(output_file, 'w') as f:
            await f.write("# Timestamp,Channel,Value (µA/cm²)\n")
            for reading in converted:
                await f.write(reading.to_csv_line())
        
        logger.info(f"Processed {len(converted)} readings to {output_file}")


# Example usage
async def main():
    """Example usage of AsyncPotentiostatReader"""
    
    # Create reader in mock mode
    reader = AsyncPotentiostatReader(
        port=HARDWARE.SERIAL_PORT,
        use_mock=True
    )
    
    # Connect
    if await reader.connect():
        print("Connected!")
        
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
