"""
Tests for async modules.

Run with: pytest test_async.py -v
"""

import pytest
import asyncio
import logging
from pathlib import Path

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

# Import modules to test
from config import HARDWARE, FILES, ASYNC_CONFIG
from app_state import AppState
from async_utils import AsyncTaskManager, interruptible_sleep, AsyncRateLimiter
from amuza_async import AsyncAmuzaConnection, Method, Sequence, CommandPriority
from sensor_reader_async import AsyncPotentiostatReader, SensorReading


class TestAppState:
    """Test AppState functionality"""
    
    @pytest.mark.asyncio
    async def test_well_selection(self):
        """Test well selection operations"""
        state = AppState()
        
        # Add wells
        await state.add_selected_well("A1")
        await state.add_selected_well("B2")
        
        wells = await state.get_selected_wells()
        assert "A1" in wells
        assert "B2" in wells
        assert len(wells) == 2
        
        # Remove well
        await state.remove_selected_well("A1")
        wells = await state.get_selected_wells()
        assert "A1" not in wells
        assert len(wells) == 1
        
        # Clear all
        await state.clear_selections()
        wells = await state.get_selected_wells()
        assert len(wells) == 0
    
    @pytest.mark.asyncio
    async def test_timing_params(self):
        """Test timing parameter operations"""
        state = AppState()
        
        # Set timing
        await state.set_timing_params(120, 180)
        
        buffer, sampling = await state.get_timing_params()
        assert buffer == 120
        assert sampling == 180
    
    @pytest.mark.asyncio
    async def test_stop_event(self):
        """Test stop event"""
        state = AppState()
        
        assert not state.stop_event.is_set()
        
        await state.request_stop()
        assert state.stop_event.is_set()
        
        await state.clear_stop()
        assert not state.stop_event.is_set()


class TestAsyncUtils:
    """Test async utility functions"""
    
    @pytest.mark.asyncio
    async def test_interruptible_sleep(self):
        """Test interruptible sleep"""
        stop_event = asyncio.Event()
        
        # Normal completion
        start = asyncio.get_event_loop().time()
        result = await interruptible_sleep(1.0, stop_event)
        elapsed = asyncio.get_event_loop().time() - start
        
        assert result is True
        assert elapsed >= 1.0
        
        # Interrupted
        stop_event.set()
        start = asyncio.get_event_loop().time()
        result = await interruptible_sleep(10.0, stop_event)
        elapsed = asyncio.get_event_loop().time() - start
        
        assert result is False
        assert elapsed < 1.0
    
    @pytest.mark.asyncio
    async def test_task_manager(self):
        """Test AsyncTaskManager"""
        manager = AsyncTaskManager()
        
        # Add tasks
        async def dummy_task(duration):
            await asyncio.sleep(duration)
            return "done"
        
        task1 = asyncio.create_task(dummy_task(0.1))
        task2 = asyncio.create_task(dummy_task(0.2))
        
        manager.add_task(task1, "task1")
        manager.add_task(task2, "task2")
        
        assert len(manager.get_running_tasks()) == 2
        
        # Wait for completion
        await asyncio.sleep(0.3)
        await manager.cleanup()
        
        assert len(manager.get_running_tasks()) == 0
    
    @pytest.mark.asyncio
    async def test_rate_limiter(self):
        """Test AsyncRateLimiter"""
        limiter = AsyncRateLimiter(max_calls=5, period=1.0)
        
        # Should allow 5 calls quickly
        start = asyncio.get_event_loop().time()
        for i in range(5):
            async with limiter:
                pass
        elapsed = asyncio.get_event_loop().time() - start
        
        assert elapsed < 0.5
        
        # 6th call should be delayed
        start = asyncio.get_event_loop().time()
        async with limiter:
            pass
        elapsed = asyncio.get_event_loop().time() - start
        
        assert elapsed >= 0.5


class TestAmuzaAsync:
    """Test async AMUZA connection"""
    
    @pytest.mark.asyncio
    async def test_mock_connection(self):
        """Test connection in mock mode"""
        conn = AsyncAmuzaConnection(
            device_address="00:00:00:00:00:00",
            use_mock=True
        )
        
        # Connect
        connected = await conn.connect()
        assert connected
        assert conn.is_connected
        
        # Disconnect
        await conn.disconnect()
        assert not conn.is_connected
    
    @pytest.mark.asyncio
    async def test_command_queue(self):
        """Test command queuing"""
        conn = AsyncAmuzaConnection(
            device_address="00:00:00:00:00:00",
            use_mock=True
        )
        
        await conn.connect()
        
        # Send commands
        await conn.send_command("@Q", priority=CommandPriority.STATUS_QUERY)
        await conn.send_command("@PA1", priority=CommandPriority.MOVEMENT)
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        await conn.disconnect()
    
    @pytest.mark.asyncio
    async def test_method_execution(self):
        """Test method execution"""
        conn = AsyncAmuzaConnection(
            device_address="00:00:00:00:00:00",
            use_mock=True
        )
        
        await conn.connect()
        
        # Create method
        method = Method(pos="A1", wait=1, eject=False, insert=False)
        
        # Execute
        stop_event = asyncio.Event()
        completed = await conn.execute_method(method, stop_event)
        
        assert completed
        
        await conn.disconnect()
    
    @pytest.mark.asyncio
    async def test_sequence_execution(self):
        """Test sequence execution"""
        conn = AsyncAmuzaConnection(
            device_address="00:00:00:00:00:00",
            use_mock=True
        )
        
        await conn.connect()
        
        # Create sequence
        sequence = Sequence("Test")
        sequence.add_method(Method("A1", wait=0.5))
        sequence.add_method(Method("B2", wait=0.5))
        
        # Execute
        stop_event = asyncio.Event()
        completed = await conn.execute_sequence(sequence, stop_event)
        
        assert completed
        
        await conn.disconnect()
    
    @pytest.mark.asyncio
    async def test_interruptible_sequence(self):
        """Test sequence can be interrupted"""
        conn = AsyncAmuzaConnection(
            device_address="00:00:00:00:00:00",
            use_mock=True
        )
        
        await conn.connect()
        
        # Create long sequence
        sequence = Sequence("Long Test")
        for i in range(10):
            sequence.add_method(Method(f"A{i+1}", wait=5))
        
        # Execute with interrupt
        stop_event = asyncio.Event()
        
        # Start execution
        execute_task = asyncio.create_task(
            conn.execute_sequence(sequence, stop_event)
        )
        
        # Interrupt after 2 seconds
        await asyncio.sleep(2.0)
        stop_event.set()
        
        # Wait for completion
        completed = await execute_task
        
        assert not completed  # Should be interrupted
        
        await conn.disconnect()


class TestSensorReaderAsync:
    """Test async sensor reader"""
    
    @pytest.mark.asyncio
    async def test_mock_reading(self):
        """Test mock sensor reading"""
        reader = AsyncPotentiostatReader(
            port="COM1",
            use_mock=True,
            output_file="test_output.csv"
        )
        
        connected = await reader.connect()
        assert connected
        
        # Start reading
        read_task = asyncio.create_task(reader.start_reading())
        
        # Run for 2 seconds
        await asyncio.sleep(2.0)
        
        # Stop
        await reader.stop()
        await read_task
        
        # Check readings
        assert reader.get_readings_count() > 0
        
        await reader.disconnect()
        
        # Cleanup
        Path("test_output.csv").unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_data_validation(self):
        """Test data validation"""
        reader = AsyncPotentiostatReader(
            port="COM1",
            use_mock=True
        )
        
        # Valid data
        assert reader._validate_data_line("1234567890.5,1,0.000001")
        
        # Invalid data
        assert not reader._validate_data_line("invalid,data,line")
        assert not reader._validate_data_line("# comment")
        assert not reader._validate_data_line("")


class TestSensorReading:
    """Test SensorReading dataclass"""
    
    def test_csv_conversion(self):
        """Test CSV line conversion"""
        reading = SensorReading(
            timestamp=1234567890.5,
            channel=3,
            value=0.000001
        )
        
        csv_line = reading.to_csv_line()
        assert "1234567890.5" in csv_line
        assert "3" in csv_line
        assert "0.000001" in csv_line


def test_config_constants():
    """Test configuration constants are defined"""
    # Hardware config
    assert hasattr(HARDWARE, 'BLUETOOTH_DEVICE_ADDRESS')
    assert hasattr(HARDWARE, 'SERIAL_PORT')
    assert hasattr(HARDWARE, 'SERIAL_BAUD_RATE')
    
    # Files config
    assert hasattr(FILES, 'OUTPUT_FILE_PATH')
    
    # Async config
    assert hasattr(ASYNC_CONFIG, 'COMMAND_QUEUE_SIZE')
    assert hasattr(ASYNC_CONFIG, 'TASK_TIMEOUT_S')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
