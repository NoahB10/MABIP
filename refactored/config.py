"""
Configuration constants for MABIP system.
Centralizes all magic numbers and settings.
"""
from dataclasses import dataclass
from typing import Dict

@dataclass
class HardwareConfig:
    """Hardware timing and connection settings."""
    # Bluetooth identity
    BLUETOOTH_DEVICE_ADDRESS: str = "FC:90:00:34"
    BT_DEVICE_NAME: str = 'FC90-0034'
    
    # Serial communication
    SERIAL_BAUD_RATE: int = 9600
    SERIAL_TIMEOUT: float = 0.5
    SERIAL_PACKAGE_LENGTH: int = 25
    SERIAL_PORT: str = "COM3"
    SERIAL_READ_TIMEOUT_S: float = 1.0
    
    # Bluetooth
    BT_SCAN_TIMEOUT: float = 10.0
    BT_QUERY_INTERVAL: float = 1.0  # seconds between status queries
    QUERY_INTERVAL_S: float = 1.0
    
    # AMUZA movement timing (seconds)
    BUFFER_TIME_DEFAULT: int = 60  # Rest time between wells
    SAMPLING_TIME_DEFAULT: int = 90  # Sampling duration per well
    INSERT_DELAY: int = 6  # Wait after tray insertion
    MOVE_COMPLETION_DELAY: int = 9  # Extra delay after move command
    MOVE_FINAL_DELAY: int = 1  # Additional buffer
    
    # Connection timing
    CONNECTION_TIMEOUT_S: float = 30.0
    
    # Temperature limits
    TEMP_MIN: float = 0.0
    TEMP_MAX: float = 99.9


@dataclass
class UIConfig:
    """UI layout and behavior settings."""
    # Well plate
    WELL_SIZE: int = 50  # pixels
    WELL_ROWS: str = "ABCDEFGH"
    WELL_COLUMNS: int = 12
    WELL_LABEL_MIN_WIDTH: int = 50
    WELL_LABEL_MIN_HEIGHT: int = 30
    SELECTED_WELL_COLOR: str = "lightblue"  # Blue wells for RUNPLATE (drag selection)
    CTRL_WELL_COLOR: str = "lightgreen"  # Green wells for MOVE (Ctrl+Click)
    
    # Plot window
    PLOT_UPDATE_INTERVAL_MS: int = 2000  # milliseconds
    PLOT_WINDOW_MINUTES_DEFAULT: int = 10  # rolling window size
    PLOT_WINDOW_OPTIONS: list = None  # Will be set in __post_init__
    PLOT_WINDOW_WIDTH: int = 1200
    PLOT_WINDOW_HEIGHT: int = 800
    PLOT_WINDOW_MINUTES: int = 10
    
    # Display
    DISPLAY_HISTORY_MAX: int = 50  # messages
    DISPLAY_HEIGHT: int = 230  # pixels
    INSTRUCTIONS_WIDTH: int = 300  # pixels
    
    # Button styling
    BUTTON_MAX_WIDTH: int = 170
    BUTTON_MAX_HEIGHT: int = 32
    
    # Main window
    MAIN_WINDOW_WIDTH: int = 1050
    MAIN_WINDOW_HEIGHT: int = 566
    
    def __post_init__(self):
        if self.PLOT_WINDOW_OPTIONS is None:
            self.PLOT_WINDOW_OPTIONS = [5, 10, 15, 30]


@dataclass
class SensorConfig:
    """Sensor calibration and gain settings."""
    # Channel mapping
    SENSOR_CHANNELS: tuple = (
        '#1ch1', '#1ch2', '#1ch3', '#1ch4', 
        '#1ch5', '#1ch6', '#1ch7'
    )
    
    # Default gain values
    DEFAULT_GAINS: Dict[str, float] = None
    
    # Default calibration values (mM)
    DEFAULT_CALIBRATIONS: Dict[str, float] = None
    
    # Sensor conversion factor
    GAIN_CONVERSION: float = 50 / (2**15 - 1)
    
    def __post_init__(self):
        if self.DEFAULT_GAINS is None:
            self.DEFAULT_GAINS = {
                "Glutamate": 3.394,
                "Glutamine": 0.974,
                "Glucose": 1.5,
                "Lactate": 0.515,
            }
        if self.DEFAULT_CALIBRATIONS is None:
            self.DEFAULT_CALIBRATIONS = {
                "Glutamate": 0.996,
                "Glutamine": 1.0,
                "Glucose": 17.38,
                "Lactate": 9.94,
            }


@dataclass
class FileConfig:
    """File I/O and logging settings."""
    # Folders
    SENSOR_READINGS_FOLDER: str = "Sensor_Readings"
    AMUZA_LOGS_FOLDER: str = "Amuza_Logs"
    OUTPUT_FILE_PATH: str = "Sensor_Readings/output.csv"
    SETTINGS_FOLDER: str = ".mabip"  # In user home directory

    # File formats
    SENSOR_FILENAME_FORMAT: str = "Sensor_readings_{timestamp}.txt"
    AMUZA_LOG_FORMAT: str = "AMUZA-{timestamp}.log"
    TIMESTAMP_FORMAT: str = "%d_%m_%y_%H_%M"
    LOG_TIMESTAMP_FORMAT: str = "%Y-%m-%d_%H-%M-%S"
    SETTINGS_FILENAME: str = "settings.json"

    # Log rotation
    MAX_LOG_FILES: int = 10
    MAX_LOG_SIZE_MB: int = 10

    # Data format
    DATA_SEPARATOR: str = "\t"
    HEADER_ROWS_SKIP: int = 3


@dataclass
class AsyncConfig:
    """Async operation timeouts and settings."""
    # Task timeouts (seconds)
    OPERATION_TIMEOUT: float = 300.0  # 5 minutes max per operation
    TASK_TIMEOUT_S: float = 300.0
    SHUTDOWN_TIMEOUT: float = 5.0  # Wait for graceful shutdown
    CONNECTION_TIMEOUT: float = 30.0  # Bluetooth/serial connection
    
    # Sleep intervals
    INTERRUPTIBLE_SLEEP_CHECK: float = 0.5  # Check stop flag every 0.5s
    
    # Queue sizes
    COMMAND_QUEUE_SIZE: int = 100
    DATA_QUEUE_SIZE: int = 1000
    
    # Debug mode
    DEBUG_SLOW_CALLBACK_MS: float = 100.0  # Log callbacks slower than this


# Singleton instances
HARDWARE = HardwareConfig()
UI = UIConfig()
SENSOR = SensorConfig()
FILES = FileConfig()
ASYNC = AsyncConfig()
ASYNC_CONFIG = ASYNC
