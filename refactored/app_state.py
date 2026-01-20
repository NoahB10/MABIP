"""
Application state management - replaces global variables.
Thread-safe shared state using asyncio locks.

Includes:
- Connection state
- Well selections
- Timing parameters
- Calibration gains
- Temperature/heater settings
- Completed wells tracking
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, List
from datetime import datetime
from pathlib import Path


# Settings file location
SETTINGS_FILE = Path.home() / ".mabip" / "settings.json"


@dataclass
class AppState:
    """
    Centralized application state.
    All mutable state should be accessed with appropriate locks.
    """
    # AMUZA connection
    connection: Optional[object] = None
    connection_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Well plate selections
    selected_wells: Set[str] = field(default_factory=set)
    ctrl_selected_wells: Set[str] = field(default_factory=set)
    completed_wells: Set[str] = field(default_factory=set)
    selection_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Timing parameters (user configurable)
    t_buffer: int = 60
    t_sampling: int = 90
    timing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Temperature settings
    target_temperature: float = 37.0
    heater_enabled: bool = False
    temperature_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Calibration gains
    calibration_gains: Dict[str, float] = field(default_factory=lambda: {
        "Glutamate": 3.394,
        "Glutamine": 0.974,
        "Glucose": 1.5,
        "Lactate": 0.515,
    })
    calibration_values: Dict[str, float] = field(default_factory=lambda: {
        "Glutamate": 0.996,
        "Glutamine": 1.0,
        "Glucose": 17.38,
        "Lactate": 9.94,
    })
    calibration_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # Operation state
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    is_running: bool = False
    current_well_index: int = 0

    # Session tracking
    session_start: Optional[datetime] = None

    # Connection state
    async def set_connection(self, conn):
        """Thread-safe connection setter."""
        async with self.connection_lock:
            self.connection = conn

    async def get_connection(self):
        """Thread-safe connection getter."""
        async with self.connection_lock:
            return self.connection

    # Well selection
    async def add_selected_well(self, well_id: str, ctrl: bool = False):
        """Thread-safe well selection."""
        async with self.selection_lock:
            if ctrl:
                self.ctrl_selected_wells.add(well_id)
            else:
                self.selected_wells.add(well_id)

    async def remove_selected_well(self, well_id: str, ctrl: bool = False):
        """Thread-safe well deselection."""
        async with self.selection_lock:
            if ctrl:
                self.ctrl_selected_wells.discard(well_id)
            else:
                self.selected_wells.discard(well_id)

    async def clear_selections(self):
        """Thread-safe clear all selections."""
        async with self.selection_lock:
            self.selected_wells.clear()
            self.ctrl_selected_wells.clear()

    async def get_selected_wells(self, ctrl: bool = False) -> Set[str]:
        """Thread-safe get selected wells."""
        async with self.selection_lock:
            if ctrl:
                return self.ctrl_selected_wells.copy()
            else:
                return self.selected_wells.copy()

    # Completed wells tracking
    async def mark_well_completed(self, well_id: str):
        """Mark a well as completed."""
        async with self.selection_lock:
            self.completed_wells.add(well_id)

    async def clear_completed_wells(self):
        """Clear completed wells."""
        async with self.selection_lock:
            self.completed_wells.clear()

    async def get_completed_wells(self) -> Set[str]:
        """Get completed wells."""
        async with self.selection_lock:
            return self.completed_wells.copy()

    async def is_well_completed(self, well_id: str) -> bool:
        """Check if well is completed."""
        async with self.selection_lock:
            return well_id in self.completed_wells

    # Timing parameters
    async def get_timing_params(self):
        """Thread-safe timing getter."""
        async with self.timing_lock:
            return self.t_buffer, self.t_sampling

    async def set_timing_params(self, buffer: int, sampling: int):
        """Thread-safe timing setter."""
        async with self.timing_lock:
            self.t_buffer = buffer
            self.t_sampling = sampling

    # Temperature settings
    async def get_temperature_settings(self):
        """Get temperature and heater settings."""
        async with self.temperature_lock:
            return self.target_temperature, self.heater_enabled

    async def set_temperature_settings(self, temperature: float, heater_on: bool):
        """Set temperature and heater settings."""
        async with self.temperature_lock:
            self.target_temperature = temperature
            self.heater_enabled = heater_on

    # Calibration
    async def get_calibration_gains(self) -> Dict[str, float]:
        """Get calibration gains."""
        async with self.calibration_lock:
            return self.calibration_gains.copy()

    async def set_calibration_gains(self, gains: Dict[str, float]):
        """Set calibration gains."""
        async with self.calibration_lock:
            self.calibration_gains.update(gains)

    async def get_calibration_values(self) -> Dict[str, float]:
        """Get calibration expected values."""
        async with self.calibration_lock:
            return self.calibration_values.copy()

    async def set_calibration_values(self, values: Dict[str, float]):
        """Set calibration expected values."""
        async with self.calibration_lock:
            self.calibration_values.update(values)

    # Operation control
    async def request_stop(self):
        """Request operation stop (safe to call from any context)."""
        self.stop_event.set()

    async def clear_stop(self):
        """Clear stop request."""
        self.stop_event.clear()

    async def wait_stopped(self):
        """Wait until stop is requested."""
        await self.stop_event.wait()

    # Settings persistence
    async def save_settings(self):
        """Save current settings to file."""
        settings = {
            "t_buffer": self.t_buffer,
            "t_sampling": self.t_sampling,
            "target_temperature": self.target_temperature,
            "heater_enabled": self.heater_enabled,
            "calibration_gains": self.calibration_gains,
            "calibration_values": self.calibration_values,
        }

        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(SETTINGS_FILE, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save settings: {e}")

    async def load_settings(self):
        """Load settings from file."""
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, 'r') as f:
                    settings = json.load(f)

                if "t_buffer" in settings:
                    self.t_buffer = settings["t_buffer"]
                if "t_sampling" in settings:
                    self.t_sampling = settings["t_sampling"]
                if "target_temperature" in settings:
                    self.target_temperature = settings["target_temperature"]
                if "heater_enabled" in settings:
                    self.heater_enabled = settings["heater_enabled"]
                if "calibration_gains" in settings:
                    self.calibration_gains.update(settings["calibration_gains"])
                if "calibration_values" in settings:
                    self.calibration_values.update(settings["calibration_values"])

                print(f"Settings loaded from {SETTINGS_FILE}")
        except Exception as e:
            print(f"Warning: Could not load settings: {e}")

    def get_progress_info(self) -> dict:
        """Get current operation progress info."""
        total = len(self.selected_wells) + len(self.ctrl_selected_wells)
        completed = len(self.completed_wells)
        return {
            "total_wells": total,
            "completed_wells": completed,
            "current_index": self.current_well_index,
            "is_running": self.is_running,
        }
