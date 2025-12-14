"""
Application state management - replaces global variables.
Thread-safe shared state using asyncio locks.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Set
from datetime import datetime


@dataclass
class AppState:
    """
    Centralized application state.
    All mutable state should be accessed with state_lock.
    """
    # AMUZA connection
    connection: Optional[object] = None
    connection_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Well plate selections
    selected_wells: Set[str] = field(default_factory=set)
    ctrl_selected_wells: Set[str] = field(default_factory=set)
    selection_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Timing parameters (user configurable)
    t_buffer: int = 60
    t_sampling: int = 90
    timing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    
    # Operation state
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    is_running: bool = False
    current_well_index: int = 0
    
    # Session tracking
    session_start: Optional[datetime] = None
    
    async def set_connection(self, conn):
        """Thread-safe connection setter."""
        async with self.connection_lock:
            self.connection = conn
    
    async def get_connection(self):
        """Thread-safe connection getter."""
        async with self.connection_lock:
            return self.connection
    
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
    
    async def get_selected_wells(self, ctrl: bool = False):
        """Thread-safe get selected wells."""
        async with self.selection_lock:
            if ctrl:
                return self.ctrl_selected_wells.copy()
            else:
                return self.selected_wells.copy()
    
    async def get_timing_params(self):
        """Thread-safe timing getter."""
        async with self.timing_lock:
            return self.t_buffer, self.t_sampling
    
    async def set_timing_params(self, buffer: int, sampling: int):
        """Thread-safe timing setter."""
        async with self.timing_lock:
            self.t_buffer = buffer
            self.t_sampling = sampling
    
    async def request_stop(self):
        """Request operation stop (safe to call from any context)."""
        self.stop_event.set()
    
    async def clear_stop(self):
        """Clear stop request."""
        self.stop_event.clear()
    
    async def wait_stopped(self):
        """Wait until stop is requested."""
        await self.stop_event.wait()
