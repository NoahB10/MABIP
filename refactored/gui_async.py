"""
Async version of AMUZA GUI using qasync for PyQt5 async integration.

Key improvements:
- Uses qasync for proper async/await in PyQt5
- @asyncSlot decorators for signal handlers
- AppState for thread-safe state management
- AsyncTaskManager for background task tracking
- Incremental file reading for plot updates
- Proper resource cleanup
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Set
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QMessageBox, QDialog,
    QDialogButtonBox, QFormLayout, QSpinBox, QDoubleSpinBox, QCheckBox,
    QTextEdit, QSizePolicy, QFileDialog, QListWidget, QAction, QMenuBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPalette, QColor

import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

# Serial port listing
try:
    from serial.tools import list_ports
    SERIAL_PORTS_AVAILABLE = True
except ImportError:
    SERIAL_PORTS_AVAILABLE = False

# Import qasync
try:
    import qasync
    from qasync import asyncSlot, QEventLoop
    QASYNC_AVAILABLE = True
except ImportError:
    QASYNC_AVAILABLE = False
    # Fallback decorators
    def asyncSlot(*args):
        def decorator(func):
            return func
        return decorator if args and callable(args[0]) else decorator

# Import our async modules
from config import HARDWARE, UI, SENSOR, FILES, ASYNC_CONFIG
from app_state import AppState
from async_utils import AsyncTaskManager, interruptible_sleep
from amuza_async import AsyncAmuzaConnection, Method, Sequence
from sensor_reader_async import AsyncPotentiostatReader


logger = logging.getLogger(__name__)


class WellLabel(QLabel):
    """Interactive well label for well plate selection"""

    COMPLETED_COLOR = "#ffd700"  # Gold color for completed wells

    def __init__(self, well_id: str, row: int, col: int, size: int):
        super().__init__(well_id)
        self.well_id = well_id
        self.row = row
        self.col = col
        self.is_selected = False
        self.is_ctrl_selected = False
        self.is_completed = False

        # Styling
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(size, size)
        self.setStyleSheet("""
            QLabel {
                border: 1px solid black;
                background-color: white;
                border-radius: 0px;
                font-weight: 600;
            }
            QLabel:hover {
                background-color: #e9f2ff;
                border-color: #4a90e2;
            }
        """)
        # Mouse events will be handled by parent widget
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)

    def set_selected(self, selected: bool):
        """Update selection state"""
        self.is_selected = selected
        self._update_appearance()

    def set_ctrl_selected(self, ctrl_selected: bool):
        """Update control selection state"""
        self.is_ctrl_selected = ctrl_selected
        self._update_appearance()

    def set_completed(self, completed: bool):
        """Update completed state"""
        self.is_completed = completed
        self._update_appearance()

    def _update_appearance(self):
        """Update visual appearance based on state"""
        if self.is_completed:
            color = self.COMPLETED_COLOR
            text_color = "black"
        elif self.is_ctrl_selected:
            color = UI.CTRL_WELL_COLOR
            text_color = "black"
        elif self.is_selected:
            color = UI.SELECTED_WELL_COLOR
            text_color = "black"
        else:
            color = "white"
            text_color = "black"

        self.setStyleSheet(f"""
            QLabel {{
                border: 1px solid black;
                background-color: {color};
                border-radius: 0px;
                font-weight: 600;
                color: {text_color};
            }}
            QLabel:hover {{
                background-color: {color};
                border-color: #4a90e2;
            }}
        """)


class PlotWindow(QMainWindow):
    """
    Real-time plotting window with OPTIMIZED rendering.

    Optimizations:
    - Pre-created line artists with set_data() instead of clear/replot
    - draw_idle() for deferred rendering
    - Direct data callback support (no file polling needed)
    - Efficient numpy-based rolling window
    - Proper y-axis scaling (allows negative values)
    """

    # Maximum points to keep in memory for rolling display
    MAX_POINTS = 10000

    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state

        # File reading state (for backward compatibility)
        self.data_file = None
        self.last_file_position = 0
        self.last_line_count = 0
        self.cached_data = pd.DataFrame()
        self.full_data = pd.DataFrame()
        self.loaded_file_path = None
        self.header_lines_skipped = False

        # OPTIMIZED: Use deque with maxlen for efficient bounded data storage
        # This automatically discards old data when limit is reached - no memory growth!
        from collections import deque
        self._time_data = deque(maxlen=self.MAX_POINTS)  # Time in seconds
        self._channel_data = {i: deque(maxlen=self.MAX_POINTS) for i in range(1, 8)}  # Channels 1-7

        # Track if we're receiving data via callback (skip file polling if so)
        self._using_callback = False

        # Plot configuration
        self.rolling_window_minutes = UI.PLOT_WINDOW_MINUTES
        self.show_full_graph = False

        # Metabolite gain values (from app_state)
        self.gain_values = app_state.calibration_gains.copy()

        # OPTIMIZED: Pre-created line artists (will be set in _init_ui)
        self._lines = {}  # {metabolite_name: Line2D}
        self._needs_full_redraw = True  # Flag for when we need to redraw everything

        # User interaction tracking - when True, don't auto-scroll
        self._user_panned = False

        self._init_ui()

        # Update timer - only used for file polling fallback
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_update)
        self.timer.start(UI.PLOT_UPDATE_INTERVAL_MS)

        logger.info("PlotWindow initialized with optimized rendering")

    def update_gains(self, gains: dict):
        """Update gain values and refresh plot"""
        self.gain_values.update(gains)
        self._needs_full_redraw = True
        self._update_plots()
        logger.info(f"Plot gains updated: {self.gain_values}")

    def set_sensor_file(self, file_path: str):
        """Set the sensor data file to read from and reset reading state"""
        self.data_file = file_path
        self.last_file_position = 0
        self.last_line_count = 0
        self.header_lines_skipped = False
        self.cached_data = pd.DataFrame()
        self.full_data = pd.DataFrame()
        self.loaded_file_path = None
        self._clear_data_arrays()
        logger.info(f"PlotWindow now reading from: {file_path}")

    def add_reading(self, reading):
        """
        OPTIMIZED: Add a single reading directly from sensor callback.
        This is more efficient than file polling.
        Uses deque with maxlen - automatically discards old data, no memory growth!
        """
        self._using_callback = True  # Mark that we're using callback mode

        # Add to internal deques (auto-trimmed by maxlen)
        time_seconds = reading.time_minutes * 60
        self._time_data.append(time_seconds)

        for i, ch_val in enumerate(reading.channels[:6], start=1):
            self._channel_data[i].append(ch_val)

        # Temperature as channel 7
        self._channel_data[7].append(reading.temperature)

        # No manual trimming needed - deque maxlen handles it automatically!

    def _clear_data_arrays(self):
        """Clear internal data arrays (deques)"""
        from collections import deque
        self._time_data = deque(maxlen=self.MAX_POINTS)
        self._channel_data = {i: deque(maxlen=self.MAX_POINTS) for i in range(1, 8)}
        self._needs_full_redraw = True
        # Don't reset _using_callback - sensor may still be running
    
    def _init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Sensor Data - Real-time")
        self.setGeometry(100, 100, UI.PLOT_WINDOW_WIDTH, UI.PLOT_WINDOW_HEIGHT)
        
        # Menu bar
        menu_bar = self.menuBar()
        
        # File Menu
        file_menu = menu_bar.addMenu("File")
        
        load_action = QAction("Load Saved", self)
        load_action.triggered.connect(self._on_load_file)
        file_menu.addAction(load_action)
        
        save_action = QAction("Save As", self)
        save_action.triggered.connect(self._on_save_file)
        file_menu.addAction(save_action)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QVBoxLayout(main_widget)
        
        # Create matplotlib figure with single plot
        self.figure = Figure(figsize=(12, 8))
        self.canvas = FigureCanvas(self.figure)
        
        # Navigation toolbar for zoom/pan
        self.nav_toolbar = NavigationToolbar(self.canvas, self)

        # Override the home button action to show full graph from 0
        # Find and disconnect the home action, then reconnect to our custom method
        for action in self.nav_toolbar.actions():
            if action.text() == 'Home':
                action.triggered.disconnect()
                action.triggered.connect(self._on_home_clicked)
                break

        # Connect to axes change events to detect user pan/zoom
        self.ax_xlim_changed_cid = None
        self.ax_ylim_changed_cid = None

        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        
        # Create single subplot for all channels
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Metabolite Concentrations")
        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Concentration (mM)")
        self.ax.grid(True, alpha=0.3)

        # OPTIMIZED: Pre-create line artists for each metabolite
        # This avoids recreating them on every update
        colors = {'Glutamate': 'b', 'Glutamine': 'g', 'Glucose': 'r', 'Lactate': 'c'}
        for metabolite, color in colors.items():
            line, = self.ax.plot([], [], color=color, linewidth=1, label=metabolite)
            self._lines[metabolite] = line

        self.ax.legend()
        self.figure.tight_layout()

        # Connect to navigation toolbar events to detect user pan/zoom
        # When user interacts, pause auto-scrolling
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        
        # Control buttons
        button_layout = QHBoxLayout()

        self.auto_follow_btn = QPushButton("Auto-Follow")
        self.auto_follow_btn.setToolTip("Resume auto-scrolling after pan/zoom")
        self.auto_follow_btn.clicked.connect(self._on_auto_follow)
        button_layout.addWidget(self.auto_follow_btn)

        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.clicked.connect(self._on_clear_data)
        button_layout.addWidget(self.clear_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self._on_export_data)
        button_layout.addWidget(self.export_btn)

        layout.addLayout(button_layout)

    def _on_auto_follow(self):
        """Resume auto-scrolling to latest data"""
        self._user_panned = False
        self.show_full_graph = False  # Back to rolling window mode
        self._update_plots()
    
    def _on_home_clicked(self):
        """Custom home button handler - shows full graph from 0 and resumes auto-scroll"""
        self.show_full_graph = True
        self._user_panned = False  # Resume auto-scrolling
        self._update_plots()

    def _on_mouse_release(self, event):
        """Detect when user finishes panning/zooming"""
        # Check if pan or zoom mode is active in the toolbar
        if self.nav_toolbar.mode in ('pan/zoom', 'zoom rect'):
            self._user_panned = True
            logger.debug("User panned - auto-scroll paused. Click 'Home' or 'Auto-Follow' to resume.")
    
    def _on_timer_update(self):
        """Timer callback for plot updates"""
        # Use asyncio to run update
        loop = asyncio.get_event_loop()
        loop.create_task(self._update_plot_async())
    
    async def _update_plot_async(self):
        """Async plot update - handles both callback mode and file polling fallback"""

        # If using callback mode, just update the plot (data already in deques)
        # No file polling needed - much more efficient and memory-safe!
        if self._using_callback:
            self._update_plots()
            return

        # FALLBACK: File polling mode (only used when callback not active)
        if not self.data_file:
            return

        try:
            # Read new data incrementally from file
            new_data = await self._read_new_data()

            if new_data is not None and not new_data.empty:
                # NOTE: We do NOT accumulate full_data anymore!
                # The sensor file already has complete data - no need to duplicate in memory.
                # This prevents memory exhaustion during long sessions.

                # Only keep cached_data for display (trimmed to rolling window)
                self.cached_data = pd.concat([self.cached_data, new_data], ignore_index=True)

                # Trim to rolling window to prevent memory growth
                if not self.cached_data.empty and not self.show_full_graph:
                    current_time = self.cached_data['Time'].max()
                    cutoff_time = current_time - (self.rolling_window_minutes * 60)
                    self.cached_data = self.cached_data[self.cached_data['Time'] >= cutoff_time]

                    # Also limit total rows as safety net
                    if len(self.cached_data) > self.MAX_POINTS:
                        self.cached_data = self.cached_data.tail(self.MAX_POINTS)

                # Update plots
                self._update_plots()

        except Exception as e:
            logger.error(f"Error updating plot: {e}")
    
    def _auto_save_raw_data(self, new_data: pd.DataFrame):
        """No longer needed - sensor reader saves data directly"""
        # The AsyncPotentiostatReader already saves data to the output file
        # This method is kept for backwards compatibility but does nothing
        pass
    
    async def _read_new_data(self) -> Optional[pd.DataFrame]:
        """
        Read only new lines from sensor file since last read.

        Expects tab-separated legacy format:
        Line 1: Created: MM/DD/YYYY\tHH:MM:SS AM/PM
        Line 2: counter\tt[min]\t#1ch1\t#1ch2\t...
        Line 3: Start: MM/DD/YYYY\tHH:MM:SS AM/PM
        Line 4+: Data rows (counter\ttime\tch1\tch2\t...)

        Returns:
            DataFrame with new data, or None if no new data
        """
        if not self.data_file:
            return None

        try:
            import aiofiles
            from pathlib import Path

            # Check if file exists
            if not Path(self.data_file).exists():
                return None

            async with aiofiles.open(self.data_file, 'r') as f:
                # Read all lines
                all_lines = await f.readlines()

            # Skip header (first 3 lines) and get only new data lines
            data_start_line = 3  # Lines 0, 1, 2 are header

            # If no new lines since last read, return None
            total_data_lines = len(all_lines) - data_start_line
            if total_data_lines <= 0:
                return None

            # Check if we have new data
            new_line_count = total_data_lines
            if new_line_count <= self.last_line_count:
                return None

            # Get only new lines (from last_line_count to end)
            new_lines = all_lines[data_start_line + self.last_line_count:]
            self.last_line_count = new_line_count

            # Parse tab-separated data lines
            data_rows = []
            for line in new_lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split('\t')

                # Need at least: counter, time, ch1-ch7
                if len(parts) < 9:
                    continue

                try:
                    # Parse counter and time
                    counter = int(parts[0])
                    time_min = float(parts[1])
                    time_seconds = time_min * 60  # Convert to seconds for internal use

                    # Parse channels 1-7 (indices 2-8)
                    row_data = {'Time': time_seconds}
                    for ch in range(1, 8):
                        col_idx = ch + 1  # Channel 1 is at index 2
                        if col_idx < len(parts):
                            try:
                                row_data[f'Channel {ch}'] = float(parts[col_idx])
                            except ValueError:
                                row_data[f'Channel {ch}'] = 0.0
                        else:
                            row_data[f'Channel {ch}'] = 0.0

                    data_rows.append(row_data)

                except (ValueError, IndexError) as e:
                    logger.debug(f"Skipping invalid line: {e}")
                    continue

            if data_rows:
                logger.debug(f"Read {len(data_rows)} new data rows from sensor file")
                return pd.DataFrame(data_rows)

            return None

        except FileNotFoundError:
            return None
        except Exception as e:
            logger.error(f"Error reading sensor data file: {e}")
            return None
    
    def _update_plots(self):
        """
        OPTIMIZED: Update plot using line.set_data() instead of clear/replot.
        This is much faster for real-time updates.
        """
        import numpy as np

        # Determine data source: internal arrays or cached DataFrame
        if self._time_data:
            # Use internal arrays (from direct callback)
            time_arr = np.array(self._time_data)
            channels = {i: np.array(self._channel_data[i]) for i in range(1, 7)}
        elif not self.cached_data.empty and 'Channel 1' in self.cached_data.columns:
            # Use cached DataFrame (from file polling)
            time_arr = self.cached_data['Time'].values
            channels = {i: self.cached_data[f'Channel {i}'].values for i in range(1, 7)}
        else:
            # No data
            return

        if len(time_arr) == 0:
            return

        # Convert time to relative minutes
        start_time = time_arr.min()
        time_minutes = (time_arr - start_time) / 60.0

        # Calculate metabolites from channel differences
        metabolites = {}
        if 1 in channels and 2 in channels:
            metabolites['Glutamate'] = (channels[1] - channels[2]) * self.gain_values.get('Glutamate', 1.0)
        if 3 in channels and 1 in channels:
            metabolites['Glutamine'] = (channels[3] - channels[1]) * self.gain_values.get('Glutamine', 1.0)
        if 5 in channels and 4 in channels:
            metabolites['Glucose'] = (channels[5] - channels[4]) * self.gain_values.get('Glucose', 1.0)
        if 6 in channels and 4 in channels:
            metabolites['Lactate'] = (channels[6] - channels[4]) * self.gain_values.get('Lactate', 1.0)

        # Apply rolling window filter to time
        max_time = time_minutes.max() if len(time_minutes) > 0 else 0

        if self.show_full_graph:
            mask = np.ones(len(time_minutes), dtype=bool)
            x_min, x_max = 0, max(1, max_time)
        else:
            if max_time > self.rolling_window_minutes:
                cutoff = max_time - self.rolling_window_minutes
                mask = time_minutes >= cutoff
                x_min, x_max = cutoff, max_time
            else:
                mask = np.ones(len(time_minutes), dtype=bool)
                x_min, x_max = 0, max(self.rolling_window_minutes, max_time)

        # OPTIMIZED: Update line data without clearing
        filtered_time = time_minutes[mask]

        y_min, y_max = float('inf'), float('-inf')

        for metabolite, line in self._lines.items():
            if metabolite in metabolites:
                filtered_values = metabolites[metabolite][mask]
                line.set_data(filtered_time, filtered_values)

                # Track y-range for autoscaling
                if len(filtered_values) > 0:
                    y_min = min(y_min, np.nanmin(filtered_values))
                    y_max = max(y_max, np.nanmax(filtered_values))
            else:
                line.set_data([], [])

        # Update axis limits ONLY if user hasn't manually panned
        if not self._user_panned:
            self.ax.set_xlim(x_min, x_max)

            # Auto-scale Y axis with padding (allow negative values!)
            if y_min != float('inf') and y_max != float('-inf'):
                y_range = y_max - y_min
                padding = max(0.1, y_range * 0.1)  # At least 0.1 padding
                self.ax.set_ylim(y_min - padding, y_max + padding)

        # Use draw_idle for deferred rendering (more efficient)
        self.canvas.draw_idle()
    
    def _on_clear_data(self):
        """
        Clear PLOT display only - does NOT affect the sensor log file.
        The sensor continues recording to the same file in the background.
        After clear, the plot shows only new data from this point onwards.
        """
        # Clear display-related caches (NOT the sensor file!)
        self.cached_data = pd.DataFrame()
        self.full_data = pd.DataFrame()
        self.last_file_position = 0
        self.last_line_count = 0
        self.header_lines_skipped = False
        self.loaded_file_path = None
        self.show_full_graph = False
        self._user_panned = False

        # Clear internal plot arrays - sensor file continues unaffected
        self._clear_data_arrays()

        # Reset line data on plot
        for line in self._lines.values():
            line.set_data([], [])

        self.ax.set_xlim(0, self.rolling_window_minutes)
        self.ax.set_ylim(0, 1)

        self.canvas.draw_idle()
        logger.info("Plot display cleared (sensor file continues recording)")
    
    def _on_export_data(self):
        """
        Export what's currently shown on the plot to CSV.
        This exports the internal arrays (post-clear data if cleared).
        The full sensor log file is always preserved separately.
        """
        import numpy as np

        # Check if we have data in internal arrays (from direct callback)
        if not self._time_data:
            # Fallback to cached_data if available
            if self.cached_data.empty:
                QMessageBox.warning(self, "No Data", "No data to export. The plot is empty.")
                return
            # Use cached data
            export_data = pd.DataFrame()
            export_data['Time (min)'] = self.cached_data['RelativeTime'] if 'RelativeTime' in self.cached_data.columns else self.cached_data['Time'] / 60
            for ch in range(1, 7):
                col_name = f'Channel {ch}'
                if col_name in self.cached_data.columns:
                    export_data[col_name] = self.cached_data[col_name]
            if 'Channel 7' in self.cached_data.columns:
                export_data['Temperature'] = self.cached_data['Channel 7']
        else:
            # Export from internal arrays (what's shown on plot)
            time_arr = np.array(self._time_data)
            start_time = time_arr.min() if len(time_arr) > 0 else 0
            time_minutes = (time_arr - start_time) / 60.0

            export_data = pd.DataFrame()
            export_data['Time (min)'] = time_minutes

            for ch in range(1, 7):
                if ch in self._channel_data:
                    export_data[f'Channel {ch}'] = self._channel_data[ch]

            if 7 in self._channel_data:
                export_data['Temperature'] = self._channel_data[7]

        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_data.to_csv(filename, index=False)
        QMessageBox.information(
            self, "Export Complete",
            f"Plot data exported to {filename}\n\n"
            f"Note: Full sensor log is always preserved in the Sensor_Readings folder."
        )
        logger.info(f"Plot data exported to {filename}")
    
    def _on_load_file(self):
        """Load data from a saved .txt or .csv file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", 
            "All Supported (*.txt *.csv);;Text Files (*.txt);;CSV Files (*.csv);;All Files (*)"
        )
        if not file_path:
            return
        
        try:
            if file_path.lower().endswith('.csv'):
                # Load CSV file (exported format)
                self._load_csv_file(file_path)
            else:
                # Load TXT file (original format)
                self._load_txt_file(file_path)
            
            # Track loaded file and stop auto-reading from sensor
            self.loaded_file_path = file_path
            self.data_file = None  # Stop auto-reading from sensor file
            self.last_file_position = 0
            self.last_line_count = 0
            self.header_lines_skipped = False
            
            # Show full graph when loading a saved file
            self.show_full_graph = True
            self._update_plots()
            
            QMessageBox.information(self, "File Loaded", f"Loaded {len(self.cached_data)} data points from {file_path}")
            logger.info(f"Loaded data from {file_path}")
            
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            QMessageBox.critical(self, "Load Error", f"Failed to load file: {e}")
    
    def _load_csv_file(self, file_path: str):
        """Load CSV file (exported channel format)"""
        df = pd.read_csv(file_path)
        
        self.cached_data = pd.DataFrame()
        self.full_data = pd.DataFrame()
        
        # Check format based on columns
        if 'Time (min)' in df.columns:
            # Exported format with Time (min)
            time_values = df['Time (min)'].values * 60  # Convert to seconds
            plot_data = {'Time': time_values}
            
            # Load individual channels
            for ch in range(1, 7):
                col_name = f'Channel {ch}'
                if col_name in df.columns:
                    plot_data[col_name] = df[col_name].values
            
            # Load temperature as Channel 7
            if 'Temperature' in df.columns:
                plot_data['Channel 7'] = df['Temperature'].values
            
            self.cached_data = pd.DataFrame(plot_data)
        elif 'Time' in df.columns:
            # Raw format with Time in seconds
            plot_data = {'Time': df['Time'].values}
            
            for ch in range(1, 8):
                col_name = f'Channel {ch}'
                if col_name in df.columns:
                    plot_data[col_name] = df[col_name].values
            
            if 'Temperature' in df.columns:
                plot_data['Channel 7'] = df['Temperature'].values
            
            self.cached_data = pd.DataFrame(plot_data)
        else:
            raise ValueError("Unrecognized CSV format - expected 'Time (min)' or 'Time' column")
        
        self.full_data = self.cached_data.copy()
    
    def _load_txt_file(self, file_path: str):
        """Load TXT file (original tab-separated format)"""
        with open(file_path, "r", newline="", encoding='utf-8') as file:
            lines = file.readlines()
        
        if len(lines) < 4:
            raise ValueError("Insufficient data in file")
        
        # Parse tab-separated data
        data = [line.strip().split("\t") for line in lines]
        
        # Clean data like original: take first 9 columns (0-8), skip first 3 rows
        df = pd.DataFrame(data)
        # Use iloc for positional indexing - take first 9 columns
        df = df.iloc[:, :9]
        # Skip first 3 rows (Created, header, Start lines)
        df = df.iloc[3:]
        # Reset index after slicing
        df = df.reset_index(drop=True)
        # Set column names
        df.columns = ['counter', 't[min]', '#1ch1', '#1ch2', '#1ch3', '#1ch4', '#1ch5', '#1ch6', '#1ch7']
        
        # Remove comments at the end if they appear
        end_idx = len(df)
        for i in range(len(df)):
            a = str(df.iloc[i]['counter'])
            if not a.isdigit():
                end_idx = i
                break
        
        if end_idx < len(df):
            df = df.iloc[:end_idx]
        
        # Convert to numeric
        df = df.apply(pd.to_numeric, errors="coerce")
        
        # Convert to our internal format for plotting
        self.cached_data = pd.DataFrame()
        self.full_data = pd.DataFrame()
        
        time_values = df['t[min]'].values * 60  # Convert minutes to seconds
        
        plot_data = {'Time': time_values}
        for ch in range(1, 8):
            col_name = f'#1ch{ch}'
            if col_name in df.columns:
                plot_data[f'Channel {ch}'] = df[col_name].values
        
        self.cached_data = pd.DataFrame(plot_data)
        self.full_data = self.cached_data.copy()
    
    def _on_save_file(self):
        """Save a copy of the current data file to a specified location (like original)"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", "Text Files (*.txt)"
        )
        if not file_path:
            return
        
        if not file_path.endswith(".txt"):
            QMessageBox.warning(self, "Warning", "Please use a .txt extension to save the data.")
            return
        
        try:
            # If we have a sensor file connected, copy it
            if self.data_file and Path(self.data_file).exists():
                with open(self.data_file, "r", encoding='utf-8') as source_file:
                    with open(file_path, "w", encoding='utf-8') as dest_file:
                        dest_file.write(source_file.read())
                QMessageBox.information(self, "Success", f"Data successfully saved to {file_path}")
                logger.info(f"Saved sensor data to {file_path}")
            # If we loaded a file, save that
            elif self.loaded_file_path and Path(self.loaded_file_path).exists():
                with open(self.loaded_file_path, "r", encoding='utf-8') as source_file:
                    with open(file_path, "w", encoding='utf-8') as dest_file:
                        dest_file.write(source_file.read())
                QMessageBox.information(self, "Success", f"Data successfully saved to {file_path}")
                logger.info(f"Saved loaded file to {file_path}")
            # Otherwise generate from full_data
            elif not self.full_data.empty:
                self._save_data_to_file(file_path, self.full_data)
                QMessageBox.information(self, "Success", f"Data successfully saved to {file_path}")
                logger.info(f"Generated and saved data to {file_path}")
            else:
                QMessageBox.warning(self, "Warning", "No data is available to save.")
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save file: {e}")
    
    def _save_data_to_file(self, file_path: str, data: pd.DataFrame):
        """Save data in original tab-separated format"""
        with open(file_path, 'w', encoding='utf-8') as f:
            # Write header with creation time
            now = datetime.now()
            f.write(f"Created: {now.strftime('%m/%d/%Y')}\t{now.strftime('%I:%M:%S %p')}\n")
            
            # Write column headers
            # Format: counter\tt[min]\t#1ch1\t#1ch2\t...
            cols = ['counter', 't[min]']
            for ch in range(1, 8):  # channels 1-7
                cols.append(f'#1ch{ch}')
            f.write('\t'.join(cols) + '\n')
            
            # Write start time
            f.write(f"Start: {now.strftime('%m/%d/%Y')}\t{now.strftime('%I:%M:%S %p')}\n")
            
            # Write data rows
            for idx, row in data.iterrows():
                counter = idx + 1
                time_min = row['Time'] / 60.0 if 'Time' in row else 0
                values = [str(counter), f"{time_min:.4f}"]
                
                # Get channel values
                for ch in range(1, 8):
                    col_name = f'Channel {ch}'
                    if col_name in row:
                        values.append(f"{row[col_name]:.2f}")
                    else:
                        values.append("0.00")
                
                f.write('\t'.join(values) + '\n')
    
    def closeEvent(self, event):
        """Handle window close"""
        self.timer.stop()
        event.accept()


class SettingsDialog(QDialog):
    """Settings dialog for timing and temperature parameters"""

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._init_ui()

    def _init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Settings")
        self.setMinimumWidth(350)

        layout = QVBoxLayout(self)

        # Timing section
        layout.addWidget(QLabel("<b>Timing Settings</b>"))

        timing_form = QFormLayout()

        # Sampling time first (time spent at each well)
        sampling_layout = QHBoxLayout()
        self.sampling_spin = QSpinBox()
        self.sampling_spin.setRange(1, 3600)
        self.sampling_spin.setValue(self.app_state.t_sampling)
        self.sampling_spin.setToolTip("Time to spend at each well (1-3600 seconds)")
        sampling_layout.addWidget(self.sampling_spin)
        sampling_layout.addWidget(QLabel("seconds"))
        sampling_layout.addStretch()
        timing_form.addRow("Sampling Time:", sampling_layout)

        # Buffer time second (time before moving to next well)
        buffer_layout = QHBoxLayout()
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(1, 3600)
        self.buffer_spin.setValue(self.app_state.t_buffer)
        self.buffer_spin.setToolTip("Time to wait before moving to next well (1-3600 seconds)")
        buffer_layout.addWidget(self.buffer_spin)
        buffer_layout.addWidget(QLabel("seconds"))
        buffer_layout.addStretch()
        timing_form.addRow("Buffer Time:", buffer_layout)

        layout.addLayout(timing_form)

        # Temperature section
        layout.addWidget(QLabel("<b>Temperature Settings</b>"))

        temp_form = QFormLayout()

        # Temperature
        temp_layout = QHBoxLayout()
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.0, 50.0)
        self.temp_spin.setDecimals(1)
        self.temp_spin.setSingleStep(0.5)
        self.temp_spin.setValue(self.app_state.target_temperature)
        self.temp_spin.setToolTip("Target temperature (0-50 C)")
        temp_layout.addWidget(self.temp_spin)
        temp_layout.addWidget(QLabel("C"))
        temp_layout.addStretch()
        temp_form.addRow("Target Temperature:", temp_layout)

        # Heater checkbox
        self.heater_checkbox = QCheckBox("Enable Heater")
        self.heater_checkbox.setChecked(self.app_state.heater_enabled)
        self.heater_checkbox.setToolTip("Turn on the heater to maintain temperature")
        temp_form.addRow("", self.heater_checkbox)

        layout.addLayout(temp_form)

        # Validation message
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: red;")
        layout.addWidget(self.validation_label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate_and_accept(self):
        """Validate inputs before accepting"""
        buffer = self.buffer_spin.value()
        sampling = self.sampling_spin.value()
        temp = self.temp_spin.value()

        # Validation
        if sampling < 1:
            self.validation_label.setText("Sampling time must be at least 1 second")
            return
        if buffer < 1:
            self.validation_label.setText("Buffer time must be at least 1 second")
            return
        if temp < 0 or temp > 50:
            self.validation_label.setText("Temperature must be between 0 and 50 C")
            return

        self.accept()

    def get_values(self):
        """Get current values"""
        return {
            'buffer': self.buffer_spin.value(),
            'sampling': self.sampling_spin.value(),
            'temperature': self.temp_spin.value(),
            'heater_enabled': self.heater_checkbox.isChecked()
        }


class SensorConnectDialog(QDialog):
    """Dialog for selecting and connecting to a sensor serial port"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected_port = None
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Connect to Sensor")
        self.setMinimumWidth(350)
        
        layout = QVBoxLayout(self)
        
        # Port selection
        layout.addWidget(QLabel("Available Serial Ports:"))
        
        self.port_list = QListWidget()
        self._refresh_ports()
        layout.addWidget(self.port_list)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Ports")
        refresh_btn.clicked.connect(self._refresh_ports)
        layout.addWidget(refresh_btn)
        
        # Mock mode checkbox
        self.mock_checkbox = QCheckBox("Use Mock Mode (for testing)")
        layout.addWidget(self.mock_checkbox)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        connect_btn = QPushButton("Connect")
        connect_btn.clicked.connect(self._on_connect)
        button_layout.addWidget(connect_btn)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)
        
        layout.addLayout(button_layout)
    
    def _refresh_ports(self):
        """Refresh the list of available serial ports"""
        self.port_list.clear()
        
        if SERIAL_PORTS_AVAILABLE:
            ports = list_ports.comports()
            for port in ports:
                self.port_list.addItem(f"{port.device} - {port.description}")
        else:
            self.port_list.addItem("COM1 (serial.tools not available)")
            self.port_list.addItem("COM2 (serial.tools not available)")
            self.port_list.addItem("COM3 (serial.tools not available)")
    
    def _on_connect(self):
        """Handle connect button"""
        if self.mock_checkbox.isChecked():
            self.selected_port = "MOCK"
            self.accept()
            return
        
        current_item = self.port_list.currentItem()
        if current_item:
            # Extract port name (before the dash)
            self.selected_port = current_item.text().split(" - ")[0].strip()
            self.accept()
        else:
            QMessageBox.warning(self, "No Port Selected", "Please select a port or use mock mode")
    
    def get_selected_port(self):
        """Get the selected port"""
        return self.selected_port


class CalibrationSettingsDialog(QDialog):
    """Dialog for adjusting calibration values for each metabolite"""

    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._init_ui()

    def _init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Calibration Settings")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)

        # Gain values section
        layout.addWidget(QLabel("<b>Gain Values (scaling factors):</b>"))

        gain_group = QFormLayout()
        self.gain_inputs = {}

        # Use current gains from app_state
        current_gains = self.app_state.calibration_gains

        for metabolite in ["Glutamate", "Glutamine", "Glucose", "Lactate"]:
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 100.0)
            spin.setDecimals(3)
            spin.setValue(current_gains.get(metabolite, SENSOR.DEFAULT_GAINS.get(metabolite, 1.0)))
            spin.setToolTip(f"Scaling factor for {metabolite} measurement")
            gain_group.addRow(f"{metabolite}:", spin)
            self.gain_inputs[metabolite] = spin

        layout.addLayout(gain_group)

        # Calibration values section
        layout.addWidget(QLabel("<b>Expected Concentration Values (mM):</b>"))

        cal_group = QFormLayout()
        self.calibration_inputs = {}

        # Use current calibration values from app_state
        current_calibrations = self.app_state.calibration_values

        for metabolite in ["Glutamate", "Glutamine", "Glucose", "Lactate"]:
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 1000.0)
            spin.setDecimals(3)
            spin.setValue(current_calibrations.get(metabolite, SENSOR.DEFAULT_CALIBRATIONS.get(metabolite, 1.0)))
            spin.setToolTip(f"Expected concentration for {metabolite} calibration standard")
            cal_group.addRow(f"{metabolite}:", spin)
            self.calibration_inputs[metabolite] = spin

        layout.addLayout(cal_group)

        # Validation message
        self.validation_label = QLabel("")
        self.validation_label.setStyleSheet("color: red;")
        layout.addWidget(self.validation_label)

        # Buttons
        button_layout = QHBoxLayout()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)

        button_layout.addStretch()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        button_layout.addWidget(buttons)

        layout.addLayout(button_layout)

    def _reset_to_defaults(self):
        """Reset all values to defaults"""
        for metabolite, spin in self.gain_inputs.items():
            spin.setValue(SENSOR.DEFAULT_GAINS.get(metabolite, 1.0))
        for metabolite, spin in self.calibration_inputs.items():
            spin.setValue(SENSOR.DEFAULT_CALIBRATIONS.get(metabolite, 1.0))

    def _validate_and_accept(self):
        """Validate inputs before accepting"""
        # Check all gains are positive
        for metabolite, spin in self.gain_inputs.items():
            if spin.value() <= 0:
                self.validation_label.setText(f"{metabolite} gain must be positive")
                return

        # Check all calibrations are positive
        for metabolite, spin in self.calibration_inputs.items():
            if spin.value() <= 0:
                self.validation_label.setText(f"{metabolite} calibration must be positive")
                return

        self.accept()

    def get_values(self):
        """Get current calibration values"""
        return {
            'gains': {k: v.value() for k, v in self.gain_inputs.items()},
            'calibrations': {k: v.value() for k, v in self.calibration_inputs.items()}
        }


class AsyncAMUZAGUI(QMainWindow):
    """
    Main async GUI for AMUZA control.
    
    Uses qasync for proper async/await integration with PyQt5.
    """
    
    def __init__(self, app_state: AppState, task_manager: AsyncTaskManager):
        super().__init__()
        
        self.app_state = app_state
        self.task_manager = task_manager
        self.cell_size = 42
        
        # Connection
        self.connection: Optional[AsyncAmuzaConnection] = None
        self.sensor_reader: Optional[AsyncPotentiostatReader] = None
        
        # Wells
        self.well_labels: Dict[str, WellLabel] = {}
        self.drag_start: Optional[tuple[int, int]] = None
        self.drag_active: bool = False
        self._pending_selection: Set[str] = set()
        
        # Plot window
        self.plot_window: Optional[PlotWindow] = None
        
        # Experiment timer
        self.experiment_timer = QTimer()
        self.experiment_timer.timeout.connect(self._update_experiment_timer)
        self.experiment_remaining_seconds = 0
        self.experiment_total_seconds = 0

        # Stop/Resume state
        self.is_paused = False
        self.remaining_wells: List[str] = []  # Wells not yet completed when paused
        self.current_sequence_wells: List[str] = []  # All wells in current sequence

        # Timer calibration (measure actual well duration)
        self.first_well_start_time: Optional[float] = None
        self.measured_well_duration: Optional[float] = None  # Actual time per well
        self.wells_completed_count = 0

        # Well completion log (synchronized with sensor log)
        self.sensor_log_start_time: Optional[datetime] = None  # When sensor started logging
        self.well_log_file: Optional[Path] = None  # Path to well completion log
        self.well_log_initialized = False  # Whether header has been written

        self._init_ui()
        
        logger.info("AsyncAMUZAGUI initialized")
    
    def _init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("AMUZA Controller - Async")
        self.setGeometry(100, 100, UI.MAIN_WINDOW_WIDTH, UI.MAIN_WINDOW_HEIGHT)
        self.setStyleSheet("""
            QMainWindow { background-color: #f1f4f7; }
            QPushButton {
                background-color: #fdfdfd;
                border: 1px solid #cfd6de;
                border-radius: 4px;
                padding: 6px 10px;
                font-weight: 600;
            }
            QPushButton:disabled { color: #9aa3ad; background-color: #f5f5f5; }
            QPushButton:hover { background-color: #e7eef5; }
            QPushButton:pressed { background-color: #dfe7ef; }
            QTextEdit { background: #ffffff; border: 1px solid #d0d7de; }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left column: controls only
        left_col = QVBoxLayout()
        
        self.connect_btn = QPushButton("Connect to AMUZA")
        self.connect_btn.clicked.connect(self._on_connect)
        left_col.addWidget(self.connect_btn)

        self.start_btn = QPushButton("Start Sampling")
        self.start_btn.clicked.connect(self._on_start)
        self.start_btn.setEnabled(False)
        left_col.addWidget(self.start_btn)

        self.insert_btn = QPushButton("Insert")
        self.insert_btn.setEnabled(False)
        self.insert_btn.clicked.connect(self._on_insert)
        left_col.addWidget(self.insert_btn)

        self.eject_btn = QPushButton("Eject")
        self.eject_btn.setEnabled(False)
        self.eject_btn.clicked.connect(self._on_eject)
        left_col.addWidget(self.eject_btn)

        self.move_btn = QPushButton("Move (Ctrl wells)")
        self.move_btn.setEnabled(False)
        self.move_btn.clicked.connect(self._on_move_ctrl)
        left_col.addWidget(self.move_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet("QPushButton { background:#e53935; color:white; font-weight:700; } QPushButton:hover{ background:#d32f2f; }")
        self.stop_btn.clicked.connect(self._on_stop_resume)
        self.stop_btn.setEnabled(False)
        left_col.addWidget(self.stop_btn)

        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self._on_settings)
        left_col.addWidget(self.settings_btn)

        self.calibration_btn = QPushButton("Calibration")
        self.calibration_btn.clicked.connect(self._on_calibration)
        left_col.addWidget(self.calibration_btn)

        self.sensor_btn = QPushButton("Connect Sensor")
        self.sensor_btn.clicked.connect(self._on_sensor_connect)
        left_col.addWidget(self.sensor_btn)
        
        self.plot_btn = QPushButton("Show Plot")
        self.plot_btn.clicked.connect(self._on_show_plot)
        left_col.addWidget(self.plot_btn)
        
        # Status labels
        self.status_label = QLabel("AMUZA: Not Connected")
        self.status_label.setAlignment(Qt.AlignCenter)
        left_col.addWidget(self.status_label)
        
        self.sensor_status_label = QLabel("Sensor: Not Connected")
        self.sensor_status_label.setAlignment(Qt.AlignCenter)
        left_col.addWidget(self.sensor_status_label)
        
        # Experiment timer display
        self.timer_label = QLabel("Experiment Timer:\n--:--:--")
        self.timer_label.setAlignment(Qt.AlignCenter)
        self.timer_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #2c3e50; border: 2px solid #3498db; border-radius: 6px; padding: 8px; background-color: #ecf0f1; }")
        left_col.addWidget(self.timer_label)
        
        # Window size display (hidden by default)
        self.size_label = QLabel()
        self.size_label.setAlignment(Qt.AlignCenter)
        self.size_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
        self.size_label.setVisible(False)
        left_col.addWidget(self.size_label)
        self._update_size_label()
        
        left_col.addStretch(1)
        
        # Center: well grid
        center_col = QVBoxLayout()
        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(4, 4, 4, 4)
        grid_layout.setHorizontalSpacing(6)
        grid_layout.setVerticalSpacing(6)
        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)
        grid_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        
        for row in range(8):
            for col in range(12):
                row_letter = chr(65 + row)  # A-H
                col_number = col + 1
                well_id = f"{row_letter}{col_number}"
                
                well_label = WellLabel(well_id, row, col, self.cell_size)
                
                grid_layout.addWidget(well_label, row, col)
                self.well_labels[well_id] = well_label
        
        total_w = 12 * self.cell_size + (12 - 1) * grid_layout.horizontalSpacing() + grid_layout.contentsMargins().left() + grid_layout.contentsMargins().right()
        total_h = 8 * self.cell_size + (8 - 1) * grid_layout.verticalSpacing() + grid_layout.contentsMargins().top() + grid_layout.contentsMargins().bottom()
        grid_widget.setFixedSize(total_w, total_h)
        
        center_col.addWidget(grid_widget, alignment=Qt.AlignCenter)
        
        # Clear button directly below grid
        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self._clear_selections)
        center_col.addWidget(clear_btn)
        
        # Display log below clear button
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        center_col.addWidget(self.display, stretch=1)
        
        # Right instructions
        right_col = QVBoxLayout()
        instructions = QTextEdit()
        instructions.setReadOnly(True)
        instructions.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        instructions.setText(
            "Instructions:\n"
            "1. Connect to AMUZA using the 'Connect to AMUZA' button.\n"
            "2. Use 'Eject' to remove the tray and 'Insert' to insert it.\n"
            "3. Drag over wells to select a block (blue).\n"
            "4. Ctrl+Click wells to build a MOVE list (green).\n"
            "5. 'Start Sampling' runs the block sequence. 'Move' runs ctrl list.\n"
            "6. 'Stop' interrupts current run.\n"
            "7. 'Show Plot' opens live data; 'Settings' adjusts timing.\n"
            "Coded By: Noah Bernten   Noah.Bernten@mail.huji.ac.il"
        )
        instructions.setMinimumWidth(260)
        right_col.addWidget(instructions, stretch=1)
        
        main_layout.addLayout(left_col, 2)
        main_layout.addLayout(center_col, 5)
        main_layout.addLayout(right_col, 2)
    
    @asyncSlot()
    async def _on_connect(self):
        """Handle connect/disconnect/reconnect button"""
        # If already connected, disconnect
        if self.connection and self.connection.is_connected:
            await self._disconnect_amuza()
            return

        # Connect (or reconnect)
        await self._connect_amuza()

    async def _connect_amuza(self):
        """Connect to AMUZA device"""
        try:
            self.connect_btn.setText("Connecting...")
            self.connect_btn.setEnabled(False)
            self.status_label.setText("AMUZA: Connecting...")
            self.add_to_display("Connecting to AMUZA...")

            # Create connection
            self.connection = AsyncAmuzaConnection(
                device_address=HARDWARE.BLUETOOTH_DEVICE_ADDRESS,
                use_mock=False
            )

            # Set timeout callback to update GUI on errors
            self.connection.set_timeout_callback(self._on_amuza_timeout)

            # Connect
            if await self.connection.connect():
                await self.app_state.set_connection(self.connection)

                self.status_label.setText("AMUZA: Connected")
                self.connect_btn.setText("Disconnect")
                self.connect_btn.setEnabled(True)
                self.insert_btn.setEnabled(True)
                self.eject_btn.setEnabled(True)
                self.move_btn.setEnabled(True)
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
                self.add_to_display("Connected to AMUZA.")
                logger.info("Connected to AMUZA")
            else:
                self.status_label.setText("AMUZA: Connection Failed")
                self.connect_btn.setText("Reconnect")
                self.connect_btn.setEnabled(True)
                self.add_to_display("Failed to connect to AMUZA.")
                QMessageBox.warning(self, "Connection Failed", "Could not connect to device")

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.status_label.setText("AMUZA: Error")
            self.connect_btn.setText("Reconnect")
            self.connect_btn.setEnabled(True)
            self.add_to_display(f"Connection error: {e}")
            QMessageBox.critical(self, "Error", f"Connection failed: {e}")

    async def _disconnect_amuza(self):
        """Disconnect from AMUZA device"""
        try:
            self.add_to_display("Disconnecting from AMUZA...")

            if self.connection:
                await self.connection.disconnect()
                self.connection = None

            await self.app_state.set_connection(None)

            self.status_label.setText("AMUZA: Disconnected")
            self.connect_btn.setText("Connect to AMUZA")
            self.connect_btn.setEnabled(True)
            self.insert_btn.setEnabled(False)
            self.eject_btn.setEnabled(False)
            self.move_btn.setEnabled(False)
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.add_to_display("Disconnected from AMUZA.")
            logger.info("Disconnected from AMUZA")

        except Exception as e:
            logger.error(f"Disconnect error: {e}")
            self.add_to_display(f"Disconnect error: {e}")

    def _on_amuza_timeout(self, command: str, attempts: int):
        """Called when AMUZA command times out after all retries"""
        self.status_label.setText("AMUZA: Timeout Error")
        self.connect_btn.setText("Reconnect")
        self.connect_btn.setEnabled(True)
        self.add_to_display(f"AMUZA timeout: {command} failed after {attempts} attempts")
        logger.warning(f"AMUZA timeout callback: {command} failed after {attempts} attempts")
    
    def _on_well_clicked(self, well_id: str):
        """Handle well click"""
        well_label = self.well_labels[well_id]
        
        # Toggle selection
        if not well_label.is_selected:
            asyncio.create_task(self.app_state.add_selected_well(well_id))
            well_label.set_selected(True)
        else:
            asyncio.create_task(self.app_state.remove_selected_well(well_id))
            well_label.set_selected(False)
        
        logger.info(f"Well {well_id} clicked")
    
    def mousePressEvent(self, event):
        """Handle mouse press for drag selection"""
        if event.button() == Qt.LeftButton:
            # Find which well was clicked
            for well_id, label in self.well_labels.items():
                if label.geometry().contains(label.parent().mapFromGlobal(event.globalPos())):
                    if event.modifiers() & Qt.ControlModifier:
                        # Ctrl+Click for MOVE command
                        asyncio.create_task(self._toggle_ctrl_well(well_id))
                    else:
                        # Start drag selection for RUNPLATE
                        self.drag_start = (label.row, label.col)
                        self.drag_active = True
                        self._apply_drag_selection(label.row, label.col)
                    break
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for drag selection"""
        if self.drag_active and (event.buttons() & Qt.LeftButton) and not (event.modifiers() & Qt.ControlModifier):
            # Find which well we're over
            for _, label in self.well_labels.items():
                if label.geometry().contains(label.parent().mapFromGlobal(event.globalPos())):
                    self._apply_drag_selection(label.row, label.col)
                    break
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to complete drag selection"""
        if self.drag_active:
            self.drag_active = False
            asyncio.create_task(self._commit_drag_selection())
        super().mouseReleaseEvent(event)
    
    def _on_well_pressed(self, row: int, col: int, modifiers: Qt.KeyboardModifiers):
        """Handle start of selection or ctrl toggle"""
        well_id = self._well_id_from_rc(row, col)
        if modifiers & Qt.ControlModifier:
            asyncio.create_task(self._toggle_ctrl_well(well_id))
            return
        
        self.drag_start = (row, col)
        self.drag_active = True
        self._apply_drag_selection(row, col)
    
    def _on_well_dragged(self, row: int, col: int, modifiers: Qt.KeyboardModifiers):
        """Handle drag selection updates"""
        if not self.drag_active or (modifiers & Qt.ControlModifier):
            return
        self._apply_drag_selection(row, col)
    
    def _on_well_released(self, row: int, col: int, modifiers: Qt.KeyboardModifiers):
        """Commit selection on release"""
        if self.drag_active and not (modifiers & Qt.ControlModifier):
            self.drag_active = False
            asyncio.create_task(self._commit_drag_selection())
    
    def _apply_drag_selection(self, row: int, col: int):
        """Preview selection during drag"""
        if not self.drag_start:
            return
        r0, c0 = self.drag_start
        r_min, r_max = sorted((r0, row))
        c_min, c_max = sorted((c0, col))
        selected = set()
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                selected.add(self._well_id_from_rc(r, c))
        self._update_selection_preview(selected)
    
    async def _commit_drag_selection(self):
        """Persist drag selection to state"""
        selected = {wid for wid, lbl in self.well_labels.items() if lbl.is_selected}
        ctrl_wells = await self.app_state.get_selected_wells(ctrl=True)
        await self.app_state.clear_selections()
        # Restore ctrl wells
        for wid in ctrl_wells:
            await self.app_state.add_selected_well(wid, ctrl=True)
            if wid in self.well_labels:
                self.well_labels[wid].set_ctrl_selected(True)
        for wid in selected:
            await self.app_state.add_selected_well(wid)
        logger.info(f"Selected wells: {sorted(selected)}")
    
    @asyncSlot()
    async def _clear_selections(self):
        """Clear all selections and completed wells. Only reset pause state if paused."""
        await self.app_state.clear_selections()
        await self.app_state.clear_completed_wells()
        for lbl in self.well_labels.values():
            lbl.set_selected(False)
            lbl.set_ctrl_selected(False)
            lbl.set_completed(False)

        # Only reset pause/resume state if currently paused
        if self.is_paused:
            self.remaining_wells = []
            self.current_sequence_wells = []
            self._reset_stop_button()
            self._stop_experiment_timer()
            self.add_to_display("Selections cleared. Paused sequence cancelled.")
        else:
            self.add_to_display("Selections cleared.")
    
    async def _toggle_ctrl_well(self, well_id: str):
        """Toggle control (MOVE) selection with green highlight"""
        label = self.well_labels[well_id]
        if not label.is_ctrl_selected:
            await self.app_state.add_selected_well(well_id, ctrl=True)
            label.set_ctrl_selected(True)
        else:
            await self.app_state.remove_selected_well(well_id, ctrl=True)
            label.set_ctrl_selected(False)
        logger.info(f"Ctrl well toggled: {well_id}")
    
    def _update_selection_preview(self, selected: set[str]):
        """Update UI selection colors without committing ctrl wells"""
        for wid, lbl in self.well_labels.items():
            lbl.set_selected(wid in selected)
    
    def _well_id_from_rc(self, row: int, col: int) -> str:
        """Convert row/col to well id like A1"""
        row_letter = chr(65 + row)
        return f"{row_letter}{col + 1}"
    
    def _start_experiment_timer(self, num_wells: int, t_buffer: int, t_sampling: int):
        """Start the experiment countdown timer"""
        # Calculate total time: (buffer + sampling) * number of wells
        self.experiment_total_seconds = (t_buffer + t_sampling) * num_wells
        self.experiment_remaining_seconds = self.experiment_total_seconds
        
        # Start timer (update every second)
        self.experiment_timer.start(1000)
        self._update_experiment_timer()
    
    def _update_experiment_timer(self):
        """Update the experiment timer display"""
        if self.experiment_remaining_seconds > 0:
            self.experiment_remaining_seconds -= 1
            
            # Format as HH:MM:SS
            hours = self.experiment_remaining_seconds // 3600
            minutes = (self.experiment_remaining_seconds % 3600) // 60
            seconds = self.experiment_remaining_seconds % 60
            
            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.timer_label.setText(f"Experiment Timer:\n{time_str}")
            
            # Change color as timer approaches zero
            if self.experiment_remaining_seconds < 60:
                self.timer_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #c0392b; border: 2px solid #e74c3c; border-radius: 6px; padding: 8px; background-color: #fadbd8; }")
            elif self.experiment_remaining_seconds < 300:  # 5 minutes
                self.timer_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #d68910; border: 2px solid #f39c12; border-radius: 6px; padding: 8px; background-color: #fdebd0; }")
        else:
            # Timer finished
            self.experiment_timer.stop()
            self.timer_label.setText(f"Experiment Timer:\nCOMPLETE!")
            self.timer_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #27ae60; border: 2px solid #2ecc71; border-radius: 6px; padding: 8px; background-color: #d5f4e6; }")
    
    def _stop_experiment_timer(self):
        """Stop and reset the experiment timer"""
        self.experiment_timer.stop()
        self.experiment_remaining_seconds = 0
        self.timer_label.setText("Experiment Timer:\n--:--:--")
        self.timer_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; color: #2c3e50; border: 2px solid #3498db; border-radius: 6px; padding: 8px; background-color: #ecf0f1; }")

    def _format_time(self, seconds: int) -> str:
        """Format seconds as HH:MM:SS"""
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    async def _start_resumed_sequence(self):
        """Start a sequence from remaining wells (after resume)"""
        if not self.remaining_wells:
            self.add_to_display("No wells to resume.")
            return

        t_buffer, t_sampling = await self.app_state.get_timing_params()

        # Create sequence from remaining wells only
        sequence = Sequence("Resumed Sequence")
        for well_id in self.remaining_wells:
            method = Method(
                pos=well_id,
                wait=t_sampling,
                buffer_time=t_buffer,
                eject=False,
                insert=False
            )
            sequence.add_method(method)

        # Recalculate timer based on measured duration if available
        if self.measured_well_duration:
            new_remaining = int(self.measured_well_duration * len(self.remaining_wells))
            self.experiment_remaining_seconds = new_remaining
        else:
            # Use estimate
            self.experiment_remaining_seconds = (t_buffer + t_sampling) * len(self.remaining_wells)

        # Reset first well timing for this segment
        self.first_well_start_time = time.time()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Run in background task
        task = asyncio.create_task(self._execute_sequence(sequence))
        self.task_manager.add_task(task, "sampling_sequence")

    @asyncSlot()
    async def _on_start(self):
        """Handle start sampling"""
        try:
            wells = await self.app_state.get_selected_wells()

            if not wells:
                QMessageBox.warning(self, "No Wells", "Please select wells to sample")
                return

            # Clear any previous stop state and ensure clean start
            await self.app_state.clear_stop()
            self._reset_stop_button()  # Reset button to STOP state

            # Cancel any ACTUALLY RUNNING sampling tasks from previous runs
            for task in self.task_manager.get_running_tasks():
                if task.get_name() == "sampling_sequence":
                    # Only cancel if task is not done
                    if not task.done():
                        logger.info("Cancelling previous running sampling task")
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    else:
                        logger.info("Previous sampling task already completed, skipping cancel")

            # Track wells for pause/resume
            # Sort wells naturally: A1, A2, ... A10, A11, A12, B1, B2, ... (not A1, A10, A11...)
            wells_list = sorted(list(wells), key=lambda x: (x[0], int(x[1:])))
            self.current_sequence_wells = wells_list.copy()
            self.remaining_wells = wells_list.copy()
            self.wells_completed_count = 0
            self.first_well_start_time = time.time()  # Start timing
            self.measured_well_duration = None  # Will be set after first well

            # Create sequence
            sequence = Sequence("Sampling Sequence")

            t_buffer, t_sampling = await self.app_state.get_timing_params()

            # Start experiment timer with estimate (will be recalibrated after first well)
            self._start_experiment_timer(len(wells_list), t_buffer, t_sampling)

            for well_id in wells_list:
                method = Method(
                    pos=well_id,
                    wait=t_sampling,  # Sampling time at well
                    buffer_time=t_buffer,  # Buffer time before move
                    eject=False,
                    insert=False
                )
                sequence.add_method(method)

            # Execute sequence
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.add_to_display(f"Running plate on wells: {', '.join(wells_list)}")

            # Run in background task
            task = asyncio.create_task(self._execute_sequence(sequence))
            self.task_manager.add_task(task, "sampling_sequence")

        except Exception as e:
            logger.error(f"Start error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start: {e}")
    
    async def _execute_sequence(self, sequence: Sequence):
        """Execute sampling sequence with real-time well completion updates"""
        try:
            if not self.connection:
                logger.error("No connection")
                return

            # Get stop event from app state
            stop_event = self.app_state.stop_event
            stop_event.clear()

            # Clear previous completed wells (only on fresh start, not resume)
            if not self.is_paused:
                await self.app_state.clear_completed_wells()
                for label in self.well_labels.values():
                    label.set_completed(False)

            # Check if plate is ejected and notify user
            if hasattr(self.connection, 'status') and self.connection.status.state != 1:
                self.add_to_display("Plate not ready. Automatically inserting plate...")

            # Define callback for well completion updates
            def on_well_completed(well_id: str, completed_wells: list):
                """Callback to update GUI when each well is completed"""
                # Update display
                total_wells = len(self.current_sequence_wells)
                progress = len(completed_wells)
                self.add_to_display(f"Well {well_id} completed ({progress}/{total_wells})")

                # Mark well as completed in app_state
                asyncio.create_task(self.app_state.mark_well_completed(well_id))

                # Update well label visual
                if well_id in self.well_labels:
                    self.well_labels[well_id].set_completed(True)

                # Remove from remaining wells
                if well_id in self.remaining_wells:
                    self.remaining_wells.remove(well_id)

                # Log well completion with sensor-synchronized timestamp
                self._log_well_completion(well_id, sequence.name)

                # Track completion count and calibrate timer after first well
                self.wells_completed_count += 1
                if self.wells_completed_count == 1 and self.first_well_start_time:
                    # Measure actual duration of first well
                    self.measured_well_duration = time.time() - self.first_well_start_time
                    logger.info(f"First well took {self.measured_well_duration:.1f}s - recalibrating timer")

                    # Recalibrate experiment timer based on actual measurement
                    remaining_count = len(self.remaining_wells)
                    if remaining_count > 0:
                        new_remaining_time = int(self.measured_well_duration * remaining_count)
                        self.experiment_remaining_seconds = new_remaining_time
                        self.experiment_total_seconds = int(self.measured_well_duration * total_wells)
                        self.add_to_display(f"Timer recalibrated: ~{self._format_time(new_remaining_time)} remaining")

                # Update progress label
                self._update_progress_display(progress, total_wells, completed_wells)

            # Timing provider - returns current settings for dynamic updates
            def get_current_timing():
                """Return current (buffer, sampling) times from app_state"""
                return (self.app_state.t_buffer, self.app_state.t_sampling)

            # Execute sequence with callback (connection will handle auto-insert)
            completed = await self.connection.execute_sequence(
                sequence,
                stop_event,
                well_completed_callback=on_well_completed,
                timing_provider=get_current_timing
            )

            # Handle completion vs stopped
            if completed:
                self.add_to_display(f"Sampling sequence completed successfully ({len(self.current_sequence_wells)} wells)")
                self.remaining_wells = []  # Clear remaining
                self._reset_stop_button()
                logger.info("Sampling sequence completed")
            else:
                # Stopped by user - don't reset button (it's now RESUME)
                logger.info(f"Sampling sequence stopped. {len(self.remaining_wells)} wells remaining.")

        except Exception as e:
            logger.error(f"Sequence error: {e}")
            self.add_to_display(f"Sequence failed: {e}")
            self._stop_experiment_timer()
            self._reset_stop_button()

        finally:
            self.start_btn.setEnabled(True)
            # Only disable stop button if sequence fully completed (not paused)
            if not self.is_paused:
                self.stop_btn.setEnabled(False)

    def _update_progress_display(self, completed: int, total: int, completed_wells: list):
        """Update progress information in timer label"""
        # Show progress in timer label subtitle
        progress_pct = int((completed / total) * 100) if total > 0 else 0
        wells_str = ", ".join(completed_wells[-3:])  # Show last 3 completed
        if len(completed_wells) > 3:
            wells_str = "..." + wells_str
        logger.debug(f"Progress: {completed}/{total} ({progress_pct}%)")
    
    @asyncSlot()
    async def _on_stop_resume(self):
        """Handle stop/resume toggle button"""
        if not self.is_paused:
            # STOP: Request stop, will finish current well then pause
            await self.app_state.request_stop()
            self.is_paused = True

            # Change button to RESUME
            self.stop_btn.setText("RESUME")
            self.stop_btn.setStyleSheet("QPushButton { background:#2e7d32; color:white; font-weight:700; } QPushButton:hover{ background:#1b5e20; }")

            # Pause the experiment timer
            self.experiment_timer.stop()

            logger.info("Stop requested - will finish current well then pause")
            self.add_to_display("STOPPED - Finishing current well, then pausing. Press RESUME to continue.")
        else:
            # RESUME: Continue from remaining wells
            if not self.remaining_wells:
                self.add_to_display("No wells remaining to resume.")
                self._reset_stop_button()
                return

            # Clear stop flag
            await self.app_state.clear_stop()
            self.is_paused = False

            # Change button back to STOP
            self._reset_stop_button()

            # Resume experiment timer
            self.experiment_timer.start(1000)

            logger.info(f"Resuming sequence with {len(self.remaining_wells)} wells remaining")
            self.add_to_display(f"RESUMED - Continuing with {len(self.remaining_wells)} wells: {', '.join(self.remaining_wells[:5])}{'...' if len(self.remaining_wells) > 5 else ''}")

            # Start the resumed sequence
            await self._start_resumed_sequence()

    def _reset_stop_button(self):
        """Reset stop button to default STOP state"""
        self.is_paused = False
        self.stop_btn.setText("STOP")
        self.stop_btn.setStyleSheet("QPushButton { background:#e53935; color:white; font-weight:700; } QPushButton:hover{ background:#d32f2f; }")

    def _init_well_log(self):
        """Initialize well completion log file with header (only when sensor is logging)"""
        if not self.sensor_log_start_time:
            return

        # Create well log file with same timestamp pattern as sensor log
        timestamp = self.sensor_log_start_time.strftime("%d_%m_%y_%H_%M")
        self.well_log_file = Path(FILES.SENSOR_READINGS_FOLDER) / f"Well_Log_{timestamp}.csv"

        # Ensure directory exists
        self.well_log_file.parent.mkdir(parents=True, exist_ok=True)

        # Write header
        with open(self.well_log_file, 'w') as f:
            f.write(f"# Well completion log - Sensor started: {self.sensor_log_start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("well_id,completed_at,sensor_elapsed_min,sequence_name\n")

        self.well_log_initialized = True
        self.add_to_display(f"Well log started: {self.well_log_file.name}")
        logger.info(f"Well log initialized: {self.well_log_file}")

    def _log_well_completion(self, well_id: str, sequence_name: str):
        """Log well completion with sensor-synchronized timestamp"""
        # Only log if sensor is connected and logging
        if not self.sensor_log_start_time or not self.well_log_file:
            return

        now = datetime.now()

        # Calculate sensor-relative time (same time base as sensor log t[min] column)
        sensor_elapsed_min = (now - self.sensor_log_start_time).total_seconds() / 60.0

        # Append to log file
        try:
            with open(self.well_log_file, 'a') as f:
                f.write(f"{well_id},{now.strftime('%Y-%m-%d %H:%M:%S')},{sensor_elapsed_min:.4f},{sequence_name}\n")
            logger.debug(f"Logged well {well_id} at sensor_elapsed={sensor_elapsed_min:.4f} min")
        except Exception as e:
            logger.error(f"Failed to write well log: {e}")

    def _reset_well_log(self):
        """Reset well log tracking (called when sensor disconnects)"""
        self.sensor_log_start_time = None
        self.well_log_file = None
        self.well_log_initialized = False

    @asyncSlot()
    async def _on_insert(self):
        """Insert tray"""
        if not self.connection:
            QMessageBox.warning(self, "Not Connected", "Connect first.")
            return
        await self.connection.insert()
        self.add_to_display("Insert command sent.")

    @asyncSlot()
    async def _on_eject(self):
        """Eject tray"""
        if not self.connection:
            QMessageBox.warning(self, "Not Connected", "Connect first.")
            return
        await self.connection.eject()
        self.add_to_display("Eject command sent.")

    @asyncSlot()
    async def _on_move_ctrl(self):
        """Run MOVE sequence on ctrl-selected wells"""
        try:
            wells = await self.app_state.get_selected_wells(ctrl=True)
            if not wells:
                QMessageBox.warning(self, "No Ctrl Wells", "Ctrl+click wells to move.")
                return

            # Sort wells naturally: A1, A2, ... A10, A11, A12, B1, B2, ... (not A1, A10, A11...)
            wells_list = sorted(list(wells), key=lambda x: (x[0], int(x[1:])))

            # Clear any previous stop state and ensure clean start
            await self.app_state.clear_stop()
            self._reset_stop_button()

            # Cancel any ACTUALLY RUNNING move tasks from previous runs
            for task in self.task_manager.get_running_tasks():
                if task.get_name() == "move_sequence":
                    # Only cancel if task is not done
                    if not task.done():
                        logger.info("Cancelling previous running move task")
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    else:
                        logger.info("Previous move task already completed, skipping cancel")

            sequence = Sequence("Move Sequence")
            t_buffer, t_sampling = await self.app_state.get_timing_params()

            # Start experiment timer (includes buffer time for accurate estimation)
            self._start_experiment_timer(len(wells_list), t_buffer, t_sampling)

            # Track wells for pause/resume
            self.current_sequence_wells = wells_list.copy()
            self.remaining_wells = wells_list.copy()
            self.wells_completed_count = 0
            self.first_well_start_time = time.time()
            self.measured_well_duration = None

            for well_id in wells_list:
                method = Method(pos=well_id, wait=t_sampling, buffer_time=t_buffer, eject=False, insert=False)
                sequence.add_method(method)

            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.add_to_display(f"Moving to wells: {', '.join(wells_list)}")
            task = asyncio.create_task(self._execute_sequence(sequence))
            self.task_manager.add_task(task, "move_sequence")
        except Exception as e:
            logger.error(f"Move error: {e}")
            QMessageBox.critical(self, "Error", f"Move failed: {e}")
    
    def _on_show_plot(self):
        """Show plot window"""
        if not self.plot_window:
            self.plot_window = PlotWindow(self.app_state)

        # If sensor is already running, connect plot to the sensor's output file
        if self.sensor_reader and self.sensor_reader.is_running:
            sensor_output_file = self.sensor_reader.get_output_file()
            self.plot_window.set_sensor_file(sensor_output_file)
            self.plot_window._using_callback = True  # Enable callback mode for live data
            self.add_to_display(f"Plot connected to live sensor data")

        self.plot_window.show()
        self.plot_window.raise_()
    
    def _on_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self.app_state, self)

        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            new_buffer = values['buffer']
            new_sampling = values['sampling']

            # Update timing params
            asyncio.create_task(
                self.app_state.set_timing_params(new_buffer, new_sampling)
            )

            # Update temperature settings
            asyncio.create_task(
                self.app_state.set_temperature_settings(values['temperature'], values['heater_enabled'])
            )

            # Apply temperature to AMUZA if connected
            if self.connection and self.connection.is_connected:
                asyncio.create_task(self._apply_temperature_settings(values['temperature'], values['heater_enabled']))

            # RECALCULATE TIMER if experiment is running or paused with remaining wells
            if self.remaining_wells:
                self._recalculate_experiment_timer(new_buffer, new_sampling)

            # Save settings to file
            asyncio.create_task(self.app_state.save_settings())

            logger.info(f"Settings updated: {values}")
            self.add_to_display(f"Settings saved: Buffer={values['buffer']}s, Sampling={values['sampling']}s, Temp={values['temperature']}C")

    def _recalculate_experiment_timer(self, t_buffer: int, t_sampling: int):
        """Recalculate experiment timer when settings change mid-experiment"""
        remaining_count = len(self.remaining_wells)
        if remaining_count <= 0:
            return

        # Calculate new remaining time based on updated settings
        time_per_well = t_buffer + t_sampling
        new_remaining_seconds = time_per_well * remaining_count

        old_remaining = self.experiment_remaining_seconds
        self.experiment_remaining_seconds = new_remaining_seconds

        # Update total for percentage calculations if needed
        total_wells = len(self.current_sequence_wells) if self.current_sequence_wells else remaining_count
        self.experiment_total_seconds = time_per_well * total_wells

        # Log the change
        old_time_str = self._format_time(old_remaining)
        new_time_str = self._format_time(new_remaining_seconds)
        self.add_to_display(f"Timer updated: {old_time_str} → {new_time_str} ({remaining_count} wells × {time_per_well}s)")
        logger.info(f"Experiment timer recalculated: {old_remaining}s → {new_remaining_seconds}s")

    def _format_time(self, seconds: int) -> str:
        """Format seconds as HH:MM:SS or MM:SS"""
        if seconds < 0:
            seconds = 0
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    async def _apply_temperature_settings(self, temperature: float, heater_on: bool):
        """Apply temperature settings to AMUZA device"""
        try:
            if self.connection:
                await self.connection.set_temperature(temperature)
                await self.connection.set_heater(heater_on)
                self.add_to_display(f"Temperature set to {temperature}C, Heater {'ON' if heater_on else 'OFF'}")
        except Exception as e:
            logger.error(f"Failed to apply temperature settings: {e}")

    def _on_calibration(self):
        """Show calibration settings dialog"""
        dialog = CalibrationSettingsDialog(self.app_state, self)

        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()

            # Update app_state with new gains and calibrations
            asyncio.create_task(
                self.app_state.set_calibration_gains(values['gains'])
            )
            asyncio.create_task(
                self.app_state.set_calibration_values(values['calibrations'])
            )

            # Update plot window if it exists
            if self.plot_window:
                self.plot_window.update_gains(values['gains'])

            # Save settings to file
            asyncio.create_task(self.app_state.save_settings())

            logger.info(f"Calibration updated: {values}")
            self.add_to_display("Calibration settings updated and saved.")
    
    def _on_sensor_connect(self):
        """Handle sensor connect/disconnect - uses sync dialog then schedules async work"""
        # If already connected, disconnect
        if self.sensor_reader and self.sensor_reader.is_running:
            asyncio.create_task(self._disconnect_sensor())
            return

        # Show connect dialog (synchronous - safe with exec_())
        dialog = SensorConnectDialog(self)

        if dialog.exec_() == QDialog.Accepted:
            port = dialog.get_selected_port()
            if not port:
                return

            # Schedule async connection
            asyncio.create_task(self._connect_sensor(port))

    async def _disconnect_sensor(self):
        """Async helper to disconnect sensor"""
        try:
            await self.sensor_reader.stop()
            await self.sensor_reader.disconnect()
            self.sensor_reader = None

            # Reset well log tracking (log file remains, but new wells won't be logged)
            if self.well_log_file:
                self.add_to_display(f"Well log saved: {self.well_log_file.name}")
            self._reset_well_log()

            self.sensor_btn.setText("Connect Sensor")
            self.sensor_status_label.setText("Sensor: Disconnected")
            self.add_to_display("Sensor disconnected.")
        except Exception as e:
            logger.error(f"Sensor disconnect error: {e}")

    async def _connect_sensor(self, port: str):
        """Async helper to connect sensor"""
        use_mock = (port == "MOCK")

        try:
            # OPTIMIZED: Set up data callback for direct plot updates
            # This is more efficient than file polling
            def on_sensor_data(reading):
                if self.plot_window:
                    self.plot_window.add_reading(reading)

            # Create sensor reader with data callback
            self.sensor_reader = AsyncPotentiostatReader(
                port=port if not use_mock else "COM1",
                use_mock=use_mock,
                data_callback=on_sensor_data
            )

            # Connect
            if await self.sensor_reader.connect():
                # Start reading in background
                task = asyncio.create_task(self.sensor_reader.start_reading())
                self.task_manager.add_task(task, "sensor_reading")

                # Initialize well completion log (synchronized with sensor start)
                self.sensor_log_start_time = datetime.now()
                self._init_well_log()

                # Auto-open plot window if not already open
                if not self.plot_window:
                    self.plot_window = PlotWindow(self.app_state)

                # Update PlotWindow with the sensor's output file path
                sensor_output_file = self.sensor_reader.get_output_file()
                self.plot_window.set_sensor_file(sensor_output_file)
                self.plot_window._using_callback = True  # Enable callback mode
                self.plot_window.show()
                self.plot_window.raise_()
                self.add_to_display(f"Plot window opened - receiving data")

                self.sensor_btn.setText("Disconnect Sensor")
                self.sensor_status_label.setText(f"Sensor: {port}")
                self.add_to_display(f"Sensor connected on {port}")
                self.add_to_display(f"Data file: {Path(sensor_output_file).name}")
                logger.info(f"Sensor connected on {port}, output: {sensor_output_file}")
            else:
                QMessageBox.warning(self, "Connection Failed", "Could not connect to sensor")
                self.sensor_reader = None

        except Exception as e:
            logger.error(f"Sensor connection error: {e}")
            QMessageBox.critical(self, "Error", f"Sensor connection failed: {e}")
            self.sensor_reader = None
    
    def resizeEvent(self, event):
        """Handle window resize to update size display"""
        super().resizeEvent(event)
        self._update_size_label()
    
    def _update_size_label(self):
        """Update the size label with current dimensions"""
        width = self.width()
        height = self.height()
        self.size_label.setText(f"Window: {width} × {height}")
    
    def closeEvent(self, event):
        """Handle window close - ensure sensor stops saving"""
        logger.info("Main window closing - stopping sensor")

        # Stop sensor reader synchronously to ensure file is flushed
        if self.sensor_reader and self.sensor_reader.is_running:
            self.sensor_reader.stop_event.set()
            self.add_to_display("Stopping sensor on application close...")

        # Close plot window
        if self.plot_window:
            self.plot_window.close()

        event.accept()

    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up GUI resources")

        # Cancel all tasks
        await self.task_manager.cancel_all_tasks()

        # Disconnect AMUZA
        if self.connection:
            await self.connection.disconnect()

        # Disconnect sensor - ensure final buffer flush
        if self.sensor_reader:
            await self.sensor_reader.stop()
            await self.sensor_reader.disconnect()
            logger.info("Sensor reader stopped and disconnected")

        # Close plot window
        if self.plot_window:
            self.plot_window.close()

    def add_to_display(self, message: str):
        """Append message to display panel"""
        current = self.display.toPlainText().splitlines()
        current.append(message)
        # keep last 50
        current = current[-50:]
        self.display.setPlainText("\n".join(current))
        self.display.moveCursor(self.display.textCursor().End)


async def async_main():
    """Async main function"""
    # Create app state
    app_state = AppState()

    # Load saved settings
    await app_state.load_settings()

    # Create task manager
    task_manager = AsyncTaskManager()

    # Ensure Sensor_Readings directory exists
    Path(FILES.SENSOR_READINGS_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(FILES.AMUZA_LOGS_FOLDER).mkdir(parents=True, exist_ok=True)

    # Create and show GUI
    gui = AsyncAMUZAGUI(app_state, task_manager)
    gui.show()

    # Wait for window to close
    try:
        while gui.isVisible():
            await asyncio.sleep(0.1)
    finally:
        # Save settings before closing
        await app_state.save_settings()
        await gui.cleanup()


def main():
    """Main entry point"""
    if not QASYNC_AVAILABLE:
        print("ERROR: qasync is required for async GUI")
        print("Install with: pip install qasync")
        sys.exit(1)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create Qt application
    app = QApplication(sys.argv)
    
    # Create async event loop with qasync
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Run async main
    with loop:
        loop.run_until_complete(async_main())


if __name__ == "__main__":
    main()
