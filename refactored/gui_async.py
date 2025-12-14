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
    
    def __init__(self, well_id: str, row: int, col: int, size: int):
        super().__init__(well_id)
        self.well_id = well_id
        self.row = row
        self.col = col
        self.is_selected = False
        self.is_ctrl_selected = False
        
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
    
    def _update_appearance(self):
        """Update visual appearance based on state"""
        if self.is_ctrl_selected:
            color = UI.CTRL_WELL_COLOR
        elif self.is_selected:
            color = UI.SELECTED_WELL_COLOR
        else:
            color = "white"
        
        self.setStyleSheet(f"""
            QLabel {{
                border: 1px solid black;
                background-color: {color};
                border-radius: 0px;
                font-weight: 600;
            }}
            QLabel:hover {{
                background-color: {color};
                border-color: #4a90e2;
            }}
        """)


class PlotWindow(QMainWindow):
    """Real-time plotting window with incremental file reading"""
    
    def __init__(self, app_state: AppState):
        super().__init__()
        self.app_state = app_state
        
        # File reading state
        self.data_file = FILES.OUTPUT_FILE_PATH
        self.last_file_position = 0
        self.cached_data = pd.DataFrame()
        self.full_data = pd.DataFrame()  # Store ALL data for saving
        self.loaded_file_path = None  # Track loaded file for save
        
        # Raw file auto-save
        self.raw_file_path = Path(FILES.OUTPUT_FILE_PATH).parent / f"raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.raw_file_initialized = False
        
        # Plot configuration
        self.rolling_window_minutes = UI.PLOT_WINDOW_MINUTES
        self.show_full_graph = False  # When True, show entire graph instead of rolling window
        
        # Metabolite gain values (calibration)
        self.gain_values = {
            "Glutamate": 3.394,
            "Glutamine": 0.974,
            "Glucose": 1.5,
            "Lactate": 0.515,
        }
        
        self._init_ui()
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_update)
        self.timer.start(UI.PLOT_UPDATE_INTERVAL_MS)
        
        logger.info(f"PlotWindow initialized, raw data will be saved to {self.raw_file_path}")
    
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
        
        layout.addWidget(self.nav_toolbar)
        layout.addWidget(self.canvas)
        
        # Create single subplot for all channels
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Metabolite Concentrations")
        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Concentration (mM)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.figure.tight_layout()
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear Data")
        self.clear_btn.clicked.connect(self._on_clear_data)
        button_layout.addWidget(self.clear_btn)
        
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.clicked.connect(self._on_export_data)
        button_layout.addWidget(self.export_btn)
        
        layout.addLayout(button_layout)
    
    def _on_home_clicked(self):
        """Custom home button handler - shows full graph from 0"""
        self.show_full_graph = True
        self._update_plots()
    
    def _on_timer_update(self):
        """Timer callback for plot updates"""
        # Use asyncio to run update
        loop = asyncio.get_event_loop()
        loop.create_task(self._update_plot_async())
    
    async def _update_plot_async(self):
        """Async plot update with incremental file reading"""
        try:
            # Read new data incrementally
            new_data = await self._read_new_data()
            
            if new_data is not None and not new_data.empty:
                # Append to full data (keep ALL data for saving)
                self.full_data = pd.concat([self.full_data, new_data], ignore_index=True)
                
                # Auto-save raw data
                self._auto_save_raw_data(new_data)
                
                # Append to cached data for plotting
                self.cached_data = pd.concat([self.cached_data, new_data], ignore_index=True)
                
                # Trim cached data for display (keep only rolling window)
                if not self.cached_data.empty:
                    current_time = self.cached_data['Time'].max()
                    cutoff_time = current_time - (self.rolling_window_minutes * 60)
                    self.cached_data = self.cached_data[self.cached_data['Time'] >= cutoff_time]
                
                # Update plots
                self._update_plots()
        
        except Exception as e:
            logger.error(f"Error updating plot: {e}")
    
    def _auto_save_raw_data(self, new_data: pd.DataFrame):
        """Auto-save new data to raw file in original tab-separated format"""
        try:
            with open(self.raw_file_path, 'a') as f:
                if not self.raw_file_initialized:
                    # Write header for new file
                    now = datetime.now()
                    f.write(f"Created: {now.strftime('%m/%d/%Y')}\t{now.strftime('%I:%M:%S %p')}\n")
                    
                    # Write column headers
                    cols = ['counter', 't[min]']
                    for ch in range(1, 8):
                        cols.append(f'#1ch{ch}')
                    f.write('\t'.join(cols) + '\n')
                    
                    f.write(f"Start: {now.strftime('%m/%d/%Y')}\t{now.strftime('%I:%M:%S %p')}\n")
                    self.raw_file_initialized = True
                
                # Write new data rows
                for idx, row in new_data.iterrows():
                    counter = len(self.full_data) - len(new_data) + idx + 1
                    time_min = row['Time'] / 60.0 if 'Time' in row else 0
                    values = [str(int(counter)), f"{time_min:.4f}"]
                    
                    for ch in range(1, 8):
                        col_name = f'Channel {ch}'
                        if col_name in row:
                            values.append(f"{row[col_name]:.2f}")
                        else:
                            values.append("0.00")
                    
                    f.write('\t'.join(values) + '\n')
        except Exception as e:
            logger.error(f"Error auto-saving raw data: {e}")
    
    async def _read_new_data(self) -> Optional[pd.DataFrame]:
        """
        Read only new lines from file since last read.
        
        Returns:
            DataFrame with new data, or None if no new data
        """
        try:
            import aiofiles
            
            async with aiofiles.open(self.data_file, 'r') as f:
                # Seek to last position
                await f.seek(self.last_file_position)
                
                # Read new lines
                new_lines = await f.readlines()
                
                # Update position
                self.last_file_position = await f.tell()
            
            if not new_lines:
                return None
            
            # Parse lines
            data_rows = []
            for line in new_lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        data_rows.append({
                            'Time': float(parts[0]),
                            'Channel': int(parts[1]),
                            'Value': float(parts[2])
                        })
                    except ValueError:
                        continue
            
            if data_rows:
                return pd.DataFrame(data_rows)
            
            return None
        
        except FileNotFoundError:
            # File doesn't exist yet
            return None
        except Exception as e:
            logger.error(f"Error reading data file: {e}")
            return None
    
    def _update_plots(self):
        """Update single plot with metabolite data"""
        if self.cached_data.empty:
            return
        
        # Convert timestamp to relative minutes
        start_time = self.cached_data['Time'].min()
        self.cached_data['RelativeTime'] = (self.cached_data['Time'] - start_time) / 60
        
        # Clear and replot
        self.ax.clear()
        
        # Colors for metabolites
        colors = {'Glutamate': 'b', 'Glutamine': 'g', 'Glucose': 'r', 'Lactate': 'c'}
        
        # Check data format
        if 'Channel 1' in self.cached_data.columns:
            # Column-per-channel format (from loaded .txt or .csv files)
            # Calculate metabolites from channels
            metabolites = {}
            if 'Channel 1' in self.cached_data.columns and 'Channel 2' in self.cached_data.columns:
                metabolites['Glutamate'] = self.cached_data['Channel 1'] - self.cached_data['Channel 2']
            if 'Channel 3' in self.cached_data.columns and 'Channel 1' in self.cached_data.columns:
                metabolites['Glutamine'] = self.cached_data['Channel 3'] - self.cached_data['Channel 1']
            if 'Channel 5' in self.cached_data.columns and 'Channel 4' in self.cached_data.columns:
                metabolites['Glucose'] = self.cached_data['Channel 5'] - self.cached_data['Channel 4']
            if 'Channel 6' in self.cached_data.columns and 'Channel 4' in self.cached_data.columns:
                metabolites['Lactate'] = self.cached_data['Channel 6'] - self.cached_data['Channel 4']
            
            # Plot each metabolite with gain scaling
            for metabolite, values in metabolites.items():
                scaled_values = values * self.gain_values.get(metabolite, 1.0)
                self.ax.plot(
                    self.cached_data['RelativeTime'],
                    scaled_values,
                    color=colors.get(metabolite, 'k'),
                    linewidth=1,
                    label=metabolite
                )
        elif 'Channel' in self.cached_data.columns:
            # Row-per-channel format (from real-time data) - needs different handling
            for i in range(1, 7):
                channel_data = self.cached_data[self.cached_data['Channel'] == i]
                if not channel_data.empty:
                    self.ax.plot(
                        channel_data['RelativeTime'],
                        channel_data['Value'],
                        linewidth=1,
                        label=f'Channel {i}'
                    )
        
        # Set labels and title
        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Concentration (mM)")
        self.ax.set_title("Metabolite Concentrations")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.ax.set_ylim(bottom=0)  # Start y-axis at 0
        
        # Set x-axis limits based on mode
        if not self.cached_data.empty:
            max_time = self.cached_data['RelativeTime'].max()
            if self.show_full_graph:
                # Show entire graph from 0 to max
                self.ax.set_xlim(0, max(1, max_time))
            else:
                # Rolling window mode
                if max_time > self.rolling_window_minutes:
                    self.ax.set_xlim(max_time - self.rolling_window_minutes, max_time)
                else:
                    self.ax.set_xlim(0, max(self.rolling_window_minutes, max_time))
        
        self.canvas.draw()
    
    def _on_clear_data(self):
        """Clear cached data and reset plots"""
        self.cached_data = pd.DataFrame()
        self.full_data = pd.DataFrame()
        self.last_file_position = 0
        self.loaded_file_path = None  # Clear loaded file reference
        self.show_full_graph = False  # Reset to rolling window mode
        
        # Start a new raw file for next data collection
        self.raw_file_path = Path(FILES.OUTPUT_FILE_PATH).parent / f"raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.raw_file_initialized = False
        
        self.ax.clear()
        self.ax.set_title("Metabolite Concentrations")
        self.ax.set_xlabel("Time (min)")
        self.ax.set_ylabel("Concentration (mM)")
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
        logger.info("Plot data cleared, new raw file will be created")
    
    def _on_export_data(self):
        """Export raw channel data to CSV"""
        if self.cached_data.empty:
            QMessageBox.warning(self, "No Data", "No data to export")
            return
        
        # Export raw channel data with time and temperature
        export_data = pd.DataFrame()
        export_data['Time (min)'] = self.cached_data['RelativeTime'] if 'RelativeTime' in self.cached_data.columns else self.cached_data['Time'] / 60
        
        # Add individual channels (1-6)
        for ch in range(1, 7):
            col_name = f'Channel {ch}'
            if col_name in self.cached_data.columns:
                export_data[col_name] = self.cached_data[col_name]
        
        # Add temperature (Channel 7)
        if 'Channel 7' in self.cached_data.columns:
            export_data['Temperature'] = self.cached_data['Channel 7']
        
        filename = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        export_data.to_csv(filename, index=False)
        QMessageBox.information(self, "Export Complete", f"Data exported to {filename}")
        logger.info(f"Data exported to {filename}")
    
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
            
            # Track loaded file
            self.loaded_file_path = file_path
            self.last_file_position = 0
            
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
            # If we have a raw file being recorded, copy it
            if self.raw_file_initialized and self.raw_file_path.exists():
                with open(self.raw_file_path, "r", encoding='utf-8') as source_file:
                    with open(file_path, "w", encoding='utf-8') as dest_file:
                        dest_file.write(source_file.read())
                QMessageBox.information(self, "Success", f"Data successfully saved to {file_path}")
                logger.info(f"Saved raw data to {file_path}")
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
    """Settings dialog for timing parameters"""
    
    def __init__(self, app_state: AppState, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Settings")
        
        layout = QFormLayout(self)
        
        # Buffer time
        buffer_layout = QHBoxLayout()
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(0, 600)
        self.buffer_spin.setValue(self.app_state.t_buffer)
        buffer_layout.addWidget(self.buffer_spin)
        buffer_layout.addWidget(QLabel("seconds"))
        buffer_layout.addStretch()
        layout.addRow("Buffer Time:", buffer_layout)
        
        # Sampling time
        sampling_layout = QHBoxLayout()
        self.sampling_spin = QSpinBox()
        self.sampling_spin.setRange(0, 600)
        self.sampling_spin.setValue(self.app_state.t_sampling)
        sampling_layout.addWidget(self.sampling_spin)
        sampling_layout.addWidget(QLabel("seconds"))
        sampling_layout.addStretch()
        layout.addRow("Sampling Time:", sampling_layout)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
    
    def get_values(self):
        """Get current values"""
        return {
            'buffer': self.buffer_spin.value(),
            'sampling': self.sampling_spin.value()
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
        gain_group = QFormLayout()
        layout.addWidget(QLabel("Gain Values:"))
        
        self.gain_inputs = {}
        gains = SENSOR.DEFAULT_GAINS
        for metabolite, default_value in gains.items():
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 100.0)
            spin.setDecimals(3)
            spin.setValue(default_value)
            gain_group.addRow(f"{metabolite}:", spin)
            self.gain_inputs[metabolite] = spin
        
        layout.addLayout(gain_group)
        
        # Calibration values section
        layout.addWidget(QLabel("\nExpected Concentration Values (mM):"))
        
        cal_group = QFormLayout()
        self.calibration_inputs = {}
        calibrations = SENSOR.DEFAULT_CALIBRATIONS
        for metabolite, default_value in calibrations.items():
            spin = QDoubleSpinBox()
            spin.setRange(0.001, 1000.0)
            spin.setDecimals(3)
            spin.setValue(default_value)
            cal_group.addRow(f"{metabolite}:", spin)
            self.calibration_inputs[metabolite] = spin
        
        layout.addLayout(cal_group)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
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
        self.connect_btn.clicked.connect(lambda: asyncio.create_task(self._on_connect()))
        left_col.addWidget(self.connect_btn)
        
        self.start_btn = QPushButton("Start Sampling")
        self.start_btn.clicked.connect(lambda: asyncio.create_task(self._on_start()))
        self.start_btn.setEnabled(False)
        left_col.addWidget(self.start_btn)
        
        self.insert_btn = QPushButton("Insert")
        self.insert_btn.setEnabled(False)
        self.insert_btn.clicked.connect(lambda: asyncio.create_task(self._on_insert()))
        left_col.addWidget(self.insert_btn)
        
        self.eject_btn = QPushButton("Eject")
        self.eject_btn.setEnabled(False)
        self.eject_btn.clicked.connect(lambda: asyncio.create_task(self._on_eject()))
        left_col.addWidget(self.eject_btn)
        
        self.move_btn = QPushButton("Move (Ctrl wells)")
        self.move_btn.setEnabled(False)
        self.move_btn.clicked.connect(lambda: asyncio.create_task(self._on_move_ctrl()))
        left_col.addWidget(self.move_btn)
        
        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setStyleSheet("QPushButton { background:#e53935; color:white; font-weight:700; } QPushButton:hover{ background:#d32f2f; }")
        self.stop_btn.clicked.connect(lambda: asyncio.create_task(self._on_stop()))
        self.stop_btn.setEnabled(False)
        left_col.addWidget(self.stop_btn)
        
        self.settings_btn = QPushButton("Settings")
        self.settings_btn.clicked.connect(self._on_settings)
        left_col.addWidget(self.settings_btn)
        
        self.sensor_btn = QPushButton("Connect Sensor")
        self.sensor_btn.clicked.connect(lambda: asyncio.create_task(self._on_sensor_connect()))
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
        clear_btn.clicked.connect(lambda: asyncio.create_task(self._clear_selections()))
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
    
    async def _on_connect(self):
        """Handle connect button"""
        try:
            # Create connection
            self.connection = AsyncAmuzaConnection(
                device_address=HARDWARE.BLUETOOTH_DEVICE_ADDRESS,
                use_mock=True  # Use mock for testing
            )
            
            # Connect
            if await self.connection.connect():
                await self.app_state.set_connection(self.connection)
                
                self.status_label.setText("Connected")
                self.connect_btn.setEnabled(False)
                self.insert_btn.setEnabled(True)
                self.eject_btn.setEnabled(True)
                self.move_btn.setEnabled(True)
                self.start_btn.setEnabled(True)
                self.stop_btn.setEnabled(True)
                self.add_to_display("Connected to AMUZA.")
                
                logger.info("Connected to AMUZA")
            else:
                QMessageBox.warning(self, "Connection Failed", "Could not connect to device")
        
        except Exception as e:
            logger.error(f"Connection error: {e}")
            QMessageBox.critical(self, "Error", f"Connection failed: {e}")
    
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
            for well_id, label in self.well_labels.items():
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
    
    async def _clear_selections(self):
        """Clear all selections"""
        await self.app_state.clear_selections()
        for lbl in self.well_labels.values():
            lbl.set_selected(False)
            lbl.set_ctrl_selected(False)
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
    
    async def _on_start(self):
        """Handle start sampling"""
        try:
            wells = await self.app_state.get_selected_wells()
            
            if not wells:
                QMessageBox.warning(self, "No Wells", "Please select wells to sample")
                return
            
            await self.app_state.clear_stop()
            
            # Create sequence
            sequence = Sequence("Sampling Sequence")
            
            t_buffer, t_sampling = await self.app_state.get_timing_params()
            
            # Start experiment timer
            self._start_experiment_timer(len(wells), t_buffer, t_sampling)
            
            for well_id in wells:
                method = Method(
                    pos=well_id,
                    wait=t_buffer + t_sampling,
                    eject=True,
                    insert=True
                )
                sequence.add_method(method)
            
            # Execute sequence
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.add_to_display(f"Running plate on wells: {', '.join(sorted(wells))}")
            
            # Run in background task
            task = asyncio.create_task(self._execute_sequence(sequence))
            self.task_manager.add_task(task, "sampling_sequence")
        
        except Exception as e:
            logger.error(f"Start error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to start: {e}")
    
    async def _execute_sequence(self, sequence: Sequence):
        """Execute sampling sequence"""
        try:
            if not self.connection:
                logger.error("No connection")
                return
            
            # Get stop event from app state
            stop_event = self.app_state.stop_event
            stop_event.clear()
            
            # Execute
            completed = await self.connection.execute_sequence(sequence, stop_event)
            
            if completed:
                QMessageBox.information(self, "Complete", "Sampling sequence completed")
                self.add_to_display("Sampling sequence completed.")
            else:
                QMessageBox.information(self, "Stopped", "Sampling sequence stopped")
                self.add_to_display("Sampling sequence stopped.")
                self._stop_experiment_timer()
        
        except Exception as e:
            logger.error(f"Sequence error: {e}")
            QMessageBox.critical(self, "Error", f"Sequence failed: {e}")
            self._stop_experiment_timer()
        
        finally:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
    
    async def _on_stop(self):
        """Handle stop button"""
        await self.app_state.request_stop()
        logger.info("Stop requested")
        self.add_to_display("Stop requested.")
    
    async def _on_insert(self):
        """Insert tray"""
        if not self.connection:
            QMessageBox.warning(self, "Not Connected", "Connect first.")
            return
        await self.connection.insert()
        self.add_to_display("Insert command sent.")
    
    async def _on_eject(self):
        """Eject tray"""
        if not self.connection:
            QMessageBox.warning(self, "Not Connected", "Connect first.")
            return
        await self.connection.eject()
        self.add_to_display("Eject command sent.")
    
    async def _on_move_ctrl(self):
        """Run MOVE sequence on ctrl-selected wells"""
        try:
            wells = await self.app_state.get_selected_wells(ctrl=True)
            if not wells:
                QMessageBox.warning(self, "No Ctrl Wells", "Ctrl+click wells to move.")
                return
            await self.app_state.clear_stop()
            sequence = Sequence("Move Sequence")
            t_buffer, t_sampling = await self.app_state.get_timing_params()
            
            # Start experiment timer (for Move, only use sampling time, no buffer)
            self._start_experiment_timer(len(wells), 0, t_sampling)
            
            for well_id in wells:
                method = Method(pos=well_id, wait=t_sampling, eject=True, insert=True)
                sequence.add_method(method)
            
            self.add_to_display(f"Moving to wells: {', '.join(sorted(wells))}")
            task = asyncio.create_task(self._execute_sequence(sequence))
            self.task_manager.add_task(task, "move_sequence")
        except Exception as e:
            logger.error(f"Move error: {e}")
            QMessageBox.critical(self, "Error", f"Move failed: {e}")
    
    def _on_show_plot(self):
        """Show plot window"""
        if not self.plot_window:
            self.plot_window = PlotWindow(self.app_state)
        
        self.plot_window.show()
        self.plot_window.raise_()
    
    def _on_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self.app_state, self)
        
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            asyncio.create_task(
                self.app_state.set_timing_params(values['buffer'], values['sampling'])
            )
            logger.info(f"Settings updated: {values}")
    
    def _on_calibration(self):
        """Show calibration settings dialog"""
        dialog = CalibrationSettingsDialog(self.app_state, self)
        
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            # Store calibration values (could be added to app_state if needed)
            logger.info(f"Calibration updated: {values}")
            self.add_to_display("Calibration settings updated.")
    
    async def _on_sensor_connect(self):
        """Handle sensor connect/disconnect"""
        # If already connected, disconnect
        if self.sensor_reader and self.sensor_reader.is_running:
            await self.sensor_reader.stop()
            await self.sensor_reader.disconnect()
            self.sensor_reader = None
            self.sensor_btn.setText("Connect Sensor")
            self.sensor_status_label.setText("Sensor: Disconnected")
            self.add_to_display("Sensor disconnected.")
            return
        
        # Show connect dialog
        dialog = SensorConnectDialog(self)
        
        if dialog.exec_() == QDialog.Accepted:
            port = dialog.get_selected_port()
            if not port:
                return
            
            use_mock = (port == "MOCK")
            
            try:
                # Create sensor reader
                self.sensor_reader = AsyncPotentiostatReader(
                    port=port if not use_mock else "COM1",
                    use_mock=use_mock,
                    output_file=FILES.OUTPUT_FILE_PATH
                )
                
                # Connect
                if await self.sensor_reader.connect():
                    # Start reading in background
                    task = asyncio.create_task(self.sensor_reader.start_reading())
                    self.task_manager.add_task(task, "sensor_reading")
                    
                    self.sensor_btn.setText("Disconnect Sensor")
                    self.sensor_status_label.setText(f"Sensor: {port}")
                    self.add_to_display(f"Sensor connected on {port}")
                    logger.info(f"Sensor connected on {port}")
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
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up GUI resources")
        
        # Cancel all tasks
        await self.task_manager.cancel_all_tasks()
        
        # Disconnect AMUZA
        if self.connection:
            await self.connection.disconnect()
        
        # Disconnect sensor
        if self.sensor_reader:
            await self.sensor_reader.stop()
            await self.sensor_reader.disconnect()
        
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
    
    # Create task manager
    task_manager = AsyncTaskManager()
    
    # Create and show GUI
    gui = AsyncAMUZAGUI(app_state, task_manager)
    gui.show()
    
    # Wait for window to close
    try:
        while gui.isVisible():
            await asyncio.sleep(0.1)
    finally:
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
