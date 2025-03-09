import sys
import os
import time
import threading
from datetime import datetime
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
import serial
from serial.tools import list_ports
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QMainWindow,
    QDialog,
    QTextEdit,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QSpinBox,
    QSpacerItem,
    QSizePolicy,
    QPushButton,
    QLineEdit,
    QFileDialog,
    QAction,
    QMessageBox,
    QComboBox,
)

# Custom Imports
from SIX_SERVER_READER import PotentiostatReader
import AMUZA_Master

# Global variables
t_buffer = 60  # 65
t_sampling = 90  # 91
sample_rate = 1
connection = None  # This will be initialized after the user clicks 'Connect'
selected_wells = (
    set()
)  # Set to store wells selected with click-and-drag (used for RUNPLATE)
ctrl_selected_wells = (
    set()
)  # Set to store wells selected with Ctrl+Click (used for MOVE)

class WellLabel(QLabel):
    """Custom QLabel for well plate cells that supports click-and-drag and Ctrl+Click selection."""

    def __init__(self, well_id):
        super().__init__(well_id)
        self.well_id = well_id
        self.setFixedSize(50, 50)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: white; border: 1px solid black;")
        self.selected = False
        self.ctrl_selected = False

    def select(self):
        """Mark this cell as selected and change its color."""
        self.selected = True
        self.setStyleSheet("background-color: lightblue; border: 1px solid black;")

    def deselect(self):
        """Mark this cell as deselected and change its color."""
        self.selected = False
        self.setStyleSheet("background-color: white; border: 1px solid black;")

    def ctrl_select(self):
        """Mark this cell as Ctrl+selected for MOVE command."""
        self.ctrl_selected = True
        self.setStyleSheet("background-color: lightgreen; border: 1px solid black;")

    def ctrl_deselect(self):
        """Deselect this cell for MOVE command."""
        self.ctrl_selected = False
        self.setStyleSheet("background-color: white; border: 1px solid black;")


class PlotWindow(QMainWindow):
    """Window for displaying and saving sensor data in real-time with automatic logging."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent  # Explicitly assign the parent
        self.setWindowTitle("Data Plot")
        self.setGeometry(200, 200, 1100, 800)
        self.data_list = []
        self.is_recording = False
        self.connection_status = False
        self.serial_connection = None
        self.thread = None
        self.stop_event = threading.Event()
        self.default_file_path = None
        self.loaded_file_path = None  # Keep track of the loaded file
        self.header = ['counter', 't[min]', '#1ch1', '#1ch2', '#1ch3', '#1ch4', '#1ch5', '#1ch6', '#1ch7']
        self.gain_values = {
                    "Glutamate": 3.394,
                    "Glutamine": 0.974,
                    "Glucose": 1.5,
                    "Lactate": 0.515,
                }
        self.mock_data_mode = False  # Add mock data mode flag
        self.mock_data_df = pd.DataFrame()  # Add mock data DataFrame

        # Calibration values
        self.calibration_glutamate = 0.996
        self.calibration_glutamine = 1.0
        self.calibration_glucose = 17.38
        self.calibration_lactate = 9.94

        main_layout = QVBoxLayout()

        # Create a vertical layout for the graph (canvas + toolbar)
        graph_layout = QVBoxLayout()

        # Set up the matplotlib figure and canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.nav_toolbar = NavigationToolbar(self.canvas, self)

        # Add the canvas and navigation toolbar to the graph layout
        graph_layout.addWidget(self.canvas)
        graph_layout.addWidget(self.nav_toolbar)

        # Create a QTextEdit widget for instructions
        self.instructions_text = QTextEdit(self)
        self.instructions_text.setReadOnly(True)
        self.instructions_text.setStyleSheet(
            "background-color: #F7F7F7; border: 1px solid #D0D0D0;"
        )
        self.instructions_text.setText(
            "Plot Instructions:\n"
            "1. Connect the Sensor:\n"
            "    o Click 'Sensor' at the top and 'Connect' \n"
            "\n"
            "    o Select which port it is plugged into and 'Connect' \n"
            "\n"
            "Connecting the Sensor will automatical log data readings into a file under folder called sensor readings.\n"
            "\n"
            "Clicking the 'Connect/Disconnect when Connected will Disconnect the Sensor'\n"
            "\n"
            "2. Click 'Start Record' to save the data to a specific file.\n"
            "\n"
            "3. Calibrate the Gain Values:\n"
            "    o Click 'Calibration Settings' at the bottom right \n"
            "\n"
            "    o Set the expected concentration values [mM] for each metabolite \n"
            "\n"
            "    o Run the calibration fluid in the system and wait for the values to steady. \n"
            "\n"
            "    o Click 'Sensor' then 'Calibrate' to update the gain\n"
            "\n"
            "4. Use 'File' and 'Open' to load a saved graph. \n"
            "\n"
            "5. Use the toolbar for zooming and panning.\n"
            "\n"
            "6. Modify the gain values at the bottom for quick changes to the plot"
        )
        self.instructions_text.setFixedWidth(300)

        # Create a horizontal layout to hold the graph and instructions side by side
        plot_instructions_layout = QHBoxLayout()
        plot_instructions_layout.addLayout(
            graph_layout
        )  # Add the graph layout (canvas + toolbar)
        plot_instructions_layout.addWidget(
            self.instructions_text
        )  # Add the instructions text panel

        self.log_file_path = None

        # Set up the menu bar with "File" and "Sensor" dropdown menus
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("File")
        load_action = QAction("Load Saved", self)
        load_action.triggered.connect(self.pick_file)
        file_menu.addAction(load_action)

        save_action = QAction("Save As", self)
        save_action.triggered.connect(self.save_file)
        file_menu.addAction(save_action)

        # Sensor Menu
        sensor_menu = menu_bar.addMenu("Sensor")

        connect_action = QAction("Connect/Disconnect", self, checkable=True)
        connect_action.triggered.connect(self.connect_to_sensor)
        sensor_menu.addAction(connect_action)

        # Mock Data Mode action
        mock_data_action = QAction("Mock Data Mode", self, checkable=True)
        mock_data_action.triggered.connect(self.toggle_mock_data_mode)
        sensor_menu.addAction(mock_data_action)

        # Add Calibrate action to the Sensor Menu
        calibrate_action = QAction("Calibrate", self)
        calibrate_action.triggered.connect(self.calibrate_sensors)
        sensor_menu.addAction(calibrate_action)

        # Status label for connection state
        self.status_label = QLabel("Disconnected")
        self.status_label.setAlignment(Qt.AlignRight)

        # Create a widget to hold the status label and add it to the menu bar
        status_widget = QWidget(self)
        status_layout = QHBoxLayout()
        status_layout.addWidget(self.status_label)
        status_layout.setContentsMargins(0, 3, 20, 0)
        status_widget.setLayout(status_layout)

        # Add the status widget to the right side of the menu bar
        menu_bar.setCornerWidget(status_widget, Qt.TopRightCorner)

        # Create the gain layout
        gain_layout = QHBoxLayout()
        self.gain_inputs = {}
        for metabolite in ["Glutamate", "Glutamine", "Glucose", "Lactate"]:
            label = QLabel(f"{metabolite} Gain:")
            input_field = QLineEdit()
            input_field.setText(str(self.gain_values[metabolite]))
            input_field.setFixedWidth(60)
            input_field.setFixedHeight(22)
            input_field.returnPressed.connect(self.update_gain_values)
            gain_layout.addWidget(label)
            gain_layout.addWidget(input_field)
            self.gain_inputs[metabolite] = input_field
  
        # Add a spacer with a specific width to position the OK button
        spacer = QSpacerItem(16, 20, QSizePolicy.Fixed, QSizePolicy.Minimum)
        gain_layout.addSpacerItem(spacer)

        # Create an "OK" button
        ok_button = QPushButton("OK")
        ok_button.setFixedWidth(60)
        ok_button.clicked.connect(self.update_gain_values)  # Connect button to update method
        
        # Add button to the layout
        gain_layout.addWidget(ok_button)
        # Add a horizontal stretch to push the Calibration Settings button to the right
        gain_layout.addStretch()

        # Create and add the Calibration Settings button
        calibration_button = QPushButton("Calibration Settings", self)
        calibration_button.setFixedWidth(150)
        calibration_button.clicked.connect(self.open_calibration_settings)
        gain_layout.addWidget(calibration_button)

        # Combine the gain layout and the horizontal plot + instructions layout
        main_layout.addLayout(
            plot_instructions_layout
        )  # Add the combined graph + instructions layout
        main_layout.addLayout(gain_layout)  # Add the gain layout

        # Set the central widget for the window
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Set up FuncAnimation for continuous plotting
        self.anim = FuncAnimation(
            self.figure, self.update_plot, interval=2000, save_count=100
        )

    def toggle_mock_data_mode(self):
        """Toggle mock data mode for testing."""
        self.mock_data_mode = not self.mock_data_mode
        if self.mock_data_mode:
            self.status_label.setText("Mock Data Mode")
        else:
            self.status_label.setText("Disconnected")

    def generate_mock_data(self):
        """Generate and append mock data for testing."""
        # find existing time
        if self.mock_data_df.empty:
            t = np.linspace(0, 1, 30)
        else:
            last_t = self.mock_data_df["t[min]"].values[-1]
            t = np.linspace(last_t, last_t + 1, 30)
        new_data = {
            "t[min]": t,
            "#1ch1": np.sin(t),
            "#1ch2": np.cos(t),
            "#1ch3": np.sin(t) + np.random.normal(0, 0.1, len(t)),
            "#1ch4": np.cos(t) + np.random.normal(0, 0.1, len(t)),
            "#1ch5": np.sin(t) * 2,
            "#1ch6": np.cos(t) * 2,
        }
        new_df = pd.DataFrame(new_data)
        self.mock_data_df = pd.concat([self.mock_data_df, new_df], ignore_index=True)

    def run_datalogger(self, file_path):
        """Run the data logger, save data to file, and log updates to the command line."""
        while not self.stop_event.is_set():
            try:
                print(f"Starting data logger on COM port: {self.selected_port}")
                self.DataLogger = PotentiostatReader(
                    com_port=self.selected_port,
                    baud_rate=9600,
                    timeout=0.5,
                    output_filename=file_path,
                )
                self.DataLogger.run()
            except Exception as e:
                print(f"Error during data logging: {str(e)}")
            
    def update_initial_plot(self, df):
        if df is None:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            x = np.linspace(0, 10, 100)
            y = np.sin(x)
            ax.plot(x, y, label="Default Sine Wave")
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_title("Default Plot: Sine Wave")
            ax.legend()
            ax.grid(True)
            self.current_plot_type = "default"  # Set the plot type to default
            self.figure.subplots_adjust(
                top=0.955, bottom=0.066, left=0.079, right=0.990
            )
            self.canvas.draw()
        else:
            # Calculate metabolites
            metabolites = {
                "Glutamate": df["#1ch1"] - df["#1ch2"],
                "Glutamine": df["#1ch3"] - df["#1ch1"],
                "Glucose": df["#1ch5"] - df["#1ch4"],
                "Lactate": df["#1ch6"] - df["#1ch4"],
            }

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            for metabolite, values in metabolites.items():
                scaled_values = values * self.gain_values[metabolite]
                ax.plot(df["t[min]"], scaled_values, label=metabolite)
            ax.set_xlabel("Time (minutes)")
            ax.set_ylabel("mA")
            ax.set_title("Time Series Data for Selected Channels")
            ax.legend()
            ax.grid(True)
            # set to start at 0
            ax.set_xlim(0, df["t[min]"].max())
            self.canvas.draw_idle()

    def update_plot(self, frame):
        """Update the plot with the given dataset."""
        if self.mock_data_mode:
            df = self.generate_mock_data()
            df = self.mock_data_df

        else:
            if self.log_file_path is None or not os.path.exists(self.log_file_path):
                return
            with open(self.log_file_path, "r", newline="") as file:
                data = [
                    line.strip().split("\t")
                    for line in file
                ]
            if not data:
                return
            df = self.clean_data(data)
            df = df.apply(pd.to_numeric, errors="coerce")
        try:
            # Calculate metabolites
            metabolites = {
                "Glutamate": df["#1ch1"] - df["#1ch2"],
                "Glutamine": df["#1ch3"] - df["#1ch1"],
                "Glucose": df["#1ch5"] - df["#1ch4"],
                "Lactate": df["#1ch6"] - df["#1ch4"],
            }
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            for metabolite, values in metabolites.items():
                scaled_values = values * self.gain_values[metabolite]
                ax.plot(df["t[min]"], scaled_values, label=metabolite) #
            ax.set_xlim(0, df["t[min]"].max())
            ax.set_ylim(0, df.max())

        except Exception as e:
            #print(e)
            False
        self.canvas.draw_idle()
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("mA")
        ax.set_title("Time Series Data for Selected Channels")
        ax.legend()
        ax.grid(True)

    def calibrate_sensors(self):
        """Perform calibration of the sensors based on current data values."""
        if not self.is_recording:
            QMessageBox.warning(
                self,
                "Calibration Error",
                "Calibration can only be performed during data recording.",
            )
            return
        try:
            # Extract the current values from data_list
            current_glutamate = self.data_list[0] - self.data_list[1]
            current_glutamine = self.data_list[2] - self.data_list[0]
            current_glucose = self.data_list[4] - self.data_list[3]
            current_lactate = self.data_list[5] - self.data_list[3]

            # Update gain values based on calibration
            if self.calibration_glutamate > 0:
                self.gain_values["Glutamate"] = (
                    self.calibration_glutamate / current_glutamate
                )
            if self.calibration_glutamine > 0:
                self.gain_values["Glutamine"] = (
                    self.calibration_glutamine / current_glutamine
                )
            if self.calibration_glucose > 0:
                self.gain_values["Glucose"] = self.calibration_glucose / current_glucose
            if self.calibration_lactate > 0:
                self.gain_values["Lactate"] = self.calibration_lactate / current_lactate

            QMessageBox.information(
                self, "Calibration", "Calibration completed successfully."
            )
            if self.parent:
                self.parent.add_to_display(
                    "Calibration completed and gain values updated."
                )
            self.update_gain_values()
        except Exception as e:
            QMessageBox.critical(
                self, "Calibration Error", f"Failed to calibrate sensors: {str(e)}"
            )

    def open_calibration_settings(self):
        """Open the Calibration Settings dialog."""
        dialog = CalibrationSettingsDialog(self)
        dialog.exec_()
        if self.parent:
            self.parent.add_to_display("Calibration settings updated.")

    def save_file(self):
        """Save a copy of the current data file to a specified location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save File", "", "Text Files (*.txt)"
        )
        if file_path:
            try:
                if file_path.endswith(".txt"):
                    if self.default_file_path and os.path.exists(self.default_file_path):
                        # Save the recorded data file
                        with open(self.default_file_path, "r") as source_file:
                            with open(file_path, "w") as dest_file:
                                dest_file.write(source_file.read())
                        QMessageBox.information(
                            self, "Success", f"Data successfully saved to {file_path}"
                        )
                    elif self.loaded_file_path and os.path.exists(self.loaded_file_path):
                        # Save the loaded data file
                        with open(self.loaded_file_path, "r") as source_file:
                            with open(file_path, "w") as dest_file:
                                dest_file.write(source_file.read())
                        QMessageBox.information(
                            self, "Success", f"Data successfully saved to {file_path}"
                        )
                    else:
                        QMessageBox.warning(
                            self, "Warning", "No data is available to save."
                        )
                else:
                    QMessageBox.warning(
                        self, "Warning", "Please use a .txt extension to save the data."
                    )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file: {e}")


    def pick_file(self):
        """Open a file dialog to select a file and load it into the plot."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "Text Files (*.txt);;All Files (*)"
        )
        if not file_path:
            print("No file selected.")
            return

        if not os.path.exists(file_path):
            print("Invalid file path.")
            return
        self.load_file(file_path)

    def load_file(self,file_path):
        with open(file_path, "r", newline="") as file:
            lines = file.readlines()

        if len(lines) < 4:  # Ensure there is enough data for processing
            print("Insufficient data in log file.")
            # show dialog
            QMessageBox.warning(
                self,
                "Insufficient Data",
                "The selected file does not contain enough data for plotting.",
            )
            return
        data = [line.strip().split("\t") for line in lines]
        df = self.clean_data(data)
        # Remove comments at the end if they appear
        index = []
        for i in range(3, len(df) + 3):
            a = df.loc[i, "counter"]
            if not a.isdigit():
                index.append(i)
                break

        if index:
            df = df.loc[: index[0] - 1, :]
            df = df.apply(pd.to_numeric, errors="coerce")
        else:
            df = df.apply(pd.to_numeric, errors="coerce")

        # Track the loaded file path
        self.loaded_file_path = file_path
        self.update_initial_plot(df)

    def connect_to_sensor(self):
        """Toggle between connecting and disconnecting the sensor."""
        if self.connection_status:  # If already connected, disconnect
            """Disconnect from the sensor and update the status label."""
            try:
                if self.serial_connection:
                    self.serial_connection.close()
                self.serial_connection = None
                self.connection_status = False
                self.DataLogger.close_serial_connection()
                self.stop_event.set()
                self.thread.join()
                self.status_label.setText("Disconnected")  # Update status label
                QMessageBox.information(
                    self, "Disconnected", "Sensor disconnected successfully."
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Disconnection Error", f"Failed to disconnect: {str(e)}"
                )
        else:
            ports = [port.device for port in list_ports.comports()]
            if not ports:
                QMessageBox.warning(
                    self, "No Ports Found", "No COM ports are available."
                )
                return

            dialog = QDialog(self)
            dialog.setWindowTitle("Select COM Port")

            layout = QVBoxLayout()
            port_selector = QComboBox()
            port_selector.addItems(ports)
            layout.addWidget(QLabel("Available COM Ports:"))
            layout.addWidget(port_selector)

            connect_button = QPushButton("Connect")
            connect_button.clicked.connect(
                lambda: self.establish_connection(dialog, port_selector.currentText())
            )
            layout.addWidget(connect_button)

            dialog.setLayout(layout)
            dialog.exec_()
    
    def clean_data(self,data):
        df = pd.DataFrame(data)
        df = df.loc[:, :8]
        df = df[3:]
        df.columns = self.header
        return df
    
    def establish_connection(self, dialog, selected_port):
        """Establish a connection to the selected COM port and start continuous logging and plotting."""
        try:
            self.serial_connection = serial.Serial(
                selected_port, baudrate=9600, timeout=1
            )
            self.selected_port = selected_port
            self.connection_status = True
            self.status_label.setText("Connected")  # Update status label
            dialog.accept()

            print(f"Connected to COM port: {self.selected_port}")

            # Start continuous logging in a separate thread
            logger_folder = "Sensor_Readings"
            os.makedirs(logger_folder, exist_ok=True)
            current_time = datetime.now()
            filename = f"Sensor_readings_{current_time.strftime('%d_%m_%y_%H_%M')}.txt"
            self.default_file_path = os.path.join(logger_folder, filename)
            self.log_file_path = self.default_file_path
            self.thread = threading.Thread(
                target=self.run_datalogger, args=(self.default_file_path,), daemon=True
            )
            self.thread.start()
            # run the plot start before continous plotting just to get it started
            QMessageBox.information(
                self, "Info", "Connected to sensor and started logging and plotting."
            )
        except serial.SerialException as e:
            QMessageBox.critical(
                self,
                "Connection Error",
                f"Could not connect to {selected_port}.\nError: {e}",
            )
            self.status_label.setText("Disconnected")  # Ensure status is reset
            self.selected_port = None

    def update_gain_values(self):
        """Update gain values based on user input and re-plot the data."""
        for metabolite, input_field in self.gain_inputs.items():
            try:
                new_value = float(input_field.text())
                self.gain_values[metabolite] = new_value
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    f"Please enter a valid number for {metabolite} gain.",
                )
                return
        # Re-plot the data with updated gains
        if hasattr(self, "data") and not self.data.empty:
            self.update_plot(self.data)

        elif self.loaded_file_path:
            self.load_file(self.loaded_file_path)


class SettingsDialog(QDialog):
    """Settings window to adjust t_sampling and t_buffer."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.parent = parent  # Store the reference to the parent AMUZAGUI

        # Layout for the form
        layout = QFormLayout()

        # Spin boxes for t_sampling and t_buffer
        self.sampling_time_spinbox = QSpinBox()
        self.sampling_time_spinbox.setRange(1, 1000)
        self.sampling_time_spinbox.setValue(t_sampling)

        self.buffer_time_spinbox = QSpinBox()
        self.buffer_time_spinbox.setRange(1, 1000)
        self.buffer_time_spinbox.setValue(t_buffer)

        # Add to layout
        layout.addRow("Sampling Time:", self.sampling_time_spinbox)
        layout.addRow("Buffer Time:", self.buffer_time_spinbox)

        # Add Ok and Cancel buttons
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept_settings)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def accept_settings(self):
        """Update t_sampling and t_buffer when OK is pressed."""
        global t_sampling, t_buffer
        t_sampling = self.sampling_time_spinbox.value()
        t_buffer = self.buffer_time_spinbox.value()
        super().accept()


class CalibrationSettingsDialog(QDialog):
    """Dialog for adjusting calibration values for each metabolite."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration Settings")

        # Layout for the form
        layout = QFormLayout()

        # Get the current calibration values from the parent (PlotWindow)
        self.parent = parent

        # Calibration input fields for each metabolite
        self.calibration_inputs = {}
        for metabolite in ["Glutamate", "Glutamine", "Glucose", "Lactate"]:
            label = QLabel(f"{metabolite} [mM]")
            input_field = QLineEdit()

            # Set the input field to the current calibration value
            current_value = getattr(
                self.parent, f"calibration_{metabolite.lower()}", 0.0
            )
            input_field.setText(str(current_value))

            layout.addRow(label, input_field)
            self.calibration_inputs[metabolite] = input_field

        # Ok button
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def accept(self):
        """Save calibration values when OK is pressed."""
        parent = self.parent
        if parent:
            try:
                # Update calibration values in the parent
                parent.calibration_glutamate = float(
                    self.calibration_inputs["Glutamate"].text()
                )
                parent.calibration_glutamine = float(
                    self.calibration_inputs["Glutamine"].text()
                )
                parent.calibration_glucose = float(
                    self.calibration_inputs["Glucose"].text()
                )
                parent.calibration_lactate = float(
                    self.calibration_inputs["Lactate"].text()
                )
                QMessageBox.information(
                    self, "Success", "Calibration values updated successfully."
                )
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    "Please enter valid numbers for calibration values.",
                )
        super().accept()


class AMUZAGUI(QWidget):
    well_complete_signal = pyqtSignal(int)  # Signal for well completion
    move_complete_signal = pyqtSignal()    # Signal for move completion
    process_stopped_signal = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Set up the window
        self.setWindowTitle("AMUZA Controller")
        self.setGeometry(100, 100, 1250, 400)
        self.setFixedSize(1250, 500)  # Prevents the window from being resized

        # Main layout - Horizontal
        self.main_layout = QHBoxLayout(self)

        # Left side layout for commands
        self.command_layout = QVBoxLayout()

        # Display screen at the top left for showing output text with history
        self.display_screen = QTextEdit(self)
        self.display_screen.setReadOnly(True)
        self.display_screen.setFixedHeight(230)  # Set height to 160 pixels
        self.display_screen.setVerticalScrollBarPolicy(
            Qt.ScrollBarAlwaysOn
        )  # Add vertical scroll bar
        self.command_layout.addWidget(self.display_screen)

        # Store display history
        self.display_history = []

        rounded_button_style = """
            QPushButton {
                background-color: #FDFDFD ; /* Light grey background */
                border: 1px solid #D0D0D0; /* Neutral grey border */
                border-radius: 10px; /* Slight rounding of the corners */
                padding: 2px 8px; /* Adjusted padding for a more compact look */
                font-size: 13px; /* Smaller font size */
                max-width: 170px; /* Maximum width to fit the text comfortably */
                max-height: 32px; /* Maximum height for a smaller button */
            }
            QPushButton:hover {
                background-color: #C0C0C0; /* Darker grey on hover */
            }
            QPushButton:pressed {
                background-color: #D3D3D3; /* Even darker grey when pressed */
            }
        """
        stop_button_style = """
            QPushButton {
                background-color: red ; /* Light grey background */
                border: 1px solid black; /* Neutral grey border */
                border-radius: 10px; /* Slight rounding of the corners */
                padding: 2px 8px; /* Adjusted padding for a more compact look */
                font-size: 13px; /* Smaller font size */
                max-width: 170px; /* Maximum width to fit the text comfortably */
                max-height: 32px; /* Maximum height for a smaller button */
            }
            QPushButton:hover {
                background-color: #C0C0C0; /* Darker grey on hover */
            }
            QPushButton:pressed {
                background-color: #D3D3D3; /* Even darker grey when pressed */
            }
        """

        # Connect button
        # Connect button
        self.connect_button = QPushButton("Connect to AMUZA", self)
        self.connect_button.setStyleSheet(rounded_button_style)
        self.connect_button.clicked.connect(self.connect_to_amuza)
        self.command_layout.addWidget(self.connect_button)

        # Control buttons (initially greyed out and disabled)
        self.start_datalogger_button = QPushButton("Start DataLogger", self)
        self.start_datalogger_button.setEnabled(False)
        self.start_datalogger_button.setStyleSheet(rounded_button_style)
        self.start_datalogger_button.clicked.connect(self.open_plot_window)
        self.command_layout.addWidget(self.start_datalogger_button)

        self.insert_button = QPushButton("INSERT", self)
        self.insert_button.setEnabled(False)
        self.insert_button.setStyleSheet(rounded_button_style)
        self.insert_button.clicked.connect(self.on_insert)
        self.command_layout.addWidget(self.insert_button)

        self.eject_button = QPushButton("EJECT", self)
        self.eject_button.setEnabled(False)
        self.eject_button.setStyleSheet(rounded_button_style)
        self.eject_button.clicked.connect(self.on_eject)
        self.command_layout.addWidget(self.eject_button)

        self.runplate_button = QPushButton("RUNPLATE", self)
        self.runplate_button.setEnabled(False)
        self.runplate_button.setStyleSheet(rounded_button_style)
        self.runplate_button.clicked.connect(self.on_runplate)
        self.command_layout.addWidget(self.runplate_button)

        self.move_button = QPushButton("MOVE", self)
        self.move_button.setEnabled(False)
        self.move_button.setStyleSheet(rounded_button_style)
        self.move_button.clicked.connect(self.on_move)
        self.command_layout.addWidget(self.move_button)

        # Stop button
        self.stop_button = QPushButton("STOP", self)
        self.stop_button.setStyleSheet(stop_button_style)
        self.stop_button.clicked.connect(self.toggle_stop_flag)
        self.command_layout.addWidget(self.stop_button)

        # Settings button
        self.settings_button = QPushButton("Settings", self)
        self.settings_button.setStyleSheet(rounded_button_style)
        self.settings_button.clicked.connect(self.open_settings_dialog)
        self.command_layout.addWidget(self.settings_button)

        # Add the command layout to the main layout (on the left side)
        self.main_layout.addLayout(self.command_layout)

        # Right side layout for the well plate
        self.plate_layout = QGridLayout()
        self.well_labels = {}
        self.start_row, self.start_col = None, None
        self.is_dragging = False

        self.setup_well_plate()
        # Instructions panel (right side)
        self.instructions_panel = QTextEdit(self)
        self.instructions_panel.setReadOnly(True)
        self.instructions_panel.setFixedWidth(350)  # Adjust width as needed
        self.instructions_panel.setStyleSheet(
            "background-color: #F7F7F7; border: 1px solid #D0D0D0;"
        )
        self.instructions_panel.setText(
            "Instructions:\n"
            "1. Connect to AMUZA using the 'Connect to AMUZA' button.\n"
            "\n"
            "2. Use 'EJECT' to remove the tray from inside the AMUZA and 'INSERT' to insert it.\n"
            "\n"
            "3. Select the well sampling area by clicking and dragging across the figure. ('Clear' clears the selection)\n"
            "\n"
            "4. Use 'RUNPLATE' to sample the selected wells (blue) in a combing sequence as displayed.\n"
            "\n"
            "5. Select individual wells by Ctrl+Click for 'MOVE'.\n"
            "\n"
            "6. Use 'MOVE' to sample the selected wells (green) in order.\n"
            "\n"
            "7. Use 'Settings' to adjust sampling and buffer rest times.\n"
            "\n"
            "8. Use 'Start DataLogger' to open up the plotting window.\n"
            "\n"
            "9. Review messages and logs in the display panel."
            "\n"
            "Coded By: Noah Bernten         Noah.Bernten@mail.huji.ac.il"
        )

        # Add the well plate layout to the main layout (on the right side)
        self.main_layout.addLayout(self.plate_layout)
        self.main_layout.addWidget(self.instructions_panel)

        # Set the layout for the QWidget
        self.setLayout(self.main_layout)
        self.stop_flag = False

        # Connect signals to slots
        self.well_complete_signal.connect(self.update_display_for_well)
        self.move_complete_signal.connect(self.on_moves_complete)
        self.process_stopped_signal.connect(self.on_process_stopped)
        self.inserted = True

    def toggle_stop_flag(self):
        """Toggle the stop flag."""
        if not self.stop_flag:
            self.stop_flag = True
            self.add_to_display(f"Stopping")
        
    def setup_well_plate(self):
        """Create a grid of 8x12 QLabel items representing the well plate."""
        rows = "ABCDEFGH"
        columns = range(1, 13)

        for i, row in enumerate(rows):
            for j, col in enumerate(columns):
                well_id = f"{row}{col}"
                label = WellLabel(well_id)
                self.well_labels[(i, j)] = label
                self.plate_layout.addWidget(label, i, j)

        # Add Clear button at the bottom spanning the full width
        clear_button = QPushButton("Clear", self)
        clear_button.clicked.connect(self.clear_plate_selection)
        self.plate_layout.addWidget(clear_button, len(rows), 0, 1, len(columns))

    def clear_plate_selection(self):
        """Clear all selected wells (both drag and Ctrl+Click selections)."""
        for label in self.well_labels.values():
            label.deselect()
            label.ctrl_deselect()
        selected_wells.clear()
        ctrl_selected_wells.clear()

    def add_to_display(self, message):
        """Add a new message to the display history and update the display screen."""
        self.display_history.append(message)
        # Limit history to last 50 messages for readability
        self.display_history = self.display_history[-50:]
        # Display the history in the QTextEdit
        self.display_screen.setPlainText("\n".join(self.display_history))

    def order(self, well_positions):
        """Orders the selction sequence to run from A1 to A12 down till G12"""
        sorted_well_positions = sorted(well_positions, key=lambda x: (x[0], int(x[1:])))
        return sorted_well_positions

    def apply_button_style(self, button):
        """Reapply the custom rounded style to the button."""
        rounded_button_style = """
            QPushButton {
                background-color: #FDFDFD ; /* Light grey background */
                border: 1px solid #D0D0D0; /* Neutral grey border */
                border-radius: 10px; /* Slight rounding of the corners */
                padding: 2px 8px; /* Adjusted padding for a more compact look */
                font-size: 13px; /* Smaller font size */
                max-width: 170px; /* Maximum width to fit the text comfortably */
                max-height: 32px; /* Maximum height for a smaller button */
            }
            QPushButton:hover {
                background-color: #C0C0C0; /* Darker grey on hover */
            }
            QPushButton:pressed {
                background-color: #D3D3D3; /* Even darker grey when pressed */
            }
        """
        button.setStyleSheet(rounded_button_style)

    def on_runplate(self):
        """Display the selected wells for RUNPLATE in the console and display screen."""
        self.stop_flag = False
        if selected_wells:
            if not self.inserted:
                self.on_insert()
                time.sleep(6)
                self.inserted = True
            self.well_list = self.order(list(selected_wells))
            self.add_to_display(
                f"Running Plate on wells: {', '.join(self.well_list)}\nSampled:"
            )
            if connection is None:
                QMessageBox.critical(self, "Error", "Please connect to AMUZA first!")
                return
            # Reset method list
            self.method = []
            # Adjust temperature before starting
            connection.AdjustTemp(6)
            # Map the wells and create method sequences
            locations = connection.well_mapping(self.well_list)
            for loc in locations:
                # Append the method sequence for each location
                self.method.append(
                    AMUZA_Master.Sequence([AMUZA_Master.Method([loc], t_sampling)])
                )

            # Start the Control_Move process in a separate thread
            thread = threading.Thread(
                target=self.Control_Move, args=(self.method, t_sampling), daemon=True
            )
            thread.start()
        else:
            self.add_to_display("No wells selected for RUNPLATE.")

    def on_move(self):
            """Display the selected wells for MOVE in the console and display screen."""
            self.stop_flag = False
            if ctrl_selected_wells:
                if not self.inserted:
                    self.on_insert()
                    time.sleep(6)
                    self.inserted = True
                self.well_list = self.order(list(ctrl_selected_wells))
                self.add_to_display(f"Moving to wells: {', '.join(self.well_list)}")
                self.add_to_display("Sampled: ")
                if connection is None:
                    QMessageBox.critical(self, "Error", "Please connect to AMUZA first!")
                    return
                # Reset method list
                self.method = []

                # Adjust temperature before moving
                connection.AdjustTemp(6)

                # Map the wells and move
                locations = connection.well_mapping(self.well_list)
                for loc in locations:
                    # Append the method sequence for each location
                    self.method.append(
                        AMUZA_Master.Sequence([AMUZA_Master.Method([loc], t_sampling)])
                    )

                # Start the Control_Move process in a separate thread
                thread = threading.Thread(target=self.Control_Move, args=(self.method, t_sampling), daemon=True)
                thread.start()
            else:
                self.add_to_display("No wells selected for MOVE.")


    def Control_Move(self, method, duration):
        """Simulate movement of the AMUZA system."""
        for i, step in enumerate(method):
            if self.stop_flag:
                self.process_stopped_signal.emit()  # Emit stop signal
                return  # Exit the method immediately
            
            time.sleep(t_buffer)  # Simulate buffer time
            connection.Move(step)  # Perform the move operation
            self.well_complete_signal.emit(i)  # Emit signal for well completion
            
            delay = 1
            time.sleep(duration + 9 + delay)  # Simulate move duration

        self.move_complete_signal.emit()  # Emit completion signal if not stopped

    def update_display_for_well(self, well_index):
        """Update the display when a well is completed."""
        current_text = self.display_screen.toPlainText()
        updated_text = f"{current_text}{self.well_list[well_index]}, "
        self.display_screen.setPlainText(updated_text)
        self.display_screen.moveCursor(self.display_screen.textCursor().End)

    def on_moves_complete(self):
        """Handle completion of all moves."""
        self.add_to_display(f"{', '.join(self.well_list)} Complete.")

    def on_process_stopped(self):
        """Handle the process being stopped."""
        self.add_to_display("Process stopped by the user.")

    def open_settings_dialog(self):
        """Open the settings dialog to adjust t_sampling and t_buffer."""
        dialog = SettingsDialog(self)
        dialog.exec_()

    def open_plot_window(self):
        """Open a new window to display the data plot."""
        self.plot_window = PlotWindow(self)
        self.plot_window.show()

    def connect_to_amuza(self):
        """Connect to the AMUZA system."""
        global connection
        try:
            connection = AMUZA_Master.AmuzaConnection(True)
            connection.connect()
            QMessageBox.information(self, "Info", "Connected to AMUZA successfully!")
            self.enable_control_buttons()
            self.add_to_display("Connected to AMUZA.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to connect to AMUZA: {str(e)}")
            connection = None
            self.add_to_display("Failed to connect to AMUZA.")

    def enable_control_buttons(self):
        """Enable the control buttons after successful connection."""
        buttons = [
            self.start_datalogger_button,
            self.insert_button,
            self.eject_button,
            self.runplate_button,
            self.move_button,
        ]
        for button in buttons:
            button.setEnabled(True)
            self.apply_button_style(button)

    def on_insert(self):
        if connection is None:
            QMessageBox.critical(self, "Error", "Please connect to AMUZA first!")
            return
        connection.Insert()
        self.add_to_display("Inserting tray.")
        self.inserted = True

    def on_eject(self):
        if connection is None:
            QMessageBox.critical(self, "Error", "Please connect to AMUZA first!")
            return
        connection.Eject()
        self.add_to_display("Ejecting tray.")
        self.inserted = False

    def resizeEvent(self, event):
        """Lock the aspect ratio of the window."""
        width = event.size().width()
        aspect_ratio = 9 / 4
        new_height = int(width / aspect_ratio)
        self.resize(QSize(width, new_height))
        super().resizeEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press event to start a selection or toggle a single well."""
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ControlModifier:
                for (i, j), label in self.well_labels.items():
                    if label.geometry().contains(self.mapFromGlobal(event.globalPos())):
                        self.toggle_ctrl_well(i, j)
                        break
            else:
                for (i, j), label in self.well_labels.items():
                    if label.geometry().contains(self.mapFromGlobal(event.globalPos())):
                        self.start_row, self.start_col = i, j
                        self.is_dragging = True
                        self.update_selection(i, j)
                        break

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move event to update the selection during drag."""
        if self.is_dragging:
            for (i, j), label in self.well_labels.items():
                if label.geometry().contains(self.mapFromGlobal(event.globalPos())):
                    self.update_selection(i, j)
                    break

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release event to finish the selection."""
        if self.is_dragging:
            self.is_dragging = False

    def update_selection(self, end_row, end_col):
        """Update the selection from the start position to the current cursor position."""
        selected_wells.clear()
        min_row, max_row = min(self.start_row, end_row), max(self.start_row, end_row)
        min_col, max_col = min(self.start_col, end_col), max(self.start_col, end_col)
        for label in self.well_labels.values():
            label.deselect()
        for i in range(min_row, max_row + 1):
            for j in range(min_col, max_col + 1):
                self.well_labels[(i, j)].select()
                selected_wells.add(self.well_labels[(i, j)].well_id)

    def toggle_ctrl_well(self, row, col):
        """Toggle the well selection for the MOVE command (Ctrl+Click functionality)."""
        label = self.well_labels[(row, col)]
        well_id = label.well_id
        if well_id in ctrl_selected_wells:
            ctrl_selected_wells.remove(well_id)
            label.ctrl_deselect()
        else:
            ctrl_selected_wells.add(well_id)
            label.ctrl_select()

# Start the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AMUZAGUI()
    window.show()
    sys.exit(app.exec_())
