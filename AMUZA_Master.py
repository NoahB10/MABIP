import threading
import time
import os
import sys
import logging
from datetime import datetime

bt = 0  # Set to zero when editing program without the AMUZA connection and 1 for regular use
logs = False  # Add this line to toggle logging on/off
if bt:
    import bluetooth

class MockBluetoothSocket:
    """Mock Bluetooth socket for testing without actual Bluetooth hardware."""
    def __init__(self):
        print("Mock Bluetooth socket created")

    def connect(self, address):
        print(f"Mock connecting to {address}")
        time.sleep(1)
        print("Mock connection established")

    def send(self, data):
        print(f"Mock sending data: {data}")

    def recv(self, buffer_size):
        #print(f"Mock receiving data with buffer size: {buffer_size}")
        # Simulate data reception
        return b"Mock data received"

    def close(self):
        print("Mock Bluetooth socket closed")

class Method:
    def __init__(self, ports, time):
        if not isinstance(ports, list):
            raise TypeError("'ports' must be of type list")
        self.ports = ports
        if not isinstance(time, int):
            raise TypeError("'time' must be of type int")
        if time > 9999 or time < 0:
            raise ValueError("'time' must be between 0 and 9999")
        self.time = time

    def __str__(self):
        toReturn = f"{self.timeStringFormat()},"
        for port in self.ports:
            toReturn += f'{str(port).zfill(2)},'
        return toReturn

    def timeStringFormat(self):
        return str(self.time).zfill(4)

class Sequence:
    def __init__(self, methods):
        if not isinstance(methods, list):
            raise TypeError("'methods' must be of type list")
        if len(methods) < 1:
            raise ValueError("'methods' must be a list of length >= 1")
        self.methods = methods

    def __str__(self):
        toReturn = "@P,"
        for i in range(len(self.methods)):
            toReturn += f'M{i+1},{str(self.methods[i])}'
        return toReturn + "\n\n"

class AmuzaConnection:   
    isInProgress = False
    currentState = 0
    stateList = ["Resting", "Ejected Tray", "Unknown", "Unknown", "Moving Tray",
                 "Unknown", "Unknown", "Unknown", "Unknown", "Moving",
                 "Unknown", "Unknown", "Unknown", "Unknown", "Unknown",
                 "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"]
    
    def __init__(self, showOutputInConsole):
        if logs:
            # Ensure the amuza_logs directory exists
            log_folder = 'Amuza_Logs'
            os.makedirs(log_folder, exist_ok=True)
            
            # Configure logging to save in the amuza_logs folder
            currentTime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_path = os.path.join(log_folder, f'AMUZA-{currentTime}.log')
            
            logging.basicConfig(level=logging.DEBUG,
                                format='%(asctime)s %(levelname)s: %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
            
            # Set up the file handler for logging
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
            
            # Clear existing handlers before adding new ones
            for handler in logging.getLogger().handlers:
                logging.getLogger().removeHandler(handler)
            
            logging.getLogger().addHandler(file_handler)
            logging.info("AMUZA Interface Initiated - Detailed Logs can be found in amuza_logs folder")
        
        self.showOutput = showOutputInConsole
        print("AMUZA Interface Initiated")


    def queryThread(self, threadEvent, socket):
        while threadEvent.is_set():
            socket.send("@Q\n")
            logging.debug("Sent Query")
            time.sleep(1)
    
    def receptionThread(self, threadEvent, socket):
        currentCmd = ""
        while threadEvent.is_set():
            data = socket.recv(1024)
            decoded = data
            try:
                decoded = data.decode()
            except:
                logging.warning("Failed to Decrypt")
            logging.info(f"Received: {decoded}")
            currentCmd += str(decoded)
            if str(decoded).endswith("\n"):
               self.handleRecieved(currentCmd)
               currentCmd = ""
    
    def loopThread(self, threadEvent, sequence):
        while threadEvent.is_set():
            if not self.checkProgress():
                self.Move(sequence)
                logging.info(f"Moving To: {str(sequence)}")
    
    def handleRecieved(self, cmd):
        logging.info(f"Handling: {cmd}")
        if cmd[:2] == "@E":
            logging.info(f"Exited with Exit Code {cmd[3]}")
        elif cmd[:2] == "@q":
            data = cmd[3:].split(',')
            if data[1] == '0':
                self.isInProgress = False
            else:
                self.isInProgress = True
                print(f"Method number {data[1]}")
                print(f"Time left at well {data[2].strip('0')}: {data[3].strip('0')} seconds")
            logging.info(f"Status Update: {cmd}")
            self.currentState = int(data[0])
        else:
            print(f"? {cmd}")
    
    def connect(self):
        if logs:
            print("Attempting to Connect to AMUZA")
            logging.info("Attempting to Connect to AMUZA")
        
        if bt:  # Use the real Bluetooth socket
            if logs:
                print(f"Scanning")
                logging.info("Scanning")
            nearby_devices = bluetooth.discover_devices(lookup_names=True,lookup_class=True)
            if logs:
                print("Found {} devices.".format(len(nearby_devices)))
                logging.info("Found {} devices.".format(len(nearby_devices)))
            address = ""
            for addr, name, device_class in nearby_devices:
                if logs:
                    print(f"  Address: {addr}")
                    print(f"  Name: {name}")
                    print(f"  Class: {device_class}")
                    logging.info(f"  Address: {addr}")
                    logging.info(f"  Name: {name}")
                    logging.info(f"  Class: {device_class}")
                if(name=='FC90-0034'):
                    address=addr
            if(address==""):
                if logs:
                    print("AMUZA not found, press ENTER to exit")
                    logging.critical("AMUZA not found, press ENTER to exit")
                input()
                exit()
            if logs:    
                print("Attempting to Connect to AMUZA")
                logging.info("Attempting to Connect to AMUZA")
            socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
            try:
                socket.connect((address,1))
            except:
                if logs:
                    print("Connection Failure, Press ENTER to exit")
                    logging.critical("Connection Failure, Press ENTER to exit")
                input()
                exit()
            if logs:
                print("Connection Success")
                logging.info("Connection Success")
            socket.send("@?\n")
            time.sleep(0.2)
            socket.send("@Q\n")
            time.sleep(0.2)
            socket.send("@Z\n")
        else:  # Use the mock socket for testing
            socket = MockBluetoothSocket()
            socket.connect(("Mock Address", 1))
            if logs:
                print("Mock connection success")
                logging.info("Mock connection success")
        
        self.socket = socket
        threads = threading.Event()
        threads.set()
        _queryThread = threading.Thread(target=self.queryThread, args=(threads, self.socket))
        _queryThread.setDaemon(True)
        _queryThread.start()
        
        if self.showOutput:
            _receptionThread = threading.Thread(target=self.receptionThread, args=(threads, self.socket))
            _receptionThread.setDaemon(True)
            _receptionThread.start()

    def well_mapping(self, locations):
        self.well_map = {}  # Initialize the dictionary
        rows = "ABCDEFGH"
        columns = range(1, 13)
        counter = 1
        
        # Generate the well_map dictionary
        for column in columns:
            for row in rows:
                well_location = f"{row}{column}"
                self.well_map[well_location] = counter
                counter += 1

        # Create a list of numeric values based on input locations
        result = []
        for location in locations:
            result.append(self.well_map.get(location, None))  # Return None if location is not found
        return result  # Return the list of values
    
    def consoleInterface(self):
        """Interactive console interface for debugging and controlling the connection."""
        while True:
            command = input()
            if logs: logging.info(f"User Input: {command}")
            if(self.checkProgress() and command != ("STOP" or "EXIT" or "STATUS")):
                print("Machine is currently doing something, send STOP to make your command work")
            if(command=="EXIT"):
                if logs:
                    logging.info(f"Exiting...")
                    print("Exiting...")
                return
            if(command=="DEMO MOVE"):
                if logs:
                    logging.info("Sent Move Command")
                    print("Sent Move Command")
                method1 = Method([1,5,13,71],15)
                sequence = Sequence([method1])
                self.Move(sequence)

            if(command=="SAMPLING"):
                if logs:
                    logging.info("Sent Sampling Command")
                    print("Sent Sampling Command")
                # One way to write the methods are from well location names and time
                loc = ['A7','B7','C7','D7']
                loc_m = self.well_mapping(loc)
                time = [197,167,197,177]
                method = []
                for i in range(0, len(loc)):
                    print(loc[i])
                    print(loc_m[i])
                    method.append(Sequence([Method([loc_m[i]],time[i])]))
                    print(method[i])
                #self.Move(method)

            if(command[:4]=="TEMP"):
                if logs: logging.info(f"Adjusting Temp To {command[5:]}") # extra char to remove space
                self.AdjustTemp(float(command[5:]))
            if(command=="MOVE"):
                print("How long at each well? (Seconds)")
                length = input()
                wellList = []
                while(True):
                    print("Enter a comma-seperated list of wells you want to enter")
                    rec = input()
                    rec = rec.split(',')
                    for entry in rec:
                        if(entry.isdigit()):
                            num = int(entry)
                            if(num < 1 or num > 96):
                                print(f"Please only input a number from 1-96. You have inputted {num} in your list.")
                            else:
                                wellList.append(num)
                        else:
                            print(f"You have a non-integer ({entry}) in your list. Please only input numbers from 1-96.")
                            break
                    print(f"Final well list: {wellList}")
                    print("Confirm? Y/N")
                    confirm = input()
                    if(confirm=="Y"):
                        break
                    print("Redoing...")
                print("Do you want to loop this command? Y/N")
                loop = False
                while(True):
                    rec = input()
                    if(rec == "Y"):
                        loop = True
                        break
                    elif(rec == "N"):
                        break
                    else:
                        print("Invalid Input. Do you want to loop this command? Y/N")
                method = Method(wellList,int(length))
                sequence = Sequence([method])
                self.Move(sequence)
                if logs:
                    print(f"Moving To: {str(sequence)}")
                    logging.info(f"Moving To: {str(sequence)}")
                if(loop):
                    print("Type END to end the loop")
                    loopEvent = threading.Event()
                    loopEvent.set()
                    loopThread = threading.Thread(target = self.loopThread,args = (loopEvent, sequence))
                    loopThread.start()
                    while(True):
                        rec = input()
                        if(rec == "END"):
                            loopEvent.clear()
                            break
                        else:
                            print("Type END to end the loop")
            if(command=="STOP"):
                if logs: logging.info("Stopping...")
                self.Stop()
            if(command=="STATUS"):
                if logs:
                    logging.info(f"Machine is currently: {self.stateList[self.currentState-1]} (ID: {self.currentState})") #-1 to adjust for the shift
                    print(f"Machine is currently: {self.stateList[self.currentState-1]}  (ID: {self.currentState})")
            if(command=="CUSTOM"):
                print("What do you want to send?")
                cmd = input()
                if logs:
                    print(f"Sending command \"{cmd}\"")
                    logging.info(f"Sending Custom Command: {cmd}")
                self.socket.send(cmd)
            if(command=="EJECT"):
                if logs: logging.info("Ejecting...")
                self.Eject()
            if(command=="INSERT"):
                if logs: logging.info("Inserting...")
                self.Insert()
            if(command=="HELP"):
                print("EXIT - Exit the program\nDEMO MOVE - quick preprogrammed move command for debugging\nTEMP <float value between 0 and 99.9> - Adjust temperature\nMOVE - Wizard to move machine\nSTOP - Stop current action\nSTATUS - Get current status of the machine\nCUSTOM - Send custom command. Start it with @, end it with \\n\nEJECT - Eject the tray\nINSERT - Insert the tray\nHELP - Open this menu\nNEEDLE - Adjust the needle height")
            
            if(command=="NEEDLE"):
                print("UP to move needle up, DOWN to move needle down, FINISH to exit this wizard")
                self.socket.send("@N\n")
                while True:
                    cmd = input()
                    if(cmd=="UP"):
                        if logs: logging.info("Moving Needle Up")
                        self.NeedleUp()
                    elif(cmd=="DOWN"):
                        if logs: logging.info("Moving Needle Down")
                        self.NeedleDown()
                    elif(cmd=="FINISH"):
                        if logs:
                            logging.info("Finished Needle Adjustments")
                            print("Finished Needle Adjustments")
                        self.socket.send("@V,180\n")
                        time.sleep(0.2)
                        self.socket.send("@T\n")
                        break
                    else:
                        print("Unknown Command - UP to move needle up, DOWN to move needle down, FINISH to exit this wizard")
    
    def Eject(self):
        """Send the command to eject the tray."""
        self.socket.send("@Y\n")
        if logs: logging.info("Eject command sent")
        print("Tray ejected")

    def Insert(self):
        """Send the command to insert the tray."""
        self.socket.send("@Z\n")
        if logs: logging.info("Insert command sent")
        print("Tray inserted")

    def Stop(self):
        """Send the command to stop the current operation."""
        self.socket.send("@T\n")
        if logs: 
            logging.info("Stop command sent")
            print("Operation stopped")

    def Move(self, sequence):
        """Send the move command with a sequence."""
        self.socket.send(str(sequence))
        if logs: 
            logging.info(f"Move command sent with sequence: {sequence}")
            print("Moving according to sequence")

    def NeedleUp(self):
        """Send the command to move the needle up."""
        self.socket.send("@U01\n")
        if logs: 
            logging.info("Needle up command sent")
            print("Needle moved up")

    def NeedleDown(self):
        """Send the command to move the needle down."""
        self.socket.send("@D01\n")
        if logs: 
            logging.info("Needle down command sent")
            print("Needle moved down")

    def AdjustTemp(self, temperature):
        """Send the command to adjust the temperature."""
        if temperature < 0 or temperature > 99.9:
            raise ValueError("Temperature must be between 0 and 99.9")
        self.socket.send(f"@V,{temperature}\n")
        if logs:
            logging.info(f"AdjustTemp command sent with temperature: {temperature}")
            print(f"Temperature adjusted to {temperature}")

if __name__ == '__main__':
    connection = AmuzaConnection(True)
    connection.connect()
    connection.consoleInterface()
