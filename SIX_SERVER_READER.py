import serial
import argparse
import os
from datetime import datetime

class PotentiostatReader:
    def __init__(self, com_port, baud_rate=9600, timeout=0.5, package_length=25, output_filename="out_data.txt"):
        self.com_port = com_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.package_length = package_length
        self.output_filename = output_filename
        self.data_block = [b'\x00'] * package_length
        self.start_timestamp = None
        self.serial_connection = None
        self.sample_number = 1
        self.channels = [f"#1ch{i}" for i in range(1, 17)] + \
                        [f"#2ch{i}" for i in range(1, 17)] + \
                        [f"#3ch{i}" for i in range(1, 17)] + \
                        [f"#4ch{i}" for i in range(1, 17)]
        self.fit_values = [f"Fit{k}a{i}" for k in range(1, 65) for i in range(1, 5)]

    def open_serial_connection(self):
        if self.serial_connection is None:
            self.serial_connection = serial.Serial(self.com_port, baudrate=self.baud_rate, timeout=self.timeout)

    def close_serial_connection(self):
        if self.serial_connection is not None:
            self.serial_connection.close()
            self.serial_connection = None

    def validate_data_block(self):
        header = [b'\x04', b'\x68', b'\x13', b'\x13', b'\x68']
        cks = 0
        for x in [int.from_bytes(x, 'big') for x in self.data_block[2:-4]]:
            cks = (cks + x) & 0xFF
        return self.data_block[-5:] == header and self.data_block[0] == b'\x16' and int.from_bytes(self.data_block[1], 'big') == cks

    def process_data_block(self):
        data_inv = [x for x in self.data_block[2:-5]]
        data_inv.reverse()
        it = iter(data_inv)
        out_data = [int.from_bytes(b''.join([x, next(it)]), byteorder='big', signed=True) for x in it]
        return out_data


    def convert_data(self, out_data):
        gain = 50 / (2**15 - 1)
        sensed_values = [str(round(int(x) * gain, 3)) for x in out_data[:6]]
        
        # Handle temperature (Channel 7)
        if len(out_data) > 6 and out_data[6] is not None:
            temperature = str(round(float(out_data[6]) / 16, 3))
        else:
            temperature = "1"  # Default temperature value if missing

        # Initialize all channels with '0'
        channel_data = ['0'] * len(self.channels)

        # Fill in the sensed values for #1ch1 to #1ch6
        for i in range(6):
            channel_data[i] = sensed_values[i]

        # Set Channel 7 to the temperature value
        channel_data[6] = temperature

        # Set Channel 8 to '1' as a flag for a complete read
        channel_data[7] = "1"

        # Return the complete line of data including temperature and status
        return channel_data

    def get_data(self):
        self.open_serial_connection()
        accumulated_bytes = b''

        while len(accumulated_bytes) < self.package_length:
            remaining_bytes = self.package_length - len(accumulated_bytes)
            new_data = self.serial_connection.read(remaining_bytes)
            accumulated_bytes += new_data

        if accumulated_bytes:
            for byte in accumulated_bytes:
                self.data_block.insert(0, bytes([byte]))
                self.data_block.pop()

            if self.validate_data_block():
                out_data = self.process_data_block()
                return self.convert_data(out_data)
        return None


    def run(self):
        created_time = None

        with open(self.output_filename, 'a') as file:
            if self.sample_number == 1:
                # Write the "Created" line
                created_time = datetime.now().strftime("%m/%d/%Y\t%I:%M:%S %p")
                file.write(f"Created: {created_time}\n")

                # Write the full header
                header = (
                    "counter\tt[min]\t" + "\t".join(self.channels) +
                    "\t" + "\t".join([f"X(Fit{k})" for k in range(1, 65)]) +
                    "\t" + "\t".join(self.fit_values) + "\n"
                )
                file.write(header)

                # Write the "Start" line
                start_time = datetime.now().strftime("%m/%d/%Y\t%I:%M:%S %p")
                file.write(f"Start: {start_time}\n")

            while True:
                data = self.get_data()
                if data is not None:
                    # Calculate time elapsed in minutes
                    time_elapsed = round((datetime.now() - datetime.strptime(created_time, "%m/%d/%Y\t%I:%M:%S %p")).total_seconds() / 60, 4)

                    # Set fit values to '0'
                    fit_values = ['0'] * len(self.fit_values)

                    # Construct the data line
                    data_line = f"{self.sample_number}\t{time_elapsed}\t" + "\t".join(data) + "\t" + "\t".join(fit_values) + "\n"

                    # Write the data line
                    file.write(data_line)
                    file.flush()  # Ensure data is written immediately

                    # Increment the sample number
                    self.sample_number += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Potentiostat Data Reader")
    parser.add_argument("--com_port", type=str, required=True, help="COM port for the potentiostat")
    parser.add_argument("--baud_rate", type=int, default=9600, help="Baud rate for serial communication")
    parser.add_argument("--timeout", type=float, default=0.5, help="Timeout for serial communication")
    parser.add_argument("--package_length", type=int, default=25, help="Expected package length for data")
    parser.add_argument("--output_filename", type=str, default="out_data.txt", help="File to save the output data")

    args = parser.parse_args()
    reader = PotentiostatReader(
        com_port=args.com_port,
        baud_rate=args.baud_rate,
        timeout=args.timeout,
        package_length=args.package_length,
        output_filename=args.output_filename,
    )

    try:
        reader.run()
    except KeyboardInterrupt:
        print("Data collection stopped by user.")
    finally:
        reader.close_serial_connection()
