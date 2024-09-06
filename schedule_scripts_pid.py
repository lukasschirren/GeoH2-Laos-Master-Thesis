import time
import subprocess
import psutil

# Replace with the actual PID of the process you want to monitor
target_pid = 9916  # Example PID, replace with the actual one

# Function to check if the process is still running by PID
def is_process_running(pid):
    return psutil.pid_exists(pid)

# Function to run a python script
def run_script(script_name):
    subprocess.run(["python", script_name])

# Wait until the process is no longer running
print(f"Waiting for process with PID {target_pid} to complete...")
while is_process_running(target_pid):
    time.sleep(10)  # Check every 10 seconds

print(f"Process with PID {target_pid} has completed. Starting the scripts...\n")


print("WOW!")
# Run the scripts in the specified order
run_script("assign_country.py")
run_script("optimize_transport_and_conversion.py")
run_script("optimize_hydrogen_plant_temporal.py")
