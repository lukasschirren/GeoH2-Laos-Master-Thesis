import time
import subprocess

# Time delay in seconds (6 hours = 6 * 60 * 60)
delay = 7 * 60 * 60

# Function to run a python script
def run_script(script_name):
    subprocess.run(["python", script_name])

# Countdown function
def countdown(seconds):
    while seconds > 0:
        hrs, rem = divmod(seconds, 3600)
        mins, secs = divmod(rem, 60)
        time_format = f"{hrs:02}:{mins:02}:{secs:02}"
        print(f"Time remaining: {time_format}", end='\r')
        time.sleep(1)
        seconds -= 1
    print("\nCountdown finished. Starting the scripts...\n")

# Start the countdown
countdown(delay)

# Run the scripts in the specified order
run_script("assign_country.py")
run_script("optimize_transport_and_conversion.py")
run_script("optimize_hydrogen_plan_temporal.py")