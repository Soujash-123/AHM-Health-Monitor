import psutil
import platform

# Get number of physical cores
physical_cores = psutil.cpu_count(logical=False)

# Get number of logical cores
logical_cores = psutil.cpu_count(logical=True)

# Get CPU frequency
cpu_freq = psutil.cpu_freq()

# Get CPU usage per core
cpu_usage_per_core = psutil.cpu_percent(interval=1, percpu=True)

# Get overall CPU usage
overall_cpu_usage = psutil.cpu_percent(interval=1)

# Print CPU details
print(f"Physical cores: {physical_cores}")
print(f"Logical cores: {logical_cores}")
print(f"Max Frequency: {cpu_freq.max:.2f}Mhz")
print(f"Min Frequency: {cpu_freq.min:.2f}Mhz")
print(f"Current Frequency: {cpu_freq.current:.2f}Mhz")
print(f"CPU Usage Per Core: {cpu_usage_per_core}")
print(f"Overall CPU Usage: {overall_cpu_usage}%")

# Get CPU times
cpu_times = psutil.cpu_times()
print("\nCPU Times:")
print(f"User time: {cpu_times.user} seconds")
print(f"System time: {cpu_times.system} seconds")
print(f"Idle time: {cpu_times.idle} seconds")
if platform.system() != "Windows":
    print(f"Interrupt time: {getattr(cpu_times, 'interrupt', 'N/A')} seconds")
    print(f"DMA transfer time: {getattr(cpu_times, 'dpc', 'N/A')} seconds")

# Get CPU statistics
cpu_stats = psutil.cpu_stats()
print("\nCPU Stats:")
print(f"Context switches: {cpu_stats.ctx_switches}")
print(f"Interrupts: {cpu_stats.interrupts}")
print(f"Soft interrupts: {cpu_stats.soft_interrupts}")
print(f"Syscalls: {cpu_stats.syscalls}")

# Get CPU load averages (1, 5, 15 minutes)
load_avg = psutil.getloadavg()
print("\nCPU Load Averages (1, 5, 15 min):")
print(f"1 min: {load_avg[0]}")
print(f"5 min: {load_avg[1]}")
print(f"15 min: {load_avg[2]}")

def get_cpu_temperature():
    try:
        # Check if sensors_temperatures is available
        temperatures = psutil.sensors_temperatures()
        if not temperatures:
            return "Temperature sensors are not available on this system."

        # Iterate over sensors and check for CPU temperature
        for name, entries in temperatures.items():
            print(f"Sensor: {name}")
            for entry in entries:
                if 'core' in entry.label.lower() or 'cpu' in entry.label.lower():
                    print(f"Label: {entry.label}, Temperature: {entry.current}°C")
        return "Temperature data retrieved successfully."
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Get and print CPU temperature
print(get_cpu_temperature())
