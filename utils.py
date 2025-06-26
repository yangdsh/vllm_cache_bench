import os
import psutil
import socket
import subprocess
import math
import time
import signal

procs = {}

def add_proc(port, p):
    procs[port] = p

def kill_proc_on_port(port):
    if port in procs:
        procs[port].kill()
        procs[port].wait()
        del procs[port]

client_procs = []

def add_client_procs(p):
    client_procs.append(p)


def restart_ray():
    """Stop and start Ray on the head node."""
    subprocess.run("ray stop --force", shell=True, check=True)
    subprocess.run(f"VLLM_PP_LAYER_PARTITION={os.environ['VLLM_PP_LAYER_PARTITION']} RAY_DEDUP_LOGS=0 ray start --head", shell=True, check=True)


def restart_ray_remote(hostname):
    """Stop and start Ray on a remote node."""
    head_node_ip = socket.gethostbyname(socket.gethostname())
    stop_command = f"ssh {hostname} 'ray stop --force'"
    start_command = (
        f"ssh {hostname} 'VLLM_PP_LAYER_PARTITION={os.environ['VLLM_PP_LAYER_PARTITION']} "
        f"RAY_DEDUP_LOGS=0 ray start --address=\"{head_node_ip}:6379\"'"
    )
    subprocess.run(stop_command, shell=True, check=True)
    subprocess.run(start_command, shell=True, check=True)


def set_pp_layers(pp, num_layers, original_pp_partition):
    """Set pipeline-parallel layers for the model."""
    if pp <= 2 or original_pp_partition:
        os.environ["VLLM_PP_LAYER_PARTITION"] = ""
        return
    else:
        layer_per_stage = math.ceil(num_layers / pp)
        first_last = num_layers - (pp - 2) * layer_per_stage
        first = first_last // 2
        last = first_last - first
        layer_partition = [first] + [layer_per_stage] * (pp - 2) + [last]
        os.environ["VLLM_PP_LAYER_PARTITION"] = ','.join(map(str, layer_partition))
    restart_ray()
    restart_ray_remote("node2")


def is_port_in_use(port, host=None):
    """
    Check if a port is in use locally or on a remote node.
    :param port: Port number to check.
    :param node: Node to check (None for local).
    :return: True if port is in use, False otherwise.
    """
    try:
        result = subprocess.run(
            ["ssh", host, f"lsof -i:{port}"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        return result.returncode == 0  # lsof returns 0 if the port is in use
    except subprocess.CalledProcessError as e:
        print(f"Failed to check port on {host}: {e}")
        return False

def kill_server(host):
    """Kill all processes using the GPUs on a host."""
    try:
        # Kill any vllm processes for the current user
        user = os.environ.get('USER')
        if user:
            pkill_cmd = f"pkill -u {user} -f vllm"
            subprocess.run(pkill_cmd, shell=True, check=True)
            print(f"Killed vllm processes for user {user}")
        # Command to find and kill all processes running on NVIDIA GPUs.
        # It first checks if nvidia-smi command exists.
        # Then, it gets the PIDs of GPU processes.
        # If any PIDs are found, it attempts to kill them with SIGKILL.
        kill_cmd = (
            "if command -v nvidia-smi &> /dev/null; then "
            "pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader); "
            "if [ -n \"$pids\" ]; then "
            "echo \"$pids\" | xargs -r kill -9; "
            "fi; "
            "fi"
        )
        subprocess.run(kill_cmd, shell=True, check=True)
        print("Killed any processes using GPUs on the local machine.")
    except subprocess.CalledProcessError as e:
        print(f"No processes to kill or an error occurred: {e}")
        return
    print("Waiting for 30 seconds to ensure all processes are killed...")
    time.sleep(30)
