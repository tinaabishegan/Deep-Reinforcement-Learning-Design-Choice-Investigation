# runner.py

import os
import subprocess
import json
import pandas as pd
import re  # Import regex for accurate parsing

# Define combinations
high_architectures = [
    # "DoubleDQNPER",
    # "DoubleDQNPERDueling",
    "DoubleDQNPERDuelingNoisy",
    "DoubleDQNPERDuelingNROWAN"
]

low_architectures = [
    "DQN",
    # "DQNNoisy",
    # "DQNNROWAN",
    # "DoubleDQN",
    # "DoubleDQNPER",
    "DoubleDQNPERDueling",
    "DoubleDQNPERDuelingNoisy",
    "DoubleDQNPERDuelingNROWAN"
]

environments = ["CartPole-v1", "LunarLander-v3"]
noise_options = ["on"]
use_hrl_options = ["yes", "no"]
mode_options = ["train", "validate"]
# Directory for results
results_dir = "/results/"
os.makedirs(results_dir, exist_ok=True)

def run_command(cmd):
    """Run a command and print its output."""
    print(f"Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
    else:
        print("Command completed successfully.")

def train_and_validate():
    """Automate training and validation for all combinations."""
    for noise in noise_options:
        for use_hrl in use_hrl_options:
            for env in environments:
                for low_arch in low_architectures:
                    if use_hrl == "yes":
                        for high_arch in high_architectures:
                            # Train
                            train_cmd = (
                                f"python main.py --use_hrl {use_hrl} --high_arch {high_arch} "
                                f"--low_arch {low_arch} --env {env} --noise {noise} --mode train"
                            )
                            run_command(train_cmd)

                            # Validate
                            validate_cmd = (
                                f"python main.py --use_hrl {use_hrl} --high_arch {high_arch} "
                                f"--low_arch {low_arch} --env {env} --noise {noise} --mode validate"
                            )
                            run_command(validate_cmd)
                    else:
                        # Train
                        train_cmd = (
                            f"python main.py --use_hrl {use_hrl} --low_arch {low_arch} "
                            f"--env {env} --noise {noise} --mode train"
                        )
                        run_command(train_cmd)

                        # Validate
                        validate_cmd = (
                            f"python main.py --use_hrl {use_hrl} --low_arch {low_arch} "
                            f"--env {env} --noise {noise} --mode validate"
                        )
                        run_command(validate_cmd)

def collect_metrics_to_excel():
    """Collect metrics from /results/ and save to a single Excel file, adding a rolling average of the last 100 episodes."""
    metrics = []

    # Temporary storage to compute rolling averages
    # Key: (use_hrl, high_arch, low_arch, env, noise, mode)
    # Value: list of episode dicts
    data_by_config = {}

    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith("training_log.json"):
                log_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, results_dir)

                # Regex pattern to extract metadata
                pattern = r"use_hrl_(yes|no)(?:_high_([^_]+))?_low_([^_]+)_env_([^_]+)_noise_([^_]+)_mode_([^_]+)"
                match = re.match(pattern, relative_path)
                if not match:
                    print(f"Warning: Couldn't parse metadata from path: {relative_path}")
                    continue

                use_hrl = match.group(1)
                high_arch = match.group(2) if match.group(2) else None
                low_arch = match.group(3)
                env = match.group(4)
                noise = match.group(5)
                mode = match.group(6)

                config_key = (use_hrl, high_arch, low_arch, env, noise, mode)

                # Read all lines and store them
                with open(log_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        # We'll store these first without rolling averages
                        if config_key not in data_by_config:
                            data_by_config[config_key] = []
                        data_by_config[config_key].append(data)

    # Now compute rolling averages and build final metrics
    for config_key, episodes_data in data_by_config.items():
        # Sort by episode number to ensure correct order
        episodes_data.sort(key=lambda x: x.get('episode', 0))

        # Extract scores to compute rolling average
        scores = [d.get('score', 0) for d in episodes_data]

        # Compute rolling average over last 100 episodes
        rolling_avgs = []
        window_size = 100
        for i in range(len(scores)):
            start = max(0, i - window_size + 1)
            window_scores = scores[start:i+1]
            rolling_avg = sum(window_scores) / len(window_scores)
            rolling_avgs.append(rolling_avg)

        # Unpack config_key
        use_hrl, high_arch, low_arch, env, noise, mode = config_key

        # Add data to metrics with rolling average included
        for data, r_avg in zip(episodes_data, rolling_avgs):
            metric = {
                "use_hrl": use_hrl,
                "high_arch": high_arch,
                "low_arch": low_arch,
                "env": env,
                "noise": noise,
                "mode": mode,
                "episode": data.get('episode'),
                "score": data.get('score'),
                "rolling_avg_100": r_avg,  # newly added rolling average field
                "noise_levels": data.get('noise_levels'),
                "loss": data.get('loss'),
                "time": data.get('time'),
                "losses": data.get('losses')
            }
            metrics.append(metric)

    if not metrics:
        print("No metrics found to collect.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(metrics)

    # Save to Excel
    output_file = os.path.join(results_dir, "results_summary.xlsx")
    df.to_excel(output_file, index=False)
    print(f"Metrics collected and saved to {output_file}")


if __name__ == "__main__":
    print("Starting training and validation runs...")
    train_and_validate()
    print("Training and validation completed.")

    print("Collecting metrics into a single Excel file...")
    collect_metrics_to_excel()
    print("Metrics collection completed.")
