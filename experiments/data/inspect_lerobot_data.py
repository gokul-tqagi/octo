# inspect_lerobot_data.py: Inspect LeRobot v3 format data for Octo conversion planning.
# inspect_lerobot_data.py: Reads parquet metadata, data schema, and episode structure.

import json
import os
import sys

import pyarrow.parquet as pq


def inspect_record(record_path):
    """Inspect a single LeRobot record directory."""
    name = os.path.basename(record_path)

    with open(os.path.join(record_path, "meta", "info.json")) as f:
        info = json.load(f)

    tasks_table = pq.read_table(os.path.join(record_path, "meta", "tasks.parquet"))
    tasks_dict = tasks_table.to_pydict()

    data_table = pq.read_table(
        os.path.join(record_path, "data", "chunk-000", "file-000.parquet")
    )

    ep_indices = set()
    for i in range(len(data_table)):
        ep_indices.add(data_table.column("episode_index")[i].as_py())

    print("=" * 60)
    print(name)
    print("=" * 60)
    print("Episodes:", info["total_episodes"])
    print("Frames:", info["total_frames"])
    print("FPS:", info["fps"])
    print("Robot:", info["robot_type"])
    print("Codebase:", info["codebase_version"])
    print()

    print("Tasks:", tasks_dict)
    print()

    cameras = [k for k in info["features"] if "images" in k]
    print("Cameras:", cameras)
    for cam in cameras:
        ci = info["features"][cam]
        print(f"  {cam}: {ci['shape']}, codec={ci.get('info', {}).get('video.codec', '?')}")
    print()

    act_info = info["features"]["action"]
    print("Action dim:", act_info["shape"][0])
    print("Action names:", act_info["names"])
    print()

    state_info = info["features"]["observation.state"]
    print("State dim:", state_info["shape"][0])
    print("State names:", state_info["names"])
    print()

    print("Data columns:", data_table.column_names)
    print("Data rows:", len(data_table))
    print("Episode indices:", sorted(ep_indices))
    print()

    # Sample values
    act0 = data_table.column("action")[0].as_py()
    state0 = data_table.column("observation.state")[0].as_py()
    print("Action[0]:", [round(v, 4) for v in act0])
    print("State[0]:", [round(v, 4) for v in state0])
    print()


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/aloha"
    for rec in sorted(os.listdir(data_dir)):
        rec_path = os.path.join(data_dir, rec)
        if os.path.isdir(rec_path) and os.path.exists(
            os.path.join(rec_path, "meta", "info.json")
        ):
            inspect_record(rec_path)


if __name__ == "__main__":
    main()
