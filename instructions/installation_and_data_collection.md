## Installation Instructions

- Clone the repository with the submodules.
```
git clone git@github.com:siddhanthaldar/Point-Policy.git --recurse-submodules
```

- You can download and install the co-tracker and dift submodules and relevant packages by running the setup.sh file. Make sure to run this from the root repository or the models may get installed in the wrong location.
```
sudo chmod 777 setup.sh # make executable
./setup.sh
```

### TAPIR point tracking (tapnet)

The TAPIR-based tracker (`point_policy/point_utils/tapir_tracker.py`) uses the `tapnet` submodule and is not installed by `setup.sh` yet. Install it and fetch its checkpoint/DIFT weights with:
```
pip install -e tapnet
bash download_tapir_models.sh
```

### hamer / OmniHands (optional, isolated environments)

`hamer` and `OmniHands` are only used to preprocess human demonstration videos into hand-keypoint tracks (via `convert_to_pkl_human_hamer.py` / `convert_to_pkl_human_omnihands.py`) — they're never imported by the main training/eval code. Their required `torch`/`numpy`/`mmcv` versions conflict with the `point-policy` environment above, so each gets its own venv instead of being installed into it.

**hamer:**
```
cd hamer
python3.10 -m venv .hamer_venv
.hamer_venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117  # match your CUDA version
.hamer_venv/bin/pip install -e .[all]
.hamer_venv/bin/pip install -v -e third-party/ViTPose
cd ..
```
Then download hamer's checkpoints per [hamer/README.md](../hamer/README.md#installation). Run the worker (from inside `hamer/`) with:
```
.hamer_venv/bin/python -I hamer_hand_worker.py --frames_npy /path/to/frames.npy --out_npy /path/to/out.npy
```

**OmniHands:**
```
cd OmniHands
python3.10 -m venv .omnihands_venv
.omnihands_venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117  # match your CUDA version
.omnihands_venv/bin/pip install -e .[all]
.omnihands_venv/bin/pip install -v -e third-party/ViTPose
cd ..
```
Then download `Demo_Multiview.pth` per [OmniHands/README.md](../OmniHands/README.md#download-checkpoints) into `OmniHands/checkpoints/`. Run the worker (from inside `OmniHands/`) with:
```
.omnihands_venv/bin/python -I omnihands_hand_worker.py --frames_npy /path/to/frames.npy --out_npy /path/to/out.npy
```

These are deliberately left out of the Docker image/container-init automation (rosdep etc.) since they need their own isolated environments rather than a system-wide install — set them up manually per-machine as needed.

## Data Collection Instructions
- Instructions for data collection are provided in the [Franka Teach submodule](Franka-Teach/README.md). This is a fork of [Open Teach](https://github.com/aadhithya14/Open-Teach) modified to only work with Franka robots.

## Data Preprocessing Instructions

NOTE: Set 'root_dir' in `point_policy/cfg/suite/point_cfg.yaml` to `path/to/repo`.

- Go to the robot utils Franka directory.
```
cd point-policy/robot_utils/franka
```
- Once you have collected the human data using Franka Teach, process it to remove pauses and save it in a nicer format.
```
python process_data_human.py --data_dir path/to/data --task_names <task_name> --process_depth
```
- Convert the data to a pickle file (without processing key points first).
```
python convert_to_pkl_human.py --data_dir path/to/data --calib_path path/to/calib_file --task_names <task_name>
```
- NOTE: Before generating task data, we first need generate the calibration file.
    - For calibration, generate the pkl file without points for the calibration data (collected using Franka Teach) and make sure to set the `PATH_DATA_PKL` to the data pickle file for the calib data first.
    - Next generate the calib file using the following command
    ```
    cd calibration
    python generate_r2c_extrinsic.py
    cd ..
    ```
    - This will generate the calib file in `point_policy/calib/calib.npy`.

- Label semantically meaningful points for each task following commands in `point-policy/robot_utils/franka/label_points.ipynb`
- Save pickle data with key point labels, both for the human hand and object points obtained through human annotations.
```
python convert_to_pkl_human.py --data_dir path/to/data --calib_path path/to/calib_file --task_names <task_name> --process_points
```
- Convert human hand poses to robot actions in the data.
```
python convert_pkl_human_to_robot.py --data_dir path/to/data --calib_path path/to/calib_file --task_name <task_name>
```

NOTE:
- The `calib_path` must be set to `<root_dir>/calib/calib.npy` where `<root_dir>` is the root directory of the repository. The `data_dir` must be set to the directory where the data is stored during teleoperation (`path/to/data`).
- The generated pkl files with robot actions will be stored in `path/to/data/expert_demos/franka_env`. The variable `data_dir` in `config.yaml` and `config_eval.yaml` in `point_policy/cfg` must be set to `path/to/data/expert_demos`.
