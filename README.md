# Learning-Driven DCM for Biped Locomotion

**What’s new in this repo (and where to find it)**  
- **DCM/ICP preview + learned offset (Δξ) teacher pipeline**: `dcm_teacher.py` (DCM predictor, Δξ offset net, consistent CoM update, IK).  
- **On-policy runner that both samples and optimizes the offset net**: `on_policy_runner.py` (rollouts + per-step/per-epoch Δξ updates; stability signals are kept differentiable).  
- **PPO with teacher imitation loss**: `ppo.py` extends the standard PPO loss by adding a teacher imitation term that aligns the policy with DCM-corrected references.  
- **TriMesh-based locomotion environment (configurable; file name kept as `pointfoot_flat.py` for compatibility)**: `pointfoot_flat.py` + `pointfoot_flat_config.py`.

This design supplies physics-plausible stepping references while letting learned corrections receive gradients throughout training, improving convergence and robustness to modeling errors and disturbances.

---

## Installation

> Works with Python **3.8** (recommended). CUDA example below uses **cu121** wheels.

1. **Create a virtual environment**
```bash
conda create -n your_virtual_env python=3.8
conda activate your_virtual_env
```

2. **Install PyTorch (CUDA 12.1 example)**
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu121
```

3. **Install Isaac Gym (Preview 3)**
```bash
cd isaacgym/python
pip install -e .
# quick test
cd ../examples
python 1080_balls_of_solitude.py
```

4. **Install this repo (editable)**
```bash
git clone <your-repo-url>
cd legged_gym
pip install -e .
```

---

## Code Structure

- **Environment & Config (TriMesh)**  
  - `pointfoot_flat.py` — **TriMesh-backed** locomotion environment (the file name remains for backward compatibility).  
    Terrain shape/usage (e.g., near-flat vs. discrete-height “stair-like” profiles) is **configured in** `pointfoot_flat_config.py`.  
    You can switch terrain/backend/profile via config without changing the task name.  
  - `pointfoot_flat_config.py` — environment & training hyper-parameters (observations, rewards, curriculum, devices; and **terrain/backend/profile switches**).

- **Teacher Pipeline**  
  - `dcm_teacher.py` — closed-form preview stepping (DCM/ICP), Δξ offset network for small corrective shifts, consistent CoM update, IK targets.

- **Training & Optimization**  
  - `on_policy_runner.py` — integrates the teacher *inside* rollouts **and** performs Δξ offset-network updates (e.g., after each collection or at configured intervals), ensuring gradients flow from stability signals to Δξ.  
  - `ppo.py` — PPO implementation with a **teacher imitation loss** added to the original PPO objective. The final objective is a weighted sum: PPO (policy, value, entropy) + λ·Imitation (aligning actions/latents to teacher-guided references). Storage supports teacher/stability tensors; interfaces remain compatible with the original scripts.

- **Task Registration**  
  - Tasks can be registered via `task_registry.register(name, EnvClass, EnvCfg, TrainCfg)`.

> **Note on naming**  
> The task is still invoked as `--task=pointfoot_flat` for compatibility, but the **actual environment is TriMesh-based**. Terrain backend/profile is controlled in `pointfoot_flat_config.py`.

---

## Usage

### Train (TriMesh env; task name unchanged)
```bash
export ROBOT_TYPE=PF_TRON1A

python legged_gym/scripts/train.py --task=pointfoot_flat --headless
```

**Common arguments**
- `--sim_device=cpu`, `--rl_device=cpu` — run on CPU (you can mix sim=CPU, rl=GPU).  
- `--headless` — disable rendering. Once training starts, press **v** to stop/resume rendering for better performance.  
- **Overrides:** `--task`, `--resume`, `--experiment_name`, `--run_name`,  
  `--load_run`, `--checkpoint`, `--num_envs`, `--seed`, `--max_iterations`.

**Checkpoints**
```
pointfoot-legged-gym/logs/<experiment_name>/<ROBOT_TYPE>/<date_time>_<run_name>/model_<iteration>.pt
```

### Evaluate a trained policy
```bash
python legged_gym/scripts/play.py \
  --task=pointfoot_flat \
  --load_run <your_model_folder> \
  --checkpoint <iter>
# example:
# --load_run Apr18_15-48-46_  --checkpoint 10000  (for model_10000.pt)
```

---

## Highlights

- **Physics-guided + learned correction**: Closed-form DCM stepping supplies a stable baseline; the Δξ net learns small shifts that improve clearance and phase timing.  
- **End-to-end differentiability**: Δξ directly affects CoM updates and stability loss; gradients back-propagate through the teacher-in-the-loop rollouts.  
- **Runner optimizes Δξ**: The on-policy runner not only collects data but also **updates the offset network** at configured intervals.  
- **Teacher imitation in PPO**: PPO uses an additional **imitation loss** (weighted by λ) to encourage the policy to match teacher-guided references while retaining PPO’s exploration/stability trade-offs.  
- **TriMesh environment**: Configurable TriMesh terrain (flat-like or stair-like profiles) without changing the task name; suitable for reproducible benchmarking; optional Vstab constraints.

---

## File Overview

- `dcm_teacher.py` — DCM predictor, Δξ offset network, IK/teacher action composition.  
- `on_policy_runner.py` — Rollouts **and** Δξ optimization in the on-policy loop.  
- `ppo.py` — PPO core with **teacher imitation loss** and teacher/stability tensor support.  
- `pointfoot_flat.py` — **TriMesh** locomotion environment (name kept for compatibility); differentiable stability loss; Vstab options; privileged heights (training only).  
- `pointfoot_flat_config.py` — Env/training configs and **terrain/backend/profile** switches.

---

## License

Recommend **MIT** (or Apache-2.0). Add a `LICENSE` file accordingly.

---

## Acknowledgments

Built on the Isaac Gym stack and legged locomotion training pipelines; extended with a learning-driven DCM teacher, **offset-network optimization inside the runner**, and a **teacher imitation loss** in PPO, running on a **TriMesh-based environment** configured via `pointfoot_flat_config.py`.

---

*Questions or issues?* Please open an issue or PR.
