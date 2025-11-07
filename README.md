Learning-Driven DCM for Biped Locomotion

What’s new in this repo (and where to find it)

DCM/ICP preview + learned offset (Δξ) teacher pipeline: dcm_teacher.py (DCM predictor, Δξ offset net, consistent CoM update, IK).

On-policy runner that both samples and optimizes the offset net: on_policy_runner.py (rollouts + per-step/per-epoch Δξ updates; stability signals are kept differentiable).

PPO with teacher imitation loss: ppo.py extends the standard PPO loss by adding a teacher imitation term that aligns the policy with DCM-corrected references.

Point-foot flat environment with training-only privileged terrain heights and optional Vstab soft constraints: pointfoot_flat.py + pointfoot_flat_config.py.

This design supplies physics-plausible stepping references while letting learned corrections receive gradients throughout training, improving convergence and robustness to modeling errors and disturbances.

Installation

Works with Python 3.8 (recommended). CUDA example below uses cu121 wheels.

Create a virtual environment
