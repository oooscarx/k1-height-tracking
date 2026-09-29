## rsl_rl Modifications

This directory is based on
[rsl_rl v2.3.3](https://github.com/leggedrobotics/rsl_rl/releases/tag/v2.3.3)
and remains under its BSD-3-Clause license.

The K1 stand-up task carries the algorithm changes used by the
WBC-AGILE Stand-Up task:

- L2C2 actor and critic regularization over adjacent non-terminal states.
- Return-variance reward normalization, including checkpoint persistence.
- Value bootstrapping plus normalized sigma penalties/bonuses for explicitly
  classified bad/good terminations.
- Optional exact Gaussian policy-KL regularization for low-noise continuation
  runs where auxiliary actor losses can otherwise bypass PPO ratio clipping.
- Optional post-update actor projection onto a configured rollout-policy KL
  trust region while leaving critic learning untouched.

Those changes are derived from
[NVIDIA WBC-AGILE](https://github.com/nvidia-isaac/WBC-AGILE), whose additions
are available under Apache-2.0. TensorDict, distillation, and entropy
annealing changes from the newer WBC-AGILE training stack are intentionally
not included because this repository targets Isaac Lab 2.2 and tensor
observations.
