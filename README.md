# Moatless Tools v0.0.2 - Fork

[SWE-Gym Dataset + Models](https://huggingface.co/SWE-Gym)

[Project page](https://swe-gym.github.io/)

This fork is based on `a1017b7` commit of [aorwall/moatless-tools](https://github.com/aorwall/moatless-tools).

We have made the following changes to enable dramatically faster iteration speed:

To run moatless-tools agent, you need the vector index of the repos too, which we pre-computed for SWE-Gym Lite and SWE-Bench Lite. Download through [this link](https://huggingface.co/datasets/SWE-Gym/Codebase-Index-Lite).

## Parallel Sampling
Instead of running a single agent task by task, we run multiple agents in parallel, by `scripts/parallel_sample.sh`

This enables us to sample the entire SWE-Gym Lite/SWE-Bench Lite within 30 mins for 32B model and 10 mins for 7B model with a single H100 GPU.

We also evaluate patches through OpenHands Cloud Service by `scripts/eval_preds.sh`

## Parallel Vector Indexing