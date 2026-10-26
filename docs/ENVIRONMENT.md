# Persistent environment reference

This workspace contains several related Git repositories under `/home/group2`; the same tree is mounted at `/workspace/group2` in the validated container. The workspace-level `AGENTS.md` is the onboarding entry point for sessions started here or below. This document records stable setup facts and the commands that were checked against the current repository.

## Runtime boundary

The validated container is `va-train`, using image `registry.mthreads.com/presale/devtech/vllm-mt:d0930`, MThreads runtime, host networking, host IPC, privileged mode, and 32 GiB shared memory. Check that it exists before using it:

```bash
test -f /.dockerenv && echo container || echo host
docker inspect -f '{{.Name}} {{.Config.Image}} {{.HostConfig.Runtime}}' va-train
```

From the host, enter it with `docker exec -it va-train bash`; inside it, do not run `docker exec` again. The host path `/home/group2` maps to `/workspace/group2`. Core trees are `dl-group2`, `GR00T-WholeBodyControl`, `canonical_va`, `va_training`, `expert_datasets`, and `outputs`.

## Python and MUSA

The previously verified container baseline is Python 3.10.12, torch 2.9.0, torch_musa 2.9.0+8432052, `torch.musa.is_available() == True`, eight MTT S4000 devices, and MUSA SDK `/usr/local/musa-4.3.6`. Verify the live values without changing anything:

```bash
python3 - <<'PY'
import sys, torch
print(sys.executable, sys.version.split()[0])
print('torch', torch.__version__)
print('torch_musa', __import__('torch_musa').__version__)
print('musa_available', torch.musa.is_available())
print('musa_devices', torch.musa.device_count())
PY
```

Training and offline VA evaluation use this environment and MUSA. Do not create a new conda/venv or install, upgrade, downgrade, or replace torch or torch_musa. If imports fail, first check host/container, interpreter, image/runtime, and mounted libraries.

## Hardware EGL and MuJoCo

The checked graphics profile is `canonical_va/graphics_profile.json`. Its files and hashes are under `outputs/mujoco_gpu_rendering/driver_stage`; the existing `canonical_va.rendering.bootstrap()` validates them and re-execs before MuJoCo is imported when loader variables need to change. The equivalent shell setup is:

```bash
source outputs/mujoco_gpu_rendering/enable_mthreads_egl.sh
python3 -m canonical_va.rendering --profile auto --mode production --output /tmp/egl_probe.json
```

A valid production probe reports `GL_VENDOR=Moore Threads` and `GL_RENDERER=MTT S4000`. `llvmpipe`, `softpipe`, and other software renderers are rejected in production. `torch.musa.is_available()` does not validate graphics hardware. The profile is machine-dependent: when the host or driver changes, recheck the profile file hashes and renderer identity; do not copy drivers or alter the host graphics stack.

## DDS namespace

Torch-only training, offline tensor evaluation, manifest inspection, and reports do not need DDS. Native SONIC replay, closed-loop rollout, and native canonical compilation do. `canonical_va.batch` already launches each native episode with `unshare --net`, brings up loopback, and then starts the existing pipeline; reuse that orchestration. Do not change DDS scheduling, SONIC controller behavior, or `TabletopDDS.step`. A manual capability check is read-only:

```bash
command -v unshare
unshare --net bash -c 'ip link set lo up && echo isolated-network-namespace-ok'
```

Only run native jobs in their isolated namespace so concurrent jobs cannot consume one another's DDS traffic. Load the EGL bootstrap in the same process when the native task renders RGB.

## Work map

| Work | Environment |
|---|---|
| Code inspection, Git, reports | Host or container as appropriate |
| VA training and offline evaluation | `va-train` + MUSA |
| Native SONIC or closed loop | `va-train` + isolated DDS namespace |
| Native canonical compiler | `va-train`; isolated DDS in native mode |
| MuJoCo RGB rendering | validated MThreads EGL profile |
| Manifest analysis | ordinary Python; no DDS or GPU required |

Use the repository's actual CLI help before invoking a command. The lightweight checker below is the first diagnostic; it never installs software, starts a simulation, restarts Docker, or mutates configuration.

## What must remain unchanged

Do not modify SONIC native control, encoder/decoder, G1 model or joint order, `TabletopDDS.step`, canonical timing/mappings, VA V4 architecture, datasets, checkpoints, DDS semantics, the verified graphics bootstrap, or approved critical entries in `canonical_va/source_policy.json` as part of environment onboarding. New documentation and the checker are outside the runtime fingerprint.

## Revalidation triggers

Re-run the checker after changing the container image/runtime, host MUSA driver, graphics profile, mount mapping, or DDS/network setup. Reuse the documented environment for ordinary new sessions. Human confirmation is still required for selecting an idle GPU, starting a native job, and accepting a changed driver/profile hash.
