# Torch 2.6 Compatibility Spike

This branch validates GT4SD source compatibility with CPU Torch 2.6.0 while
preserving the current Torch 1.12 production dependency manifests.

## Changes

- Updates the TorchDrug scheduler compatibility patch for Torch 2's public
  `LRScheduler` base class while retaining Torch 1.x support.
- Installs a compatibility wrapper that restores the historical implicit
  `torch.load(..., weights_only=False)` behavior for trusted legacy GT4SD
  checkpoints. Explicit `weights_only=True` calls remain unchanged.
- Pins the Moses and GuacaMol-baselines testing forks to immutable commits that
  fix Torch 2's boolean-mask requirement in Moses generation and ORGAN
  training.

## Security boundary

The `torch.load` wrapper is a compatibility measure, not a hardening change.
Full pickle loading is appropriate only for trusted checkpoint artifacts.

## Verification

With CPU Torch 2.6.0, TorchVision 0.21.0, matching PyG CPU wheels, and a
`pytorch-fast-transformers` rebuild against Torch 2.6, the published testing
fork pins passed the full suite:

- 400 passed, 61 skipped
- Black and Flake8 passed
- all six CLI entry points passed

## Review request

The existing GitHub Actions workflow remains a Torch 1.12 job. **Request for
@drugilsberg / Matteo:** please add or advise on a dedicated Torch 2.x Actions
job so this compatibility path is validated remotely rather than only in the
local spike environment.

## Scope

This does not migrate the project's production manifests to Torch 2, and it
does not claim Torch-related advisories are closed. A production runtime
upgrade, including its dependent Transformers/Diffusers stack, remains a
separate effort.
