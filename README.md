# carbonsix-python-template

[![build-and-upload-wheel](https://github.com/Carbon6Robotics/carbonsix-python-template/actions/workflows/build-and-upload.yml/badge.svg?branch=main)](https://github.com/Carbon6Robotics/carbonsix-python-template/actions/workflows/build-and-upload.yml)

Template repository for python projects

## LIBERO LCM Environment

This repository now includes a separate LIBERO runtime that can be controlled
over LCM, without modifying anything under `reference/`.

### 1) Start the LIBERO simulator server

```bash
libero-lcm run-env --scenario config/libero_lcm_scenario.yaml
```

- The simulator listens for action commands on `LIBERO_COMMAND`.
- This mode streams camera frames over LCM for `libero-lcm visualize`.
- If `lcm_show_native_viewer: true`, it also opens native MuJoCo viewer (3D camera).

### 1-1) Run local MuJoCo interactive viewer

```bash
libero-lcm run-interactive --scenario config/libero_lcm_scenario.yaml
```

- This mode opens the native robosuite / MuJoCo interactive viewer locally.
- This mode uses the native `mujoco.viewer` window (mouse orbit / pan / zoom).

### 1-2) Viewer smoke test (mouse-control check)

```bash
libero-lcm test-viewer --steps 10000
```

- This launches a minimal `Lift` scene in native `mujoco.viewer`.
- Use this first to verify mouse camera interaction works on your machine.

### 2) Visualize the running environment stream

In another terminal:

```bash
libero-lcm visualize
```

- This subscribes to `LIBERO_FRAME` and shows a live frame window.
- It also reads `LIBERO_STATE` and overlays step / done info.

### 3) Control the simulator over LCM

In another terminal:

```bash
# Reset env
libero-lcm control reset

# Single step action (7-DoF example)
libero-lcm control step --action 0,0,0,0,0,0,0

# Random test actions
libero-lcm control random --steps 200 --rate-hz 10
```

### Notes

- `reference/` is unchanged and independent from this flow.
- LIBERO Python package must be installed in your environment to run the
  simulator server.
- `control` commands affect `run-env` (LCM server mode), not `run-interactive`.

## Directory Structure
An example directory structure is provided in the this template. However, structure may change depending on the needs of the project.

## Recommended Software Packages
Following softwares are recommended for the development, and the template assumes you use the followings. If you don't use these tools, please make necessary changes.
- `uv`: for managing virtual environment and python dependencies
- `ruff`: for linting and formatting
- `pyright`: for static type checking

## Setting up GitHub Actions

This template includes a pre-configured GitHub Actions workflow for building wheels and running tests. The workflow file is located at [.github/workflows/build-and-upload.yml](.github/workflows/build-and-upload.yml) and is commented out by default.

To enable the workflow:
1. Open [.github/workflows/build-and-upload.yml](.github/workflows/build-and-upload.yml)
2. Uncomment all lines (remove the `#` prefix from each line)
3. Modify Github Actions workflow for your repository
4. Update the GitHub Actions badge in this README.md file:
   - Replace `Carbon6Robotics/carbonsix-python-template` with your repository's owner and name

The workflow will automatically:
- Run code formatting checks via pre-commit
- Execute tests using pytest
- Build wheel packages
- Upload wheels to AWS CodeArtifact
