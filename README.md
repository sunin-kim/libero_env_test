# carbonsix-python-template

[![build-and-upload-wheel](https://github.com/Carbon6Robotics/carbonsix-python-template/actions/workflows/build-and-upload.yml/badge.svg?branch=main)](https://github.com/Carbon6Robotics/carbonsix-python-template/actions/workflows/build-and-upload.yml)

Template repository for python projects

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
