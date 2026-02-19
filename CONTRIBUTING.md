# Contributing to <REPOSITORY_NAME>

This guide will help you set up your development environment and contribute to the project.

## Development Environment Setup

### 1. Install UV

UV is the package manager used in this project. Install it using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Setup AWS CodeArtifact using keyring

This project uses AWS CodeArtifact as a private PyPI repository. You need to configure authentication to access private packages.

#### Install AWS CLI

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

#### Configure AWS Credentials

Ask `kyehoon.jeon@carbon6robotics.com` or `daeson.park@carbon6robotics.com` for AWS account credentials.

```bash
aws configure set aws_access_key_id <your_access_key>
aws configure set aws_secret_access_key <your_secret_access_key>
aws configure set region ap-northeast-2
echo "export UV_INDEX_CARBONSIX_PYPI_USERNAME=aws" >> ~/.bashrc
source ~/.bashrc
```

#### Install keyring with CodeArtifact support

```bash
uv tool install keyring --with keyrings.codeartifact
```

### 3. Install Project Dependencies

Clone the repository and install dependencies:

```bash
git clone <your_repository_url>
cd <repository_folder>
uv sync
```

This will:
- Create a virtual environment (if using `.venv`)
- Install all project dependencies including dev dependencies
- Install the project in editable mode

### 4. Activate Virtual Environment (if needed)

If you're using a `.venv` virtual environment:

```bash
source .venv/bin/activate
```

Or use `uv run` prefix for commands:

```bash
uv run python <script>
```

### 5. Test it

Run tests using pytest:

```bash
uv run pytest
```


## Code Quality

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. Install and set up pre-commit:

```bash
uv run pre-commit install
```

The pre-commit hooks will automatically run on commit and include:
- Ruff linting and formatting
- YAML validation
- Large file checks
- Documentation linting (pydoclint)

You can manually run pre-commit hooks on all files:

```bash
uv run pre-commit run --all-files
```

## Continuous Integration and Testing

This project uses GitHub Actions for continuous integration. The build and test process is automated via [build-and-upload.yml](.github/workflows/build-and-upload.yml).

### Automated Build and Test

Once enabled, when you push to the `main` branch or create a pull request, the following actions are automatically triggered:

1. **Code Formatting**: Pre-commit hooks run to ensure code quality and formatting
2. **Test**: Tests are executed using pytest
3. **Build**: The project builds wheel packages
4. **Upload**: Built wheels are uploaded to AWS CodeArtifact

You can check the status of builds and tests in the [GitHub Actions tab](https://github.com/Carbon6Robotics/<your_repository_name>/actions) of the repository.
