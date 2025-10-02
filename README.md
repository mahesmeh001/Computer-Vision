# Computer Vision (CSE252A) Development Environment Setup

This README provides instructions for setting up a Python development environment for the Computer Vision class using pyenv.

## Prerequisites

- macOS (tested on macOS 14.3.0)
- Homebrew installed
- Terminal access

## Installation Steps

### 1. Install pyenv and pyenv-virtualenv

```bash
# Install pyenv
brew install pyenv

# Install pyenv-virtualenv plugin
brew install pyenv-virtualenv
```

### 2. Configure Shell

Add the following lines to your `~/.zshrc` file:

```bash
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
```

Then reload your shell:
```bash
source ~/.zshrc
```

### 3. Create Virtual Environment

```bash
# Create a virtual environment for Computer Vision
pyenv virtualenv 3.9.18 computer-vision

# Activate the environment
pyenv activate computer-vision
```

### 4. Install Required Packages

```bash
# Install essential packages for computer vision work
pip install numpy matplotlib opencv-python jupyter ipykernel scipy scikit-image pillow
```

### 5. Create Jupyter Kernel

```bash
# Create a Jupyter kernel from the virtual environment
python -m ipykernel install --user --name=computer-vision --display-name="Computer Vision (CSE252A)"
```

## Usage

### Activating the Environment

To activate the virtual environment for development:

```bash
pyenv activate computer-vision
```

### Running Jupyter Notebooks

1. Start Jupyter Lab or Jupyter Notebook:
   ```bash
   jupyter lab
   # or
   jupyter notebook
   ```

2. When creating a new notebook, select the "Computer Vision (CSE252A)" kernel from the kernel selection menu.

### Deactivating the Environment

To deactivate the virtual environment:

```bash
pyenv deactivate
```

## Installed Packages

The environment includes the following packages commonly used in computer vision:

- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **opencv-python**: Computer vision library
- **scipy**: Scientific computing
- **scikit-image**: Image processing
- **pillow**: Python Imaging Library
- **jupyter**: Jupyter notebook environment
- **ipykernel**: Jupyter kernel for Python

## Troubleshooting

### If pyenv commands are not found

Make sure your shell configuration is properly loaded:
```bash
source ~/.zshrc
```

### If the virtual environment is not activated

Check available environments:
```bash
pyenv versions
```

Activate the environment:
```bash
pyenv activate computer-vision
```

### If Jupyter kernel is not available

Recreate the kernel:
```bash
python -m ipykernel install --user --name=computer-vision --display-name="Computer Vision (CSE252A)"
```

## Project Structure

```
Comp Vision/
├── CSE252A_FA25_assignment_0/
│   ├── CSE252A_FA25_assignment_0.ipynb
│   ├── california_map.png
│   ├── california_map_mask.png
│   ├── SunGod.jpg
│   └── usa_map.png
└── README.md
```

## Notes

- The virtual environment is isolated from your system Python installation
- All packages are installed locally within the virtual environment
- The Jupyter kernel allows you to use this environment in Jupyter notebooks
- Remember to activate the environment before working on assignments

## Support

If you encounter any issues with the setup, please check:
1. pyenv is properly installed and configured
2. The virtual environment is activated
3. All required packages are installed
4. The Jupyter kernel is properly registered
