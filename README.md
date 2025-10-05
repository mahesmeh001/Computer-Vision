# CSE252A Computer Vision - Dev Setup

Quick setup for Computer Vision class using pyenv.

## Setup (One-time)

```bash
# Install pyenv + virtualenv
brew install pyenv pyenv-virtualenv

# Add to ~/.zshrc
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.zshrc
source ~/.zshrc

# Create environment
pyenv virtualenv 3.9.18 computer-vision
pyenv activate computer-vision

# Install packages
pip install numpy matplotlib opencv-python jupyter ipykernel scipy scikit-image pillow

# Create Jupyter kernel
python -m ipykernel install --user --name=computer-vision --display-name="Computer Vision (CSE252A)"
```

## Usage

```bash
# Activate environment
pyenv activate computer-vision

# Start Jupyter
jupyter lab
```

**In Cursor/VS Code:** Select "Computer Vision (CSE252A)" kernel or Python interpreter at `/Users/mehul/.pyenv/versions/computer-vision/bin/python`

## Quick Test

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
print("✅ Ready to go!")
```

## Troubleshooting

- **Kernel not showing?** Try `Cmd+Shift+P` → "Python: Refresh Interpreters"
- **Packages missing?** Make sure environment is activated: `pyenv activate computer-vision`
