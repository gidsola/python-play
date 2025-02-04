

# Python Play  <img src="https://github.com/gidsola/python-play/actions/workflows/pylint.yml/badge.svg"/>

Welcome to the Python Play repository!  

This workspace is dedicated to experimenting with Python code and exploring various programming concepts using Huggingface and locally hosted models as well as other custom endpoint implements.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/python-play.git
    ```
2. Navigate to the project directory:
    ```bash
    cd python-play
    ```
3. Install the required dependencies:
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install "fastapi[standard]"
    ```

## Getting Online
You can start the fastapi server by typing: 
```bash 
fastapi dev main.py
```

Happy coding!
