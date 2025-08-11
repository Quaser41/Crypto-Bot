# Crypto-Bot


## Installation

1. Install the native TA-Lib library **before** installing Python packages.
   - **Debian/Ubuntu**:
     ```bash
     sudo apt-get update && sudo apt-get install -y build-essential ta-lib
     ```
   - **macOS (Homebrew)**:
     ```bash
     brew install ta-lib
     ```
   - **Windows**: Download precompiled TA-Lib binaries from [https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and install them, ensuring the library is on your PATH.

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

=======
## Running the Bot

### Unix-like systems
Use the provided shell script:

```bash
./run_bot.sh
```

### Windows
Run the batch script:

```cmd
run_bot.bat
```

Both scripts change to the project directory and execute `python main.py`.

