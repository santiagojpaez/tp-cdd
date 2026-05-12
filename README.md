# TP1 Environment Setup

This folder contains the practical assignments (`numpy_practice.ipynb` and `ia_mlp_maintenance.py`). Follow the instructions below to set up your environment to run them properly.

## Prerequisites

- [Python](https://www.python.org/downloads/) 3.8 or higher installed on your system.

## Setup Instructions

### 1. Create a virtual environment

It's highly recommended to use a virtual environment to avoid dependency conflicts. Open your terminal in the `tp1` folder and run:

**On Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

With the virtual environment activated, upgrade `pip` and install the required packages using the provided `requirements.txt` file:

```bash
pip install -U pip
pip install -r requirements.txt
```

### 3. Run the files

**Using Visual Studio Code (Recommended):**
1. Open [`numpy_practice.ipynb`](numpy_practice.ipynb) or [`ia_mlp_maintenance.py`](ia_mlp_maintenance.py).
2. Click on the kernel selector in the top right corner of the editor.
3. Select **Python Environments** -> select the `.venv` that you just created.
4. You can now execute the notebook cells (or `# %%` interactive cells in the python script) directly in the editor.

**Using Jupyter Notebook via Terminal:**
If you prefer the browser-based Jupyter interface, run:
```bash
jupyter notebook
```
Navigate to your `.ipynb` file in the browser window that opens.
