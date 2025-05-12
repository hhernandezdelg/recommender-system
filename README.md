# deep-learning-assignment1
Home work assignment for fashion recommender

## 🚀 Environment Setup

This repository uses a shared environment with GPU-ready PyTorch, Jupyter, and common ML libraries. Please follow the steps below based on your operating system.

---

### 🧩 1. Clone the Repository (All Systems)

```bash
git clone git@github.com:hhernandezdelg/deep-learning-assignment1.git
cd deep-learning-assignment1

### 2. Set up your environment (Windows)
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

### 2. Set up your environment (macOS)
bash envsetup.sh
source .venv/bin/activate       

## 🧠 Always Activate the Environment
** Before running any Python or Jupyter code, make sure the .venv environment is active. You should see (.venv) in your terminal prompt**

### ▶️ On Linux/macOS
source .venv/bin/activate


### 🪟 On Windows (PowerShell)
.venv\Scripts\Activate.ps1


```
✅ After activation, you can run:
    jupyter notebook to open notebooks


📂 Folder Structure
├── data/               # (empty) Place datasets here
├── notebooks/          # Jupyter experiments
├── envsetup.sh         # Script for Linux/macOS setup
├── requirements.txt    # Dependencies
├── .env                # Project settings (auto-created)
├── .gitignore
├── .vscode/
└── README.md


🔧 Git Workflow (Team Guidelines)

    Work in a branch:
    git checkout -b yourname-feature

    Push and open a pull request (PR) to main when done.
    Keep .venv/, data/, and .env out of Git!

📦 Key Dependencies

    Python 3.8+
    PyTorch 2.2.0 (CUDA 12.1)
    torchvision, torchaudio
    numpy, pandas, matplotlib, scikit-learn
    Jupyter, ipykernel

See requirements.txt for the full list.