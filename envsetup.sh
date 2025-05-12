#!/bin/bash

# 🧠 Make sure script is run using Bash
if [ -z "$BASH_VERSION" ]; then
    echo "❌ Please run this script using Bash (e.g., bash envsetup.sh)"
    exit 1
fi

# ⚠️ Warn if running in native Windows shell
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        echo "⚠️ Detected Windows shell. Please use Git Bash or WSL to run this script."
        ;;
esac

ENV_NAME=$(basename "$PWD")
START_TIME=$(date +%s)

# 🛑 Prevent running in unsafe folders
BAD_FOLDERS=("$HOME" "$HOME/bin" "/" "/usr" "/opt" "/tmp")
for bad in "${BAD_FOLDERS[@]}"; do
    if [[ "$PWD" == "$bad" ]]; then
        echo "🚫 You are in an unsafe folder: $PWD"
        echo "💡 Please run envsetup inside a project folder (e.g., ~/my-repo)"
        exit 1
    fi
done

echo "🔍 Setting up project: $ENV_NAME"

# 🛑 Handle existing venv
if [ -d ".venv" ]; then
    read -p "⚠️  .venv already exists. Overwrite? [y/N]: " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        echo "❌ Aborting setup."
        exit 1
    fi
    rm -rf .venv
fi

# ✅ Create & activate venv
echo "🔧 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "📦 Upgrading pip..."
pip install --upgrade pip

# 📦 Install core packages (curated)
echo "📦 Installing core packages..."
pip install \
    "numpy>=1.25" \
    "pandas>=2.2" \
    "scikit-learn>=1.3" \
    "scipy>=1.11" \
    ipykernel \
    "ipython>=8.0" \
    jupyter_client \
    matplotlib-inline \
    "matplotlib>=3.8" \
    "pillow>=10.0" \
    joblib \
    tqdm \
    psutil \
    sympy \
    networkx

# ⚙️ Install PyTorch GPU version unless --minimal passed
if [[ "$1" != "--minimal" ]]; then
    echo "🚀 Installing PyTorch 2.2.0 (GPU, CUDA 12.1 compatible)..."
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
else
    echo "⚠️ Skipping PyTorch installation (--minimal mode)"
fi

# 🧪 Register Jupyter kernel
python3 -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)"

# 📝 Create .env file
# Set data path and create the folder
datapath="./data"
mkdir -p "$datapath"

cat <<EOF > .env
# Environment variables
PROJECT_NAME=$ENV_NAME
PYTHON_ENV=development
DATA_PATH=$datapath
EOF

echo "🧾 .env file created."

# ⚙️ VS Code settings
mkdir -p .vscode
cat <<EOF > .vscode/settings.json
{
  "python.envFile": "\${workspaceFolder}/.env"
}
EOF

echo "🧠 VS Code will auto-load .env"

# 🧹 .gitignore
if [ -f ".gitignore" ]; then
    read -p "⚠️ .gitignore already exists. Overwrite? [y/N]: " gignore_choice
    if [[ "$gignore_choice" =~ ^[Yy]$ ]]; then
        create_gitignore=true
    else
        echo "ℹ️ Skipping .gitignore overwrite."
        create_gitignore=false
    fi
else
    create_gitignore=true
fi

if [[ "$create_gitignore" = true ]]; then
    cat <<EOF > .gitignore
# Virtual environment
.venv/

# Python cache
__pycache__/
*.pyc

# Jupyter
.ipynb_checkpoints/

# Environment config
.env

#Data
data/

# VS Code settings
.vscode/
EOF
    echo "✅ .gitignore created."
fi

# 📦 Write clean requirements.txt
echo "📦 Writing clean requirements.txt..."

cat <<EOF > requirements.txt
# Core packages
numpy>=1.25
pandas>=2.2
scikit-learn>=1.3
scipy>=1.11

# Jupyter + IPython
ipykernel
ipython>=8.0
jupyter_client
matplotlib-inline

# Visualization
matplotlib>=3.8
pillow>=10.0

# Utility
joblib
tqdm
psutil

# Optional (symbolic math / graph)
sympy
networkx

# NOTE: PyTorch is installed separately via envsetup.sh
EOF

echo "✅ Clean requirements.txt saved."

# ⏱️ Done
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "✅ Environment setup complete in ${ELAPSED}s!"
echo "💡 In VS Code, select kernel: Python ($ENV_NAME)"
