# extras

## 📥 Installation:

## Install [uv](https://docs.astral.sh/uv/getting-started/installation/#installing-uv):
### 1. For Linux/MacOS:
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### 2. For Windows:
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Clone the repository:
```
mkdir -p $HOME/Desktop/QL/
cd $HOME/Desktop/QL/
git clone https://github.com/aashish-ql/extras.git
cd extras
```

## Install all `uv` dependencies:
```
uv sync
```

## Activate python environment:
```
source .venv/bin/activate
```

## Start working on the project:
```
labelImg images/ labels/classes.txt
```

## To extact `images` and `labels`:
```
uv run scripts/extract_images_and_labels.py
```

## To update labels:
```
uv run scripts/update_labels.py
```

## To split data:
```
uv run scripts/split_data.py
```

## To train models:
```
uv run scrpits/train_script.py
```

## To do batch analysis and convert data to AVA:
```
uv run scripts/batch_analysis_and_convert_dataset_to_ava.py
```