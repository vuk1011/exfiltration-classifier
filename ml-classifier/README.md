# ML Classifier for DNS exfiltration

## Setup

Create the virtual environment and install the dependencies (run once, from this directory `ml-classifier/`):

```
python3 -m venv . # if preexisting binaries are not for your OS
./bin/pip install -r requirements.txt
```

Only direct dependencies are pinned in `requirements.txt`; pip installs its subdependencies automatically.

To activate virtual environment, always first run:
```
source ./bin/activate
```

Optionally also add [dataset](https://data.mendeley.com/datasets/c4n7fckkz3/3) to `dns_dataset` folder.

## Project itself

Project trains
