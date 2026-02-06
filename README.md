# MGTEval

MGTEval is a unified framework for machine generated text detection. It covers dataset building, adversarial text attacks, detector training, and detector evaluation in one workflow.

---

## Introduction
MGTEval focuses on a single workflow for dataset building, attacks, training, calibration, and evaluation. It brings metric based and model based detectors into one CLI with consistent inputs and outputs. It includes twelve text attack methods for robustness checks and reports standard metrics such as AUROC AUPR ECE Brier TPR at FPR and ASR.

---

## Installation

### Setup with conda
Step 1 Create a new environment
```bash
conda create -n mgteval python=3.12 -y
```

Step 2 Activate the environment
```bash
conda activate mgteval
```

Step 3 Upgrade pip
```bash
pip install -U pip
```

Step 4 Install MGTEval in editable mode
```bash
pip install -e .
```

[Optional] Install VLLM: 
```bash
pip install -e '.[vllm]'
```

Install Server Package:
```bash
pip install -e '.[server]'
```

Step 5 Verify the installation
```bash
mgteval-cli --help
mgteval-cli list
```

Step 6 Start Server
```bash
./start_dev.sh
```
### Setup with venv
Step 1 Create a virtual environment
```bash
python -m venv .venv
```

Step 2 Activate the environment
```bash
source .venv/bin/activate
```

Step 3 Upgrade pip
```bash
pip install -U pip
```

Step 4 Install MGTEval in editable mode
```bash
pip install -e .
```

Step 5 Verify the installation
```bash
mgteval-cli --help
mgteval-cli list
```

---

## Workflow and examples

This section explains the full pipeline from data construction to evaluation. Use the example YAML files in the examples directory. The commands below reference those files without repeating their contents.

### Step 1 Build a dataset from human text
Use the build subcommand with the example file to generate machine text from human prompts.

Example file
- examples/build/build_dataset.yaml

Command
- mgteval-cli build examples/build/build_dataset.yaml

What this step does
- Reads human text from the input dataset
- Builds prompts from the selected label
- Generates machine text with the configured backend
- Writes a paired dataset with human and machine records

### Step 2 Apply attacks to an existing dataset
Use the attack subcommand with the example file to perturb an existing dataset.

Example file
- examples/attack/build_attack_dataset.yaml

Command
- mgteval-cli attack examples/attack/build_attack_dataset.yaml

What this step does
- Loads the dataset produced in step 1 or any existing dataset
- Applies a selected set of text attacks from the attack config
- Writes an attacked dataset for robustness evaluation

### Step 3 Train a detector with datasets
Choose one branch to follow first. You can return to the other branch later.

Branch A Metric based training
Example file
- examples/train/binoculars.yaml
Command
- mgteval-cli train examples/train/binoculars.yaml
What this step does
- Runs the detector on training data
- Fits a calibrator to map scores to probabilities
- Saves calibration outputs for later evaluation

Branch B Model based training
Example file
- examples/train/coco.yaml
Command
- mgteval-cli train examples/train/coco.yaml
What this step does
- Fine tunes the detector on the training dataset
- Saves checkpoints and training summaries

### Step 4 Detect datasets with the trained detector
Use the detection branch that matches the training branch you chose.

Branch A Metric based detection
Example file
- examples/detect/binoculars.yaml
Command
- mgteval-cli detect examples/detect/binoculars.yaml
What this step does
- Runs evaluation on the test dataset
- Produces metrics and curves
- Can evaluate on attacked datasets if configured

Branch B Model based detection
Example file
- examples/detect/coco.yaml
Command
- mgteval-cli detect examples/detect/coco.yaml
What this step does
- Loads the trained checkpoint
- Evaluates on the test dataset
- Produces metrics and curves

---

## Supported detectors

### Metric based detectors

| Detector full name | Key |
|---|---|
| Binoculars | binoculars |
| DetectGPT | detectgpt |
| Fast DetectGPT | fastdetectgpt |
| GLTR | gltr |
| Likelihood | likelihood |
| Rank | rank |
| LogRank | logrank |
| Entropy | entropy |
| LRR | lrr |
| NPR | npr |
| RAIDAR | raidar |
| TOCSIN | tocsin |
| DNA DetectLLM | dnadetectllm |
| DNA GPT | dnagpt |
| LASTDE | lastde |
| LASTDE plus plus | lastdepp |

### Model based detectors

| Detector full name | Key |
|---|---|
| Pretrained Generic | pretrained |
| OpenAI Detector RoBERTa base | openai-detector-base |
| OpenAI Detector RoBERTa large | openai-detector-large |
| SimpleAI Detector | simpleai-detector |
| RADAR | radar |
| GREATER | greater |
| DeTeCtive | detective |
| CoCo | coco |
| ImBD | imbd |
| Longformer | longformer |
| Longerformer | longerformer |
| MPU | mpu |
| PECOLA | pecola |

---

## Supported attack approaches

| Attack type full name | Key | Description |
|---|---|
| Span perturbation | span | Mask and fill spans to rewrite parts of a sentence while preserving meaning |
| Paraphrase | para | Rephrase text with paraphrasing models or LLM backends |
| Typo mixed | typo | Mixed character noise including insert delete substitute and transpose |
| Typo insertion | inse | Insert extra characters into words |
| Typo deletion | dele | Delete characters from words |
| Typo substitution | subs | Replace characters with nearby or random alternatives |
| Typo transposition | tran | Swap adjacent characters inside words |
| Homoglyph alteration | homo | Replace characters with visually similar glyphs |
| Format character editing | form | Insert formatting noise such as zero width or case shifts |
| Synonym substitution | syno | Replace words with synonyms while keeping meaning |
| Back translation | back_trans | Translate to a pivot language and back to rewrite text |
| Humanize | humanize | Use a model to rewrite machine text into more human style |

---

## Notes
- Use the examples directory for complete configurations
- For metric based detectors, calibration improves probability quality
- For model based detectors, checkpoint choice and max_length are the strongest drivers
