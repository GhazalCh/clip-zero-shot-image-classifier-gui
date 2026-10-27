# CLIP Zero-Shot Image Classifier

A desktop application for zero-shot image classification using a pretrained CLIP vision–language model. Built with Python, PyTorch, Hugging Face Transformers, and Tkinter, the application allows users to select an image and view its scores across predefined semantic classes.

## Overview

This project implements a graphical image classification application using the pretrained `openai/clip-vit-large-patch14` model.

Instead of training a separate image classifier, the application compares an input image with multiple natural-language prompts associated with predefined classes.

The similarity scores of prompts belonging to the same class are averaged to obtain a class-level score. Softmax is then applied across the class scores to produce normalized values displayed as percentages.

The application presents the results in a desktop interface, including the selected image, scores for all candidate classes, and the highest-scoring class.

## Features

* Zero-shot image classification using a pretrained CLIP model.
* Graphical user interface built with Tkinter.
* Image selection through a file dialog.
* Image preview before displaying classification results.
* Multiple natural-language prompts per class.
* Prompt-level score aggregation into class-level scores.
* Percentage-based display of normalized scores.
* Final prediction based on the highest-scoring class.
* Background threads for model loading and image classification.

## Technologies

* **Python** — Application logic.
* **PyTorch** — Tensor operations and model inference.
* **Hugging Face Transformers** — CLIP model and processor.
* **Tkinter** — Desktop graphical interface.
* **Pillow** — Image loading, resizing, and display.
* **Requests** — Handling model download-related exceptions.

## Model

The application uses:

`openai/clip-vit-large-patch14`

The model and its processor are loaded through the Hugging Face Transformers library.

On the first run, model files may need to be downloaded from Hugging Face. An internet connection is therefore required unless the model has already been downloaded and is available locally.

## Candidate Classes

The application currently evaluates six predefined classes:

* `cat`
* `girl and fireworks`
* `light souls`
* `snowman`
* `harbor`
* `Snowy Christmas Town`

Each class is associated with multiple natural-language prompts. These prompts provide alternative descriptions of the concepts represented by the class.

The classes and prompts are defined in the `Promts_Per_Class` dictionary in the source code.

## Project Structure

```text
clip-zero-shot-image-classifier-gui/
├── zero-shot.py
├── requirements.txt
├── .gitignore
└── README.md
```

The Python filename shown above is a suggested repository filename. Rename the uploaded script to `zero-shot.py` before using this structure, or update the commands below to match its actual filename.

An `images/` directory is created automatically when needed. The user can select an image through the file dialog.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/GhazalCh/clip-zero-shot-image-classifier-gui
cd clip-zero-shot-image-classifier-gui
```

Replace the placeholders with your actual GitHub repository URL and directory name.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Install a compatible version of PyTorch:

https://pytorch.org/get-started/locally/

Install the remaining Python dependencies:

```bash
pip install transformers pillow requests
```

Tkinter is included with many standard Python installations. On some Linux distributions, it must be installed separately through the operating system's package manager.

For reproducible results, record the tested dependency versions in `requirements.txt`.

## Usage

### 1. Run the application

If your script is named `zero-shot.py`, execute:

```bash
python zero-shot.py
```

If you keep the original filename, `zero-shot(1).py`, use:

```bash
python "zero-shot(1).py"
```

### 2. Wait for the model to load

The application loads the pretrained model and its processor in a background thread. Once loading is complete, the image selection button becomes available.

### 3. Select an image

Use the image selection button to open the file dialog and choose an image.

Supported formats:

* JPG
* JPEG
* PNG

### 4. View classification results

The application displays the selected image and the normalized scores for all six classes.

The class with the highest score is shown as the final prediction.

## How It Works

1. **Model initialization:** Loads the pretrained CLIP model and processor.
2. **Prompt preparation:** Collects all natural-language prompts associated with the candidate classes.
3. **Image selection:** Allows the user to select an input image.
4. **Image preprocessing:** Converts the image to RGB and prepares it for inference.
5. **Image–text comparison:** Computes CLIP's image–text similarity logits.
6. **Prompt aggregation:** Averages the logits associated with each class.
7. **Score normalization:** Applies softmax across the aggregated class scores.
8. **Result display:** Shows the class scores and identifies the highest-scoring class.

## Understanding the Scores

The application displays class scores as percentages.

These values are produced by applying softmax to the averaged prompt logits. They represent normalized scores relative to the six candidate classes.

They should not be interpreted as calibrated probabilities that a prediction is objectively correct.

The result also depends on the wording and selection of the prompts associated with each class.

## Limitations

* The application evaluates only the six predefined classes.
* Predictions depend on the prompts used to describe each class.
* The final prediction is restricted to the class with the highest normalized score.
* The displayed percentages are not calibrated confidence estimates.
* Model loading may require an internet connection on the first run.
* The application does not include a labeled test dataset or quantitative accuracy evaluation.

## Potential Improvements

* Allow users to add or edit classes through the graphical interface.
* Allow custom prompts for each class.
* Add a progress indicator during inference.
* Display the top-ranked class more prominently.
* Export classification results to CSV.
* Add model caching and configurable model selection.
* Evaluate the application on a labeled dataset.

## Learning Objectives

This project demonstrates practical experience with:

* Vision–language models.
* Zero-shot image classification.
* Prompt-based classification.
* Pretrained model inference.
* Similarity-logit aggregation.
* PyTorch and Hugging Face Transformers.
* Desktop GUI development with Tkinter.
* Background task execution in Python.

## References

* [Hugging Face CLIP Documentation](https://huggingface.co/docs/transformers/model_doc/clip)
* [Pretrained CLIP Model](https://huggingface.co/openai/clip-vit-large-patch14)
* [CLIP Research Paper](https://arxiv.org/abs/2103.00020)
* [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## License

This repository contains an educational implementation. The licensing terms of the pretrained model, software libraries, and dependencies remain applicable.
