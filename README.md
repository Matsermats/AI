# AI

This repository contains a simple example of how to train a model that can
classify images as `dog`, `cat`, or `other`.

## Dataset

Prepare a directory with three subfolders named `dog`, `cat`, and `other`.
Each subfolder should contain images of the corresponding class.

```
/dataset
  /dog
  /cat
  /other
```

## Training

Install dependencies with `pip install torch torchvision pillow` and then run:

```
python src/train.py /path/to/dataset model.pth --epochs 5
```

This will create `model.pth` containing the trained weights.

## Prediction

To classify an image, run:

```
python src/predict.py model.pth image.jpg dog cat other
```

This prints the predicted class.

## GUI

After training a model, you can launch a small Tkinter GUI to classify images
interactively. By default the GUI looks for a file called `model.pth` in the
current directory and assumes three classes: `dog`, `cat` and `other`.

```
python src/gui.py
```

Use `--model` to specify a different model file and `--class-names` to specify
the class labels if they differ from the defaults. For example:

```
python src/gui.py --model my_model.pth --class-names dog cat
```
