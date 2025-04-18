# Homework 3

In this homework, we will explore Convolution Networks in core computer vision tasks:
- Classification
- Segmentation
- Detection

You will need to use a GPU or [Google Colab](https://colab.research.google.com/) to train your models.

## Setup + Starter Code

In this assignment, we'll be working with two datasets:
- [SuperTuxKart Classification Dataset](https://www.cs.utexas.edu/~bzhou/dl_class/classification_data.zip) for classification
- [SuperTuxKart Drive Dataset](https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip) for segmentation and detection

You can download the datasets by running the following command from the main directory:
```bash
curl -s -L https://www.cs.utexas.edu/~bzhou/dl_class/classification_data.zip -o ./classification_data.zip && unzip -qo classification_data.zip
curl -s -L https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip -o ./drive_data.zip && unzip -qo drive_data.zip
```

Make sure you see the following directories and files inside your main directory
```
bundle.py
homework
grader
classification_data
drive_data
```
You will run all scripts from inside this main directory.

In the `homework` directory, you'll find the following:
- `models.py` - where you will implement various models
- `metrics.py` - metrics to evaluate your models
- `datasets` - contains dataset loading and transformation functions

## Training

This time, you'll implement the training code from scratch!
Note that this homework consists of two parts, so we suggest splitting your training code into two parts:
* `train_classification.py` - part 1
* `train_detection.py` - part 2

The previous homework should be a good reference, but feel free to modify different parts of the training code depending on how you want to perform experiments.

Recall that a training pipeline includes:
* Creating an optimizer
* Creating a model, loss, metrics (task dependent)
* Loading the data (task dependent)
* Running the optimizer for several epochs
* Logging + saving your model (use the provided `save_model`)

The metrics are provided in `metrics.py`, so no need to implement them yourself.
You can see how to use them in `grader/tests.py`.

### Grader Instructions

You can grade your trained models by running the following command from the main directory:
- `python3 -m grader homework -v` for medium verbosity
- `python3 -m grader homework -vv` to include print statements
- `python3 -m grader homework --disable_color` for Google Colab since colors don't show up that well

## Part 1: Classification (35 points)

We will start by extending the classification model from the previous homework to use convolutional layers.

Implement the `Classifier` model in `models.py`.
The `forward` function receives a `(B, 3, 64, 64)` image tensor as an input and should return a `(B, 6)` tensor of logits (one value per class), where `B` is the batch size.

The accuracy requirements for this model will be more challenging!
To get full credit, you might need to experiment with different forms of data augmentation, a technique to artificially increase the size of your training dataset by applying various transformations to your training data.
Data augmentation are heavily tied to the specific task  and the dataset you are working with.
For instance, in classification, one common augmentation is to apply random horizontal flips to your images.

To add data augmentations, modify the `get_transform` method in `datasets/classification_datasets.py` to construct your data augmentation pipeline.
Pass the corresponding `transform_pipeline` string to the `load_data` function.
Remember to only apply data augmentations to the training set.

The accuracy cutoff for this section is 0.80 on the validation/test set.

### Hints/Tips
- Run your model on some sample data as a sanity check - do the shapes of the tensors make sense?
- A network with just a few layers should be able to achieve a decent accuracy on this task.
- Additional tricks like residual blocks, dropout, weight regularization can help improve performance.
- The solution achieves 92% validation accuracy in 10 minutes of training on Colab.
- Remember to call `model.train()` and `model.eval()` to switch between training and evaluation modes (even when using `torch.inference_mode()`).

### Relevant Operations
 - [torch.nn.Conv2d](https://pytorch.org/docs/stable/nn.html#convolution-layers)
 - [torch.nn.BatchNorm2d](https://pytorch.org/docs/stable/nn.html#normalization-layers)
 - [torchvision.transforms.Compose](https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html)
 - [torchvision.transforms.HorizontalFlip](https://pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html)

## Part 2: Road Detection (65 points)

Next, we'll focus on a more challenging and practical task: road detection.
To do this, we'll decompose the task into two parts: depth estimation and road semantic segmentation.
When combined, we can use the depth information to construct the 3D structure of the road for navigation!

Previously we've done classification where the model predicts a single output class label per image.
In semantic segmentation, the model now predicts a class for **every pixel** in the image.
For this task in particular, we'll be predicting if a pixel is part of the left/right boundary of the race-track or background for a total of 3 classes (thought experiment: think about why the boundary is more useful than predicting binary road or not for driving).

### Dataset

We'll use the [SuperTuxKart Drive Dataset](https://www.cs.utexas.edu/~bzhou/dl_class/drive_data.zip), which contains pairs of images `(3, 96, 128)`, depth `(96, 128)` and information about the road which allows us to create road segmentation masks `(96, 128)`.
The segmentation mask consists of labels in `{0, 1, 2}`, corresponding to the background and 2 lane classes (left and right boundaries of the lanes) respectively.
The depth values are real-valued numbers normalized to the range `[0, 1]`, where 0 is close and 1 is far.
Use the `load_data` function in `datasets/drive_dataset.py` to load the dataset.
This dataset yields a dictionary with keys `image`, `depth`, and `track` (segmentation labels).

### Model

Implement the `Detector` model in `models.py`.
Your `forward` function receives a `(B, 3, 96, 128)` image tensor as an input and should return both:
- `(B, 3, 96, 128)` logits for the 3 classes
- `(B, 96, 128)` tensor of depths.

Use a series of convolutions to gradually reduce the spatial dimensions of the input while increasing the number of channels, then use up-convolutions `(torch.nn.Conv2dTranspose)` to recover the original spatial dimensions.
Here's an example of how the intermediate layer outputs shapes would look like:
```
Input   (b,  3,     h,     w)    input image
Down1   (b, 16, h / 2, w / 2)    after strided conv layer
Down2   (b, 32, h / 4, h / 4)    after strided conv layer
Up1     (b, 16, h / 2, w / 2)    after up-conv layer
Up2     (b, 16,     h,     w)    after up-conv layer
Logits  (b,  n,     h,     w)    output logits, where n = num_class
Depth   (b,  1,     h,     w)    output depth, single channel
```

Additionally, the model should be able to handle arbitrary input resolutions and produce an output of the same shape as the input by using appropriate padding and striding.

To supervise the model, you will need to use two loss functions:
- Cross-entropy loss for the segmentation task
- Regression loss (e.g., absolute error, squared error) for the depth prediction task
Use a combination of the two losses to train the model.

### Evaluation

Most of the segmentation mask will belong to the background class, making it is easy to achieve a high accuracy by predicting only background.
In this task, we will additionally evaluate the model using the mean Intersection over Union (mIoU) metric, which helps to account for the class imbalance (see the provided `ConfusionMatrix` class in `metrics.py`).

$$IoU = \frac{\text{true positives}}{\text{true positives} + \text{false positives} + \text{false negatives}}$$

For depth prediction, we will use the mean absolute error (MAE) metric as well as MAE computed for the lane boundary pixels only.

The grading cutoffs for this section are:
- Segmentation IOU > 0.75
- Depth Error < 0.05
- Depth Error for lane boundaries < 0.05

### Hints/Tips
- Use a single model to process the image features, then branch out to separate heads for segmentation and depth prediction.
- Start with the simplest model, then gradually increase the complexity.
- If you are getting IOU < 0.4, it is highly likely your segmentation head is simply predicting only the background class and not learning anything!
- Build your network as a set of composable blocks (one for down convolution and one for up convolution).
- Adding residual connections between the down blocks and the up blocks (skip connections as in [U-Net](https://en.wikipedia.org/wiki/U-Net)) are particularly useful in segmentation to help recover fine boundary details.
- Print the shapes and min/max of your tensors! Are they what you expect?
- For depth prediction, try both unconstrained regression and using an activation function that scales your model output to (0,1).

## Submission

Create a submission bundle (max size **40MB**) using:
```bash
python3 bundle.py homework [YOUR UT ID]
```

Please double-check that your zip file was properly created by grading it again.
```bash
python3 -m grader [YOUR UT ID].zip
```
After verifying that the zip file grades successfully, you can submit it on Canvas.
