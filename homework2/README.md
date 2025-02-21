# Homework 2

In this homework, we will train **deep networks** to classify images from *SuperTuxKart*.

<img src="./viz.png" width="400">

This homework will require a GPU - if you don't have access to one, you can use Google Colab.

We have provided some additional instructions in this [starter colab notebook](https://colab.research.google.com/drive/1k-OTy-eM7BDHqOrRyM9yTeLFvqdjpzvd)

## Setup + Starter Code

The starter code contains a `data` directory where you'll copy (or symlink) the [SuperTuxKart Classification Dataset](https://www.cs.utexas.edu/~bzhou/dl_class/classification_data.zip).
Unzip the data directly into the homework folder, replacing the existing data directory completely.

Make sure you see the following directories and files inside your main directory
```
homework/
grader/
bundle.py
classification_data/train
classification_data/val
```
You will run all scripts from inside this main directory.

In the `homework` directory, you'll find the following starter code files
- `train.py` - code to train, evaluate and save your models
- `models.py` - where you will implement various models
- `logger.py` - utility functions to log your model's performance using Tensorboard
- `utils.py` - data loader for the SuperTuxKart dataset

### Data Loader

In `utils.py` we have provided a data loader for the SuperTuxKart dataset.
Labels and the corresponding image paths are saved in `labels.csv` and there are 6 classes of objects.
In our setting, the label `background` corresponds to 0, `kart` is 1, `pickup` is 2, `nitro` is 3, `bomb` is 4 and `projectile` 5.

Take a look at the `SuperTuxDataset` class and the `__init__`, `__len__`, and the `__getitem__` functions, as this demonstrates how to load and preprocess data for classification tasks.
- `__init__` reads the csv file and stores the image paths and labels.
- `__len__` returns the size of the dataset.
- `__getitem__` returns a tuple of image, label where image is a `torch.Tensor` of size `(3,64,64)` with range `[0,1]`, and the label is an `int`.

Note: you have access to two different directories of data:
- a training set (used to train the model)
- a validation set (used to approximate your performance on new unseen data).

When we grade your solution, we will use a third data split (a hidden test set).
We split the data into three to help you prevent overfitting (a phenomenon where the network performs very well on its training data, but cannot make sense of any new data; more on this later in class).

### Local Grader Instructions

You can grade your implementation after any part of the homework by running the following command from the main directory:
- `python3 -m grader homework -v` for medium verbosity
- `python3 -m grader homework -vv` to include print statements

## Logging (10 pts)

Logging is an important part of training models and provides a way to monitor/track your experiments.
We start by learning how to use `tensorboard`, a tool for monitoring the training of our model.

We created a dummy training procedure in `logger.py` and provided you an instance of a `tb.SummaryWriter`.
Implement the rest of `test_logging`.
Use the summary writer to log the training loss at every iteration, the training accuracy at each epoch and the validation accuracy at each epoch.
Remember to log everything in *global training steps*.

Here is a simple example of how to use the `SummaryWriter`.
```python
import torch.utils.tensorboard as tb

logger = tb.SummaryWriter('cnn')
logger.add_scalar('train/loss', t_loss, 0)
```
In `logger.py`, you should **not** create your own `SummaryWriter`, but rather use the one provided.
You can test your logger by calling
```bash
python3 -m homework.logger --exp_dir logs
```
To view the logs in tensorboard,
- Spawn a new terminal and start a tensorboard server: `tensorboard --logdir logs`.
- Open up a web browser and navigate to the provided URL (usually `localhost:6006`).

## Classification Loss (10 pts)

Next, we'll implement the `ClassificationLoss` in `models.py`.
We will later use this loss to train our classifiers.
You should implement the log-likelihood of a softmax classifier.

$$-\log\left(\frac{\exp(x_l) }{ \sum_j \exp(x_j)} \right),$$
where $x$ are the logits and $l$ is the label.
You may use existing PyTorch functions to implement this.

### Relevant Operations
 - [torch.nn.functional](https://pytorch.org/docs/stable/nn.html#torch-nn-functional)

## Linear Model (5 pts)

Let's begin building our first neural network. We will build a neural network to classify different classes in SuperTuxKart dataset.

Implement the `LinearClassifier` class in `models.py`.
Define the linear model and all layers in the `__init__` function, then implement `forward`.
Your `forward` function receives a `(B,3,64,64)` tensor as an input and should return a `(B,6)` `torch.Tensor` (one value per class), where `B` stands for batch size.
You can earn these full credits without training the model, just from the correct model definition.

You can grade your linear model using

```bash
python3 -m grader homework -v
```

### Hints/Tips

- Run the grader before training your model to make sure your definition of the model is correct.
- Use `torch.nn.Linear` to define a linear layer.
- If you are using the VSCode debugger, you might need to temporarily set `num_workers=0` in the DataLoader so that the debugger can attach to the correct process.

### Relevant Operations
 - [torch.nn.Linear](https://pytorch.org/docs/stable/nn.html#linear)
 - [torch.tensor.View](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view)
 - and all previous

## Training the Linear Model (15 pts)

Train your linear model in `train.py`.

Complete the code for a full training procedure.
This includes:
 * Creating a model, loss, optimizer
 * Loading the data: `train` and `val`
 * Running the optimizer for several epochs (the default `max_epochs` might not be enough)
 * Saving your final model, using `save_model`

Train your network using
```bash
python3 -m homework.train --model_name linear
```

You can then test your trained model using
```bash
python3 -m grader homework -v
```

The accuracy cutoff for this section is 0.70 on the validation/test set.

### Hints/Tips
- You might find it useful to store optimization parameters in the `ArgumentParser`, and quickly try a few from the command-line.
- Try to write your training code to be model agnostic. We will swap out the model below.

We will use the model checkpoint `linear.th` to grade your trained model's performance.
You can grade your trained model using
```bash
python3 -m grader homework -v
```

### Relevant Operations
 - [torch.optim.Optimizer](https://pytorch.org/docs/stable/optim.html#torch.optim.Optimizer)
 - [torch.optim.SGD](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)
 - [torch.optim.Adam](https://pytorch.org/docs/stable/optim.html#torch.optim.Adam)
 - [torch.Tensor.backward](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.backward)
 - and all previous

## MLP Model (20 pts)

Implement the `MLPClassifier` class in `models.py`.
The inputs and outputs to the multi-layer perceptron are the same as the linear classifier.
However, now you're learning a non-linear function.

Train your network using
```bash
python3 -m homework.train --model_name mlp
```

The accuracy cutoff for this section is 0.80 on the validation/test set.

### Relevant Operations

 - [torch.nn.ReLU](https://pytorch.org/docs/stable/nn.html#relu)
 - [torch.nn.Sequential](https://pytorch.org/docs/stable/nn.html#sequential)
 - and all previous

## Deep Network (15 pts)

Implement the `MLPClassifierDeep` class in `models.py`.
For this part, build a model that has at least 4 layers.

You can train your network using
```bash
python3 -m homework.train --model_name mlp_deep
```

The accuracy cutoff for this section is 0.80 on the validation/test set.

### Hints/Tips
- You can use `torch.nn.Sequential` to easily build a multi-layer model.
- This part mainly requires tuning the number of layers in your model.
- Try to pass a `num_layers` argument to your model to tune efficiently.
- You might need to tune your learning rate `lr` and `batch_size`.
- You might need to tune the hidden dimension of your model ('width' of each layer)

## Deep Network with Residual Connections (20 pts)

Implement the `MLPClassifierDeepResidual` class in `models.py`.

This time, let's try to build a model that has at least 4 layers, and with residual connections.
Residual connections are a way to help with the vanishing gradient problem in deep networks.

You can train your network using
```bash
python3 -m homework.train --model_name mlp_deep_residual
```

The accuracy cutoff for this section is 0.80 on the validation/test set.

### Hints/Tips
- Run the grader before training your model to make sure your residual connections are implemented properly.
- You can use `torch.nn.ModuleList` to store your layers. This will allow you to easily iterate over the layers in your forward function.

### Relevant Operations
 - [ResNet](https://arxiv.org/abs/1512.03385)
 - [torch.nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList)
 - and all previous

## Submission

Once you finished the assignment, create a submission bundle using
```bash
python3 bundle.py homework [YOUR UT ID]
```
and submit the zip file on canvas. Please note that the maximum file size our grader accepts is **40MB**. Please keep your model compact.

Please double-check that your zip file was properly created, by grading it again
```bash
python3 -m grader [YOUR UT ID].zip
```
