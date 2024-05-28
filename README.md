# Image Classification with ResNet18

In this task, we use **ResNet18** to classify images from the CUB-200-2011 dataset. The dataset consists of 11,788 images across 200 bird species. The images are split into 5994 training images and 5794 testing images.

## How to train the model
In the `/script/` directory, you can find a file name `run_model.sh`, you can run the script by executing the following command (Assuming you are in the `/script/` directory and using bash):
```bash
bash run_model.sh
```
Before running, you might need to set the variable:
- `workpath`: The directory where you're working
- `save_name`: The name of the model you want to save
- `pretrained`: Bool value to determine whether to use pretrained model or start from scratch
- `epochs`: Number of epochs to train the model
- `optimizer_name`: The optimizer you want to use: SGD or Adam, others are not supported
- `lr`: The learning rate of the optimizer

After running you'll find the model saved in the `workpath` directory with the name `save_name`.

## How to visualize the model
We use `tensorboard` to visualize the model. You can run the following command to start the tensorboard server:
```bash
tensorboard --logdir=/path/to/your/logdir/
```
Replace `/path/to/your/logdir/` with the path to the log directory `lightning_logs` where the model is saved.

## How to test the model
To test the model, you'll find a file named `test_model.ipynb` in the `/script/` directory. You can run the notebook to test the model. You might need to set the variable `WORKPATH` and `model_path` to the path where the model is saved.

## Dependency installation
`requirements.txt` lists the dependencies required to run the model. You can install the dependencies by running the following command:
```bash
pip install -r requirements.txt
```
To avoid possible collision, it is recommended to install pytorch following the instructions on the [official website](https://pytorch.org/get-started/locally/). So the requirements.txt does not include pytorch.