# Ovarian_Cancer
Kaggle competition to detect Ovarian Cancer

The data was downloaded from https://www.kaggle.com/competitions/UBC-OCEAN/data (737GB) and added to ./data within this project. A resized dataset can be found here for experimentation (2.54GB) https://www.kaggle.com/datasets/ryanwaka/ovarian-cancer-resized or downloaded using the following command:
kaggle datasets download ryanwaka/ovarian-cancer-resized

The resized dataset folder was renamed and follows this path data/preprocessed_images/
train.csv will need to be downloaded as well from the kaggle link and added to the data/ directory

**Data information:**
* image_id - A unique ID code for each image.
* label - The target class. One of these subtypes of ovarian cancer: CC, EC, HGSC, LGSC, MC, Other. The Other class is not present in the training set; identifying outliers is one of the challenges of this competition. Only available for the train set.
* image_width - The image width in pixels.
* image_height - The image height in pixels.
* is_tma - True if the slide is a tissue microarray. Only available for the train set.

Notes to look into:
* Where should I add moving things to gpu? - Done
* Clean up output using tqdm_notebook - should i output over each epoch? - Started
* Bring in tensorboard and logging https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-model - started: train/loss added
* Where will I use optuna? - Need to create modular files that take in parameters
* Is train (in trainer.py) the correct place for it?
* Visualize gradients, I need to know if I have a vanishing or exploding gradient flow.

Ideas to pursue:
* Should I resize images? they are over 20k pixes on length and width.
* How do I handle different image sizes when training? 
* How do I handle different image sizes when predicting?

Approaches:
1. Naive - Preprocess and resize input images for a CNN

Future improvements:
1. Allow argument parser for model selection

Training can be started by using:
python train.py --model CNN

A new model can be added and trained by adding a <model_name>.py file and a <model_name> class within the file.

View Tensorboard:
1. tensorboard --logdir runs
2. Open http://localhost:6006/ in browser
