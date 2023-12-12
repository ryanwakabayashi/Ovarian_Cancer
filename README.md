# Ovarian_Cancer
Kaggle competition to detect Ovarian Cancer

The data was downloaded from https://www.kaggle.com/competitions/UBC-OCEAN/data (737GB) and added to ./data within this project.

**Data information:**
* image_id - A unique ID code for each image.
* label - The target class. One of these subtypes of ovarian cancer: CC, EC, HGSC, LGSC, MC, Other. The Other class is not present in the training set; identifying outliers is one of the challenges of this competition. Only available for the train set.
* image_width - The image width in pixels.
* image_height - The image height in pixels.
* is_tma - True if the slide is a tissue microarray. Only available for the train set.

Notes to look into:
* Where should I add moving things to gpu? - Done
* Clean up output using tqdm_notebook - should i output over each epoch? - Started
* Bring in tensorboard and logging https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-model
* Where will I use optuna? - Need to create modular files that take in parameters
* Is train (in trainer.py) the correct place for it?

Ideas to pursue:
* If you’re loading images that large and you plan on just scaling them down, I would recommend preprocessing them all first (create another dataset where all the images are of the size used by the network) as it would greatly speed up training.
* Should I resize images? they are over 20k pixes on length and width.
* How do I handle different image sizes when training? 
* How do I handle different image sizes when predicting?

Approaches:
1. Naive - Resize input images for a CNN