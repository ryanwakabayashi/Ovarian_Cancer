# Ovarian_Cancer
Kaggle competition to detect Ovarian Cancer

The data was downloaded from https://www.kaggle.com/competitions/UBC-OCEAN/data (737GB) and added to ./data within this project.

**Data information:**
* image_id - A unique ID code for each image.
* label - The target class. One of these subtypes of ovarian cancer: CC, EC, HGSC, LGSC, MC, Other. The Other class is not present in the training set; identifying outliers is one of the challenges of this competition. Only available for the train set.
* image_width - The image width in pixels.
* image_height - The image height in pixels.
* is_tma - True if the slide is a tissue microarray. Only available for the train set.