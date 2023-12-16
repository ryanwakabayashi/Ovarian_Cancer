from PIL import Image
import os
import concurrent.futures


class ImageProcessor:
    def __init__(self, source_dir, target_dir, size=(256, 256), max_workers=4):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.size = size
        self.max_workers = max_workers
        Image.MAX_IMAGE_PIXELS = 9964795283

        # Create the target directory if it does not exist
        if not os.path.exists(self.target_dir):
            os.makedirs(self.target_dir)

    def preprocess_and_save(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a future for each file in the source directory
            futures = [executor.submit(self.resize_and_save, filename)
                       for filename in os.listdir(self.source_dir)
                       if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                future.result()  # This will re-raise any exception raised in the worker thread

    def resize_and_save(self, filename):
        img_path = os.path.join(self.source_dir, filename)
        with Image.open(img_path) as img:
            # Resize the image
            img = img.resize(self.size, Image.ANTIALIAS)

            # Save the image to the target directory
            target_path = os.path.join(self.target_dir, filename)
            img.save(target_path)