import PIL

from src.data.preprocess import ImageProcessor


def process():
    source_directory = 'data/train_images'
    target_directory = 'data/preprocessed_images'
    processor = ImageProcessor(source_directory, target_directory, size=(256, 256), max_workers=8)
    print(f'Processor initialized with source directory {source_directory} and target directory {target_directory}')

    processor.preprocess_and_save()
    print(f'Preprocessing complete and saved to {target_directory}')


if __name__ == '__main__':
    process()
