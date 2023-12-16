import torch
from PIL import Image
from torchvision.transforms import ToPILImage, PILToTensor
from src.models.ConvAutoEncoder import ConvAutoEncoder


def main():
    device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = ConvAutoEncoder().to(device)
    checkpoint = torch.load('src/checkpoints/autoencoder-checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    image = Image.open('data/preprocessed_images/4.png')
    to_tensor = PILToTensor()
    # inputs, labels = inputs.type(torch.float32).to(self.device), labels.to(self.device)

    image = to_tensor(image).type(torch.float32).to(device)

    batch = image.unsqueeze(0) # add batch dimension

    with torch.no_grad():
        encoding, decoding = model(batch)

        to_pil = ToPILImage()
        encoded_image = to_pil(encoding[0])
        decoded_image = to_pil(decoding[0])

        encoded_image.save('data/testing_encoder/4_encode_3.png')
        decoded_image.save('data/testing_encoder/4_decode_3.png')


if __name__ == '__main__':
    main()