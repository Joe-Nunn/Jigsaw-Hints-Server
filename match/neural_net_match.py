import cv2

from match import Match
from neural_network import NeuralNetwork
import torch
import torchvision.transforms as transforms

HINT_BOX_COLOUR = (255, 0, 0)  # RGB
HINT_BOX_THICKNESS = 8


class NeuralNetMatch(Match):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load("model.pt", map_location=self.device))
        self.model.eval()

    @staticmethod
    def _convert_to_tensor(image):
        image = image[:, :, :3]  # remove alpha channel
        transform = transforms.ToTensor()
        image = transform(image)
        image = image.float()
        image = image[None, :]  # Extra dimension for batch of 1
        return image

    def find_best_piece_coords(self, base, piece):
        piece = self._convert_to_tensor(piece)

        # get the base height and width
        height, width, _ = base.shape

        best_piece = (0, 0, 0)

        # loop through the base in steps of 25 pixels
        for y in range(0, height - 255, 25):
            for x in range(0, width - 255, 25):
                section = base[y:y + 255, x:x + 255]
                section = self._convert_to_tensor(section)
                match_score = self.model.forward(piece, section)
                if match_score > best_piece[2]:
                    best_piece = (x, y, match_score)

        x, y, _ = best_piece
        return x, y

    def find_match(self, base, piece, save_image, request_time, hint_accuracy, no_pieces):
        x, y = self.find_best_piece_coords(base, piece)
        cv2.rectangle(base, (x, y), (x+255, y+255), HINT_BOX_COLOUR, HINT_BOX_THICKNESS)
        return self.save_solved_image(request_time, base)
