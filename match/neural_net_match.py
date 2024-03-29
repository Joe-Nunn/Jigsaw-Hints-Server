"""
Functionality for attempting to match a jigsaw piece onto a base image using the neural network.
"""

import cv2
import torch
import torchvision.transforms as transforms
import numpy as np

from match import Match
from match.neural_network import NeuralNetwork

SECTION_SIZE = 255


class NeuralNetMatch(Match):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.model = NeuralNetwork()
        self.model.load_state_dict(torch.load("match/model.pt", map_location=self.device))
        self.model.eval()

    @staticmethod
    def _convert_to_tensor(image):
        """
        Converts image to a tensor ensuring it's the correct format (no transparency channel)
        """
        image = image[:, :, :3]  # remove alpha channel
        transform = transforms.ToTensor()
        image = transform(image)
        image = image.float()
        image = image[None, :]  # Extra dimension for batch of 1
        return image

    @staticmethod
    def _calculate_required_padding(base):
        """
        Calculates how much padding is required to make an images dimensions divisible by the section size
        """
        height, width, _ = base.shape

        # Required extra padding
        extra_padding_x = SECTION_SIZE - (width % SECTION_SIZE)
        extra_padding_y = SECTION_SIZE - (height % SECTION_SIZE)

        return extra_padding_x, extra_padding_y

    def _add_padding(self, base):
        """
        Add padding to the base in order to make it divisible by the section size
        """
        extra_padding_x, extra_padding_y = self._calculate_required_padding(base)

        # Create new image with extra padding
        padded_base = cv2.copyMakeBorder(base, 0, extra_padding_y, 0, extra_padding_x, cv2.BORDER_CONSTANT, value=0)

        return padded_base

    @staticmethod
    def _remove_padding(image, padding_x, padding_y):
        """
        Takes an image with padding and a picture of the same size before padding (original) and crops the padded one
        """
        height, width, _ = image.shape

        new_dim_x = width - padding_x
        new_dim_y = height - padding_y

        # crop the image
        return image[0:new_dim_y, 0:new_dim_x]

    def score_sections(self, base, piece):
        """
        Get a score for each section of the base for how well it matches the piece
        """
        piece = self._convert_to_tensor(piece)

        base = self._add_padding(base)

        # get the base height and width
        height, width, _ = base.shape

        scores = np.empty([int(height / SECTION_SIZE), int(width / SECTION_SIZE)], np.single)

        # loop through the base in steps of section size
        for base_y in range(0, int(height / SECTION_SIZE)):
            score_y = base_y * SECTION_SIZE
            for base_x in range(0, int(width / SECTION_SIZE)):
                score_x = base_x * SECTION_SIZE
                section = base[score_y:score_y + SECTION_SIZE, score_x:score_x + SECTION_SIZE]
                section = self._convert_to_tensor(section)
                scores[base_y, base_x] = self.model.forward(piece, section)
        return scores

    def create_heat_map(self, base, scores):
        """
        Applies a heat map based on the match probability scores to the base
        """
        # Get max and min scores for normalisation
        max_score_y, max_score_x = np.unravel_index(scores.argmax(), scores.shape)
        max_score = scores[max_score_y, max_score_x]

        min_score_y, min_score_x = np.unravel_index(scores.argmin(), scores.shape)
        min_score = scores[min_score_y, min_score_x]

        # Normalise scores between 0 and 255
        normalise = lambda score: (score - min_score / (max_score - min_score)) * 255
        scores = normalise(scores)

        # Convert scores into a grayscale image
        scores = scores.astype('uint8')
        score_image = cv2.cvtColor(scores, cv2.COLOR_GRAY2BGR)

        # Resize scores image to be same size as base
        height, width, _ = base.shape
        extra_padding_x, extra_padding_y = self._calculate_required_padding(base)
        # Resize to same size as base with padding as scores calculated on base with padding. Maintain the positioning
        score_image = cv2.resize(score_image, (width + extra_padding_x, height + extra_padding_y))
        # Crop parts that were padded as padding to get original lined up
        score_image = self._remove_padding(score_image, extra_padding_x, extra_padding_y)

        # Turn scores into a heatmap
        heatmap = cv2.applyColorMap(score_image, cv2.COLORMAP_JET)

        # Superimpose score heat map onto the base
        return cv2.addWeighted(heatmap, 0.5, base, 0.5, 0)

    def _scale_base(self, base, scale):
        """
        Scales base (increases / deceases its size) by a factor of scale. The larger the scale the more accurate unless
        too large to the point too little of the base image is being examined
        """
        width = int(base.shape[1] * scale)
        height = int(base.shape[0] * scale)
        dim = (width, height)

        # resize image
        return cv2.resize(base, dim)

    def find_match(self, base, piece, save_image, request_time, hint_accuracy, no_pieces):
        """
        Returns a heat map with higher heat where the piece matches the sections of the base image
        """
        base = self._scale_base(base, 4)
        section_scores = self.score_sections(base, piece)
        base = self.create_heat_map(base, section_scores)
        return self.save_solved_image(request_time, base)
