import math
from collections import OrderedDict

from PIL import Image
from numpy.lib.stride_tricks import as_strided
from scipy.io import loadmat
from random import choice as rand_choice
import neat
import cv2
import numpy as np
import pickle

# Library issue, search for papers, maybe other Neat implementations???, reg CNN for problem


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    # Fitness function
    def work(self):
        # Debug purposes
        # random_choices = []

        current_fitness = 0

        # Prep recurrent neural network
        rnn = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        torso_info = [12, 0]
        joint_info = OrderedDict()
        joint_info = {
            "Right Upper Leg": [1, 2, 0],
            "Left Upper Leg": [3, 4, 0],
            "Right Lower Leg": [0, 1, 0],
            "Left Lower Leg": [4, 5, 0],
            "Right Upper Arm": [7, 8, 0],
            "Left Upper Arm": [9, 10, 0],
            "Right Forearm": [6, 7, 0],
            "Left Forearm": [10, 11, 0],
            "Head": [12, 13, 0]
        }

        # Cycle through all sample images and determine fitness of genome
        for ind in range(0, sample_size):
            # Pull all relevant data for current image
            current_image, current_x_scalar, current_y_scalar, dataset_index = images[ind]

            # Feed image into neural network to receive joint location guesses
            results = rnn.activate(current_image)

            # Prep all relevant coordinates for comparison against rnn output
            x = [round((i * current_x_scalar), 3) for i in x_coordinates[:, dataset_index]]
            y = [round((i * current_y_scalar), 3) for i in y_coordinates[:, dataset_index]]

            def calculate_half_lengths_for_ground_truth():
                mid_hip_x = (x[2] + x[3]) / 2
                mid_hip_y = (y[2] + y[3]) / 2
                half_lengths = [math.sqrt(((x[12] - mid_hip_x) ** 2) + ((y[12] - mid_hip_y) ** 2)) / 2]
                half_lengths.extend([do_calculation_for_ground_truth(v[0], v[1])/2 for v in joint_info.values()])
                return half_lengths

            def do_calculation_for_ground_truth(joint_ind_one, joint_ind_two):
                return math.sqrt(((x[joint_ind_one] - x[joint_ind_two]) ** 2) + ((y[joint_ind_one] - y[joint_ind_two]) ** 2))

            half_lengths = calculate_half_lengths_for_ground_truth()

            # Torso, Upper Leg x2, Lower Leg x2, Upper Arm x2, Forearm x2, Head
            # [?, ?], R[2, 3]+L[4, 5], R[1, 2]+L[5, 6], R[8, 9]+L[10, 11], R[7, 8]+L[11, 12], [13, 14]

            # Pull isOccluded data for image
            #o = joint_occluded[:, dataset_index]

            # Parse out rnn output to compare against ground truth
            rx = [i % 30.0 for i in results[0:14]]
            ry = [i % 30.0 for i in results[14:28]]
            #ro = [round(i, 3) for i in results[28:42]]

            def calculate_distances_from_ground_truth():
                mid_hip_x = (x[2] + x[3]) / 2
                mid_hip_y = (y[2] + y[3]) / 2
                mid_hip_rx = (rx[2] + rx[3]) / 2
                mid_hip_ry = (ry[2] + ry[3]) / 2
                distances_from_ground_truth = [do_calculation_for_rnn_estimates(i) for i in range(0, 14)]
                center_hip = math.sqrt(((mid_hip_x - mid_hip_rx) ** 2) + ((mid_hip_y - mid_hip_ry) ** 2))
                distances_from_ground_truth.append(center_hip)
                return distances_from_ground_truth

            def do_calculation_for_rnn_estimates(ind):
                return math.sqrt(((x[ind] - rx[ind]) ** 2) + ((y[ind] - ry[ind]) ** 2))

            def calculate_if_joint_detected():
                distances = calculate_distances_from_ground_truth()
                detected = [do_comparisons(distances[14], distances[12], half_lengths[0])]
                detected.extend(
                    [
                        do_comparisons(
                            distances[v[0]], distances[v[1]], half_lengths[i+1]
                        ) for i, v in enumerate(joint_info.values())
                    ]
                )
                if detected[0]:
                    torso_info[1] += 1
                for i, v in enumerate(joint_info.values()):
                    if detected[i+1]:
                        v[2] += 1

            def do_comparisons(joint_one_distance, joint_two_distance, half_length):
                joint_one_detected = joint_one_distance <= half_length
                joint_two_detected = joint_two_distance <= half_length
                return True if joint_one_detected and joint_two_detected else False

            calculate_if_joint_detected()
            testing = calculate_distances_from_ground_truth()
            for entry in testing:
                if entry < 1:
                    current_fitness += 5
                elif entry < 2:
                    current_fitness += 3
                elif entry < 3:
                    current_fitness += 2
                elif entry < 5:
                    current_fitness += 1

            """for i in range(0, 14):
                if round(ro[i]) == o[i]:
                    current_fitness += 0.01"""

        def determine_fitness():
            total = 0
            total += torso_info[1]/sample_size
            for key, value in joint_info.items():
                total += value[2]/sample_size
            return total/10

        real_fitness = determine_fitness()
        if determine_fitness() > 0:
            print("Torso: " + str(torso_info[1] / sample_size))
            for key, value in joint_info.items():
                print(key + ": " + str(value[2] / sample_size))
            current_fitness *= 10
        return current_fitness

            # Debug usage
            # random_choices.append(list(x) + list(y) + list(o))
            # random_choices.append(list(x) + list(y))

        # Set genome fitness
        #genome.fitness = round(current_fitness)

        # Debug usage
        # print(str(genome_id) + ': ' + str(round(current_fitness)) + ' fitness')
        # print(random_choices)


"""def determine_fitness(pair, new_fitness):
    for i in range(0, 14, 1):
        # Determine difference between supposition and ground truth
        difference = abs(pair[0][i] - pair[1][i])

        # Determine score based on deviation
        if difference < 0.01:
            new_fitness += 20
        elif difference < 0.1:
            new_fitness += 10
        elif difference < 1:
            new_fitness += 1
    return new_fitness"""


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()


def change_contrast(img, level):
    img = Image.fromarray(img)
    factor = (259 * (level + 255)) / (255 * (259 - level))

    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)


def normalize_image(img):
    img = img.astype('float32')
    img /= 255.0
    return img


def global_centering(img):
    img_mean = img.mean()
    img = img - img_mean
    return img


def pool2d(image, kernel_size, stride, padding, pool_mode='max'):
    '''
    2D Pooling

    Parameters:
        image: input 2D array
        kernel_size: int, the FOV of the pooling
        stride: int, the step size of the kernel when sliding through image
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    image = np.pad(image, padding, mode='constant')

    # Window view of image
    new_width = (image.shape[0] - kernel_size) // stride + 1
    new_height = (image.shape[1] - kernel_size) // stride + 1
    output_shape = (new_width, new_height)
    kernel_size = (kernel_size, kernel_size)

    new_shape = output_shape + kernel_size
    stride_x = stride * image.strides[0]
    stride_y = stride * image.strides[1]
    strides_xy = (stride_x, stride_y) + image.strides
    image_formatted = as_strided(image, shape=new_shape, strides=strides_xy)
    image_formatted = image_formatted.reshape(-1, *kernel_size)

    # Return the result of pooling
    if pool_mode == 'max':
        return image_formatted.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return image_formatted.mean(axis=(1, 2)).reshape(output_shape)


# Load dataset
mat_file = loadmat('data/joints.mat')

# Grab joint information from dataset
joints = mat_file['joints']

# Separate x and y coordinates, as well as the occlusion data
x_coordinates = joints[0]
y_coordinates = joints[1]
joint_occluded = joints[2]

# Initialize image list for appending the sample size amount
images = OrderedDict()
sample_size = 500
image_total = 2000

# Image dimension to reduce processing time. Not a great solution.
image_dimensions = 30

# Start appending the image data
for i in range(0, sample_size):
    # Random choice from images
    random_index = rand_choice(range(0, image_total))

    # Load image
    image_number = 10001 + random_index
    image_number_string = str(image_number)[1:]
    image = cv2.imread('./data/images/im' + image_number_string + '.jpg')

    # Resize image for faster processing
    height, width, channels = image.shape
    new_frame_size = (image_dimensions, image_dimensions)

    # Color may not be important for joint recognition
    greyscale = cv2.COLOR_BGR2GRAY
    image = cv2.cvtColor(image, greyscale)

    #image = change_contrast(image, 50)
    #image = np.asarray(image)

    # Normalization
    #image = normalize_image(image)

    # Global centering
    #image = global_centering(image)

    # Max pooling
    #image = np.reshape(image, (height, width))
    #image = pool2d(image, kernel_size=16, stride=14, padding=0, pool_mode='max')

    image = cv2.resize(image, new_frame_size)

    # Reshape and flatten to feed to genome networks
    image = np.reshape(image, new_frame_size)
    one_dimensional_image = np.ndarray.flatten(image)

    # Determine scalar to multiply joint info with to determine location on distorted image
    x_scalar = image_dimensions / width
    y_scalar = image_dimensions / height

    # Append preprocessed image data, relevant scalars and dataset index
    images[i] = [one_dimensional_image, x_scalar, y_scalar, random_index]

# Load configuration file for the entire NEAT process
config_file_name = 'config-feedforward2'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file_name)

# Create population based on configurations
p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint("neat-checkpoint-1999")

# Add reporters to track generational progression
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))  # Save the process after each 10 frames


if __name__ == '__main__':
    pe = neat.ParallelEvaluator(4, eval_genomes)

    # Determine winner
    winner = p.run(pe.evaluate, 2000)
    with open('winner.pk1', 'wb') as output:
        pickle.dump(winner, output, 1)


# Print out the accuracy of the best fitness based on pixel distance
# Reusing config for new population
# RNN vs Feed Forward