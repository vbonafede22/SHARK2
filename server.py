'''

You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

'''

from flask import Flask, request
from flask import render_template
import time
import json

# my imports
import numpy as np
import math
from scipy.interpolate import interp1d

app = Flask(__name__)

# Centroids of 26 keys
centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

# Pre-process the dictionary and get templates of 10000 words
words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])


def generate_sample_points(points_X, points_Y):     # DONE
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template computationally.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    sample_points_X, sample_points_Y = [], []
    # ediff1d gives difference between consecutive elements of the array
    # we find the distance between coordinates and find the cumulative sum
    distance = np.cumsum(np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2))
    # basically when words like mm or ii have no path / little path, use centroid
    if distance[-1] == 0:
        for i in range(100):
            sample_points_X.append(points_X[0])
            sample_points_Y.append(points_Y[0])
    else:
        # get the proportion of line segments
        distance = distance / distance[-1]
        # scale the points to get linear interpolations along the path
        fx, fy = interp1d(distance, points_X), interp1d(distance, points_Y)
        # generate 100 equidistant points on normalized line
        alpha = np.linspace(0, 1, 100)
        # use the interpolation function to translate from normalized to real plane
        x_regular, y_regular = fx(alpha), fy(alpha)
        sample_points_X = x_regular.tolist()
        sample_points_Y = y_regular.tolist()
        # print(sample_points_X)
        # print(sample_points_Y)
    return sample_points_X, sample_points_Y


# Pre-sample every template
template_sample_points_X, template_sample_points_Y = [], []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)


def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):     # DONE
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider it reasonable)
    to narrow down the number of valid words so that the ambiguity can be avoided to some extent.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # Set your own pruning threshold
    threshold = 20
    # Do pruning (12 points)
    L = 1

    start_point_X, start_point_Y = gesture_points_X[0], gesture_points_Y[0]     # get the first point of gesture
    end_point_X, end_point_Y = gesture_points_X[99], gesture_points_Y[99]       # get the last point of gesture
    # TODO normaliziation makes program run extremly slow (around 5 mins per input)
    # gest_norm_X = max(gesture_points_X)         # get max W
    # gest_norm_Y = max(gesture_points_Y)         # get max H
    # gest_s = L / max(gest_norm_X, gest_norm_Y)  # get max W,H

    for i in range(10000):
        template_X_value, template_Y_value = template_sample_points_X[i], template_sample_points_Y[i]   # get each template value
        template_start_X, template_start_Y = template_X_value[0], template_Y_value[0]        # get first point in template value
        template_end_X, template_end_Y = template_X_value[99], template_Y_value[99]          # get last point in template value
        # temp_norm_X = max(template_X_value)         # get max W
        # temp_norm_Y = max(template_Y_value)         # get max H
        # temp_s = L / max(temp_norm_Y, temp_norm_X)  # get max W,H
        # find the euclidean distance of the template and gesture start points
        # first_distance = math.sqrt((start_point_X * gest_s - template_start_X * temp_s) ** 2 + (start_point_Y * gest_s - template_start_Y * temp_s) ** 2)
        first_distance = math.sqrt((start_point_X - template_start_X) ** 2 + (start_point_Y - template_start_Y) ** 2)
        # find the euclidean distance of the template and gesture end points
        # last_distance = math.sqrt((end_point_X * gest_s - template_end_X * temp_s) ** 2 + (end_point_Y * gest_s - template_end_Y * temp_s) ** 2)
        last_distance = math.sqrt((end_point_X - template_end_X) ** 2 + (end_point_Y - template_end_Y) ** 2)

        # if start and end distance is within the threshold add to valid words and points
        if first_distance <= threshold and last_distance <= threshold:
            valid_template_sample_points_X.append(template_X_value)
            valid_template_sample_points_Y.append(template_Y_value)
            valid_words.append(words[i])
            # print(words[i])

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y


def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # Set your own L
    L = 1

    # Calculate shape scores (12 points)

    gest_norm_Y = max(gesture_sample_points_Y)          # get max H
    gest_norm_X = max(gesture_sample_points_X)          # get max W
    gest_s = L / max(gest_norm_X, gest_norm_Y)          # get max W,H
    for i in range(len(valid_template_sample_points_X)):
        temp_Y = valid_template_sample_points_Y[i]
        temp_X = valid_template_sample_points_X[i]
        temp_norm_Y = max(temp_Y)                       # get max H
        temp_norm_X = max(temp_X)                       # get max W
        temp_s = L / max(temp_norm_X, temp_norm_Y)      # get max W,H
        distance = 0
        for j in range(100):
            gest_X = gesture_sample_points_X[j]
            gest_Y = gesture_sample_points_Y[j]
            my_X = temp_X[j]
            my_Y = temp_Y[j]
            distance += math.sqrt(((gest_X * gest_s) - (my_X * temp_s)) ** 2 + ((gest_Y * gest_s) - (my_Y * temp_s)) ** 2)      # normalize

        # print("shape score: " + str(distance))
        shape_scores.append(distance)

    return shape_scores


def min_dist(X1, Y1, X2, Y2):
    my_min = 100000
    k = 0
    for j in range(100):
        dist = math.sqrt((X1 - X2[j]) ** 2 + (Y1 - Y2[j]) ** 2)
        my_min = min(my_min, dist)
        if my_min >= dist:
            k = j

    if k is not 0:
        min_distance = math.sqrt((X1 - X2[k]) ** 2 + (Y1 - Y2[k]) ** 2)
    else:
        min_distance = 0

    return min_distance


def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    # Calculate location scores (12 points)

    for i in range(len(valid_template_sample_points_X)):
        temp_X = valid_template_sample_points_X[i]
        temp_Y = valid_template_sample_points_Y[i]
        gest_sum = 0
        temp_sum = 0
        for j in range(100):
            gest_sum += max(min_dist(gesture_sample_points_X[j], gesture_sample_points_Y[j], temp_X, temp_Y) - radius, 0)
        for j in range(100):
            temp_sum += max(min_dist(temp_X[j], temp_Y[j], gesture_sample_points_X, gesture_sample_points_Y) - radius, 0)
        # print("gest sum " + str(gest_sum))
        # print("temp sum " + str(temp_sum))
        if gest_sum is 0 and temp_sum is 0:
            location_scores.append(0)
        else:
            distance = 0
            dist_sum = 0
            for j in range(100):
                my_X = temp_X[j]
                my_Y = temp_Y[j]
                gest_X = gesture_sample_points_X[j]
                gest_Y = gesture_sample_points_Y[j]
                dist_sum += math.sqrt((my_X - gest_X) ** 2 + (my_Y - gest_Y) ** 2)
                dist = math.sqrt((my_X - gest_X) ** 2 + (my_Y - gest_Y) ** 2)
                distance += dist * dist_sum
            location_scores.append(distance)

    return location_scores


def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # Set your own shape weight
    shape_coef = 0.5
    # Set your own location weight
    location_coef = 0.5
    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores


def get_best_word(valid_words, integration_scores):     # DONE
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # Set your own range.
    # n = 3
    # Get the best word (12 points)

    if len(valid_words) != 0:
        # convert two lists to a dict
        my_dict = {valid_words[i]: integration_scores[i] for i in range(len(valid_words))}

        # multiply prob by integration score of each corresponding word
        for key in my_dict:
            if key in probabilities:
                my_dict[key] = my_dict[key] * (1 - probabilities[key])

        # sort dictionary by value
        sorted_dict = {k: v for k, v in sorted(my_dict.items(), key=lambda item: item[1])}
        print(sorted_dict)

        # get first key in sorted dict
        best_word = list(sorted_dict.keys())[0]
    else:
        best_word = "***NO BEST WORD***"

    # print("best word from best word funct: " + best_word)

    return best_word


@app.route("/")
def init():
    return render_template('index.html')


@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())
    
    # print(data)

    gesture_points_X = []
    gesture_points_Y = []
    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    # gesture_points_X = [gesture_points_X]         # doesnt make sense to me to store in list again
    # gesture_points_Y = [gesture_points_Y]

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_sample_points_X, gesture_sample_points_Y, template_sample_points_X, template_sample_points_Y)

    shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

    integration_scores = get_integration_scores(shape_scores, location_scores)

    best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    return '{"best_word":"' + best_word + '", "elapsed_time":"' + str(round((end_time - start_time) * 1000, 5)) + 'ms"}'


if __name__ == "__main__":
    app.run()
