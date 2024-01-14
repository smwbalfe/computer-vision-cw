import os
import cv2
import numpy as np
import re
import time


def create_pyramid(image_p, level_n, orientation_n):
    # Create a Gaussian pyramid
    pyramid = []
    temp = image_p.copy()
    for i in range(level_n):
        temp = cv2.pyrDown(temp)
        pyramid.append(temp)

    features_pyramid = []

    for level in pyramid:
        rows, cols = level.shape[:2]
        level_features = []
        for i in range(orientation_n):
            angle = i * (360 / orientation_n)
            m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            # Calculate the new size of the output image
            cos = np.abs(m[0, 0])
            sin = np.abs(m[0, 1])
            new_cols = int((rows * sin) + (cols * cos))
            new_rows = int((rows * cos) + (cols * sin))
            # Adjust the translation part of the matrix to ensure the entire image is visible
            m[0, 2] += (new_cols / 2) - (cols / 2)
            m[1, 2] += (new_rows / 2) - (rows / 2)
            # Apply the rotation
            rotated = cv2.warpAffine(level, m, (new_cols, new_rows))
            # Add the rotated image to the level features list

            # Create a mask of the black pixels
            mask = cv2.inRange(rotated, (0, 0, 0), (0, 0, 0))
            rotated[mask > 0] = [255, 255, 255]

            kernel = np.ones((3, 3), np.uint8)
            rotated = cv2.dilate(rotated, kernel, iterations=1)
            level_features.append(rotated)

        # Add the level features list to the features pyramid list
        features_pyramid.append(level_features)
    return features_pyramid


def show_pyramid(pyramid):
    for level in pyramid:
        for img in level:
            cv2.imshow('Result', img)
            cv2.waitKey(0)


def template_matcher(main_image, images, levels, orientations):
    matches = []
    for image_ in images:

        pyramid = create_pyramid(image_[0], levels, orientations)

        for level in pyramid:
            for orientation in level:
                result = cv2.matchTemplate(main_image, orientation, cv2.TM_CCOEFF_NORMED)
                matches.append((result, image_[1]))
    return matches


def non_maximum_suppression(current_matches, threshold):

    filtered_matches_l = []

    for i, match in enumerate(current_matches):

        # Apply threshold
        loc = np.where(match[0] >= threshold)
        for pt in zip(*loc[::-1]):
            # Check if this point is already covered by another match
            covered = False
            for other_match in filtered_matches_l:
                if other_match[0][0] <= pt[0] <= other_match[1][0] and other_match[0][1] <= pt[1] <= other_match[1][1]:
                    covered = True
                    break
            if not covered:
                # Add bounding box to filtered matches
                filtered_matches_l.append(((pt[0], pt[1]), (pt[0] + 100, pt[1] + 100), match[1]))
    return filtered_matches_l


def read_training_images(directory, label_regex):
    icon_count = 0
    images_l = []
    for filename in os.listdir(directory):

        icon_count += 1
        match = re.match(label_regex, filename)
        if re.match(regex, filename):
            label = match.group(1)
            image_ = cv2.imread(os.path.join(directory, filename))
            images_l.append((image_, label))
        else:
            print("incorrect filename , probably read an incorrect image that was not in the training set")
    return images_l, icon_count


def read_test_images(directory):
    test_images_l = []
    for filename in sorted(os.listdir(directory), key=lambda x: int(x.split("_")[-1].split(".")[0])):
        img = cv2.imread(os.path.join(directory, filename))
        test_images_l.append((img, filename))
    return test_images_l


def display(image, filtered_matches):
    for matched_image, bbox, class_ in filtered_matches:
        cv2.rectangle(image, matched_image, bbox, (255, 0, 0), 2)
        cv2.putText(image, class_, (matched_image[0], matched_image[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0),
                    thickness=1)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def point_within_square(x, y, x1, y1, x2, y2):
    return int(x1) <= int(x) <= int(x2) and int(y1) <= int(y) <= int(y2)


def eval_positive(annotations_p, point, class_p):

    # slight offset to ensure the dot we search is inside the search area
    x = point[0] + 5
    y = point[1] + 5

    # go through the labels in the images and check if our class is inside it and is also the correct class.
    for label in annotations_p:
        if point_within_square(x, y, label[1][0], label[1][1], label[2][0], label[2][1]):
            if label[0] == class_p:
                # true positive
                return True
    # false positive
    return False


def extract_labels(filtered_list):
    labels = []
    for _, _, label in filtered_list:
        labels.append(label)
    return labels


def extract_ann_labels(ann_list):
    ann_labels = []
    for label, _, _ in ann_list:
        ann_labels.append(label)
    return ann_labels


def template_match(test_image, training_images_p, annotations_p, threshold, icon_count, levels, orientations):

    # return: [(per pixel similarity score 2D, image), (...)]
    matches = template_matcher(test_image, training_images_p, levels, orientations)

    # return: [(point identified, bottom corner identified, label),(...)]
    filtered = non_maximum_suppression(matches, threshold)

    labels = extract_labels(filtered)

    tp = fp = tn = fn = 0

    for label in annotations_p:
        if label[0] not in labels:
            fn += 1

    for match_point, _, class_ in filtered:
        if eval_positive(annotations_p, match_point, class_):
            tp += 1
        else:
            fp += 1

    tn = icon_count - (fp + tp + fn)

    tpr_ = tp / (tp + fn)
    fpr_ = fp / (fp + tn)
    acc_ = (tp + tn) / (tp + fp + tn + fn)

    display(test_image, filtered)

    return acc_, tpr_, fpr_


def extract_annotations(directory):

    annotations_l = []

    for filename in sorted(os.listdir(directory), key=lambda x: int(x.split("_")[-1].split(".")[0])):
        file_path = os.path.join(directory, filename)
        temp = []
        with open(file_path, 'r') as file:
            for line in file:
                line_s = line.split(',')
                for i in range(len(line_s)):
                    line_s[i] = line_s[i].replace('(', '').replace(')', '').replace('\n', '').replace(' ', '')
                temp.append((line_s[0], (line_s[1], line_s[2]), (line_s[3], line_s[4])))
        annotations_l.append(temp)

    return annotations_l


if __name__ == '__main__':

    regex = r'\d{3}-(\w+(?:-\w+)*)\.png'

    training_images_dir = './dataset_2/training_images'

    image_type = 'rotation'
    test_image_dir = ""
    annotation_dir = ""

    if image_type == ' d rotation':
        test_image_dir = './dataset_2/test_rotations/images/'
        annotation_dir = './dataset_2/test_rotations/annotations'
    else:
        test_image_dir = './dataset_2/test_no_rotations/images'
        annotation_dir = './dataset_2/test_no_rotations/annotations'

    training_images, count = read_training_images(training_images_dir, regex)
    test_images = read_test_images(test_image_dir)
    annotations = extract_annotations(annotation_dir)

    path = './dataset_2/test_rotations/images/'

    times = []
    eval_data = []

    evaluations = [(4, 8, 0.85)]

    for item in evaluations:
        print(f"levels: {item[0]} | orientations: {item[1]} | threshold: {item[2]}")
        for index, image_t in enumerate(test_images):
            start_time = time.time()
            acc, tpr, fpr = template_match(image_t[0], training_images, annotations[index], item[2], count, item[0], item[1])
            print("eval: ", acc, tpr, fpr)
            end_time = time.time()
            runtime = end_time - start_time
            eval_data.append((acc, tpr, fpr, runtime))

        total_acc = total_tpr = total_fpr = total_runtime = 0

        for acc, tpr, fpr, runtime in eval_data:
            total_acc += acc
            total_tpr += tpr
            total_fpr += fpr
            total_runtime += runtime

        print("average accuracy: ", total_acc / count)
        print("average tpr: ", total_tpr / count)
        print("average fpr: ", total_fpr / count)
        print("average runtime: ", total_runtime / count)


