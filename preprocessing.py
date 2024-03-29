import cv2
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import re

preprocess = {'March': 'Mar', 'Sept': 'Sep'}
month_to_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'April': 4, 'May': 5, 'Jun': 6,
                'July': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

# resized_shape = (400, 600)


def sorting_function(ra_file):
    splitted_ra_file = ra_file.split("_")
    date = splitted_ra_file[2]
    month, year = re.split('(\d+)', date)[:-1]
    splitted_ra_file[2] = f'{year}{month}'

    new_ra_file = '_'.join(splitted_ra_file)

    for old, new in preprocess.items():
        new_ra_file = new_ra_file.replace(old, new)

    for old, new in month_to_num.items():
        new_ra_file = new_ra_file.replace(old, '%02d_' % new)

    return new_ra_file


def sorting_function_two(file):
    file_name = '_'.join(file.rsplit('/', 1))
    file_name = os.path.basename(file_name)
    return sorting_function(file_name)


def get_right_format_two():
    """"""



def get_only_feet(ra_files):
    """

    :param ra_files:    all the filenames of the xray in a list
    :return:            Return only filenames that contain "feet"
    """
    return [ra_file for ra_file in ra_files if 'feet' in ra_file]


def get_only_hands(ra_files):
    """

    :param ra_files:    all the filenames of the xray in a list
    :return:            Return only filenames that contain "hand"
    """
    return [ra_file for ra_file in ra_files if 'hands' in ra_file]


def split_by_patients(ra_patient_files):
    """

    :param ra_patient_files:   Contains the file names of the xray

    :return:                    the xray separated as a list of patients
                                [
                                [patient1_xray_image1, patient1_xray_image2, patient1_xray_image3, ...],
                                [patient2_xray_image1, patient2_xray_image2, patient2_xray_image3, ...]
                                ]
    """

    previous_patient = '_'.join(ra_patient_files[0].split("_")[:2])

    patients = []

    current_patient_images = [ra_patient_files[0]]
    for ra_file in ra_patient_files[1:]:
        current_patient = '_'.join(ra_file.split("_")[:2])
        if previous_patient == current_patient:
            current_patient_images.append(ra_file)
        else:
            previous_patient = current_patient
            patients.append(current_patient_images)
            current_patient_images = [ra_file]

    return patients


def split_by_leftright(patients):
    """


    :param patients:        the xray as a list of patients
                        [
                            [patient1_xray_image1, patient1_xray_image2, patient1_xray_image3, ...],
                            [patient2_xray_image1, patient2_xray_image2, patient2_xray_image3, ...]
                        ]
    :return:                the xray as a list of patients separated by left and right
                        [
                            [
                                [patient1_xray_image1left, patient1_xray_image2left, patient1_xray_image3left],
                                [patient1_xray_image1right, patient1_xray_image2right, patient1_xray_image3right]
                            ],
                            [
                                [patient2_xray_image1left, patient2_xray_image2left, patient2_xray_image3left],
                                [patient2_xray_image1right, patient2_xray_image2right, patient2_xray_image3right]
                            ],
                        ]
    """

    patients_leftright = []

    for patient in patients:
        left_images = []
        right_images = []
        for ra_file in patient:
            if 'left' in ra_file:
                left_images.append(ra_file)
            elif 'right' in ra_file:
                right_images.append(ra_file)
        patients_leftright.append([left_images, right_images])

    return patients_leftright


def load_image(image_name, dirname="", resize_shape=None, grayscale=False):
    image_filename = os.path.join(dirname, image_name)

    if grayscale:
        flags = 0
    else:
        flags = None
    image = cv2.imread(image_filename, flags)
    if resize_shape is not None:
        image = cv2.resize(image, resize_shape)
    return image


def plot_images(patient_leftright_images):
    """

    :param patient_leftright_images: the xray as a list of patients separated by left and right

                        [
                            [
                                [patient1_xray_image1left, patient1_xray_image2left, patient1_xray_image3left],
                                [patient1_xray_image1right, patient1_xray_image2right, patient1_xray_image3right]
                            ],
                            [
                                [patient2_xray_image1left, patient2_xray_image2left, patient2_xray_image3left],
                                [patient2_xray_image1right, patient2_xray_image2right, patient2_xray_image3right]
                            ],
                        ]
    :return:
    """
    for patient in patient_leftright_images:
        # get all dates
        left_dates = [ra_file.split("_")[2] for ra_file in patient[0]]  # left
        right_dates = [ra_file.split("_")[2] for ra_file in patient[1]]  # right
        all_dates = np.unique(left_dates + right_dates).tolist()

        # look at all the left/right image (images is either all left images, or all right images)
        # patient name
        patient_name = '_'.join(patient[0][0].split("_")[:2])

        fig, ax = plt.subplots(nrows=2, ncols=len(all_dates))

        for idx, images in enumerate(patient):
            for image_idx, image in enumerate(images):
                date = image.split("_")[2]
                col_index = all_dates.index(date)
                ax[idx][col_index].set_title(date)
                dirname = os.path.join('all_images', 'focused_processed_xray_images')
                ax[idx][col_index].imshow(load_image(image, dirname, (400, 600)))
        fig.suptitle(patient_name)
        plt.show()


def plot_unprocessed_images(patients_images, resize_shape=(128, 128), grayscale=False):
    """

    :param patients_images:  [
                                [patient1_image1, patient1_image2, ...],
                                [patient2_image1, patient2_image2, ...].
                                ...
                            ]
    :param resize_shape:   The shape to resize the images to
    :param grayscale:       Load as grayscale (True or False)
    :return:
    """

    for patient_images in patients_images:
        dates = [img.split("/")[3].split("_")[-1] for img in patient_images]
        fig, ax = plt.subplots(nrows=1, ncols=len(dates))

        for image_idx, image in enumerate(patient_images):
            date = image.split("_")[-1].split("/")[0]
            col_index = dates.index(date)
            ax[col_index].set_title(date)
            ax[col_index].imshow(load_image(image, resize_shape=(400, 600)))

        patient_name = patient_images[0].split("/")[2]
        fig.suptitle(patient_name)
        plt.show()


def load_images(patient_leftright_images, resize_shape=(128, 128), grayscale=False):
    """

    load as numpy array from filename

    :param patient_leftright_images: the xray as a list of patients separated by left and right

                        [
                            [
                                [patient1_xray_image1left, patient1_xray_image2left, patient1_xray_image3left],
                                [patient1_xray_image1right, patient1_xray_image2right, patient1_xray_image3right]
                            ],
                            [
                                [patient2_xray_image1left, patient2_xray_image2left, patient2_xray_image3left],
                                [patient2_xray_image1right, patient2_xray_image2right, patient2_xray_image3right]
                            ],
                        ]
    :return:            list of patients containing shape [direction, number of images, height, width, channel]
                        direction is 2 because left and right
    """
    all_patients = []
    for patient in patient_leftright_images:
        all_images = []
        for images in patient:
            current_direction = []
            for image in images:
                dirname = os.path.join('all_images', 'focused_processed_xray_images')
                current_direction.append(load_image(image, resize_shape, grayscale))
            all_images.append(np.array(current_direction))
        all_patients.append(np.array(all_images))

    return all_patients


def load_patient_feet(resize_shape=(128,128), grayscale=False):
    ra_files = os.listdir('all_images/focused_processed_xray_images')
    ra_files = sorted(ra_files, key=sorting_function)
    feet_files = get_only_feet(ra_files)
    patient_feet = split_by_patients(feet_files)
    patient_feet = split_by_leftright(patient_feet)
    return load_images(patient_feet, resize_shape, grayscale)


def load_patient_hands(resize_shape=(128,128), grayscale=False):
    ra_files = os.listdir('all_images/focused_processed_xray_images')
    ra_files = sorted(ra_files, key=sorting_function)

    # split patient hand
    hand_files = get_only_hands(ra_files)
    patient_hand = split_by_patients(hand_files)
    patient_hand = split_by_leftright(patient_hand)
    return load_images(patient_hand, resize_shape, grayscale)


def plot_patient_feet():
    ra_files = os.listdir('all_images/focused_processed_xray_images')
    ra_files = sorted(ra_files, key=sorting_function)
    feet_files = get_only_feet(ra_files)
    patient_feet = split_by_patients(feet_files)
    patient_feet = split_by_leftright(patient_feet)
    plot_images(patient_feet)


def plot_patient_hands():
    ra_files = os.listdir('all_images/focused_processed_xray_images')
    ra_files = sorted(ra_files, key=sorting_function)

    # split patient hand
    hand_files = get_only_hands(ra_files)
    patient_hand = split_by_patients(hand_files)
    patient_hand = split_by_leftright(patient_hand)
    plot_images(patient_hand)


def plot_unprocessed_hands():
    # new files
    files = glob.glob('RA_Xray/2/**/*.jpeg', recursive=True)
    files = [file for file in files if 'hand' in file]
    files = sorted(files, key=sorting_function_two)
    patient_hand = split_by_patients(files)
    plot_unprocessed_images(patient_hand)


plot_unprocessed_hands()


