import cv2
import os
import random
import xml.etree.ElementTree as ET


def parse_xml_annotations(folder_dir):
    files = sorted(os.listdir(folder_dir))

    img_coords = {}

    for file in files:
        filename = os.path.join(folder_dir, file)
        et = ET.parse(filename)

        all_coords = []

        img_name = et.find('filename').text
        all_object_coords = et.findall('object')
        for coord in all_object_coords:
            xmin = int(coord.find('bndbox').find('xmin').text)
            xmax = int(coord.find('bndbox').find('xmax').text)
            ymin = int(coord.find('bndbox').find('ymin').text)
            ymax = int(coord.find('bndbox').find('ymax').text)

            all_coords.append([(xmin, ymin), (xmax, ymax)])

        img_coords[img_name] = all_coords

    return img_coords


def create_haar_cascade_info_dat(img_coords, positive_samples_dir, negative_samples_dir):

    # for positive samples
    with open('info.dat', 'w') as f:
        for img_name, coords in img_coords.items():
            text = f'{os.path.join(positive_samples_dir, img_name)} {len(coords)} '
            for coord in coords:
                start_x = coord[0][0]
                start_y = coord[0][1]

                width = coord[1][0] - start_x
                height = coord[1][1] - start_y

                text += f'{start_x} {start_y} {width} {height} '
            f.write(text)
            f.write('\n')

    with open('bg.txt', 'w') as f:
        negative_samples = os.listdir(negative_samples_dir)
        for negative_sample in negative_samples:
            f.write(os.path.join(negative_samples_dir, negative_sample))
            f.write('\n')


def create_negative_images(img_dir, save_dir, img_coords, iter=100):
    """

    :param img_dir:             image folder location
    :param img_coords:          The coordinates of the joint location so we don't get these part as negative images
    :param iter:                The number of times to crop negative part of a single image
    :return:
    """
    num_img = 0
    os.makedirs(save_dir, exist_ok=True)

    for img_name, coords in img_coords.items():

        img = cv2.imread(os.path.join(img_dir, img_name))
        img_height, img_width, img_channel = img.shape

        for i in range(iter):
            crop_start_x = random.randint(0, img_width)
            crop_start_y = random.randint(0, img_height)
            crop_end_x = crop_start_x + random.randint(50, 100)
            crop_end_y = crop_start_y + random.randint(50, 100)

            is_joint_location = False
            for coord in coords:

                start_coord = coord[0]
                start_x = start_coord[0]
                start_y = start_coord[1]

                end_coord = coord[1]
                end_x = end_coord[0]
                end_y = end_coord[1]

                # if the random points is at the joint location then we skip, because we want to crop parts of the image
                # that is not joint
                # top left
                if start_x < crop_start_x < end_x and start_y < crop_start_y < end_y:
                    is_joint_location = True
                    break
                # bottom right
                if start_x < crop_end_x < end_x and start_y < crop_end_y < end_y:
                    is_joint_location = True
                    break
                # top right
                if start_x < crop_end_x < end_x and start_y < crop_start_y < end_y:
                    is_joint_location = True
                    break
                # bottom left
                if start_x < crop_start_x < end_x and start_y < crop_end_y < end_y:
                    is_joint_location = True
                    break

                # if the joint coords are located inside this crop part we want to skip this too because it is at the
                # joint location
                if crop_start_x < start_x < crop_end_x and crop_start_y < start_y < crop_end_y:
                    is_joint_location = True
                    break
                # bottom right
                if crop_start_x < end_x < crop_end_x and crop_start_y < end_y < crop_end_y:
                    is_joint_location = True
                    break
                # top right
                if crop_start_x < end_x < crop_end_x and crop_start_y < start_y < crop_end_y:
                    is_joint_location = True
                    break
                # bottom left
                if crop_start_x < start_x < crop_end_x and crop_start_y < end_y < crop_end_y:
                    is_joint_location = True
                    break

            if is_joint_location:
                continue

            crop_img = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
            cv2.imwrite(os.path.join(save_dir, f'{num_img}.png'), crop_img)
            num_img += 1


train_img_location = os.path.join('joint_detection', 'boneage-training-dataset')

img_coords = parse_xml_annotations('train')
create_negative_images(train_img_location, 'negative_images', img_coords)
create_haar_cascade_info_dat(img_coords, positive_samples_dir=train_img_location, negative_samples_dir='negative_images')
