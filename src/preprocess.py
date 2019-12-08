import numpy as np
import tensorflow as tf
import os

from imageio import imwrite
from random import uniform


# Sets up tensorflow graph to load images
# (This is the version using new-style tf.data API)
def load_image_batch(dir_name, batch_size, shuffle_buffer_size=250000, n_threads=2):
    """
    Given a directory and a batch size, the following method returns a dataset iterator that can be queried for 
    a batch of images

    :param dir_name: a batch of images
    :param batch_size: the batch size of images that will be trained on each time
    :param shuffle_buffer_size: representing the number of elements from this dataset from which the new dataset will 
    sample
    :param n_thread: the number of threads that will be used to fetch the data

    :return: an iterator into the dataset
    """
    # Function used to load and pre-process image files
    # (Have to define this ahead of time b/c Python does allow multi-line
    #    lambdas, *grumble*)
    def load_and_process_image(file_path):
        """
        Given a file path, this function opens and decodes the image stored in the file.

        :param file_path: a batch of images

        :return: an rgb image
        """
        # Load image
        image = tf.io.decode_jpeg(tf.io.read_file(file_path), channels=3)
        # Convert image to normalized float (0, 1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Rescale data to range (-1, 1)
        image = (image - 0.5) * 2
        return image

    # List file names/file paths
    dir_path = dir_name + '/*.jpg'
    dataset = tf.data.Dataset.list_files(dir_path)

    # Shuffle order
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # Load and process images (in parallel)
    dataset = dataset.map(map_func=load_and_process_image, num_parallel_calls=n_threads)

    # Create batch, dropping the final one which has less than batch_size elements and finally set to reshuffle
    # the dataset at the end of each iteration
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the next batch while the GPU is training
    dataset = dataset.prefetch(1)

    # Return an iterator over this dataset
    return dataset

def random_jitter_and_mirroring(images):
    input_, ground_truth = tf.split(images, 2, 2)
    batch_size, crop_height, crop_width, _ = input_.shape
    assert(crop_height == 256 and crop_width == 256)

    if uniform(0, 1) < 0.5:
        tf.image.flip_left_right(input_) 
        tf.image.flip_left_right(ground_truth) 

    resized_input = tf.image.resize(input_, tf.constant([286, 286]))
    resized_ground_truth = tf.image.resize(ground_truth, tf.constant([286, 286]))
    cropped_input = tf.image.random_crop(resized_input, [batch_size, crop_height, crop_width, 3])
    cropped_ground_truth = tf.image.random_crop(resized_ground_truth, [batch_size, crop_height, crop_width, 3])
    return cropped_input, cropped_ground_truth

def main():
    img_dir = "../../UnetGenerator/data/facades"
    out_dir = "./output"
    batchs = load_image_batch(img_dir + '/test', 1)
    for i, img in enumerate(batchs):
        assert(np.all(-1.0 <= img) and np.all(img <= 1.0))
        img = tf.concat(random_jitter_and_mirroring(img), 2).numpy()
        # Rescale the image from (-1, 1) to (0, 255)
        img = ((img / 2) + 0.5) * 255
        img = img.astype(np.uint8)
        # Save images to disk
        s = out_dir+'/'+str(i)+'.png'
        imwrite(s, img[0])

if __name__ == '__main__':
   main()
