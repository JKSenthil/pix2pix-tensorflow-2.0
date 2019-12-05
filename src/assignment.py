import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, LeakyReLU, Reshape, Conv2DTranspose
from preprocess import load_image_batch
import generator.py
import discriminator.py
import tensorflow_gan as tfgan
import tensorflow_hub as hub

import numpy as np

from imageio import imwrite
import os
import argparse

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

gpu_available = tf.test.is_gpu_available()
print("GPU Available: ", gpu_available)

## --------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='PIX2PIX')

parser.add_argument('--img-dir', type=str, default='./data/celebA',
                    help='Data where training images live')

parser.add_argument('--outline-dir', type=str, default='./data/celebA',
                    help='Data where training images live')

parser.add_argument('--out-dir', type=str, default='./output',
                    help='Data where sampled output images will be written')

parser.add_argument('--mode', type=str, default='train',
                    help='Can be "train" or "test"')

parser.add_argument('--restore-checkpoint', action='store_true',
                    help='Use this flag if you want to resuming training from a previously-saved checkpoint')

parser.add_argument('--z-dim', type=int, default=100,
                    help='Dimensionality of the latent space')

parser.add_argument('--batch-size', type=int, default=128,
                    help='Sizes of image batches fed through the network')

parser.add_argument('--num-data-threads', type=int, default=2,
                    help='Number of threads to use when loading & pre-processing training images')

parser.add_argument('--num-epochs', type=int, default=10,
                    help='Number of passes through the training data to make before stopping')

parser.add_argument('--learn-rate', type=float, default=0.0002,
                    help='Learning rate for Adam optimizer')

parser.add_argument('--beta1', type=float, default=0.5,
                    help='"beta1" parameter for Adam optimizer')

parser.add_argument('--num-gen-updates', type=int, default=2,
                    help='Number of generator updates per discriminator update')

parser.add_argument('--log-every', type=int, default=7,
                    help='Print losses after every [this many] training iterations')

parser.add_argument('--save-every', type=int, default=500,
                    help='Save the state of the network after every [this many] training iterations')

parser.add_argument('--device', type=str, default='GPU:0' if gpu_available else 'CPU:0',
                    help='specific the device of computation eg. CPU:0, GPU:0, GPU:1, GPU:2, ... ')

args = parser.parse_args()


# Train the model for one epoch.
def train(generator, discriminator, image_iterator, outline_iterator, manager):
    """
    Train the model for one epoch. Save a checkpoint every 500 or so batches.

    :param generator: generator model
    :param discriminator: discriminator model
    :param image_iterator: iterator over real images
    :param outline_iterator: iterator over outlined images
    :param manager: the manager that handles saving checkpoints by calling save()
    """
    # Loop over our data until we run out
    for iteration, (img_batch, outline_batch) in zip(image_iterator,outline_iterator):
        with tf.GradientTape() as tape_d:
            gen_output = generator(outline_batch)
            real_probs = discriminator(outline_batch,img_batch)
            fake_probs = discriminator(outline_batch,gen_output)
            disc_loss = discriminator.loss_function(real_probs,fake_probs)
        d_grad = tape_d.gradient(disc_loss,discriminator.trainable_variables)
        discriminator.optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))
        for i in range(args.num_gen_updates):
            with tf.GradientTape() as tape_g:
                gen_output = generator(outline_batch)
                fake_probs = discriminator(gen_output)
                gen_loss= generator.loss_function(fake_probs)
            g_grad = tape_g.gradient(gen_loss,generator.trainable_variables)
            generator.optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))

        # Save
        if iteration % args.save_every == 0:
            manager.save()

        if iteration % 100 == 0:
            test(generator,outline_batch)


# Test the model by generating some samples.
def test(generator,outline_batch):
    """
    Test the model.

    :param generator: generator model
    :param outline_batch: batch of outlines to test on

    :return: None
    """
    img = np.array(generator(outline_batch))

    ### Below, we've already provided code to save these generated images to files on disk
    # Rescale the image from (-1, 1) to (0, 255)
    img = ((img / 2) - 0.5) * 255
    # Convert to uint8
    img = img.astype(np.uint8)
    # Save images to disk
    for i in range(0, args.batch_size):
        img_i = img[i]
        s = args.out_dir+'/'+str(i)+'.png'
        imwrite(s, img_i)


## --------------------------------------------------------------------------------------

def main():
    # Load a batch of images (to feed to the discriminator)
    real_iterator,outline_iterator = load_image_batch(args.img_dir,args.outline_dir, batch_size=args.batch_size, n_threads=args.num_data_threads)

    # Initialize generator and discriminator models
    generator = UnetGenerator()
    discriminator = discriminator()

    # For saving/loading models
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    # Ensure the output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.restore_checkpoint or args.mode == 'test':
        # restores the latest checkpoint using from the manager
        checkpoint.restore(manager.latest_checkpoint)

    try:
        # Specify an invalid GPU device
        with tf.device('/device:' + args.device):
            if args.mode == 'train':
                for epoch in range(0, args.num_epochs):
                    print('========================== EPOCH %d  ==========================' % epoch)
                    # Save at the end of the epoch, too
                    print("**** SAVING CHECKPOINT AT END OF EPOCH ****")
                    manager.save()
            if args.mode == 'test':
                test(generator)
    except RuntimeError as e:
        print(e)

if __name__ == '__main__':
   main()
