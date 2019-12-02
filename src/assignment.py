import numpy as np
import tensorflow as tf

def train_step(generator,discriminator,real_images,outlined_images,batch_size):
    num_images = tf.shape(real_images)[0]
    num_batches = int(num_images/batch_size)
    for iteration in range(num_batches):
        batch_real = tf.slice(real_images,[iteration * batch_size,0,0,0],[(iteration + 1) * batch_size,-1,-1,-1])
        batch_outlines = tf.slice(outlined_images, [iteration * batch_size, 0, 0, 0], [(iteration + 1) * batch_size, -1, -1, -1])
        #TODO add batch of random inputs "z"
        z = ...
        with tf.GradientTape() as tape_d:
            batch_fake = generator(z,batch_outlines)
            real_probs = discriminator(batch_outlines,batch_real)
            fake_probs = discriminator(batch_outlines,batch_fake)
            disc_loss = discriminator.loss_function(real_probs,fake_probs)
        d_grad = tape_d.gradient(disc_loss, discriminator.trainable_variables)
        discriminator.optimizer.apply_gradients(zip(d_grad, discriminator.trainable_variables))
        num_gen_updates = 2
        for update in range(num_gen_updates):
            with tf.GradientTape() as tape_g:
                batch_fake = generator(z, batch_outlines)
                fake_probs = discriminator(batch_outlines, batch_fake)
                gen_loss = generator.loss_function(fake_probs)
            g_grad = tape_g.gradient(gen_loss,generator.trainable_variables)
            generator.optimizer.apply_gradients(zip(g_grad, generator.trainable_variables))

def test(generator, out_dir, outlined_images,batch_size):
    
    #TODO add z
    z = ...
    img = np.array(generator(z,outlined_images))

    ### Below, we've already provided code to save these generated images to files on disk
    # Rescale the image from (-1, 1) to (0, 255)
    img = ((img / 2) - 0.5) * 255
    # Convert to uint8
    img = img.astype(np.uint8)
    # Save images to disk
    for i in range(0, batch_size):
        img_i = img[i]
        s = out_dir + '/' + str(i) + '.png'
        imwrite(s, img_i)

def train(generator,discriminator,real_images,outlined_images,batch_size,num_epochs,out_dir):
    for epoch in range(0,num_epochs):
        print('========================== EPOCH %d  ==========================' % epoch)
        train_step(generator,discriminator,real_images,outlined_images,batch_size)
        batch_outlined = tf.slice(outlined_images,[0,0,0,0],[batch_size,-1,-1,-1])
        test(generator,out_dir,outlined_images,batch_size)

def main():
    generator = generator()
    discriminator = discriminator()
    # TODO input real images, outlined images. Specify hyperparameters, give place to save generated images to 
    real_images = ...
    outlined_images = ...
    num_epochs = ...
    batch_size = ...
    num_epochs = ...
    out_dir = ...
    
    train(generator,discriminator,real_images,outlined_images,batch_size,num_epochs,out_dir)
main()