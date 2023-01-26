import tensorflow as tf
import numpy as np
# from scipy.misc import imsave
from imageio import imsave

import os
import shutil
from PIL import Image
import time
import random
import sys
import glob


from layers import *



to_train = True
to_test = False
to_restore = False
output_path = "./output"
check_dir = "./output/checkpoints/"


temp_check = 0

class CycleGAN():

    def __init__(self, pathA, pathB, img_dim= (256, 256, 3), batch_size = 1, ngf=32, ndf=64, pool_size = 50, save_images = True, max_images = 100) -> None:
        self.pathA = pathA
        self.pathB = pathB

        self.img_height = 256
        self.img_width = 256
        self.img_layer = 3
        self.img_size = self.img_height * self.img_width
        self.ngf = ngf
        self.ndf = ndf
        self.batch_size = batch_size
        self.pool_size = pool_size
        self.save_images = True
        self.max_images = max_images

        

    


    def build_resnet_block(self, inputres, dim, name="resnet"):
        
        with tf.variable_scope(name):

            out_res = tf.pad(inputres, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c1")
            out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
            out_res = general_conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID","c2",do_relu=False)
            
            return tf.nn.relu(out_res + inputres)


    def build_generator_resnet_6blocks(self, inputgen, name="generator"):
        with tf.variable_scope(name):
            f = 7
            ks = 3
            
            pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
            o_c1 = general_conv2d(pad_input, self.ngf, f, f, 1, 1, 0.02,name="c1")
            o_c2 = general_conv2d(o_c1, self.ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
            o_c3 = general_conv2d(o_c2, self.ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

            o_r1 = self.build_resnet_block(o_c3, self.ngf*4, "r1")
            o_r2 = self.build_resnet_block(o_r1, self.ngf*4, "r2")
            o_r3 = self.build_resnet_block(o_r2, self.ngf*4, "r3")
            o_r4 = self.build_resnet_block(o_r3, self.ngf*4, "r4")
            o_r5 = self.build_resnet_block(o_r4, self.ngf*4, "r5")
            o_r6 = self.build_resnet_block(o_r5, self.ngf*4, "r6")

            o_c4 = general_deconv2d(o_r6, [self.batch_size,64,64,self.ngf*2], self.ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
            o_c5 = general_deconv2d(o_c4, [self.batch_size,128,128,self.ngf], self.ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
            o_c5_pad = tf.pad(o_c5,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
            o_c6 = general_conv2d(o_c5_pad, self.img_layer, f, f, 1, 1, 0.02,"VALID","c6",do_relu=False)

            # Adding the tanh layer

            out_gen = tf.nn.tanh(o_c6,"t1")


            return out_gen

    def build_generator_resnet_9blocks(self, inputgen, name="generator"):
        with tf.variable_scope(name):
            f = 7
            ks = 3
            
            pad_input = tf.pad(inputgen,[[0, 0], [ks, ks], [ks, ks], [0, 0]], "REFLECT")
            o_c1 = general_conv2d(pad_input, self.ngf, f, f, 1, 1, 0.02,name="c1")
            o_c2 = general_conv2d(o_c1, self.ngf*2, ks, ks, 2, 2, 0.02,"SAME","c2")
            o_c3 = general_conv2d(o_c2, self.ngf*4, ks, ks, 2, 2, 0.02,"SAME","c3")

            o_r1 = self.build_resnet_block(o_c3, self.ngf*4, "r1")
            o_r2 = self.build_resnet_block(o_r1, self.ngf*4, "r2")
            o_r3 = self.build_resnet_block(o_r2, self.ngf*4, "r3")
            o_r4 = self.build_resnet_block(o_r3, self.ngf*4, "r4")
            o_r5 = self.build_resnet_block(o_r4, self.ngf*4, "r5")
            o_r6 = self.build_resnet_block(o_r5, self.ngf*4, "r6")
            o_r7 = self.build_resnet_block(o_r6, self.ngf*4, "r7")
            o_r8 = self.build_resnet_block(o_r7, self.ngf*4, "r8")
            o_r9 = self.build_resnet_block(o_r8, self.ngf*4, "r9")

            o_c4 = general_deconv2d(o_r9, [self.batch_size,128,128,self.ngf*2], self.ngf*2, ks, ks, 2, 2, 0.02,"SAME","c4")
            o_c5 = general_deconv2d(o_c4, [self.batch_size,256,256,self.ngf], self.ngf, ks, ks, 2, 2, 0.02,"SAME","c5")
            o_c6 = general_conv2d(o_c5, self.img_layer, f, f, 1, 1, 0.02,"SAME","c6",do_relu=False)

            # Adding the tanh layer

            out_gen = tf.nn.tanh(o_c6,"t1")


            return out_gen


    def build_gen_discriminator(self, inputdisc, name="discriminator"):

        with tf.variable_scope(name):
            f = 4

            o_c1 = general_conv2d(inputdisc, self.ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm=False, relufactor=0.2)
            o_c2 = general_conv2d(o_c1, self.ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
            o_c3 = general_conv2d(o_c2, self.ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
            o_c4 = general_conv2d(o_c3, self.ndf*8, f, f, 1, 1, 0.02, "SAME", "c4",relufactor=0.2)
            o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

            return o_c5


    def patch_discriminator(self, inputdisc, name="discriminator"):

        with tf.variable_scope(name):
            f= 4

            patch_input = tf.random_crop(inputdisc,[1,70,70,3])
            o_c1 = general_conv2d(patch_input, self.ndf, f, f, 2, 2, 0.02, "SAME", "c1", do_norm="False", relufactor=0.2)
            o_c2 = general_conv2d(o_c1, self.ndf*2, f, f, 2, 2, 0.02, "SAME", "c2", relufactor=0.2)
            o_c3 = general_conv2d(o_c2, self.ndf*4, f, f, 2, 2, 0.02, "SAME", "c3", relufactor=0.2)
            o_c4 = general_conv2d(o_c3, self.ndf*8, f, f, 2, 2, 0.02, "SAME", "c4", relufactor=0.2)
            o_c5 = general_conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5",do_norm=False,do_relu=False)

            return o_c5

    def input_setup(self):

        ''' 
        This function basically setup variables for taking image input.

        filenames_A/filenames_B -> takes the list of all training images
        self.image_A/self.image_B -> Input image with each values ranging from [-1,1]
        '''

        filenames_A = glob.glob(self.pathA + "*.jpg")    
        self.queue_length_A = tf.size(filenames_A)
        filenames_B = glob.glob(self.pathB + "*.jpg")    
        self.queue_length_B = tf.size(filenames_B)
        
        print(len(filenames_A))
        print(len(filenames_B))



        filename_queue_A = tf.train.string_input_producer(filenames_A)
        filename_queue_B = tf.train.string_input_producer(filenames_B)

        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)

        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[256,256]),127.5),1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[256,256]),127.5),1)

    

    def input_read(self, sess):


        '''
        It reads the input into from the image folder.

        self.fake_images_A/self.fake_images_B -> List of generated images used for calculation of loss function of Discriminator
        self.A_input/self.B_input -> Stores all the training images in python list
        '''

        # Loading images into the tensors
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        num_files_A = sess.run(self.queue_length_A)
        num_files_B = sess.run(self.queue_length_B)

        self.fake_images_A = np.zeros((self.pool_size,1,self.img_height, self.img_width, self.img_layer))
        self.fake_images_B = np.zeros((self.pool_size,1,self.img_height, self.img_width, self.img_layer))


        self.A_input = np.zeros((self.max_images, self.batch_size, self.img_height, self.img_width, self.img_layer))
        self.B_input = np.zeros((self.max_images, self.batch_size, self.img_height, self.img_width, self.img_layer))

        for i in range(self.max_images): 
            image_tensor = sess.run(self.image_A)
            if(image_tensor.size == self.img_size*self.batch_size*self.img_layer):
                self.A_input[i] = image_tensor.reshape((self.batch_size,self.img_height, self.img_width, self.img_layer))

        for i in range(self.max_images):
            image_tensor = sess.run(self.image_B)
            if(image_tensor.size == self.img_size*self.batch_size*self.img_layer):
                self.B_input[i] = image_tensor.reshape((self.batch_size,self.img_height, self.img_width, self.img_layer))


        coord.request_stop()
        coord.join(threads)




    def model_setup(self):

        ''' This function sets up the model to train

        self.input_A/self.input_B -> Set of training images.
        self.fake_A/self.fake_B -> Generated images by corresponding generator of input_A and input_B
        self.lr -> Learning rate variable
        self.cyc_A/ self.cyc_B -> Images generated after feeding self.fake_A/self.fake_B to corresponding generator. This is use to calcualte cyclic loss
        '''

        self.input_A = tf.placeholder(tf.float32, [self.batch_size, self.img_width, self.img_height, self.img_layer], name="input_A")
        self.input_B = tf.placeholder(tf.float32, [self.batch_size, self.img_width, self.img_height, self.img_layer], name="input_B")
        
        self.fake_pool_A = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.img_layer], name="fake_pool_A")
        self.fake_pool_B = tf.placeholder(tf.float32, [None, self.img_width, self.img_height, self.img_layer], name="fake_pool_B")

        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        self.num_fake_inputs = 0

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")

        with tf.variable_scope("Model") as scope:
            self.fake_B = self.build_generator_resnet_9blocks(self.input_A, name="g_A")
            self.fake_A = self.build_generator_resnet_9blocks(self.input_B, name="g_B")
            self.rec_A = self.build_gen_discriminator(self.input_A, "d_A")
            self.rec_B = self.build_gen_discriminator(self.input_B, "d_B")

            scope.reuse_variables()

            self.fake_rec_A = self.build_gen_discriminator(self.fake_A, "d_A")
            self.fake_rec_B = self.build_gen_discriminator(self.fake_B, "d_B")
            self.cyc_A = self.build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = self.build_generator_resnet_9blocks(self.fake_A, "g_A")

            scope.reuse_variables()

            self.fake_pool_rec_A = self.build_gen_discriminator(self.fake_pool_A, "d_A")
            self.fake_pool_rec_B = self.build_gen_discriminator(self.fake_pool_B, "d_B")

    def loss_calc(self):

        ''' In this function we are defining the variables for loss calcultions and traning model

        d_loss_A/d_loss_B -> loss for discriminator A/B
        g_loss_A/g_loss_B -> loss for generator A/B
        *_trainer -> Variaous trainer for above loss functions
        *_summ -> Summary variables for above loss functions'''

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        
        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B,1))
        
        g_loss_A = cyc_loss*10 + disc_loss_B
        g_loss_B = cyc_loss*10 + disc_loss_A

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.rec_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(self.rec_B,1)))/2.0

        
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
        
        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        for var in self.model_vars: print(var.name)

        #Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    def save_training_images(self, sess, epoch):

        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")

        for i in range(0,10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
            imsave("./output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",tf.image.resize_images(((fake_A_temp[0]+1)*127.5),[120,40]).eval().astype(np.uint8))
            imsave("./output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",tf.image.resize_images(((fake_B_temp[0]+1)*127.5),[120,40]).eval().astype(np.uint8))
            imsave("./output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",tf.image.resize_images(((cyc_A_temp[0]+1)*127.5),[120,40]).eval().astype(np.uint8))
            imsave("./output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",tf.image.resize_images(((cyc_B_temp[0]+1)*127.5),[120,40]).eval().astype(np.uint8))
            imsave("./output/imgs/inputA_"+ str(epoch) + "_" + str(i)+".jpg",tf.image.resize_images(((self.A_input[i][0]+1)*127.5),[120,40]).eval().astype(np.uint8))
            imsave("./output/imgs/inputB_"+ str(epoch) + "_" + str(i)+".jpg",tf.image.resize_images(((self.B_input[i][0]+1)*127.5),[120,40]).eval().astype(np.uint8))

    def fake_image_pool(self, num_fakes, fake, fake_pool):
        ''' This function saves the generated image to corresponding pool of images.
        In starting. It keeps on feeling the pool till it is full and then randomly selects an
        already stored image and replace it with new one.'''

        if(num_fakes < self.pool_size):
            fake_pool[num_fakes] = fake
            return fake
        else :
            p = random.random()
            if p > 0.5:
                random_id = random.randint(0,self.pool_size-1)
                temp = fake_pool[random_id]
                fake_pool[random_id] = fake
                return temp
            else :
                return fake


    def train(self):


        ''' Training Function '''


        # Load Dataset from the dataset folder
        self.input_setup()  

        #Build the network
        self.model_setup()

        #Loss function calculations
        self.loss_calc()
      
        # Initializing the global variables
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()     

        with tf.Session() as sess:
            sess.run(init)

            #Read input to nd array
            self.input_read(sess)

            #Restore the model to run the model from last checkpoint
            if to_restore:
                chkpt_fname = tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)

            writer = tf.summary.FileWriter("./output/2")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step),100):                
                print ("In the epoch ", epoch)
                saver.save(sess,os.path.join(check_dir,"cyclegan"),global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                if(epoch < 100) :
                    curr_lr = 0.0002
                else:
                    curr_lr = 0.0002 - 0.0002*(epoch-100)/100

                if(self.save_images):
                    self.save_training_images(sess, epoch)

                # sys.exit()

                for ptr in range(0,self.max_images):
                    print("In the iteration ",ptr)
                    print("Starting",time.time()*1000.0)

                    # Optimizing the G_A network

                    _, fake_B_temp, summary_str = sess.run([self.g_A_trainer, self.fake_B, self.g_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})
                    
                    writer.add_summary(summary_str, epoch*self.max_images + ptr)                    
                    fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)
                    
                    # Optimizing the D_B network
                    _, summary_str = sess.run([self.d_B_trainer, self.d_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_B:fake_B_temp1})
                    writer.add_summary(summary_str, epoch*self.max_images + ptr)
                    
                    
                    # Optimizing the G_B network
                    _, fake_A_temp, summary_str = sess.run([self.g_B_trainer, self.fake_A, self.g_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})

                    writer.add_summary(summary_str, epoch*self.max_images + ptr)
                    
                    
                    fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)

                    # Optimizing the D_A network
                    _, summary_str = sess.run([self.d_A_trainer, self.d_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_A:fake_A_temp1})

                    writer.add_summary(summary_str, epoch*self.max_images + ptr)
                    
                    self.num_fake_inputs+=1
            
                        

                sess.run(tf.assign(self.global_step, epoch + 1))

            writer.add_graph(sess.graph)

    def test(self):


        ''' Testing Function'''

        print("Testing the results")

        self.input_setup()

        self.model_setup()
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            self.input_read(sess)

            chkpt_fname = tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)

            if not os.path.exists("./output/imgs/test/"):
                os.makedirs("./output/imgs/test/")            

            for i in range(0,100):
                fake_A_temp, fake_B_temp = sess.run([self.fake_A, self.fake_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
                imsave("./output/imgs/test/fakeB_"+str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/fakeA_"+str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputA_"+str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
                imsave("./output/imgs/test/inputB_"+str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))


def main():
    
    model = CycleGAN("./input/Herstelldatum/", "./input/generatedImages/", img_dim=(256, 256, 1))
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__ == '__main__':

    main()