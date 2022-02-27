from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from adabelief_tf import AdaBeliefOptimizer

import tensorflow as tf
from dataloader import ZurichSequence
from unet3plusv3 import UNet3Plus
from PIL import Image
import logging
import time
import utils
import os



class Train(object):

    def __init__(self, model,
                epochs,
                config_dir,
                target_dir = '',
                batch_size = 5,
                input_size = (224,224,4),
                target_shape = (448,448,3),
                save_per_epoch=True,
                save_best_model=True):

        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_size = input_size
        self.target_shape = target_shape
        self.target_size = self.target_shape[0]*self.target_shape[1]*self.target_shape[2]
        self.optimizer = AdaBeliefOptimizer(learning_rate = 0.0001, weight_decay=5e-4)
        self.model_load_dir = config_dir["model_load_dir"]
        self.model_save_dir = config_dir["model_save_dir"]
        self.images_save_dir = config_dir["images_dir"]
        self.save_best_model = save_best_model
        self.save_per_epoch = save_per_epoch

    def compute_loss(self, predict, target):


        # MSE loss

        loss_mse = tf.reduce_sum(tf.pow(target[0] - predict[0], 2))/(self.target_size * self.batch_size)

        #mse = tf.keras.losses.MeanSquaredError()
        #loss_mse = mse(target[0], predict).numpy()

        # PSNR loss
        loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))
        
        
        # SSIM loss
        #loss_ssim = tf.reduce_mean(tf.image.ssim(predict, target, 1.0))

        # MS-SSIM loss
        #loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(predict, target, 1.0))

        return {"loss":loss_mse,"mse":loss_mse, "psnr": loss_psnr}

    def train_step(self, batch):

        image, target = batch

        # [loss_temp, temp] = sess.run([loss_generator, train_step_gen], feed_dict={phone_: phone_images, dslr_: d$
        # training_loss += loss_temp / eval_step

        with tf.GradientTape() as tape:
            predictions = self.model(image, training=True)
            loss_dict = self.compute_loss(target, predictions)
        gradients = tape.gradient(loss_dict["loss"], self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients,
                                        self.model.trainable_variables))

    
        return loss_dict["loss"]

    def test_step(self, batch):
        """One test step.
        Args:
        inputs: one batch input.
        """
        image, target = batch
        predictions = self.model(image, training=False)

        loss_dict = self.compute_loss(target, predictions) 
 

        return loss_dict


    def train(self, train_dataset, test_dataset):

        NUM_BATCHES = len(train_dataset)
        TEST_SIZE = test_dataset.num_images
        logging.info("Start training.")
        best_psnr=0
        #with tf.Graph().as_default(), tf.Session() as sess:
        for epoch in range(self.epochs):


            loss_mse_eval = 0
            loss_psnr_eval = 0
            #Train step
            start_estimate = time.time()
            start_loadtime = time.time()

            for i, batch in enumerate(train_dataset):
                load_time = time.time() - start_loadtime
                start_processtime = time.time()

                loss = self.train_step(batch)

                end_processtime=time.time()
                estimate_time = ((time.time() - start_estimate)/((i+1)/NUM_BATCHES))/3600
                logging.info("Process time per batch: {:.4f}s | load time per batch: {:.4f}s | estimate time per epoch: {:.2f}hrs".format((end_processtime-start_processtime),load_time, estimate_time))
                logging.info(f"Epoch: {epoch}, Loss:{loss}, batch num: {i}, Training in {int(i/NUM_BATCHES*100)}%...")
                logging.info("==================================================================================================")
                start_loadtime = time.time()
                
                
                 #Test step
            for i, batch in enumerate(test_dataset):
                loss_dict = self.test_step(batch)
                loss_mse_eval += loss_dict["mse"]
                loss_psnr_eval += loss_dict["psnr"]
                logging.info("Epoch %d, mse_val: %.4f, psnr_val: %.4f" % (epoch, loss_mse_eval/TEST_SIZE, loss_psnr_eval/TEST_SIZE))
                logging.info("====================")



            # Save the model per epoch
            model = self.model
            if self.save_per_epoch:
                logging.info(f"Saving model for epoch {epoch}...")
                model.save(os.path.join(self.model_save_dir,f"unet3plus_epoch_{epoch}"))
                logging.info(f"Finish saving.")

            if self.save_best_model and loss_psnr_eval > best_psnr:
                    logging.info(f"Saving best model...")
                    model.save(os.path.join(self.model_save_dir,f"unet3plus_the_best_model"))
                    logging.info(f"Finish saving.")

            if loss_psnr_eval > best_psnr:
                best_psnr = loss_psnr_eval





if __name__ == "__main__":
    #model = UNet3Plus(4)
    model = UNet3Plus([224, 224, 4])

    train_dataset = ZurichSequence(data_dir= "/storage/zurich-dataset/",batch_size=5)
    test_dataset = ZurichSequence(data_dir="/storage/zurich-dataset/",batch_size=5, train=False)


    log_file = "/storage/matheus/tensorflow/output/"
    os.makedirs(log_file,exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    handlers=[
                        logging.FileHandler(log_file+'test.log', mode="w"),
                        logging.StreamHandler()
                    ])

    #with tf.device('/GPU:0'):
    #    train = Train(model, epochs=50, batch_size=1, target_dir = '/storage/matheus/tensorflow/model')
    config_dir = {"model_load_dir": None,
        "model_save_dir": '/storage/matheus/tensorflow/model',
        "images_dir":None            
        }
    train = Train(model, 
                    epochs=50,
                    batch_size=5,
                    config_dir=config_dir)
    train.train(train_dataset, test_dataset)
    #https://stackoverflow.com/questions/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell



