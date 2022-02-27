import tensorflow as tf
from tensorflow import keras
from numpy.core.shape_base import _accumulate
from unet3plusv3 import UNet3Plus
import time
import os
import logging
import argparse
from dataloader import ZurichSequence 
import math
parser = argparse.ArgumentParser(description="Generate image..")

parser.add_argument("--modeldir", help="load model folder", default=None)
parser.add_argument("--datadir", help="dataset dir", default=None)
parser.add_argument("--outdir", help="name output folder", default=None)
parser.add_argument("--indir", help="name output folder", default=None)
parser.add_argument("--numimgs", help="name save model folder", default=10, type=int)
parser.add_argument("--isindividual", help="individual image", default=1, type=int)
parser.add_argument("--logpath")

args = parser.parse_args()

logdir = "/storage/rodrigo/unet3plus_tfv3/results.log"
outdir = "/storage/rodrigo/unet3plus_tfv3/images"
#local###################
outdir = args.outdir
logdir = args.logpath
#########################
outdir_jpg = os.path.join(outdir,"jpg")
outdir_png = os.path.join(outdir,"png")

os.makedirs(outdir_jpg,exist_ok=True)
os.makedirs(outdir_png, exist_ok=True)

 
if __name__ == "__main__":
    batch_size = 1
    workers = 1

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    handlers=[
                        logging.FileHandler(logdir, mode="w"),
                        logging.StreamHandler()
                    ])

    #Device available for training.
    #https://qastack.com.br/programming/38009682/how-to-tell-if-tensorflow-is-using-gpu-acceleration-from-inside-python-shell
    device = tf.test.gpu_device_name()
    logging.info(f"Using device: {device}")

    #load model
    #https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model
    model = keras.models.load_model(args.modeldir)


    accumulated_psnr = 0
    accumulated_mse = 0
    #generate by dataset
    if args.datadir:

        test_dataset = ZurichSequence(args.datadir,batch_size=batch_size, train=False, shuffle=False)

        for i, batch in enumerate(test_dataset):

            x,y = batch
            #x = x[0]
            #y = y[0]
            #y = y.to(device)
            #x = x.to(device)
            
            #Calling model https://stackoverflow.com/questions/60837962/confusion-about-keras-model-call-vs-call-vs-predict-methods
            #https://stackoverflow.com/questions/67385963/what-is-the-tensorflow-keras-equivalent-of-pytorchs-no-grad-function
            predicted = model.predict(x)

            #MSE loss https://www.tensorflow.org/api_docs/python/tf/keras/metrics/mean_squared_error
            #mse = tf.keras.metrics.mean_squared_error(predicted,y)
            mse = tf.keras.losses.MeanSquaredError()
            mse = mse(y, predicted[0]).numpy()
            #mse = tf.reduce_sum(tf.pow(y - predicted[0], 2))/(1 * 1)

            #print(mse)

            psnr = 20 * math.log10(1.0 / math.sqrt(mse))
            #psnr = 20 * ((mse)/ 1.0 )
            accumulated_mse += mse
            accumulated_psnr += psnr
            logging.info("img%d.png: mse: %.4f, psnr: %.4f" % (i,mse, psnr))
            psnr_str = "PSNR" + ("{0:.3f}".format(3.22345)).replace(".","_")

            
            #t = tf.image.grayscale_to_rgb(predicted)

            #Save array as image
            #https://stackoverflow.com/questions/67473951/image-getting-corrupted-with-tf-keras-preprocessing-image-save-img
            tf.keras.preprocessing.image.save_img(os.path.join(outdir_jpg,f"img{i}.jpg"), predicted[0], 
                                                    data_format = 'channels_last' ,  scale=True)
            tf.keras.preprocessing.image.save_img(os.path.join(outdir_png,f"img{i}.png"), predicted[0], 
                                                    data_format = 'channels_last' , scale=True)

        mse = accumulated_mse/len(test_dataset)
        psnr = accumulated_psnr/len(test_dataset)
        logging.info("============================================")
        logging.info("mse_avg: %.4f, psnr_avg: %.4f" % ( mse, psnr))
        logging.info("============================================")

    if args.indir:
        #TODO
        pass

#python3 generate_image.py --modeldir ../output/unet3plus_test/state_saved/unet3plus_the_best_model.pth --datadir /home/matheuss/datasets/zurich_test --numimgs 5 --outdir ../output/unet3plus_test/images


#python workspace/generate_image.py --modeldir /storage/rodrigo/unet3plus_tfv2/state_saved/unet3plus_the_best_model --datadir /storage/zurich-dataset --numimgs 1204  
            
            
            
            
            
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
