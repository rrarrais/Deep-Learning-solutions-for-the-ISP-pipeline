import tensorflow as tf
from tensorflow.python.eager.context import device
from tensorflow import keras
from dataloader import ZurichSequence
from unet3plusv3 import UNet3Plus
import time
import os
import logging
import argparse
from train import Train


## Arguments parser
parser = argparse.ArgumentParser(description="Train the Unet3+ model.")

parser.add_argument("--datadir", help="all data folder", default="/storage/rodrigo")
parser.add_argument("--datasetdir", help="data set dir", default="/storage/zurich-dataset")
parser.add_argument("--loadmodel", help="load model folder", default=None)
parser.add_argument("--outfolder", help="name output folder", default=None)
parser.add_argument("--savefolder", help="name save model folder", default="state_saved/")
parser.add_argument("--logfile", help="log file name", default = "unet3plus_v4.log")
parser.add_argument("--imagesfolder",help="images files folder", default = "images")
parser.add_argument("--dataloader",help="images files folder", default = 1, type = int)

parser.add_argument("--epochs", help="number of epochs", default = 30, type=int)
parser.add_argument("--batchsize", help="batch size", default = 5, type=int)

args = parser.parse_args()

if args.outfolder == None:
    year, month, day, hour, min = map(int, time.strftime("%Y %m %d %H %M").split())
    out_folder = f"unet3plus_{year}_{month}_{day}_{hour}_{min}"
else:
    out_folder = args.outfolder


#Define model directory to load
if args.loadmodel:
    model_load_folder = args.loadmodel
    model_load_dir = os.path.join(args.datadir, model_load_folder)
    files = []
    for file in os.listdir(model_load_dir):
        if file.endswith(".pth"):
            files.append(os.path.join(model_load_dir,file))

    files.sort()
    model_load_dir = files[-1]
    
else:
    model_load_dir = None

# Directories of input and output
model_load_dir = model_load_dir #/datadir/model_dir
out_dir = os.path.join(args.datadir,out_folder) #/datadir/out_folder
model_save_dir = os.path.join(out_dir,args.savefolder) #/datadir/out_folder/model_save_dir
log_file_dir = os.path.join(out_dir,args.logfile)#/datadir/out_folder/logfile.log
images_dir = os.path.join(out_dir,args.imagesfolder)

os.makedirs(out_dir,exist_ok=True)
os.makedirs(model_save_dir,exist_ok=True)
os.makedirs(images_dir,exist_ok=True)

#Defining logging file configurations.

logging.basicConfig(level=logging.INFO,
		     format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_dir, mode="w"),
                        logging.StreamHandler()
                    ])

logging.info(f"Output directory: {out_dir}")
logging.info(f"Model save directory: {model_save_dir}")
logging.info(f"log file directory: {log_file_dir}")
logging.info(f"Images output directory: {images_dir}")


config_dir = {"model_load_dir": model_load_dir,
            "model_save_dir": model_save_dir,
            "images_dir":images_dir            
            }
            

logging.info("Start Job")

#Load Dataset
try:

    DATASET_DIR = args.datasetdir

    train_dataset = ZurichSequence(
        DATASET_DIR, 
        batch_size=args.batchsize
    )
    test_dataset = ZurichSequence(
        DATASET_DIR,
        batch_size=args.batchsize,
        train=False
    )

except Exception as e:
    logging.exception(str(e))
    raise Exception()

logging.info("Datasets defined")


#########Device available for training.

#https://www.tensorflow.org/api_docs/python/tf/test/is_gpu_available
#tf.config.list_physical_devices('GPU')
#https://www.tensorflow.org/api_docs/python/tf/test/gpu_device_name
#tf.test.gpu_device_name()


#Define model
#model = UNet3Plus(4)
model = UNet3Plus([224, 224, 4])

try:
    #optimizer = AdaBelief(model.parameters())
    # with tf.Session( config = tf.ConfigProto( log_device_placement = True ) ):
    train = Train(model, 
                    epochs=args.epochs,
                    batch_size=args.batchsize,
                    config_dir=config_dir)
    train.train(train_dataset, test_dataset)

    logging.info("Finish Train.")
except Exception as e:
    logging.exception(str(e))





