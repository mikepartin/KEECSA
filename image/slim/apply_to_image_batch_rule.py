import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import flowers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from datasets import imagenet
from nets import inception
from preprocessing import inception_preprocessing

from tensorflow.contrib import slim

image_size = inception.inception_v3.default_image_size

import argparse

show_image = False

parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch',nargs=1,help="a list of images per line to process",required=True)
parser.add_argument('-m','--model',nargs=1,help="a path the where the model and labels.txt file are",required=True)
parser.add_argument('-o','--output',nargs=1,help="the output file to store the results",required=True)
parser.add_argument('-c','--checkpoint',nargs=1,help="the number of the model checkpoint (model.ckpt-xxxx)",required=True)
#TODO: MAKE THIS USE REGEX TO GRAB THE CHECKPOINT NUMBER
parser.add_argument('-s','--show',action='store_true')
args=parser.parse_args()
if args.batch:
    imageToProcess=args.batch[0]
if args.model:
    model_path=args.model[0]
if args.output:
    output_file=args.output[0]
if args.checkpoint:
    model_checkpoint=args.checkpoint[0]
if args.show:
    show_image = True

#train_dir = "/home/michael/workspace/trained/flood-nonflood"
train_dir=model_path
#flowers_data_dir = "/tmp/flood"

classes=[]
with open(train_dir + "/labels.txt", "r") as fp:
    data = fp.readlines()
    for l in data:
        classes.append(l.split(":")[1].strip("\n"))
num_class = len(classes)

flood_threshold = 0.80

with open(imageToProcess, "r") as fp:
    images_to_process = fp.readlines()
    images_to_process = [x.strip() for x in images_to_process]
    for file_path in images_to_process:

        with tf.Graph().as_default():


            image_string = tf.gfile.FastGFile(file_path, 'rb').read()
            image = tf.image.decode_jpeg(image_string, channels=3)
            processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
            processed_images  = tf.expand_dims(processed_image, 0)
            
            # Create the model, use the default arg scope to configure the batch norm parameters.
            with slim.arg_scope(inception.inception_v3_arg_scope()):
                logits, _ = inception.inception_v3(processed_images, num_classes=num_class, is_training=False)
            probabilities = tf.nn.softmax(logits)
            
            init_fn = slim.assign_from_checkpoint_fn(
                os.path.join(train_dir, 'model.ckpt-' + str(model_checkpoint)),
                #slim.get_model_variables('InceptionV3'))
                slim.get_variables_to_restore())
            
            with tf.Session() as sess:
                with slim.queues.QueueRunners(sess):
                    sess.run(tf.initialize_local_variables())
                    init_fn(sess)
                    np_image, probabilities = sess.run([image, probabilities])
                    print(np.argmax(probabilities[0, :]))
                    probabilities = probabilities[0, 0:]

                    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
            if show_image:
                plt.figure()
                plt.imshow(np_image.astype(np.uint8))
                plt.axis('off')
                plt.show()

            names = imagenet.create_readable_names_for_imagenet_labels()
            with open(output_file, "a") as fp:
                fp.write(file_path.split("/")[-1])
                print("Results for: \""+file_path+"\"")
                p_prob=[]
                p_class=[]
                for i in range(num_class):
                    index = sorted_inds[i]
                    p_prob.append(probabilities[index])
                    p_class.append(classes[index])
                    print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, classes[index]))
                    #fp.write('  Probability %0.2f%% => [%s]' % (probabilities[index] * 100, classes[index]))
                    #fp.write("\n")
                if p_class[0] == "flood" and p_prob[0] >= flood_threshold:
                    print("  --> FLOOD")
                    fp.write(",1\n")
                else:
                    print("  --> NON-FLOOD")
                    fp.write(",0\n")