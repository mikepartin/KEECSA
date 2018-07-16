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


parser = argparse.ArgumentParser()
parser.add_argument('-f','--file',nargs=1,help='the image to process',required=True)
args=parser.parse_args()
if args.file:
    for item in args.file:
        imageToProcess=args.file[0]

#train_dir = "/tmp/flood-models/inception_v3"
train_dir="/tmp/flood-nonflood/keep"
#flowers_data_dir = "/tmp/flood"

classes=[]
with open("/tmp/flood-nonflood/keep/labels.txt", "r") as fp:
    data = fp.readlines()
    for l in data:
        classes.append(l.split(":")[1].strip("\n"))


"""
dataset = flowers.get_split('train', flowers_data_dir)
data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
image_raw, label = data_provider.get(['image', 'label'])
"""

with tf.Graph().as_default():
    #url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    #image_string = urllib.urlopen(url).read()
    #file_path = '/home/michael/Pictures/nonflood/5.jpg'
    file_path = imageToProcess
    image_string = tf.gfile.FastGFile(file_path, 'rb').read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(processed_images, num_classes=2, is_training=False)
    probabilities = tf.nn.softmax(logits)
    
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(train_dir, 'model.ckpt-1000'),
        #slim.get_model_variables('InceptionV3'))
        slim.get_variables_to_restore())
    
    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            sess.run(tf.initialize_local_variables())
            init_fn(sess)
            np_image, probabilities = sess.run([image, probabilities])
            #np_probabilities, np_images_raw, np_labels = sess.run([probabilities, images_raw, labels])
            print(np.argmax(probabilities[0, :]))
            probabilities = probabilities[0, 0:]

            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
        
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(2):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, classes[index]))
        #print("index: " + str(index))