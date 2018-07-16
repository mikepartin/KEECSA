import numpy as np
import tensorflow as tf
from datasets import flowers
from nets import inception
from preprocessing import inception_preprocessing
import matplotlib.pyplot as plt
import cv2

from tensorflow.contrib import slim

flowers_data_dir = "/tmp/flowers"
train_dir = "/tmp/flowers-models/inception_v3"
#train_dir = "/tmp/flowers-models/test"


def get_single_img():
    file_path = '/tmp/rose_3.jpg'
    #image_data = tf.gfile.FastGFile(file_path, 'rb').read()
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    img = tf.cast(img, tf.float32)

    #print(pixels.shape)
    return img

def load_batch(dataset, batch_size=32, height=299, width=299, is_training=False):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', 'label'])

    #print(image_raw)
    
    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training)
    
    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)



    # Batch it up.
    images, images_raw, labels = tf.train.batch(
          [image, image_raw, label],
          batch_size=batch_size,
          num_threads=1,
          capacity=2 * batch_size)
    
    return images, images_raw, labels

image_size = inception.inception_v3.default_image_size
batch_size = 1

with tf.Graph().as_default():
    #image_reader = ImageReader()
    tf.logging.set_verbosity(tf.logging.INFO)
    
    dataset = flowers.get_split('train', flowers_data_dir)
    
    
    images, images_raw, labels = load_batch(dataset, height=image_size, width=image_size)
    print("*******************************")
    print(images_raw)
    images=get_single_img()
    images_raw=get_single_img()
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, _ = inception.inception_v3(images, num_classes=dataset.num_classes, is_training=True)

    probabilities = tf.nn.softmax(logits)
    
    checkpoint_path = tf.train.latest_checkpoint(train_dir)
    init_fn = slim.assign_from_checkpoint_fn(
      checkpoint_path,
      slim.get_variables_to_restore())
    
    with tf.Session() as sess:
        with slim.queues.QueueRunners(sess):
            sess.run(tf.initialize_local_variables())
            init_fn(sess)
            np_probabilities, np_images_raw, np_labels = sess.run([probabilities, images_raw, labels])

            
            #print("****************************")
            #print(img)
            #logits = inception.inference(img)
            #img_4d = tf.reshape(img,[img.shape[0],img.shape[1],3])
            #np_probabilities, np_images_raw, np_labels = sess.run([probabilities, img, labels])
    
            for i in range(batch_size): 
                image = np_images_raw[i, :, :, :]
                #image = cv2.cvtColor(get_single_img(), cv2.COLOR_BGR2RGB)
                #image = get_single_img()
                true_label = np_labels[i]
                predicted_label = np.argmax(np_probabilities[i, :])
                predicted_name = dataset.labels_to_names[predicted_label]
                true_name = dataset.labels_to_names[true_label]
                
                plt.figure()
                plt.imshow(image.astype(np.uint8))
                plt.title('Ground Truth: [%s], Prediction [%s]' % (true_name, predicted_name))
                plt.axis('off')
                plt.show()
