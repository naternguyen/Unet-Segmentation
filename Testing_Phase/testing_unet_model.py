import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

IMG = (slice(None), slice(None), slice(0,3))
MASK = (slice(None), slice(None), slice(3,4))
WEIGHTS = (slice(None), slice(None), slice(4,5))
def modify_image(image):
    dim = (128,128)
    resized = cv2.resize(image,dim)
    return resized

def show_image_list(images):
    if len(images) == 1:
        im = plt.imshow(np.squeeze(images[0]))
    else:
        fig, axs = plt.subplots(1, len(images), figsize=(20,20))
        for img, ax in zip(images, axs):
            im = ax.imshow(np.squeeze(img))
        fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.2)
    plt.show()

# model_path = "D:\\UNET in Tensorflow\\Unet_Tensor\\model_output\\unet\\ckpt\\model.ckpt-1"
model_path = "D:\\UNET in Tensorflow\\Unet_Tensor\\model_output\\unet\\model.ckpt-20"
# model_path = "D:\\UNET in Tensorflow\\Unet_Tensor\\data\\logs\\unet-same\\model.ckpt-2"

mask_graph = tf.Graph()
# with tf.device('/gpu:0'):
#     a = tf.constant(3.0)
with tf.Session(graph=mask_graph) as sess:
    # image = cv2.imread("D:\\UNET in Tensorflow\\Unet_Tensor\\dataset\\image_test\\PNG\\1_cv.png")
    image = cv2.imread("D:\\UNET in Tensorflow\\Unet_Tensor\\dataset\\image_test\\PNG\\2.png")
    # image = cv2.imread("D:\\UNET\\data-science-bowl-2018\\stage1_test\\31f1fbe85b8899258ea5bcf5f93f7ac8238660c386aeab40649c715bd2e38a0a\\images\\31f1fbe85b8899258ea5bcf5f93f7ac8238660c386aeab40649c715bd2e38a0a.png")
    w = image.shape[1]
    h = image.shape[0]
    print(w,h)
    in_image = modify_image(image)
    in_image = np.expand_dims(in_image,0)
    # print(in_image)
    # input("Enter!")
    # Load the graph with the trained states
    loader = tf.train.import_meta_graph(model_path+'.meta')
    loader.restore(sess, model_path)

    # Get the tensors by their variable name
    image_tensor = mask_graph.get_tensor_by_name('input_image_placeholder:0')
    mask = mask_graph.get_tensor_by_name('sigmoid_tensor:0')
    ...
    # Make predictions
    _mask = sess.run(mask, feed_dict={image_tensor: in_image})
    _mask = _mask.reshape(128,128)
    _mask *= 255
    _mask = _mask.astype(np.uint8)
    print(_mask)
    _mask = cv2.resize(_mask,(256,256))
    cv2.imshow('1',_mask)
    cv2.waitKey(0)

    fig, ax = plt.subplots()

    # min_val, max_val = 0.1, 0.7

    # intersection_matrix = np.random.randint(0, 10, size=(max_val, max_val))

    ax.matshow(_mask, cmap=plt.cm.Blues)

    for i in range(128):
        for j in range(128):
            c = _mask[j,i]
            ax.text(i, j, str(c), va='center', ha='center')
    #Testing
    # masking = _mask.reshape(128,128)
    # masking = masking*255
    # cv2.imshow('ori',image)
    # masking = cv2.resize(masking,(w,h))
    # cv2.imshow('1',masking)
    # cv2.waitKey()

    # cv2.imshow(_mask)
    # cv2.waitKey()

    # imgMask = np.zeros((h,w), dtype=np.uint8)
    # imgMask = _mask.astype(int)
    # print(imgMask)

    # _, _mask = cv2.threshold(_mask,250,255,cv2.THRESH_BINARY_INV)
    #
    # print("mask_shape:", _mask.shape)
    # cv2.imshow("show", imgMask)
    # cv2.waitKey(0)
    # show_image_list(_mask)
