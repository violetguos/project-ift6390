from keras.preprocessing import image
from keras.layers.core import Lambda
from keras.models import Sequential
from tensorflow.python.framework import ops
import keras.backend as K
import tensorflow as tf
import numpy as np
import keras
import sys
import cv2
import os
import pickle
import keras
import matplotlib.pyplot as plt

'''
opening comment:
Visualization of the gradients in CNN
adapted from https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py

we load a trained model in .h5 format, and pass an example image through the network
to find the hidden layer weights visualizations

The original implementation can ONLY work with keras pretrained VGG16

The main function takes 1sec_hd pickles and loop through 20 of them,
concatenate all the images

each layer can be visualized, or choose to visualize a particular layer
'''

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(img):
#     img_path = sys.argv[1]
#     img = image.load_img(img_path, target_size=(202, 66))
#     x = image.img_to_array(img)
    x = img
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='conv2d_37'):
    input_img = model.input
    #layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = keras.models.load_model(os.path.join(MODEL_PATH , 'val_acc_87_30_epoches.h5')) #VGG16(weights='imagenet')
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1

    x = x.astype(np.float64)
    x -= x.mean()

    #np.add(x,  x.mean(), out=x, casting="unsafe")
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, image, category_index, layer_name):
#     model = Sequential()
#     model.add(input_model)

#     nb_classes = 2
#     target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
#     model.add(Lambda(target_layer,
#                      output_shape = target_category_loss_output_shape))
    model = input_model

    loss = K.sum(model.layers[-1].output)
    print([l.name for l in model.layers])
    print([l.name for l in model.layers][0] == 'conv2d_37')

    conv_output =  [l for l in model.layers if l.name == layer_name][0].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])


    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (66, 201))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    image = np.stack((image, image, image), axis = -1)
    image = image.reshape((201, 66, 3))

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap


def concatIm(imArr):
    '''
    input: n lists of list of numpy arrays of 202 by 66 by 3 or not 3
    output: 202 by 66n by 3
    '''
    #res = np.ndarray(shape=(imArr[0][0].shape[1], len(imArr) *  imArr[0][0].shape[2], 3))


    res = np.concatenate((imArr), axis = 1)
    if len(res.shape) == 3:
        res = res.reshape(res.shape[1], res.shape[0], res.shape[2])
    else:
        res = res.reshape(res.shape[1], res.shape[0])
    print("res shape in concatIm", res.shape)
    return res


def plotSave(imFinal,figName, output_dir):
    '''
    after aggregating all 1 sec segments
    save it into a bigger plot
    '''
    plt.figure()
    if len(imFinal.shape) == 3:
        plt.imshow( cv2.cvtColor(imFinal, cv2.COLOR_BGR2RGB), aspect="auto")
    else:

        plt.imshow(imFinal, cmap = 'gray', aspect="auto")


    #plt.imshow(finalCam, aspect="auto")
    plt.xticks(np.arange(0, finalCam.shape[0], step=100))
    plt.xlabel('Time (ms)')
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '{}.png'.format(figName)))
    #plt.show()


if __name__ == '__main__':

    DATA_PATH = "./"

    with open(os.path.join(DATA_PATH, "val_1_sec_hd_feature.pickle"), 'rb') as jar:
        x_val = pickle.load(jar)

    with open(os.path.join(DATA_PATH, "val_1_sec_hd_label.pickle"), 'rb') as jar:
        y_val = pickle.load(jar)

    print("Finished loading pickle")
    print(x_val.shape)
    print(y_val.shape)


    MODEL_PATH = './'

    aggreCam = []
    aggreGrad = []

    model = keras.models.load_model(os.path.join(MODEL_PATH , 'val_acc_87_30_epoches.h5'))
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    foo = 1
    while foo:
    #for layerName, layerVal in layer_dict.items():
        foo = 0
        layerName = 'conv2d_37'
        for i in range(2):
            preprocessed_input = load_image(x_val[i])
            predictions = model.predict(preprocessed_input)
            top_1 = np.argmax(predictions, axis = 1)
            print('Predicted class:', top_1)
            print('with probability %.2f' % (top_1))

            predicted_class = top_1[0] #np.argmax(predictions)
            cam, heatmap = grad_cam(model, preprocessed_input, predicted_class, layerName)
            cam_name = "gradcam" + str(i) +".jpg"
            cv2.imwrite(cam_name, cam)
            aggreCam.append(cam)


            register_gradient()
            guided_model = modify_backprop(model, 'GuidedBackProp')
            saliency_fn = compile_saliency_function(guided_model)
            saliency = saliency_fn([preprocessed_input, 0])
            gradcam = saliency[0] * heatmap[..., np.newaxis]
            gradcam_name = "guided_gradcam" + str(i) +".jpg"
            gradcam_img = deprocess_image(gradcam)
            cv2.imwrite(gradcam_name, gradcam_img)
            aggreGrad.append(gradcam_img)

        #plot all 1 sec segments together
        output_dir = './test1'
        finalCam  = concatIm(aggreCam)
        fname = layerName + "cam"
        #out_name = os.path.join(output_dir, fname)
        #cv2.imwrite(out_name, finalCam)
        plotSave(finalCam, fname, output_dir)


        finalGrad = concatIm(aggreGrad)
        fname = layerName + "grad"
        out_name = os.path.join(output_dir, fname)
        plotSave(finalGrad, fname, output_dir)

        #cv2.imwrite(out_name, finalGrad)
