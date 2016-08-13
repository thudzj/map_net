# Standard library imports
import cPickle as pkl
import collections
import os
import random
from shutil import move, rmtree
import sys
import time

# Related third party imports
import lasagne
from lasagne.layers import get_output
import numpy as np
import theano
from theano import tensor as T

# Local layers
from layers import ReSegLayer
from mnist import iterate_minibatches
from padded import BilinearLayer, SmoothLayer, MultiScaleConvLayer

from scipy.misc import imsave
from theano.tensor.nnet.abstract_conv import bilinear_kernel_2D
from math import ceil

def getFunctions(input_var, target_var, l_map, l_pred, num_classes,
                 batch_norm=False, weight_decay=0.,
                 optimizer=lasagne.updates.adadelta,
                 learning_rate=None, momentum=None,
                 rho=None, beta1=None, beta2=None, epsilon=None, smooth = 0):
    '''Helper function to build the training function

    '''
    input_shape = input_var.shape
    # Compute BN params for prediction
    batch_norm_params = dict()
    if batch_norm:
        batch_norm_params.update(
            dict(batch_norm_update_averages=False))
        batch_norm_params.update(
            dict(batch_norm_use_averages=True))

    # Prediction function:
    # computes the deterministic distribution over the labels, i.e. we
    # disable the stochastic layers such as Dropout
    pmap = lasagne.layers.get_output(l_map, deterministic=True,
                                           **batch_norm_params)
    prediction = lasagne.layers.get_output(l_pred, deterministic=True,
                                           **batch_norm_params)
    f_map = theano.function(
        [input_var],
        pmap.reshape(
            (-1, input_shape[2], input_shape[3])))

    test_loss = lasagne.objectives.categorical_crossentropy(prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    f_pred = theano.function([input_var, target_var], [test_loss, test_acc])

    # Compute the loss to be minimized during training
    batch_norm_params = dict()
    if batch_norm:
        batch_norm_params.update(
            dict(batch_norm_update_averages=True))
        batch_norm_params.update(
            dict(batch_norm_use_averages=False))

    prediction = lasagne.layers.get_output(l_pred,
                                           **batch_norm_params)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = T.mean(loss)
    if smooth:
      #l_map_max = lasagne.layers.MaxPool2DLayer(l_map, pool_size=(3, 3), stride=(1,1), pad=(1,1))
      #pmap = lasagne.layers.get_output(l_map,**batch_norm_params)
      #pmap_max = lasagne.layers.get_output(l_map_max,**batch_norm_params)
      #loss += T.mean((pmap - pmap_max) ** 2)
      loss += lasagne.layers.get_output(SmoothLayer(l_map),**batch_norm_params)


    if weight_decay > 0:
        l2_penalty = lasagne.regularization.regularize_network_params(
            l_pred,
            lasagne.regularization.l2,
            tags={'regularizable': True})
        loss += l2_penalty * weight_decay

    params = lasagne.layers.get_all_params(l_pred, trainable=True)

    opt_params = dict()

    if optimizer.__name__ == 'sgd':
        if learning_rate is None:
            raise TypeError("Learning rate can't be 'None' with SGD")
        opt_params = dict(learning_rate=learning_rate)

    elif (optimizer.__name__ == 'momentum' or
          optimizer.__name__ == 'nesterov_momentum'):
        if learning_rate is None:
            raise TypeError("Learning rate can't be 'None' "
                            "with Momentum SGD or Nesterov Momentum")
        opt_params = dict(
            learning_rate=learning_rate,
            momentum=momentum
        )

    elif optimizer.__name__ == 'adagrad':

        if learning_rate is not None:
            opt_params.update(dict(learning_rate=learning_rate))
        if epsilon is not None:
            opt_params.update(dict(epsilon=epsilon))

    elif (optimizer.__name__ == 'rmsprop' or
          optimizer.__name__ == 'adadelta'):

        if learning_rate is not None:
            opt_params.update(dict(learning_rate=learning_rate))
        if rho is not None:
            opt_params.update(dict(rho=rho))
        if epsilon is not None:
            opt_params.update(dict(epsilon=epsilon))

    elif (optimizer.__name__ == 'adam' or
          optimizer.__name__ == 'adamax'):

        if learning_rate is not None:
            opt_params.update(dict(learning_rate=learning_rate))
        if beta1 is not None:
            opt_params.update(dict(beta1=beta1))
        if beta2 is not None:
            opt_params.update(dict(beta2=beta2))
        if epsilon is not None:
            opt_params.update(dict(epsilon=epsilon))

    else:
        raise NotImplementedError('Optimization method not implemented')

    updates = optimizer(loss, params, **opt_params)

    # Training function:
    # computes the training loss (with stochasticity, if any) and
    # updates the weights using the updates dictionary provided by the
    # optimization function
    f_train = theano.function([input_var, target_var],
                              loss, updates=updates)

    return f_pred, f_train, f_map

def build(input_s, num_classes, dim_proj=[32, 32],
          pwidth=2,
          pheight=2,
          stack_sublayers=(True, True),
          RecurrentNet=lasagne.layers.GRULayer,
          nonlinearity=lasagne.nonlinearities.rectify,
          hid_init=lasagne.init.Constant(0.),
          grad_clipping=0,
          precompute_input=True,
          mask_input=None,

          # 1x1 Conv layer for dimensional reduction
          conv_dim_red=False,
          conv_dim_red_nonlinearity=lasagne.nonlinearities.identity,

          # GRU specific params
          gru_resetgate=lasagne.layers.Gate(W_cell=None),
          gru_updategate=lasagne.layers.Gate(W_cell=None),
          gru_hidden_update=lasagne.layers.Gate(
              W_cell=None,
              nonlinearity=lasagne.nonlinearities.tanh),
          gru_hid_init=lasagne.init.Constant(0.),

          # LSTM specific params
          lstm_ingate=lasagne.layers.Gate(),
          lstm_forgetgate=lasagne.layers.Gate(),
          lstm_cell=lasagne.layers.Gate(
              W_cell=None,
              nonlinearity=lasagne.nonlinearities.tanh),
          lstm_outgate=lasagne.layers.Gate(),

          # RNN specific params
          rnn_W_in_to_hid=lasagne.init.Uniform(),
          rnn_W_hid_to_hid=lasagne.init.Uniform(),
          rnn_b=lasagne.init.Constant(0.),

          # Output upsampling layers
          out_upsampling='grad',
          out_nfilters=None,  # The last number should be the num of classes
          out_filters_size=(1, 1),
          out_filters_stride=None,
          out_W_init=lasagne.init.GlorotUniform(),
          out_b_init=lasagne.init.Constant(0.),
          out_nonlinearity=lasagne.nonlinearities.rectify,
          out_pad=1,

          # Special layers
          batch_norm=False,

          # Optimization method
          optimizer=lasagne.updates.adadelta,
          learning_rate=None,
          momentum=None,
          rho=None,
          beta1=None,
          beta2=None,
          epsilon=None,
          weight_decay=0.,  # l2 reg
          smooth=0
          ):

    rng = np.random.RandomState(0xbeef)
    if type(pwidth) != list:
        pwidth = [pwidth] * len(dim_proj)
    if type(pheight) != list:
        pheight = [pheight] * len(dim_proj)

    print("Building model ...")

    input_shape = (None, input_s[0], input_s[1], input_s[2])
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Set the RandomStream to assure repeatability
    lasagne.random.set_rng(rng)
    n_layers = len(dim_proj)

    print('Input shape: ' + str(input_shape))
    l_in = lasagne.layers.InputLayer(shape=input_shape,
                                     input_var=input_var,
                                     name="input_layer")

    # Convert to bc01 (batchsize, ch, rows, cols)
    #l_in = lasagne.layers.DimshuffleLayer(l_in, (0, 3, 1, 2))
    # l_conv_1 = lasagne.layers.Conv2DLayer(
    #         l_in, num_filters=12, filter_size=(3, 3),stride=(1, 1),pad='same',
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         W=lasagne.init.GlorotUniform())
    # l_conv_2 = lasagne.layers.Conv2DLayer(
    #         l_in, num_filters=12, filter_size=(7, 7),stride=(1, 1),pad='same',
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         W=lasagne.init.GlorotUniform())
    # l_conv_3 = lasagne.layers.Conv2DLayer(
    #         l_in, num_filters=12, filter_size=(11, 11),stride=(1, 1),pad='same',
    #         nonlinearity=lasagne.nonlinearities.rectify,
    #         W=lasagne.init.GlorotUniform())

    # l_multiconv_1 = MultiScaleConvLayer(l_conv_1, num_filters=12, filter_sizes=[(3,3), (7,7), (11,11)])
    # l_multiconv_2 = MultiScaleConvLayer(l_conv_2, num_filters=12, filter_sizes=[(3,3), (7,7), (11,11)])
    # l_multiconv_3 = MultiScaleConvLayer(l_conv_3, num_filters=12, filter_sizes=[(3,3), (7,7), (11,11)])
    # l_concat = lasagne.layers.ConcatLayer([l_multiconv_1,l_multiconv_2,l_multiconv_3])

    l_multiconv_1 = MultiScaleConvLayer(l_in, num_filters=12, filter_sizes=[(3,3), (7,7), (11,11)])
    l_multiconv_2 = MultiScaleConvLayer(l_multiconv_1, num_filters=12, filter_sizes=[(3,3), (7,7), (11,11)])
    l_map = lasagne.layers.Conv2DLayer(l_multiconv_2,num_filters=11,filter_size=(3, 3),stride=(1, 1),pad='same',nonlinearity=lasagne.nonlinearities.rectify)
    l_map = lasagne.layers.Conv2DLayer(l_map,num_filters=1,filter_size=(3, 3),stride=(1, 1),pad='same',nonlinearity=lasagne.nonlinearities.sigmoid)

    # l_map = BilinearLayer(l_map, 2)
    # # channel = nclassesdd_in
    # l_map = lasagne.layers.Conv2DLayer(
    #     l_map,
    #     num_filters=1,
    #     filter_size=(1, 1),
    #     stride=(1, 1),
    #     W=out_W_init,
    #     b=out_b_init,
    #     nonlinearity=lasagne.nonlinearities.sigmoid
    # )
    # if batch_norm:
    #     l_map = lasagne.layers.batch_norm(l_map, axes='auto')

    l_com = lasagne.layers.ElemwiseMergeLayer([l_in, l_map], T.mul)

    network = lasagne.layers.Conv2DLayer(
            l_com, num_filters=12, filter_size=(7, 7), stride=(2,2),# pad = 3,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=24, filter_size=(3, 3), stride=(2,2),# pad = 1,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=96, filter_size=(4, 4),stride=(2, 2),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=0.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    print "Compiling functions"
    f_pred, f_train, f_map = getFunctions(input_var = input_var, target_var = target_var, 
                                   l_map = l_map, l_pred = l_out, num_classes = num_classes,
                                   weight_decay = weight_decay, optimizer=optimizer,
                                   learning_rate=learning_rate,
                                   momentum=momentum, rho=rho, beta1=beta1,
                                   beta2=beta2, epsilon=epsilon, smooth = smooth)
    return f_pred, f_train, f_map

def train(params):
    f_pred, f_train, f_map = build(input_s = params['input-s'], 
          num_classes=params['num_classes'],
          dim_proj=params['dim-proj'],
          pwidth=params['pwidth'],
          pheight=params['pheight'],
          stack_sublayers=params['stack-sublayers'],
          RecurrentNet=params['RecurrentNet'],
          nonlinearity=params['nonlinearity'],
          hid_init=params['hid-init'],
          grad_clipping=params['grad-clipping'],
          precompute_input=params['precompute-input'],
          mask_input=params['mask-input'],

          # GRU specific params
          gru_resetgate=params['gru-resetgate'],
          gru_updategate=params['gru-updategate'],
          gru_hidden_update=params['gru-hidden-update'],
          gru_hid_init=params['gru-hid-init'],
          # LSTM specific params
          lstm_ingate=params['lstm-ingate'],
          lstm_forgetgate=params['lstm-forgetgate'],
          lstm_cell=params['lstm-cell'],
          lstm_outgate=params['lstm-outgate'],

          # RNN specific params
          rnn_W_in_to_hid=params['rnn-W-in-to-hid'],
          rnn_W_hid_to_hid=params['rnn-W-hid-to-hid'],
          rnn_b=params['rnn-b'],

          # Output upsampling layers
          out_upsampling=params['out-upsampling'],
          out_nfilters=params['out-nfilters'],
          out_filters_size=params['out-filters-size'],
          out_filters_stride=params['out-filters-stride'],
          out_W_init=params['out-W-init'],
          out_b_init=params['out-b-init'],
          out_nonlinearity=params['out-nonlinearity'],
          out_pad=params['out-pad'],

          # Special layers
          batch_norm=params['batch-norm'],

          # Optimization method
          optimizer=params['optimizer'],
          learning_rate=params['learning-rate'],
          momentum=params['momentum'],
          rho=params['rho'],
          beta1=params['beta1'],
          beta2=params['beta2'],
          epsilon=params['epsilon'],
          weight_decay=params['weight-decay'],
          smooth = params['smooth']
          )
    
    print("Loading data...")
    npz_file = np.load("mnist-cluttered-master/train.npz")
    X_train = npz_file['arr_0']
    y_train = npz_file['arr_1']
    y_train = np.array(y_train, dtype=np.int32)
    print(X_train.shape, y_train.shape)

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    npz_file = np.load("mnist-cluttered-master/test.npz")
    X_test = npz_file['arr_0']
    y_test = npz_file['arr_1']
    y_test = np.array(y_test, dtype=np.int32)

    print(X_test.shape, y_test.shape)

    img_id = 1
    for batch in iterate_minibatches(X_val, y_val, params['valid-batch-size'], shuffle=False):
      inputs, targets = batch
      maps = f_map(inputs[:10])
      for ma, inp in zip(maps,inputs[:10]):
          imsave(params['pre'] + str(img_id) + "_map.png", ma)
          imsave(params['pre'] + str(img_id) + ".png", inp.reshape((params['input-s'][1], params['input-s'][2])))
          img_id += 1
      break

    print "Starting training"
    for epoch in range(params['num_epochs']):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, params['batch-size'], shuffle=True):
            inputs, targets = batch
            tmp_err = f_train(inputs, targets)
            train_err += tmp_err
            train_batches += 1
            if train_batches % 50 == 0: 
                print(".epoch {} batch {} loss {:.6f}".format(epoch + 1, train_batches, float(tmp_err)))

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        if epoch % params['valid-frequence'] == 0:
          for batch in iterate_minibatches(X_val, y_val, params['valid-batch-size'], shuffle=True):
              inputs, targets = batch
              if val_batches == 0:
                maps = f_map(inputs[:10])
                for ma, inp in zip(maps,inputs[:10]):
                  imsave(params['pre'] + str(img_id) + "_map.png", ma)
                  imsave(params['pre'] + str(img_id) + ".png", inp.reshape((params['input-s'][1], params['input-s'][2])))
                  img_id += 1
              err, acc = f_pred(inputs, targets)
              val_err += err
              val_acc += acc
              val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, params['num_epochs'], time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        if epoch % params['valid-frequence'] == 0:
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, params['valid-batch-size'], shuffle=False):
        inputs, targets = batch
        err, acc = f_pred(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))


if __name__ == '__main__':
    # width = 4
    # heigh = 4
    # f = ceil(width/2.0)
    # c = (2 * f - 1 - f % 2) / (2.0 * f)
    # bilinear = np.zeros([width, heigh], dtype=np.float32)
    # for x in range(width):
    #     for y in range(heigh):
    #         value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
    #         bilinear[x, y] = value
    # print bilinear
    # out_w1 = np.zeros((256, 32, width, heigh), dtype=np.float32)
    # out_w2 = np.zeros((32, 1, width, heigh), dtype = np.float32)
    # for i in range(256):
    #   for j in range(32):
    #     out_w1[i, j, :, :] = bilinear
    # for i in range(32):
    #   for j in range(1):
    #     out_w2[i, j, :, :] = bilinear
    train({
        'num_epochs': 200,
        'input-s': [1, 100, 100],
        'num_classes': 10, 
        # RNNs layers
        'dim-proj': [128, 128],
        'pwidth': [2, 2],
        'pheight': [2, 2],
        'stack-sublayers': (True, True),
        'RecurrentNet': lasagne.layers.GRULayer,
        'nonlinearity': lasagne.nonlinearities.rectify,
        'hid-init': lasagne.init.Constant(0.),
        'grad-clipping': 0,
        'precompute-input': True,
        'mask-input': None,

        # GRU specific params
        'gru-resetgate': lasagne.layers.Gate(W_cell=None),
        'gru-updategate': lasagne.layers.Gate(W_cell=None),
        'gru-hidden-update': lasagne.layers.Gate(
            W_cell=None,
            nonlinearity=lasagne.nonlinearities.tanh),
        'gru-hid-init': lasagne.init.Constant(0.),

        # LSTM specific params
        'lstm-ingate': lasagne.layers.Gate(),
        'lstm-forgetgate': lasagne.layers.Gate(),
        'lstm-cell': lasagne.layers.Gate(
            W_cell=None,
            nonlinearity=lasagne.nonlinearities.tanh),
        'lstm-outgate': lasagne.layers.Gate(),

        # RNN specific params
        'rnn-W-in-to-hid': lasagne.init.Uniform(),
        'rnn-W-hid-to-hid': lasagne.init.Uniform(),
        'rnn-b': lasagne.init.Constant(0.),

        # Output upsampling layers
        'out-upsampling': 'grad',
        'out-nfilters': [32, 1],#[50, 50], [32, 8], 
        'out-filters-size': [(2, 2), (2, 2)],
        'out-filters-stride': [(2, 2), (2, 2)],
        'out-W-init': [lasagne.init.GlorotUniform(), lasagne.init.GlorotUniform()], #out_w
        'out-b-init': lasagne.init.Constant(0.),
        'out-nonlinearity': [lasagne.nonlinearities.rectify, lasagne.nonlinearities.sigmoid], #
        'out-pad': [0, 0],

        # Special layers
        'batch-norm': False,

        # Optimization method
        'optimizer': lasagne.updates.adadelta,
        'learning-rate': None,
        'momentum': None,
        'rho': None,
        'beta1': None,
        'beta2': None,
        'epsilon': None,
        'weight-decay': 1e-4,  # l2 reg

        'batch-size': 250,
        'valid-batch-size': 500,
        'shuffle': True,
        'valid-frequence': 1,
        'smooth': 0,
        'pre': "samples4/"
        }
    )
