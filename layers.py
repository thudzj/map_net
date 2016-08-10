from collections import Iterable

import numpy as np
import lasagne
from lasagne.layers import get_output, get_output_shape
from lasagne.layers.conv import TransposedConv2DLayer
import theano.tensor as T

from padded import DynamicPaddingLayer


def ceildiv(a, b):
    """Division rounded up

    Parameters
    ----------
    a : number
        The numerator
    b : number
        The denominator

    Reference
    ---------
    http://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent\
-of-operator-in-python
    """
    return -(-a // b)


class ReSegLayer(lasagne.layers.Layer):
    def __init__(self,
                 l_in,
                 n_layers,
                 pheight,
                 pwidth,
                 dim_proj,
                 stack_sublayers,
                 # outsampling
                 out_upsampling_type,
                 out_nfilters,
                 out_filters_size,
                 out_filters_stride,
                 out_W_init=lasagne.init.GlorotUniform(),
                 out_b_init=lasagne.init.Constant(0.),
                 out_nonlinearity=lasagne.nonlinearities.identity,
                 # common recurrent layer params
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
                 # Special layers
                 batch_norm=False,
                 name=''):
        """A ReSeg layer

        The ReSeg layer is composed by multiple ReNet layers and an
        upsampling layer

        Parameters
        ----------
        l_in : lasagne.layers.Layer
            The input layer, in bc01 format
        n_layers : int
            The number of layers
        pheight : tuple
            The height of the patches, for each layer
        pwidth : tuple
            The width of the patches, for each layer
        dim_proj : tuple
            The number of hidden units of each RNN, for each layer
        nclasses : int
            The number of classes of the data
        stack_sublayers : bool
            If True the bidirectional RNNs in the ReNet layers will be
            stacked one over the other. See ReNet for more details.
        out_upsampling_type : string
            The kind of upsampling to be used
        out_nfilters : int
            The number of hidden units of the upsampling layer
        out_filters_size : tuple
            The size of the upsampling filters, if any
        out_filters_stride : tuple
            The stride of the upsampling filters, if any
        out_W_init : Theano shared variable, numpy array or callable
            Initializer for W
        out_b_init : Theano shared variable, numpy array or callable
            Initializer for b
        out_nonlinearity : Theano shared variable, numpy array or callable
            The nonlinearity to be applied after the upsampling
        hypotetical_fm_size : float
            The hypotetical size of the feature map that would be input
            of the layer if the input image of the whole network was of
            size (100, 100)
        RecurrentNet : lasagne.layers.Layer
            A recurrent layer class
        nonlinearity : callable or None
            The nonlinearity that is applied to the output. If
            None is provided, no nonlinearity will be applied.
        hid_init : callable, np.ndarray, theano.shared or
                   lasagne.layers.Layer
            Initializer for initial hidden state
        grad_clipping : float
            If nonzero, the gradient messages are clipped to the given value
            during the backward pass.
        precompute_input : bool
            If True, precompute input_to_hid before iterating through the
            sequence. This can result in a speedup at the expense of an
            increase in memory usage.
        mask_input : lasagne.layers.Layer
            Layer which allows for a sequence mask to be input, for when
            sequences are of variable length. Default None, which means no mask
            will be supplied (i.e. all sequences are of the same length).
        gru_resetgate : lasagne.layers.Gate
            Parameters for the reset gate, if RecurrentNet is GRU
        gru_updategate : lasagne.layers.Gate
            Parameters for the update gate, if RecurrentNet is GRU
        gru_hidden_update : lasagne.layers.Gate
            Parameters for the hidden update, if RecurrentNet is GRU
        gru_hid_init : callable, np.ndarray, theano.shared or
                       lasagne.layers.Layer
            Initializer for initial hidden state, if RecurrentNet is GRU
        lstm_ingate : lasagne.layers.Gate
            Parameters for the input gate, if RecurrentNet is LSTM
        lstm_forgetgate : lasagne.layers.Gate
            Parameters for the forget gate, if RecurrentNet is LSTM
        lstm_cell : lasagne.layers.Gate
            Parameters for the cell computation, if RecurrentNet is LSTM
        lstm_outgate : lasagne.layers.Gate
            Parameters for the output gate, if RecurrentNet is LSTM
        rnn_W_in_to_hid : Theano shared variable, numpy array or callable
            Initializer for input-to-hidden weight matrix, if
            RecurrentNet is RecurrentLayer
        rnn_W_hid_to_hid : Theano shared variable, numpy array or callable
            Initializer for hidden-to-hidden weight matrix, if
            RecurrentNet is RecurrentLayer
        rnn_b : Theano shared variable, numpy array, callable or None
            Initializer for bias vector, if RecurrentNet is
            RecurrentLaye. If None is provided there will be no bias
        batch_norm: this add a batch normalization layer at the end of the
            network right after each Gradient Upsampling layers
        name : string
            The name of the layer, optional
        """

        super(ReSegLayer, self).__init__(l_in, name)
        self.l_in = l_in
        self.n_layers = n_layers
        self.pheight = pheight
        self.pwidth = pwidth
        self.dim_proj = dim_proj
        self.stack_sublayers = stack_sublayers
        # upsampling
        self.out_upsampling_type = out_upsampling_type
        self.out_nfilters = out_nfilters
        self.out_filters_size = out_filters_size
        self.out_filters_stride = out_filters_stride
        self.out_W_init = out_W_init
        self.out_b_init = out_b_init
        self.out_nonlinearity = out_nonlinearity
        # common recurrent layer params
        self.RecurrentNet = RecurrentNet
        self.nonlinearity = nonlinearity
        self.hid_init = hid_init
        self.grad_clipping = grad_clipping
        self.precompute_input = precompute_input
        self.mask_input = mask_input
        # GRU specific params
        self.gru_resetgate = gru_resetgate
        self.gru_updategate = gru_updategate
        self.gru_hidden_update = gru_hidden_update
        self.gru_hid_init = gru_hid_init
        # LSTM specific params
        self.lstm_ingate = lstm_ingate
        self.lstm_forgetgate = lstm_forgetgate
        self.lstm_cell = lstm_cell
        self.lstm_outgate = lstm_outgate
        # RNN specific params
        self.rnn_W_in_to_hid = rnn_W_in_to_hid
        self.rnn_W_hid_to_hid = rnn_W_hid_to_hid
        self.name = name
        self.sublayers = []

        # ReNet layers
        l_renet = l_in
        for lidx in xrange(n_layers):
            l_renet = ReNetLayer(l_renet,
                                 patch_size=(pwidth[lidx], pheight[lidx]),
                                 n_hidden=dim_proj[lidx],
                                 stack_sublayers=stack_sublayers[lidx],
                                 RecurrentNet=RecurrentNet,
                                 nonlinearity=nonlinearity,
                                 hid_init=hid_init,
                                 grad_clipping=grad_clipping,
                                 precompute_input=precompute_input,
                                 mask_input=mask_input,
                                 # GRU specific params
                                 gru_resetgate=gru_resetgate,
                                 gru_updategate=gru_updategate,
                                 gru_hidden_update=gru_hidden_update,
                                 gru_hid_init=gru_hid_init,
                                 # LSTM specific params
                                 lstm_ingate=lstm_ingate,
                                 lstm_forgetgate=lstm_forgetgate,
                                 lstm_cell=lstm_cell,
                                 lstm_outgate=lstm_outgate,
                                 # RNN specific params
                                 rnn_W_in_to_hid=rnn_W_in_to_hid,
                                 rnn_W_hid_to_hid=rnn_W_hid_to_hid,
                                 rnn_b=rnn_b,
                                 batch_norm=batch_norm,
                                 name=self.name + '_renet' + str(lidx))
            self.sublayers.append(l_renet)

            # Print shape
            out_shape = get_output_shape(l_renet)
            if stack_sublayers:
                msg = 'ReNet: After 2 rnns {}x{}@{} and 2 rnns 1x1@{}: {}'
                print(msg.format(pheight[lidx], pwidth[lidx], dim_proj[lidx],
                                 dim_proj[lidx], out_shape))
            else:
                print('ReNet: After 4 rnns {}x{}@{}: {}'.format(
                    pheight[lidx], pwidth[lidx], dim_proj[lidx], out_shape))

            # 1x1 conv layer : dimensionality reduction layer
            if conv_dim_red:
                l_renet = lasagne.layers.Conv2DLayer(
                    l_renet,
                    num_filters=dim_proj[lidx],
                    filter_size=(1, 1),
                    W=lasagne.init.GlorotUniform(),
                    b=lasagne.init.Constant(0.),
                    pad='valid',
                    nonlinearity=conv_dim_red_nonlinearity,
                    name=self.name + '_1x1_conv_layer' + str(lidx)
                )

                # Print shape
                out_shape = get_output_shape(l_renet)
                print('Dim reduction: After 1x1 convnet: {}'.format(out_shape))

        # Upsampling
        if out_upsampling_type == 'autograd':
            pass

        elif out_upsampling_type == 'grad':
            l_upsampling = l_renet
            for i, (nf, f_size, stride) in enumerate(zip(
                    out_nfilters, out_filters_size, out_filters_stride)):
                l_upsampling = TransposedConv2DLayer(
                    l_upsampling,
                    num_filters=nf,
                    filter_size=f_size,
                    stride=stride,
                    crop=0,
                    W=out_W_init,
                    b=out_b_init,
                    nonlinearity=out_nonlinearity)
                self.sublayers.append(l_upsampling)

                if batch_norm:
                    l_upsampling = lasagne.layers.batch_norm(
                        l_upsampling,
                        axes='auto')
                    self.sublayers.append(l_upsampling)
                    print "Batch normalization after Grad layer "

                # Print shape
                out_shape = get_output_shape(l_upsampling)
                print('Transposed conv: {}x{} (str {}x{}) @ {}:{}'.format(
                    f_size[0], f_size[1], stride[0], stride[1], nf, out_shape))

        elif out_upsampling_type == 'linear':
            # Go to b01c
            l_upsampling = lasagne.layers.DimshuffleLayer(
                l_renet,
                (0, 2, 3, 1),
                name=self.name + '_grad_undimshuffle')
            self.sublayers.append(l_upsampling)

            expand_height = np.prod(pheight)
            expand_width = np.prod(pwidth)
            l_upsampling = LinearUpsamplingLayer(l_upsampling,
                                                 expand_height,
                                                 expand_width,
                                                 1,
                                                 batch_norm=batch_norm,
                                                 name="linear_upsample_layer")
            self.sublayers.append(l_upsampling)
            print('Linear upsampling')

            if batch_norm:
                l_upsampling = lasagne.layers.batch_norm(
                    l_upsampling,
                    axes=(0, 1, 2))

                self.sublayers.append(l_upsampling)
                print "Batch normalization after Linear upsampling layer "

            # Go back to bc01
            l_upsampling = lasagne.layers.DimshuffleLayer(
                l_upsampling,
                (0, 3, 1, 2),
                name=self.name + '_grad_undimshuffle')
            self.sublayers.append(l_upsampling)

        self.l_out = l_upsampling

        # HACK LASAGNE
        # This will set `self.input_layer`, which is needed by Lasagne to find
        # the layers with the get_all_layers() helper function in the
        # case of a layer with sublayers
        if isinstance(self.l_out, tuple):
            self.input_layer = None
        else:
            self.input_layer = self.l_out

    def get_output_shape_for(self, input_shape):
        for layer in self.sublayers:
            output_shape = layer.get_output_shape_for(input_shape)
            input_shape = output_shape

        return output_shape
        # return self.l_out.get_output_shape_for(input_shape)
        # return list(input_shape[0:3]) + [self.nclasses]

    def get_output_for(self, input_var, **kwargs):
        # HACK LASAGNE
        # This is needed, jointly with the previous hack, to ensure that
        # this layer behaves as its last sublayer (namely,
        # self.input_layer)
        return input_var


class ReNetLayer(lasagne.layers.Layer):

    def __init__(self,
                 l_in,
                 patch_size=(2, 2),
                 n_hidden=50,
                 stack_sublayers=False,
                 RecurrentNet=lasagne.layers.GRULayer,
                 nonlinearity=lasagne.nonlinearities.rectify,
                 hid_init=lasagne.init.Constant(0.),
                 grad_clipping=0,
                 precompute_input=True,
                 mask_input=None,
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
                 batch_norm=False,
                 name='', **kwargs):
        """A ReNet layer

        Each ReNet layer is composed by 4 RNNs (or 2 bidirectional RNNs):
        * First SubLayer:
            2 RNNs scan the image vertically (up and down)
        * Second Sublayer:
            2 RNNs scan the image horizontally (left and right)

        The sublayers can be stacked one over the other or can scan the
        image in parallel

        Parameters
        ----------
        l_in : lasagne.layers.Layer
            The input layer, in format batches, channels, rows, cols
        patch_size : tuple
            The size of the patch expressed as (pheight, pwidth).
            Optional
        n_hidden : int
            The number of hidden units of each RNN. Optional
        stack_sublayers : bool
            If True, the sublayers (i.e. the bidirectional RNNs) will be
            stacked one over the other, meaning that the second
            bidirectional RNN will read the feature map coming from the
            first bidirectional RNN. If False, all the RNNs will read
            the input. Optional
        RecurrentNet : lasagne.layers.Layer
            A recurrent layer class
        nonlinearity : callable or None
            The nonlinearity that is applied to the output. If
            None is provided, no nonlinearity will be applied.
        hid_init : callable, np.ndarray, theano.shared or
                   lasagne.layers.Layer
            Initializer for initial hidden state
        grad_clipping : float
            If nonzero, the gradient messages are clipped to the given value
            during the backward pass.
        precompute_input : bool
            If True, precompute input_to_hid before iterating through the
            sequence. This can result in a speedup at the expense of an
            increase in memory usage.
        mask_input : lasagne.layers.Layer
            Layer which allows for a sequence mask to be input, for when
            sequences are of variable length. Default None, which means no mask
            will be supplied (i.e. all sequences are of the same length).
        gru_resetgate : lasagne.layers.Gate
            Parameters for the reset gate, if RecurrentNet is GRU
        gru_updategate : lasagne.layers.Gate
            Parameters for the update gate, if RecurrentNet is GRU
        gru_hidden_update : lasagne.layers.Gate
            Parameters for the hidden update, if RecurrentNet is GRU
        gru_hid_init : callable, np.ndarray, theano.shared or
                       lasagne.layers.Layer
            Initializer for initial hidden state, if RecurrentNet is GRU
        lstm_ingate : lasagne.layers.Gate
            Parameters for the input gate, if RecurrentNet is LSTM
        lstm_forgetgate : lasagne.layers.Gate
            Parameters for the forget gate, if RecurrentNet is LSTM
        lstm_cell : lasagne.layers.Gate
            Parameters for the cell computation, if RecurrentNet is LSTM
        lstm_outgate : lasagne.layers.Gate
            Parameters for the output gate, if RecurrentNet is LSTM
        rnn_W_in_to_hid : Theano shared variable, numpy array or callable
            Initializer for input-to-hidden weight matrix, if
            RecurrentNet is RecurrentLayer
        rnn_W_hid_to_hid : Theano shared variable, numpy array or callable
            Initializer for hidden-to-hidden weight matrix, if
            RecurrentNet is RecurrentLayer
        rnn_b : Theano shared variable, numpy array, callable or None
            Initializer for bias vector, if RecurrentNet is
            RecurrentLaye. If None is provided there will be no bias
        name : string
            The name of the layer, optional
        """
        super(ReNetLayer, self).__init__(l_in, name)
        self.l_in = l_in
        self.patch_size = patch_size
        self.n_hidden = n_hidden
        self.stack_sublayers = stack_sublayers
        self.name = name
        self.stride = self.patch_size  # for now, it's not parametrized

        # Dynamically add padding if the input is not a multiple of the
        # patch size (expected input format: bs, ch, rows, cols)
        l_in = DynamicPaddingLayer(l_in, patch_size, self.stride,
                                   name=self.name + '_padding')

        # get_output(l_in).shape will result in an error in the
        # recurrent layers
        batch_size = -1
        cchannels, cheight, cwidth = get_output_shape(l_in)[1:]
        pheight, pwidth = patch_size
        psize = pheight * pwidth * cchannels

        # Number of patches in each direction
        npatchesH = cheight / pheight
        npatchesW = cwidth / pwidth

        # Split in patches: bs, cc, #H, ph, #W, pw
        l_in = lasagne.layers.ReshapeLayer(
            l_in,
            (batch_size, cchannels, npatchesH, pheight, npatchesW, pwidth),
            name=self.name + "_pre_reshape0")

        # bs, #H, #W, ph, pw, cc
        l_in = lasagne.layers.DimshuffleLayer(
            l_in,
            (0, 2, 4, 3, 5, 1),
            name=self.name + "_pre_dimshuffle0")

        # FIRST SUBLAYER
        # The RNN Layer needs a 3D tensor input: bs*#H, #W, psize
        # bs*#H, #W, ph * pw * cc
        l_sub0 = lasagne.layers.ReshapeLayer(
            l_in,
            (-1, npatchesW, psize),
            name=self.name + "_sub0_reshape0")

        # Left/right scan: bs*#H, #W, 2*hid
        l_sub0 = BidirectionalRNNLayer(
            l_sub0,
            n_hidden,
            RecurrentNet=RecurrentNet,
            nonlinearity=nonlinearity,
            hid_init=hid_init,
            grad_clipping=grad_clipping,
            precompute_input=precompute_input,
            mask_input=mask_input,
            # GRU specific params
            gru_resetgate=gru_resetgate,
            gru_updategate=gru_updategate,
            gru_hidden_update=gru_hidden_update,
            gru_hid_init=gru_hid_init,
            batch_norm=batch_norm,
            # LSTM specific params
            lstm_ingate=lstm_ingate,
            lstm_forgetgate=lstm_forgetgate,
            lstm_cell=lstm_cell,
            lstm_outgate=lstm_outgate,
            # RNN specific params
            rnn_W_in_to_hid=rnn_W_in_to_hid,
            rnn_W_hid_to_hid=rnn_W_hid_to_hid,
            rnn_b=rnn_b,
            name=self.name + "_sub0_renetsub")

        # Revert reshape: bs, #H, #W, 2*hid
        l_sub0 = lasagne.layers.ReshapeLayer(
            l_sub0,
            (batch_size, npatchesH, npatchesW, 2 * n_hidden),
            name=self.name + "_sub0_unreshape")

        # # Invert rows and columns: #H, bs, #W, 2*hid
        # l_sub0 = lasagne.layers.DimshuffleLayer(
        #     l_sub0,
        #     (2, 1, 0, 3),
        #     name=self.name + "_sub0_undimshuffle")

        # If stack_sublayers is True, the second sublayer takes as an input the
        # first sublayer's output, otherwise the input of the ReNetLayer (e.g
        # the image)
        if stack_sublayers:
            # bs, #H, #W, 2*hid
            input_sublayer1 = l_sub0
            psize = 2 * n_hidden
        else:
            #     # #H, bs, #W, ph, pw, cc
            #     input_sublayer1 = lasagne.layers.DimshuffleLayer(
            #         l_in,
            #         (2, 1, 0, 3, 4, 5),
            #         name=self.name + "_presub1_in_dimshuffle")
            # bs, #H, #W, ph*pw*cc
            input_sublayer1 = lasagne.layers.ReshapeLayer(
                l_in,
                (batch_size, npatchesH, npatchesW, psize),
                name=self.name + "_presub1_in_dimshuffle")

        # SECOND SUBLAYER
        # Invert rows and columns: bs, #W, #H, psize
        l_sub1 = lasagne.layers.DimshuffleLayer(
            input_sublayer1,
            (0, 2, 1, 3),
            name=self.name + "_presub1_dimshuffle")

        # The RNN Layer needs a 3D tensor input: bs*#W, #H, psize
        l_sub1 = lasagne.layers.ReshapeLayer(
            l_sub1,
            (-1, npatchesH, psize),
            name=self.name + "_sub1_reshape")

        # Down/up scan: bs*#W, #H, 2*hid
        l_sub1 = BidirectionalRNNLayer(
            l_sub1,
            n_hidden,
            RecurrentNet=RecurrentNet,
            nonlinearity=nonlinearity,
            hid_init=hid_init,
            grad_clipping=grad_clipping,
            precompute_input=precompute_input,
            mask_input=mask_input,
            # GRU specific params
            gru_resetgate=gru_resetgate,
            gru_updategate=gru_updategate,
            gru_hidden_update=gru_hidden_update,
            gru_hid_init=gru_hid_init,
            # LSTM specific params
            lstm_ingate=lstm_ingate,
            lstm_forgetgate=lstm_forgetgate,
            lstm_cell=lstm_cell,
            lstm_outgate=lstm_outgate,
            # RNN specific params
            rnn_W_in_to_hid=rnn_W_in_to_hid,
            rnn_W_hid_to_hid=rnn_W_hid_to_hid,
            rnn_b=rnn_b,
            name=self.name + "_sub1_renetsub")
        psize = 2 * n_hidden

        # Revert the reshape: bs, #W, #H, 2*hid
        l_sub1 = lasagne.layers.ReshapeLayer(
            l_sub1,
            (batch_size, npatchesW, npatchesH, psize),
            name=self.name + "_sub1_unreshape")

        # Invert rows and columns: bs, #H, #W, psize
        l_sub1 = lasagne.layers.DimshuffleLayer(
            l_sub1,
            (0, 2, 1, 3),
            name=self.name + "_sub1_undimshuffle")

        # Concat all 4 layers if needed: bs, #H, #W, {2,4}*hid
        if not stack_sublayers:
            l_sub1 = lasagne.layers.ConcatLayer([l_sub0, l_sub1], axis=3)

        # Get back to bc01: bs, psize, #H, #W
        self.out_layer = lasagne.layers.DimshuffleLayer(
            l_sub1,
            (0, 3, 1, 2),
            name=self.name + "_out_undimshuffle")

        # HACK LASAGNE
        # This will set `self.input_layer`, which is needed by Lasagne to find
        # the layers with the get_all_layers() helper function in the
        # case of a layer with sublayers
        if isinstance(self.out_layer, tuple):
            self.input_layer = None
        else:
            self.input_layer = self.out_layer

    def get_output_shape_for(self, input_shape):
        pheight, pwidth = self.patch_size
        npatchesH = ceildiv(input_shape[2], pheight)
        npatchesW = ceildiv(input_shape[3], pwidth)

        if self.stack_sublayers:
            dim = 2 * self.n_hidden
        else:
            dim = 4 * self.n_hidden

        return input_shape[0], dim, npatchesH, npatchesW

    def get_output_for(self, input_var, **kwargs):
        # HACK LASAGNE
        # This is needed, jointly with the previous hack, to ensure that
        # this layer behaves as its last sublayer (namely,
        # self.input_layer)
        return input_var


class BidirectionalRNNLayer(lasagne.layers.Layer):

    # Setting a value for grad_clipping will clip the gradients in the layer
    def __init__(
            self,
            l_in,
            num_units,
            RecurrentNet=lasagne.layers.GRULayer,
            # common parameters
            nonlinearity=lasagne.nonlinearities.rectify,
            hid_init=lasagne.init.Constant(0.),
            grad_clipping=0,
            precompute_input=True,
            mask_input=None,
            # GRU specific params
            gru_resetgate=lasagne.layers.Gate(W_cell=None),
            gru_updategate=lasagne.layers.Gate(W_cell=None),
            gru_hidden_update=lasagne.layers.Gate(
                W_cell=None,
                nonlinearity=lasagne.nonlinearities.tanh),
            gru_hid_init=lasagne.init.Constant(0.),
            batch_norm=False,
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
            name='',
            **kwargs):
        """A Bidirectional RNN Layer

        Parameters
        ----------
        l_in : lasagne.layers.Layer
            The input layer
        num_units : int
            The number of hidden units of each RNN
        RecurrentNet : lasagne.layers.Layer
            A recurrent layer class
        nonlinearity : callable or None
            The nonlinearity that is applied to the output. If
            None is provided, no nonlinearity will be applied. Only for
            LSTMLayer and RecurrentLayer
        hid_init : callable, np.ndarray, theano.shared or
                   lasagne.layers.Layer
            Initializer for initial hidden state
        grad_clipping : float
            If nonzero, the gradient messages are clipped to the given value
            during the backward pass.
        precompute_input : bool
            If True, precompute input_to_hid before iterating through the
            sequence. This can result in a speedup at the expense of an
            increase in memory usage.
        mask_input : lasagne.layers.Layer
            Layer which allows for a sequence mask to be input, for when
            sequences are of variable length. Default None, which means no mask
            will be supplied (i.e. all sequences are of the same length).
        gru_resetgate : lasagne.layers.Gate
            Parameters for the reset gate, if RecurrentNet is GRU
        gru_updategate : lasagne.layers.Gate
            Parameters for the update gate, if RecurrentNet is GRU
        gru_hidden_update : lasagne.layers.Gate
            Parameters for the hidden update, if RecurrentNet is GRU
        gru_hid_init : callable, np.ndarray, theano.shared or
                       lasagne.layers.Layer
            Initializer for initial hidden state, if RecurrentNet is GRU
        lstm_ingate : lasagne.layers.Gate
            Parameters for the input gate, if RecurrentNet is LSTM
        lstm_forgetgate : lasagne.layers.Gate
            Parameters for the forget gate, if RecurrentNet is LSTM
        lstm_cell : lasagne.layers.Gate
            Parameters for the cell computation, if RecurrentNet is LSTM
        lstm_outgate : lasagne.layers.Gate
            Parameters for the output gate, if RecurrentNet is LSTM
        rnn_W_in_to_hid : Theano shared variable, numpy array or callable
            Initializer for input-to-hidden weight matrix, if
            RecurrentNet is RecurrentLayer
        rnn_W_hid_to_hid : Theano shared variable, numpy array or callable
            Initializer for hidden-to-hidden weight matrix, if
            RecurrentNet is RecurrentLayer
        rnn_b : Theano shared variable, numpy array, callable or None
            Initializer for bias vector, if RecurrentNet is
            RecurrentLaye. If None is provided there will be no bias
        name = string
            The name of the layer, optional
        """
        super(BidirectionalRNNLayer, self).__init__(l_in, name, **kwargs)
        self.l_in = l_in
        self.num_units = num_units
        self.grad_clipping = grad_clipping
        self.name = name

        # We use a bidirectional RNN, which means we combine two
        # RecurrentLayers, the second of which with backwards=True
        # Setting only_return_final=True makes the layers only return their
        # output for the final time step, which is all we need for this task

        # GRU
        if RecurrentNet.__name__ == 'GRULayer':
            if batch_norm:
                RecurrentNet = lasagne.layers.BNGRULayer

            rnn_params = dict(
                resetgate=gru_resetgate,
                updategate=gru_updategate,
                hidden_update=gru_hidden_update,
                hid_init=gru_hid_init)

        # LSTM
        elif RecurrentNet.__name__ == 'LSTMLayer':
            rnn_params = dict(
                nonlinearity=nonlinearity,
                ingate=lstm_ingate,
                forgetgate=lstm_forgetgate,
                cell=lstm_cell,
                outgate=lstm_outgate)

        # RNN
        elif RecurrentNet.__name__ == 'RecurrentLayer':
            rnn_params = dict(
                nonlinearity=nonlinearity,
                W_in_to_hid=rnn_W_in_to_hid,
                W_hid_to_hid=rnn_W_hid_to_hid,
                b=rnn_b)
        else:
            raise NotImplementedError('RecurrentNet not implemented')

        common_params = dict(
            hid_init=hid_init,
            grad_clipping=grad_clipping,
            precompute_input=precompute_input,
            mask_input=mask_input,
            only_return_final=False)
        rnn_params.update(common_params)

        l_forward = RecurrentNet(
            l_in,
            num_units,
            name=name + '_l_forward_sub',
            **rnn_params)
        l_backward = RecurrentNet(
            l_forward,
            num_units,
            backwards=True,
            name=name + '_l_backward_sub',
            **rnn_params)

        # Now we'll concatenate the outputs to combine them
        # Note that l_backward is already inverted by Lasagne
        l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward],
                                              axis=2, name=name+'_concat')

        # HACK LASAGNE
        # This will set `self.input_layer`, which is needed by Lasagne to find
        # the layers with the get_all_layers() helper function in the
        # case of a layer with sublayers
        if isinstance(l_concat, tuple):
            self.input_layer = None
        else:
            self.input_layer = l_concat

    def get_output_shape_for(self, input_shape):
        return list(input_shape[0:2]) + [self.num_units * 2]

    def get_output_for(self, input_var, **kwargs):
        # HACK LASAGNE
        # This is needed, jointly with the previous hack, to ensure that
        # this layer behaves as its last sublayer (namely,
        # self.input_layer)
        return input_var


class LinearUpsamplingLayer(lasagne.layers.Layer):

    def __init__(self,
                 incoming,
                 expand_height,
                 expand_width,
                 nclasses,
                 W=lasagne.init.Normal(0.01),
                 b=lasagne.init.Constant(.0),
                 batch_norm=False,
                 **kwargs):
        super(LinearUpsamplingLayer, self).__init__(incoming, **kwargs)
        nfeatures_in = self.input_shape[-1]
        nfeatures_out = expand_height * expand_width * nclasses

        self.nfeatures_out = nfeatures_out
        self.incoming = incoming
        self.expand_height = expand_height
        self.expand_width = expand_width
        self.nclasses = nclasses
        self.batch_norm = batch_norm

        # ``regularizable`` and ``trainable`` by default
        self.W = self.add_param(W, (nfeatures_in, nfeatures_out), name='W')
        if not batch_norm:
            self.b = self.add_param(b, (nfeatures_out,), name='b')

    def get_output_for(self, input_arr, **kwargs):
        # upsample
        pred = T.dot(input_arr, self.W)
        if not self.batch_norm:
            pred += self.b

        nrows, ncolumns = self.input_shape[1:3]
        batch_size = -1
        nclasses = self.nclasses
        expand_height = self.expand_height
        expand_width = self.expand_width

        # Reshape after the upsampling to come back to the original
        # dimensions and move the pixels in the right place
        pred = pred.reshape((batch_size,
                             nrows,
                             ncolumns,
                             expand_height,
                             expand_width,
                             nclasses))
        pred = pred.dimshuffle((0, 1, 4, 2, 5, 3))
        pred = pred.reshape((batch_size,
                             nrows * expand_height,
                             ncolumns * expand_width,
                             nclasses))
        return pred

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],
                input_shape[1] * self.expand_height,
                input_shape[2] * self.expand_width,
                self.nclasses)


class CropLayer(lasagne.layers.Layer):
    def __init__(self, l_in, crop, data_format='bc01', centered=True,
                 **kwargs):
        super(CropLayer, self).__init__(l_in, crop, **kwargs)
        assert data_format in ['bc01', 'b01c']
        if not isinstance(crop, T.TensorVariable):
            crop = lasagne.utils.as_tuple(crop, 2)
        self.crop = crop
        self.data_format = data_format
        self.centered = centered

    def get_output_shape_for(self, input_shape, **kwargs):
        # self.crop is a tensor --> we cannot know in advance how much
        # we will crop
        if isinstance(self.crop, T.TensorVariable):
            if self.data_format == 'bc01':
                input_shape = list(input_shape)
                input_shape[2] = None
                input_shape[3] = None
            else:
                input_shape = list(input_shape)
                input_shape[1] = None
                input_shape[2] = None
        # self.crop is a list of ints
        else:
            if self.data_format == 'bc01':
                input_shape = list(input_shape)
                input_shape[2] -= self.crop[0]
                input_shape[3] -= self.crop[1]
            else:
                input_shape = list(input_shape)
                input_shape[1] -= self.crop[0]
                input_shape[2] -= self.crop[1]
        return input_shape

    def get_output_for(self, input_arr, **kwargs):
        crop = self.crop.astype('int32')  # Indices have to be int
        sz = input_arr.shape

        if self.data_format == 'bc01':
            if self.centered:
                idx0 = T.switch(T.eq(-crop[0] + crop[0]/2, 0),
                                sz[2], -crop[0] + crop[0]/2)
                idx1 = T.switch(T.eq(-crop[1] + crop[1]/2, 0),
                                sz[3], -crop[1] + crop[1]/2)
                return input_arr[:, :, crop[0]/2:idx0, crop[1]/2:idx1]
            else:
                idx0 = T.switch(T.eq(crop[0], 0), sz[2], -crop[0])
                idx1 = T.switch(T.eq(crop[1], 0), sz[3], -crop[1])
                return input_arr[:, :, :idx0, :idx1]
        else:
            if self.centered:
                idx0 = T.switch(T.eq(-crop[0] + crop[0]/2, 0),
                                sz[1], -crop[0] + crop[0]/2)
                idx1 = T.switch(T.eq(-crop[1] + crop[1]/2, 0),
                                sz[2], -crop[1] + crop[1]/2)
                return input_arr[:, crop[0]/2:idx0, crop[1]/2:idx1, :]
            else:
                idx0 = T.switch(T.eq(crop[0], 0), sz[1], -crop[0])
                idx1 = T.switch(T.eq(crop[1], 0), sz[2], -crop[1])
                return input_arr[:, :idx0, :idx1, :]