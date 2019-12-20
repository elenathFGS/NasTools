#                    __
#                  / *_)
#       _.----. _ /../
#     /............./
#  __/.(...|.(...|
# /__.--|_|---|_|
#
# based on the implementation of Christos Kyrkou, PhD
#    (\~---.
#    /   (\-`-/)
#   (      ' ' )
#    \ (  \_Y_/\
#     ""\ \___//
#        `w   "
# enhanced and revised by Elenath Feng, HIT
# 2019

# Estimator for model FLOPS in keras
# Use: net_flops(model, conv_only=True, show_table=False, verbose=False)

# Supported Layers: Conv2D, DepthwiseConv2D, SeparableConv2D, Activation, BatchNormalization, InputLayer, Reshape,
#                  Concatenate, Average, pool, Flatten, Global Pooling, Add, Maximum, SpatialPyramidPooling

def net_flops(model, conv_only=False, show_table=False, verbose=False):
    """

    :param model: the built model we count flops for
    :param conv_only: only count the conv flop as the paper
    <Convolutional neural networks at constrained time cost> He at.al
    :param show_table: show the info table
    :param verbose: print the final result
    :return:
    """
    if show_table:
        print('%25s | %16s | %16s | %16s | %16s | %6s | %6s' % (
            'Layer Name', 'Input Shape', 'Output Shape', 'Kernel Size', 'Filters', 'Strides', 'FLOPS'))
        print('-' * 170)

    t_flops = 0
    t_macc = 0

    for l in model.layers:

        o_shape, i_shape, strides, ks, filters = ['', '', ''], ['', '', ''], [1, 1], [0, 0], [0, 0]
        flops = 0
        macc = 0
        name = l.name

        factor = 1000000000

        if 'InputLayer' in str(l) and not conv_only:
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = i_shape

        if 'Reshape' in str(l) and not conv_only:
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()

        if 'Add' in str(l) or 'Maximum' in str(l) or 'Concatenate' in str(l) and not conv_only:
            i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
            o_shape = l.output.get_shape()[1:4].as_list()
            flops = (len(l.input) - 1) * i_shape[0] * i_shape[1] * i_shape[2]

        if 'Average' in str(l) and 'pool' not in str(l) and not conv_only:
            i_shape = l.input[0].get_shape()[1:4].as_list() + [len(l.input)]
            o_shape = l.output.get_shape()[1:4].as_list()
            flops = len(l.input) * i_shape[0] * i_shape[1] * i_shape[2]

        if 'BatchNormalization' in str(l) and not conv_only:
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()

            bflops = 1
            for i in range(len(i_shape)):
                bflops *= i_shape[i]
            flops /= factor

        if 'Activation' in str(l) or 'activation' in str(l) and not conv_only:
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()
            bflops = 1
            for i in range(len(i_shape)):
                bflops *= i_shape[i]
            flops /= factor

        if 'SpatialPyramidPooling' in str(l) and not conv_only:
            i_shape = l.input.shape[1:4].as_list()
            out_vec = 1
            for i in range(len(i_shape)):
                out_vec *= i_shape[i]
            o_shape = out_vec
            i_shape = l.input.shape[1:4].as_list()
            # don't know how to count this flop yet

        if 'pool' in str(l) and ('Global' not in str(l)) and 'spatial' not in str(l) and not conv_only:
            i_shape = l.input.get_shape()[1:4].as_list()
            strides = l.strides
            ks = l.pool_size
            flops = ((i_shape[0] / strides[0]) * (i_shape[1] / strides[1]) * (ks[0] * ks[1] * i_shape[2]))

        if 'Flatten' in str(l) and not conv_only:
            i_shape = l.input.shape[1:4].as_list()
            flops = 0
            out_vec = 1
            for i in range(len(i_shape)):
                out_vec *= i_shape[i]
            o_shape = out_vec

        if 'Dense' in str(l) and not conv_only:
            i_shape = l.input.shape[1:4].as_list()[0]
            if i_shape is None:
                i_shape = out_vec

            o_shape = l.output.shape[1:4].as_list()
            flops = 2 * (o_shape[0] * i_shape)
            macc = flops / 2

        if 'Padding' in str(l) and not conv_only:
            flops = 0

        if 'Global' in str(l) and not conv_only:
            i_shape = l.input.get_shape()[1:4].as_list()
            flops = ((i_shape[0]) * (i_shape[1]) * (i_shape[2]))
            o_shape = [l.output.get_shape()[1:4].as_list(), 1, 1]
            out_vec = o_shape

        if 'Conv2D ' in str(l) and 'DepthwiseConv2D' not in str(l) and 'SeparableConv2D' not in str(l):
            strides = l.strides
            ks = l.kernel_size
            filters = l.filters
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()

            if filters is None:
                filters = i_shape[2]

            flops = 2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
                    o_shape[0] * o_shape[1]))
            # flops = 2 * ((filters * ks[0] * ks[1] * i_shape[2]) * (
            #         (i_shape[0] / strides[0]) * (i_shape[1] / strides[1])))
            macc = flops / 2

        if 'Conv2D ' in str(l) and 'DepthwiseConv2D' in str(l) and 'SeparableConv2D' not in str(l):
            strides = l.strides
            ks = l.kernel_size
            filters = l.filters
            i_shape = l.input.get_shape()[1:4].as_list()
            o_shape = l.output.get_shape()[1:4].as_list()

            if filters is None:
                filters = i_shape[2]

            flops = 2 * (
                    (ks[0] * ks[1] * i_shape[2]) * (o_shape[0] * o_shape[1])) / factor

            macc = flops / 2

        t_macc += macc

        t_flops += flops

        if show_table:
            print('%25s | %16s | %16s | %16s | %16s | %6s | %5.4f' % (
                name, str(i_shape), str(o_shape), str(ks), str(filters), str(strides), flops))
    t_flops = t_flops / factor
    t_macc = t_macc / factor

    if verbose:
        print('\nTotal FLOPS (G): %10.5f\n' % (t_flops))
        print('\nTotal MACCs (G): %10.5f\n' % (t_macc))

    return t_macc
