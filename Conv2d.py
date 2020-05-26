import numpy as np
import neural_element as ne

def conv2d(in_image, weight, bias):
    # in_image = sess.run(in_image)

    #print('shape of weight : ', weight.shape)

    colOfWeight = weight.shape[0]
    #print("colOfWeight : ", colOfWeight)
    rowOfWeight = weight.shape[1]
    #print("rowOfWeight : ", rowOfWeight)
    inChannelOfWeight = weight.shape[2]
    #print("inChannelOfWeight : ", inChannelOfWeight)
    outChannelOfWeight = weight.shape[3]
    #print("outChannelOfWeight : ", outChannelOfWeight)

    colOfInImage = in_image.shape[1]
    rowOfInImage = in_image.shape[2]
    channelOfInImage = in_image.shape[3]


    colOfOutImage = colOfInImage-colOfWeight + 1
    #print('colOfOutImage : ', colOfOutImage)
    rowOfOutImage = rowOfInImage-rowOfWeight + 1
    #print('rowOfOutImage : ', rowOfOutImage)
    outChannelOfImage = outChannelOfWeight
    #print('outChannelOfWeight : ', outChannelOfWeight)
    output = np.zeros((1, colOfOutImage, rowOfOutImage, outChannelOfImage), dtype='f')
    #print('output.shape : ', output.shape)
    #print('bias.shape : ', bias.shape)

    weight = weight.astype('int')
    in_image = in_image.astype('int')


    for o_channel in range(0, outChannelOfWeight):
        for i_channel in range(0, inChannelOfWeight):                            #output channel
            for in_col in range(0, colOfOutImage, 1):                             #28
                for in_row in range(0, rowOfOutImage, 1):                         #28
                    for weight_col in range(0, colOfWeight):                     #3
                        for weight_row in range(0, rowOfWeight):                 #3
                            bpSum, onesSum = ne.neural_element(in_image[0][in_col + weight_col][in_row + weight_row][i_channel], weight[weight_col][weight_row][i_channel][o_channel])
                            output[0][in_col][in_row][o_channel] += bpSum + onesSum
                            #output[0][in_col][in_row][o_channel] += (in_image[0][in_col + weight_col][in_row + weight_row][i_channel]) * (weight[weight_col][weight_row][i_channel][o_channel])
        output[0][in_col][in_row][o_channel] += bias[o_channel]

    return output
