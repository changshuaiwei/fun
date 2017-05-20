# 1_data_processing.ipynb
extract and resize images, extract information from digit_struct.mat, and save a mini_train data for model debugging

# 2_cnn_softmax.ipynb
build a CNN+multiple-softmax model. the model contain 2 conv layers. model can overfit the mini_train data

# 3_cnn_softmax_localization.ipynb
add 4 regression head for localization. can overfit mini_train

# 4_cnn_lstm.ipynb
2 layers CNN + LSTM, use adam optimizer,can over fit mini_train

# 5_cnn_lstm_clip_gradient.ipynb
2 layers CNN + LSTM, use gradiant clip, gradiant decay, can overfit mini_train. But slow

# 6_data_processing_2.ipynb
extract images, crop image and resize .extract digit_struct, and save train, test, extra for future model training and testing

# 7_cnn_softmax _real.ipynb
train the CNN+softmax model using full training data and obtain prediction accuracy using testing data

# 8_data_explore.ipynb
calculate some statistics for labels and images

# 9_cnn_softmax _deeper_reg.ipynb
CNN + softmax, with 4 layers of conv layer

# 9_cnn_softmax _deeper_reg _2.ipynb
CNN + softmax, with 6 layers of conv layer

# 10_cnn_lstm_2.ipynb
CNN + LSTM, with 6 layers of conv layer (with batch normalization)

# 11_cnn_softmax _BN.ipynb
implement batch normalization on the 2 layer CNN + softmax.

# 12_cnn_softmax _deeoer_reg_BN.ipynb
CNN + softmax, with 6 conv layers and batch normalization

