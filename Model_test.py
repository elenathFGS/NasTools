from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
# from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import SeparableConv2D
from keras_flops_estimator import net_flops
from base_line_model import BaselineModel,BaselineModel_B,BaselineModel_C

# model = BaselineModel()
# model_B = BaselineModel_B()
# model_C = BaselineModel_C()
# flop = net_flops(model, show_table=True, verbose=True, conv_only=True)
# flopB = net_flops(model_B, show_table=True, verbose=True, conv_only=True)
# flopC = net_flops(model_C, show_table=True, verbose=True, conv_only=True)
# print(f'flop_B / flop = {flopB/flop}')
# print(f'flop_C / flop = {flopC/flop}')


# model = VGG16(weights=None, include_top=True, pooling=None,input_shape=(224,224,3))
model = InceptionV3(weights=None, include_top=True, pooling=None,input_shape=(224,224,3))

# model = Sequential()
# model.add(Dense(32, activation='relu', input_dim=100))
# model.add(SeparableConv2D(filters=64, kernel_size=3))
# model.add(Dense(1, activation='sigmoid'))
flop = net_flops(model, show_table=True, verbose=True, conv_only=True)
# Prints a table with the FLOPS at each layer and total FLOPs


