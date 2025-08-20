import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

class Models:
  def __init__(self, model_name):
    self.available_models = ['vgg16', 'resnet50']
    self.include_top = True

    # verify if the model is valid
    if model_name in self.available_models:
      try:
        if model_name == 'vgg16':
          self.name = model_name
          self.layername_feature_extraction = 'fc2'
          self.img_size_model = (224, 224)
          self.model = VGG16(weights='imagenet', include_top=self.include_top)
        elif model_name == 'resnet50':
          self.name = model_name
          self.layername_feature_extraction = 'predictions'
          self.img_size_model = (224, 224)
          self.model = ResNet50(weights='imagenet', include_top=self.include_top)
      except:
        print(f">> Error while loading model '{model_name}'")
    else:
        print(f">> Error: there is no '{model_name}' in {self.available_models}")

  def features(self, video_file):
    features_list = []
    if(os.path.exists(video_file)):
      frame_list = os.listdir(video_file)
      frame_list.sort() # to garanted the time order
      features_list = [    # Compute feature vector extracted
                      self.get_feature_vector(video_file + frames) for frames in frame_list #List of frame features
                      ]

    return features_list

  def get_feature_vector(self, img_path):
      model_feature_vect = Model(inputs=self.model.input, outputs=self.model.get_layer(self.layername_feature_extraction).output)
      img = tf.keras.utils.load_img(img_path, target_size=self.img_size_model)
      img_arr = np.array(img)
      img_ = self.image_processing(img_arr)

      feature_vect = model_feature_vect.predict(img_, verbose='1')

      return feature_vect

  def image_processing(self, img_array):
      img = np.expand_dims(img_array, axis=0)
      processed_img = preprocess_input(img)

      return processed_img