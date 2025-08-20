import os
import cv2 
import numpy as np
from .Files import Files

class Frame:
  def __init__(self, model):
    self.model = model

  def load(self, video_file, delta_t, output_graph_file, features_list):
    if(os.path.exists(video_file)):
      frame_list = os.listdir(video_file)
      frame_list.sort() # to garanted the time order

      feat_list_len = len(features_list)
      weight_list = []

      for vertex1 in range(feat_list_len):
        for vertex2 in range(self.calc_init(vertex1, delta_t, feat_list_len),
                                      self.calc_end(vertex1, delta_t, feat_list_len)):
          frame1 = os.path.join(video_file, frame_list[vertex1])
          frame2 = os.path.join(video_file, frame_list[vertex2])
          w = self.compute_similarity_img(frame1, frame2, features_list[vertex1], features_list[vertex2])
          weight_list.append(w)
          output_graph_file.save_graph_data("{}, {}, {:.2f}".format(vertex1, vertex2, w) , ' ', ' ')

  def compute_similarity_img(self, img_path_1, img_path_2, fea_vec_img1, fea_vec_img2): # img_path_1, img_path_2):
      filename1 = os.path.basename(img_path_1).split(".")[0]
      filename2 = os.path.basename(img_path_2).split(".")[0]

      # Compute cosine similarity
      sim_cos = self.calculate_similarity(fea_vec_img1, fea_vec_img2)

      # Read images
      im1 = cv2.resize(cv2.imread(img_path_1), dsize=self.model.img_size_model, interpolation = cv2.INTER_AREA)
      im2 = cv2.resize(cv2.imread(img_path_2), dsize=self.model.img_size_model, interpolation = cv2.INTER_AREA)

      # Concatenate images horizontally
      im12 = cv2.hconcat([im1, im2])

      # Save concatenated image
      folder = Files()
      dst_dir_cos_sim = "../report/cos_sim"
      folder.create_folder(dst_dir_cos_sim)
      dst_dir = f"{dst_dir_cos_sim}/{self.model.name}"
      folder.create_folder(dst_dir)

      new_filename = f"{filename1}_{filename2}"
      cv2.imwrite(f"{dst_dir}/{new_filename}.jpg", im12)

      return sim_cos

  def calculate_similarity(self, vector1, vector2):
      sim_cos = np.linalg.norm(vector1-vector2, 1) * 100 / len(vector2)
      return sim_cos
  
#   def calc_init(self, i, delta_t, frame_len):
#       if((i < delta_t) or delta_t < 0):
#           return 0
#       elif((i + delta_t) > frame_len):
#           return i
#       else:
#           return i - delta_t

#   def calc_end(self, i, delta_t, frame_len):
#       if(((i + delta_t) > frame_len) or delta_t < 0):
#           return frame_len
#       else:
#           return i + delta_t
  def calc_init(self, i, delta_t, frame_len):
    # Normal behavior: if i is too low, start at 0.
    start = 0 if (i < delta_t or delta_t < 0) else i - delta_t
    # Ensure that there is at least one neighbor:
    if start == i and i > 0:
        start = i - 1
    return start

  def calc_end(self, i, delta_t, frame_len):
    # Normal behavior:
    end = frame_len if ((i + delta_t) > frame_len or delta_t < 0) else i + delta_t
    # Ensure that there is at least one neighbor (if possible):
    if end <= i + 1 and i < frame_len - 1:
        end = i + 2
    return end

  def load_features(self, features_list, input_graph_file, delta_t):
      #print("DEBUG: Entering load_features; features_list has", len(features_list), "elements.")
      if features_list is None or len(features_list) == 0:
          #print("DEBUG: Warning: Received empty features list.")
          return
      # Clear previous content:
      input_graph_file.content = ""
      feat_list_len = len(features_list)
      for vertex1 in range(feat_list_len):
          start = self.calc_init(vertex1, delta_t, feat_list_len)
          end = self.calc_end(vertex1, delta_t, feat_list_len)
          for vertex2 in range(start, end):
              if vertex1 == vertex2:
                  continue
              w = self.calculate_similarity(features_list[vertex1], features_list[vertex2])
              # Write to the in-memory content:
              input_graph_file.content += f"{vertex1}, {vertex2}, {w:.2f}\n"
      #print(f"DEBUG: Graph file (in-memory) written with edges computed from {feat_list_len} nodes.")

  # def load_features(self, features_list, input_graph_file, delta_t):
  #   print("DEBUG: Entering load_features; features_list has", len(features_list), "elements.")
  #   if features_list is None or len(features_list) == 0:
  #       print("DEBUG: Warning: Received empty features list.")
  #       return
  #   try:
  #       with open(input_graph_file.file, "w") as f:
  #           feat_list_len = len(features_list)
  #           for vertex1 in range(feat_list_len):
  #               #print("DEBUG: Processing vertex1:", vertex1)
  #               start = self.calc_init(vertex1, delta_t, feat_list_len)
  #               end = self.calc_end(vertex1, delta_t, feat_list_len)
  #               #print("DEBUG: vertex1 =", vertex1, "start =", start, "end =", end)
  #               for vertex2 in range(start, end):
  #                   if vertex1 == vertex2:
  #                       continue
  #                   w = self.calculate_similarity(features_list[vertex1], features_list[vertex2])
  #                   #print(f"DEBUG: Edge {vertex1} -> {vertex2} weight: {w:.2f}")
  #                   # Write with a space after the comma:
  #                   #f.write(f"{vertex1}, {vertex2}, {w:.2f}\n")
  #                   f.write(f"{vertex1}, {vertex2}, {w:.2f}\n")
  #       print(f"DEBUG: Graph file '{input_graph_file.file}' written with edges computed from {len(features_list)} nodes.")
  #   except Exception as e:
  #       print("DEBUG: Exception in load_features:", e)
