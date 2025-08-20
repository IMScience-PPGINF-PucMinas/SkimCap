# from .Files import Files
# from .Frame import Frame
# from .Models import Models
# from .Graph import Graph
# from .Evaluation import Evaluation
# import os 
# from PIL import Image
# import numpy as np
# from scipy import spatial
# import networkx as nx
# import cv2 as cv

# class Summary:
#     def __init__(self, dataset_frames, video, rate, time, hierarchy, selected_model, is_binary, percent, alpha, gen_summary_method):
#         self.delta_t = rate * time

#         # If input is precomputed features (full path expected)
#         if video.endswith('.npy'):
#             # Set output directories for features processing.
#             self.out_skim_dir = '/home/bernardop/recurrent-transformer/video_feature/rt_anet_feat/trainvalskim_test'
#             self.out_index_dir = '/home/bernardop/recurrent-transformer/video_feature/rt_anet_feat/trainvalindexes_test'
#             os.makedirs(self.out_skim_dir, exist_ok=True)
#             os.makedirs(self.out_index_dir, exist_ok=True)
#             # Use the directory of the input file:
#             base = os.path.splitext(os.path.basename(video))[0]
#             self.video_file = os.path.join(os.path.dirname(video), base) + os.sep
#             if not os.path.isdir(self.video_file):
#                 os.makedirs(self.video_file, exist_ok=True)
#             self.frames_path = os.path.join(self.video_file, "frames/")
#             if not os.path.isdir(self.frames_path):
#                 os.makedirs(self.frames_path, exist_ok=True)
#             # Load features directly from the input file.
#             features_list = np.load(video)
#             self.features_list = features_list
#             # Skip loading the model (avoids resnet50 load)
#             self.model = None
#         else:
#             self.video_file = os.path.join(dataset_frames, video) + os.sep
#             self.frames_path = os.path.join(self.video_file, "frames/")
#             features_list = None  
#             self.features_list = None
#             self.model = Models(selected_model)
            
#         self.video = video
#         self.alpha = alpha
#         self.percent = percent
#         self.graph = Graph(is_binary, hierarchy)
#         self.frame = Frame(self.model)
#         self.len_shot = 0
#         self.rate = rate
        
#         # Creating paths for auxiliary files.
#         self.input_graph_file = Files('{}graph.txt'.format(self.video_file))
#         self.input_mst = '{}mst_{}_{}.txt'.format(self.video_file, self.percent, self.alpha)
#         self.input_higra = Files('{}higra_{}_{}.txt'.format(self.video_file, self.percent, self.alpha))
#         self.cut_graph_file = Files('{}cut_graph_{}_{}.txt'.format(self.video_file, self.percent, self.alpha))
#         self.output_skim = '{}skim_{}_{}'.format(self.video_file, self.percent, self.alpha)
#         self.summ_path = '{}{}/'.format(self.video_file, gen_summary_method['method'])
#         self.summ_input = Files('{}{}.txt'.format(self.video_file, gen_summary_method['method']))
#         self.evaluate_summary = Evaluation(self.frames_path, dataset_frames, 'self.gt_path', self.model, gen_summary_method['method'])
#         self.fscore = 0
#         self.mean_cusa = 0
#         self.mean_cuse = 0
#         self.cov_value = 0

#         #print("----------------------")
#         #print("Processing video {}".format(self.video_file))

#         # Create the graph file if needed.
#         if not os.path.exists(self.input_graph_file.file) or os.path.getsize(self.input_graph_file.file) == 0:
#             #print("DEBUG: Graph file is missing or empty. Calling load_features...")
#             with open(self.input_graph_file.file, "w") as f:
#                 pass
#             if features_list is None:
#                 features_list = self.model.features(self.frames_path)
#                 self.frame.load(self.frames_path, self.delta_t, self.input_graph_file, features_list)
#             else:
#                 self.frame.load_features(features_list, self.input_graph_file, self.delta_t)
#         else:
#             pass
#             #print("DEBUG: Graph file exists and is non-empty.")

#        # Force reprocessing for small videos
#         if self.features_list is not None and self.features_list.shape[0] < 100:
#             #print("DEBUG: Small video detected (<100 frames). Forcing full processing with no reduction.")
#             if os.path.exists(self.input_higra.file):
#                 os.remove(self.input_higra.file)

#        # Process keyframe (or key-feature) selection.
#         if not os.path.exists(self.input_higra.file):
#             if gen_summary_method['method'] == 'n_fixed_keyframes':
#                 self.cut_number = gen_summary_method['n_keyframes'] - 1
#                 self.len_shot = round((len(os.listdir(self.frames_path))) * (self.percent/100)) - 1
#             elif gen_summary_method['method'] in ['group_central_frames', 'percent_spaced']:
#                 if self.features_list is not None:
#                     self.len_shot = round((len(self.features_list)) * (self.percent/100)) - 1
#                 else:
#                     self.len_shot = round((len(os.listdir(self.frames_path))) * (self.percent/100)) - 1
#                 self.cut_number = int(self.bestCutNumber() * (self.alpha / 100))
#                 if self.cut_number <= 2:
#                     self.cut_number = 3

#             if gen_summary_method['method'] == 'sequential_keyframe':
#                 self.sequential_keyframe(gen_summary_method['n_keyframes'])
#             else:
#                 tree = self.input_graph_file.read_graph_file(Files(), cut_graph=False, cut_number=0)
#                 leaflist = self.graph.compute_hierarchy(tree, self.input_higra)
#                 cuted_graph = self.graph.cut_graph(self.input_higra, self.cut_graph_file, cutNumber=self.cut_number)
#                 if gen_summary_method['method'] == 'group_central_frames':
#                     if self.features_list is not None:
#                         self.group_central_features(cuted_graph, leaflist)
#                     else:
#                         self.group_central_frames(cuted_graph, leaflist)
#                 else:
#                     self.get_n_frames(cuted_graph, leaflist)

#             if not os.path.exists(self.output_skim):
#                 os.mkdir(self.output_skim)

#         if os.path.exists(self.summ_input.file):
#             if video in ['Air_Force_One', 'Cooking', 'Bearpark_climbing', 'Saving_dolphines',
#                          'Cockpit_Landing', 'Bus_in_Rock_Tunnel', 'Kids_playing_in_leaves',
#                          'Scuba', 'Bike_Polo', 'Fire_Domino', 'car_over_camera', 'Eiffel_Tower',
#                          'Valparaiso_Downhill', 'Paintball', 'Statue_of_Liberty', 'Excavators_river_crossing',
#                          'St_Maarten_Landing', 'Jumps', 'playing_ball', 'Notre_Dame', 'Uncut_Evening_Flight',
#                          'Car_railcrossing', 'Playing_on_water_slide', 'Base_jumping', 'paluma_jump']:
#                 self.fscore, self.mean_cusa, self.mean_cuse, self.cov_value = self.evaluate_summary.evaluate(video)

#     # Use the precomputed features if available
#     def bestCutNumber(self, features_list=None):
#     # Use the precomputed features if available
#         if features_list is None:
#             if self.features_list is not None:
#                 features = self.features_list
#             elif os.path.exists(self.frames_path):
#                 frame_list = [f for f in os.listdir(self.frames_path) if f.endswith("jpg")]
#                 frame_list.sort()  # to guarantee time order
#                 features = []
#                 for frame in frame_list:
#                     frame_path = os.path.join(self.frames_path, frame)
#                     features.append(self.rgbSim(frame_path))
#             else:
#                 features = []
#         else:
#             features = features_list

#         if len(features) < 2:
#             return 0

#         weight_list = []
#         feat_list_len = len(features)
#         for vertex1 in range(feat_list_len):
#             start = self.calc_init(vertex1, self.delta_t, feat_list_len)
#             end = self.calc_end(vertex1, self.delta_t, feat_list_len)
#             for vertex2 in range(start, end):
#                 if vertex2 < 0 or vertex2 >= feat_list_len or vertex2 == vertex1:
#                     continue
#                 w = self.spatialSim(features[vertex1], features[vertex2]) / 100
#                 weight_list.append(w)
#         if not weight_list:
#             return 0

#         cut = np.std(weight_list)
        
#         while cut <= 1:
#             cut *= 10
#             if cut == 0:
#                 break
#         print("DEBUG: bestcutnumber:", np.round(cut) - 1)
#         return np.round(cut) - 1


#     def rgbSim(self, frame_path):
#         frame = Image.open(frame_path)
#         frame_reshape = frame.resize((round(frame.size[0] * 0.5), round(frame.size[1] * 0.5)))
#         frame_array = np.array(frame_reshape).flatten() / 255
#         return frame_array

#     def spatialSim(self, frame1, frame2):
#         similarity = 100 * (-1 * (spatial.distance.cosine(frame1, frame2) - 1))
#         return similarity if similarity >= 20 else 20

#     def sequential_keyframe(self, n):
#         for i in range(1, n + 1):
#             kf = str(i).zfill(6)
#             if not os.path.isdir(self.summ_path):
#                 os.mkdir(self.summ_path)
#             os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, kf, self.summ_path, kf))
#             self.summ_input.save_graph_data(kf, '  ', '.jpg')

#     def get_n_frames(self, graph, leaflist):
#         subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
#         for comp in subgraphs:
#             nodes_list = list(comp.nodes)
#             comp_leaf_list = [node for node in nodes_list if node in leaflist]
#             if comp_leaf_list:
#                 cn = int(len(comp_leaf_list) / 2)
#                 kf = str(comp_leaf_list[cn]).zfill(6)
#                 if not os.path.isdir(self.summ_path):
#                     os.mkdir(self.summ_path)
#                 os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, kf, self.summ_path, kf))
#                 self.summ_input.save_graph_data(kf, '  ', '.jpg')

#     def group_central_frames(self, graph, leaflist):
#         subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
#         self.len_shot = int(self.len_shot / len(subgraphs))
#         for comp in subgraphs:
#             nodes_list = list(comp.nodes)
#             comp_leaf_list = [node for node in nodes_list if node in leaflist]
#             len_leaf_list = len(comp_leaf_list)
#             if len_leaf_list:
#                 cn = int(len_leaf_list / 2)
#                 if len_leaf_list < self.len_shot:
#                     init_keyshots = 0
#                     end_keyshots = len_leaf_list - 1
#                 else:
#                     init_keyshots = cn - int(self.len_shot / 2)
#                     end_keyshots = cn + int(self.len_shot / 2)
#                 if not os.path.isdir(self.summ_path):
#                     os.mkdir(self.summ_path)
#                 for k in range(init_keyshots, end_keyshots):
#                     keyshot = str(comp_leaf_list[k]).zfill(6)
#                     os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, keyshot, self.summ_path, keyshot))
#                     self.summ_input.save_graph_data(keyshot, '  ', '.jpg')

#     def keyshot(self, graph, leaflist):
#         subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
#         self.len_shot = int(self.len_shot / len(subgraphs))
#         for comp in subgraphs:
#             nodes_list = list(comp.nodes)
#             comp_leaf_list = [node for node in nodes_list if node in leaflist]
#             if comp_leaf_list:
#                 cn = int(len(comp_leaf_list) / 2)
#                 if len(comp_leaf_list) < self.len_shot:
#                     init_keyshots = 0
#                     end_keyshots = len(comp_leaf_list) - 1
#                 else:
#                     init_keyshots = cn - int(self.len_shot / 2)
#                     end_keyshots = cn + int(self.len_shot / 2)
#                 if not os.path.isdir(self.summ_path):
#                     os.mkdir(self.summ_path)
#                 for k in range(init_keyshots, end_keyshots):
#                     keyshot = str(comp_leaf_list[k]).zfill(6)
#                     os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, keyshot, self.summ_path, keyshot))
#                     self.summ_input.save_graph_data(keyshot, '  ', '.jpg')

#     def generate_video(self):
#         image_folder = self.summ_path.rstrip(os.sep)
#         images = [img for img in os.listdir(image_folder)
#                   if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")]
#         frame = cv.imread(os.path.join(image_folder, images[0]))
#         height, width, layers = frame.shape
#         os.chdir(self.output_skim)
#         video = cv.VideoWriter(self.video + '.mp4', 0, self.rate, (width, height))
#         for image in images:
#             video.write(cv.imread(os.path.join(image_folder, image)))
#         cv.destroyAllWindows()
#         video.release()

#     def calc_init(self, i, delta_t, frame_len):
#         return 0 if (i < delta_t or delta_t < 0) else (i - delta_t if (i + delta_t) <= frame_len else i)

#     def calc_end(self, i, delta_t, frame_len):
#         return frame_len if (i + delta_t) > frame_len or delta_t < 0 else i + delta_t

#     def group_central_features(self, graph, leaflist):
#         import os
#         import numpy as np
#         import networkx as nx
#         #print("DEBUG: Entering group_central_features")
#         print("DEBUG: Initial len_shot:", self.len_shot)
#         print("DEBUG: Initial cut_number:", self.cut_number)
        
#         # Derive the base name from the full input file path.
#         base_name = os.path.splitext(os.path.basename(self.video))[0]
#         #print("DEBUG: Base name:", base_name)
        
#         if self.features_list is not None:
#             pass#print("DEBUG: Feature shape:", self.features_list.shape)
#         else:
#             print("DEBUG: No precomputed features loaded.")
        
#         # Define output paths.
#         output_resnet_path = os.path.join(self.out_skim_dir, base_name + ".npy")
#         output_txt_path = os.path.join(self.out_index_dir, base_name + ".txt")
#         base_bn_name = base_name.replace('_resnet', '_bn')
#         output_bn_path = os.path.join(self.out_skim_dir, base_bn_name + ".npy")
        
#         # Check if the number of frames is less than 100.
#         if self.features_list is not None and self.features_list.shape[0] < 100:
#             #print("DEBUG: Fewer than 100 frames detected. Keeping all frames without reduction.")
#             selected_features = self.features_list
#             selected_indexes = list(range(self.features_list.shape[0]))
#             # Save outputs without reduction.
#             np.save(output_resnet_path, selected_features)
#             #print("DEBUG: Saved combined selected resnet features to", output_resnet_path)
#             with open(output_txt_path, "w") as f:
#                 for idx in selected_indexes:
#                     f.write(str(idx) + "\n")
#             #print("DEBUG: Saved selected indexes to", output_txt_path)
#             # Process BN features.
#             bn_file_name = os.path.basename(self.video).replace('_resnet.npy', '_bn.npy')
#             bn_file_path = os.path.join(os.path.dirname(self.video), bn_file_name)
#             #print("DEBUG: BN file path:", bn_file_path)
#             if not os.path.exists(bn_file_path):
#                 #print("DEBUG: Corresponding bn file not found:", bn_file_path)
#                 return
#             bn_features = np.load(bn_file_path)
#             #print("DEBUG: BN features shape:", bn_features.shape)
#             selected_bn_features = bn_features[np.array(selected_indexes, dtype=int)]
#             #print("DEBUG: Selected BN features shape:", selected_bn_features.shape)
#             np.save(output_bn_path, selected_bn_features)
#             #print("DEBUG: Saved new bn features to", output_bn_path)
#             return

#         # If all outputs already exist, skip processing.
#         if os.path.exists(output_resnet_path) and os.path.exists(output_txt_path) and os.path.exists(output_bn_path):
#             #print(f"DEBUG: All outputs for {base_name} already exist. Skipping processing.")
#             return
        
#         # Case 2: If resnet and index exist but BN output is missing, re-run only BN processing.
#         if os.path.exists(output_resnet_path) and os.path.exists(output_txt_path) and not os.path.exists(output_bn_path):
#             #print(f"DEBUG: BN output missing for {base_name}. Re-running BN processing.")
#             selected_indexes = []
#             with open(output_txt_path, "r") as f:
#                 for line in f:
#                     try:
#                         idx = int(line.strip())
#                         selected_indexes.append(idx)
#                     except Exception as e:
#                         print("DEBUG: Error parsing index:", line, e)
#             #print("DEBUG: Loaded selected indexes from file:", selected_indexes)
            
#             bn_file_name = os.path.basename(self.video).replace('_resnet.npy', '_bn.npy')
#             bn_file_path = os.path.join(os.path.dirname(self.video), bn_file_name)
#             #print("DEBUG: BN file path:", bn_file_path)
#             if not os.path.exists(bn_file_path):
#                 #print("DEBUG: Corresponding bn file not found:", bn_file_path)
#                 return
#             bn_features = np.load(bn_file_path)
#             #print("DEBUG: BN features shape:", bn_features.shape)
#             selected_bn_features = bn_features[np.array(selected_indexes, dtype=int)]
#             #print("DEBUG: Selected BN features shape:", selected_bn_features.shape)
#             np.save(output_bn_path, selected_bn_features)
#             #print("DEBUG: Re-saved new bn features to", output_bn_path)
#             return

#         # Case 3: Full processing if outputs don't exist.
#         subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
#         #print("DEBUG: Number of subgraphs found:", len(subgraphs))
#         if len(subgraphs) == 0:
#             #print("DEBUG: No connected components found in graph.")
#             return
#         self.len_shot = int(self.len_shot / len(subgraphs))
#         #print("DEBUG: Computed len_shot after dividing by subgraphs:", self.len_shot)
        
#         selected_features = []
#         selected_indexes = []
#         for comp in subgraphs:
#             nodes_list = list(comp.nodes)
#             comp_leaf_list = [node for node in nodes_list if node in leaflist]
#             #print("DEBUG: For subgraph, total nodes:", len(nodes_list), "leaf nodes:", len(comp_leaf_list))
#             if len(comp_leaf_list):
#                 cn = int(len(comp_leaf_list) / 2)
#                 if len(comp_leaf_list) < self.len_shot:
#                     init_keyshots = 0
#                     end_keyshots = len(comp_leaf_list)
#                 else:
#                     init_keyshots = max(0, cn - int(self.len_shot / 2))
#                     end_keyshots = min(len(comp_leaf_list), cn + int(self.len_shot / 2))
#                 #print("DEBUG: For subgraph, selecting indexes from", init_keyshots, "to", end_keyshots)
#                 for k in range(init_keyshots, end_keyshots):
#                     key_index = comp_leaf_list[k]
#                     key_str = str(key_index).zfill(6)
#                     #print("DEBUG: Selecting feature for index", key_str)
#                     selected_features.append(self.features_list[int(key_index)])
#                     selected_indexes.append(key_index)
#                     self.summ_input.save_graph_data(key_str, '  ', '.npy')
        
#         #print("DEBUG: Total selected indexes:", selected_indexes)
#         selected_features_array = np.array(selected_features)
#         #print("DEBUG: Combined selected features shape:", selected_features_array.shape)
#         np.save(output_resnet_path, selected_features_array)
#         #print("DEBUG: Saved combined selected resnet features to", output_resnet_path)
        
#         with open(output_txt_path, "w") as f:
#             for idx in selected_indexes:
#                 f.write(str(idx) + "\n")
#         #print("DEBUG: Saved selected indexes to", output_txt_path)
        
#         bn_file_name = os.path.basename(self.video).replace('_resnet.npy', '_bn.npy')
#         bn_file_path = os.path.join(os.path.dirname(self.video), bn_file_name)
#         #print("DEBUG: Final BN file path:", bn_file_path)
#         if not os.path.exists(bn_file_path):
#             #print("DEBUG: Corresponding bn file not found:", bn_file_path)
#             return
#         bn_features = np.load(bn_file_path)
#         #print("DEBUG: BN features shape:", bn_features.shape)
#         selected_bn_features = bn_features[np.array(selected_indexes, dtype=int)]
#         #print("DEBUG: Final selected BN features shape:", selected_bn_features.shape)
#         np.save(output_bn_path, selected_bn_features)
#         #print("DEBUG: Saved new bn features to", output_bn_path)


# #salvar nome shape cut number e len shot
# #consertar o for do best cut number
# #lenshot é 15% dos frames totais, n usar listdir
from .Files import Files
from .Frame import Frame
from .Models import Models
from .Graph import Graph
from .Evaluation import Evaluation
import os 
from PIL import Image
import numpy as np
from scipy import spatial
import networkx as nx
import cv2 as cv

class Summary:
    def __init__(self, dataset_frames, video, rate, time, hierarchy, selected_model, is_binary, percent, alpha, gen_summary_method):
        self.delta_t = rate * time

        # If input is precomputed features (full path expected)
        if video.endswith('.npy'):
            # Set output directories for features processing.
            self.out_skim_dir = '/home/bernardop/recurrent-transformer/video_feature/rt_anet_feat/trainvalskimTESTE_1000'
            self.out_index_dir = '/home/bernardop/recurrent-transformer/video_feature/rt_anet_feat/trainvalindexesTESTE_1000'
            os.makedirs(self.out_skim_dir, exist_ok=True)
            os.makedirs(self.out_index_dir, exist_ok=True)
            # Use the dataset_frames folder as base (do not create a dedicated folder)
            # Also extract base name from the feature file.
            base = os.path.splitext(os.path.basename(video))[0]
            self.video_file = dataset_frames  # use common folder
            self.frames_path = ""  # Not used for precomputed features.
            # Load features directly from the input file.
            features_list = np.load(video)
            self.features_list = features_list
            # Skip loading the model.
            self.model = None
        else:
            self.video_file = os.path.join(dataset_frames, video) + os.sep
            self.frames_path = os.path.join(self.video_file, "frames/")
            features_list = None  
            self.features_list = None
            self.model = Models(selected_model)
            
        self.video = video
        self.alpha = alpha
        self.percent = percent
        self.graph = Graph(is_binary, hierarchy)
        self.frame = Frame(self.model)
        self.len_shot = 0
        self.rate = rate
        
        # Creating paths for auxiliary files in memory.
        # Instead of a folder per video, we now store all intermediate data in dataset_frames.
        self.input_graph_file = Files(os.path.join(dataset_frames, base + '_graph.txt'))
        self.input_mst = Files(os.path.join(dataset_frames, base + '_mst_{}_{}.txt'.format(self.percent, self.alpha)))
        self.input_higra = Files(os.path.join(dataset_frames, base + '_higra_{}_{}.txt'.format(self.percent, self.alpha)))
        self.cut_graph_file = Files(os.path.join(dataset_frames, base + '_cut_graph_{}_{}.txt'.format(self.percent, self.alpha)))
        self.output_skim = os.path.join(dataset_frames, base + '_skim_{}_{}'.format(self.percent, self.alpha))
        self.summ_path = os.path.join(dataset_frames, base + '_' + gen_summary_method['method'])
        self.summ_input = Files(os.path.join(dataset_frames, base + '_' + gen_summary_method['method'] + '.txt'))
        self.evaluate_summary = Evaluation(self.frames_path, dataset_frames, 'self.gt_path', self.model, gen_summary_method['method'])
        self.fscore = 0
        self.mean_cusa = 0
        self.mean_cuse = 0
        self.cov_value = 0

        # Create the graph data in memory if not already present.
        if self.input_graph_file.content == "":
            print("DEBUG: Graph content is empty. Calling load_features...")
            # Clear previous content:
            self.input_graph_file.content = ""
            if features_list is None:
                features_list = self.model.features(self.frames_path)
                self.frame.load(self.frames_path, self.delta_t, self.input_graph_file, features_list)
            else:
                self.frame.load_features(features_list, self.input_graph_file, self.delta_t)
        else:
            pass

        # Force reprocessing for small videos
        # if self.features_list is not None and self.features_list.shape[0] < 100:
        #     print("DEBUG: Small video detected (<100 frames). Forcing full processing with no reduction.")
        #     # Remove existing in-memory higra data to force reprocessing.
        #     self.input_higra.content = ""
        
        # Process keyframe (or key-feature) selection.
        if self.input_higra.content == "":
            if gen_summary_method['method'] == 'n_fixed_keyframes':
                self.cut_number = gen_summary_method['n_keyframes'] - 1
                # For raw frames branch, you might use os.listdir(self.frames_path);
                # for precomputed features, use len(self.features_list)
                if self.features_list is not None:
                    self.len_shot = round((len(self.features_list)) * (self.percent/100)) - 1
                else:
                    self.len_shot = round((len(os.listdir(self.frames_path))) * (self.percent/100)) - 1
            elif gen_summary_method['method'] in ['group_central_frames', 'percent_spaced']:
                if self.features_list is not None:
                    self.len_shot = round((len(self.features_list)) * (self.percent/100)) - 1
                else:
                    self.len_shot = round((len(os.listdir(self.frames_path))) * (self.percent/100)) - 1
                # IMPORTANT: Pass precomputed features explicitly.
                self.cut_number = int(self.bestCutNumber(self.features_list) * (self.alpha / 100))
                if self.cut_number <= 2:
                    self.cut_number = 3

            if gen_summary_method['method'] == 'sequential_keyframe':
                self.sequential_keyframe(gen_summary_method['n_keyframes'])
            else:
                # Use the in-memory content by calling read_graph_file on our memory file.
                tree = self.input_graph_file.read_graph_file(Files(), cut_graph=False, cut_number=0)
                leaflist = self.graph.compute_hierarchy(tree, self.input_higra)
                cuted_graph = self.graph.cut_graph(self.input_higra, self.cut_graph_file, cutNumber=self.cut_number)
                if gen_summary_method['method'] == 'group_central_frames':
                    if self.features_list is not None:
                        self.group_central_features(cuted_graph, leaflist)
                    else:
                        self.group_central_frames(cuted_graph, leaflist)
                else:
                    self.get_n_frames(cuted_graph, leaflist)
            # Instead of creating a folder on disk, we keep the summary in memory.
            # (If you need to output final results, you can later write self.summ_input.content to disk.)
        # Log final debug info:
        import logging
        logging.debug("Processed feature file: %s", self.video)
        if self.features_list is not None:
            logging.debug("Feature shape: %s", str(self.features_list.shape))
        else:
            logging.debug("No features loaded.")
        try:
            logging.debug("Cut number: %d", self.cut_number)
        except Exception as e:
            logging.debug("Cut number not set: %s", str(e))
        try:
            logging.debug("Len shot: %d", self.len_shot)
        except Exception as e:
            logging.debug("Len shot not set: %s", str(e))

        if self.summ_input.content != "":
            if video in ['Air_Force_One', 'Cooking', 'Bearpark_climbing', 'Saving_dolphines',
                         'Cockpit_Landing', 'Bus_in_Rock_Tunnel', 'Kids_playing_in_leaves',
                         'Scuba', 'Bike_Polo', 'Fire_Domino', 'car_over_camera', 'Eiffel_Tower',
                         'Valparaiso_Downhill', 'Paintball', 'Statue_of_Liberty', 'Excavators_river_crossing',
                         'St_Maarten_Landing', 'Jumps', 'playing_ball', 'Notre_Dame', 'Uncut_Evening_Flight',
                         'Car_railcrossing', 'Playing_on_water_slide', 'Base_jumping', 'paluma_jump']:
                self.fscore, self.mean_cusa, self.mean_cuse, self.cov_value = self.evaluate_summary.evaluate(video)

    def bestCutNumber(self, features_list=None):
        if features_list is None:
            if self.features_list is not None:
                features = self.features_list
            elif self.frames_path and os.path.exists(self.frames_path):
                frame_list = [f for f in os.listdir(self.frames_path) if f.endswith("jpg")]
                frame_list.sort()
                features = []
                for frame in frame_list:
                    frame_path = os.path.join(self.frames_path, frame)
                    features.append(self.rgbSim(frame_path))
            else:
                features = []
        else:
            features = features_list

        if len(features) < 2:
            return 0

        weight_list = []
        feat_list_len = len(features)
        for vertex1 in range(feat_list_len):
            start = self.calc_init(vertex1, self.delta_t, feat_list_len)
            end = self.calc_end(vertex1, self.delta_t, feat_list_len)
            for vertex2 in range(start, end):
                if vertex2 < 0 or vertex2 >= feat_list_len or vertex2 == vertex1:
                    continue
                w = self.spatialSim(features[vertex1], features[vertex2]) / 100
                weight_list.append(w)
        if not weight_list:
            return 0

        cut = np.std(weight_list)
        while cut <= 1:
            cut *= 10
            if cut == 0:
                break
        print("DEBUG: bestcutnumber:", np.round(cut) - 1)
        return np.round(cut) - 1

    def rgbSim(self, frame_path):
        frame = Image.open(frame_path)
        frame_reshape = frame.resize((round(frame.size[0] * 0.5), round(frame.size[1] * 0.5)))
        frame_array = np.array(frame_reshape).flatten() / 255
        return frame_array

    def spatialSim(self, frame1, frame2):
        similarity = 100 * (-1 * (spatial.distance.cosine(frame1, frame2) - 1))
        return similarity if similarity >= 20 else 20

    def sequential_keyframe(self, n):
        for i in range(1, n + 1):
            kf = str(i).zfill(6)
            if not os.path.isdir(self.summ_path):
                os.mkdir(self.summ_path)
            os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, kf, self.summ_path, kf))
            self.summ_input.content += f"{kf}\n"

    def get_n_frames(self, graph, leaflist):
        subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        for comp in subgraphs:
            nodes_list = list(comp.nodes)
            comp_leaf_list = [node for node in nodes_list if node in leaflist]
            if comp_leaf_list:
                cn = int(len(comp_leaf_list) / 2)
                kf = str(comp_leaf_list[cn]).zfill(6)
                if not os.path.isdir(self.summ_path):
                    os.mkdir(self.summ_path)
                os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, kf, self.summ_path, kf))
                self.summ_input.content += f"{kf}\n"

    def group_central_frames(self, graph, leaflist):
        subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        self.len_shot = int(self.len_shot / len(subgraphs))
        for comp in subgraphs:
            nodes_list = list(comp.nodes)
            comp_leaf_list = [node for node in nodes_list if node in leaflist]
            len_leaf_list = len(comp_leaf_list)
            if len_leaf_list:
                cn = int(len_leaf_list / 2)
                if len_leaf_list < self.len_shot:
                    init_keyshots = 0
                    end_keyshots = len_leaf_list - 1
                else:
                    init_keyshots = cn - int(self.len_shot / 2)
                    end_keyshots = cn + int(self.len_shot / 2)
                if not os.path.isdir(self.summ_path):
                    os.mkdir(self.summ_path)
                for k in range(init_keyshots, end_keyshots):
                    keyshot = str(comp_leaf_list[k]).zfill(6)
                    os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, keyshot, self.summ_path, keyshot))
                    self.summ_input.content += f"{keyshot}\n"

    def keyshot(self, graph, leaflist):
        subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        self.len_shot = int(self.len_shot / len(subgraphs))
        for comp in subgraphs:
            nodes_list = list(comp.nodes)
            comp_leaf_list = [node for node in nodes_list if node in leaflist]
            if comp_leaf_list:
                cn = int(len(comp_leaf_list) / 2)
                if len(comp_leaf_list) < self.len_shot:
                    init_keyshots = 0
                    end_keyshots = len(comp_leaf_list) - 1
                else:
                    init_keyshots = cn - int(self.len_shot / 2)
                    end_keyshots = cn + int(self.len_shot / 2)
                if not os.path.isdir(self.summ_path):
                    os.mkdir(self.summ_path)
                for k in range(init_keyshots, end_keyshots):
                    keyshot = str(comp_leaf_list[k]).zfill(6)
                    os.system('cp {}frames/{}.jpg {}{}.jpg'.format(self.video_file, keyshot, self.summ_path, keyshot))
                    self.summ_input.content += f"{keyshot}\n"

    def generate_video(self):
        image_folder = self.summ_path.rstrip(os.sep)
        images = [img for img in os.listdir(image_folder)
                  if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")]
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        os.chdir(self.output_skim)
        video = cv2.VideoWriter(self.video + '.mp4', 0, self.rate, (width, height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release()

    def calc_init(self, i, delta_t, frame_len):
        return 0 if (i < delta_t or delta_t < 0) else (i - delta_t if (i + delta_t) <= frame_len else i)

    def calc_end(self, i, delta_t, frame_len):
        return frame_len if (i + delta_t) > frame_len or delta_t < 0 else i + delta_t

    def group_central_features(self, graph, leaflist):
        import os
        import numpy as np
        import networkx as nx
        print("DEBUG: Initial len_shot:", self.len_shot)
        print("DEBUG: Initial cut_number:", self.cut_number)
        
        base_name = os.path.splitext(os.path.basename(self.video))[0]
        
        # Define output paths for final outputs (written to disk).
        output_resnet_path = os.path.join(self.out_skim_dir, base_name + ".npy")
        output_txt_path = os.path.join(self.out_index_dir, base_name + ".txt")
        base_bn_name = base_name.replace('_resnet', '_bn')
        output_bn_path = os.path.join(self.out_skim_dir, base_bn_name + ".npy")
        
        # If the number of features is less than 100, keep them all.
        # if self.features_list is not None and self.features_list.shape[0] < 100:
        #     selected_features = self.features_list
        #     selected_indexes = list(range(self.features_list.shape[0]))
        #     np.save(output_resnet_path, selected_features)
        #     with open(output_txt_path, "w") as f:
        #         for idx in selected_indexes:
        #             f.write(str(idx) + "\n")
        #     bn_file_name = os.path.basename(self.video).replace('_resnet.npy', '_bn.npy')
        #     bn_file_path = os.path.join(os.path.dirname(self.video), bn_file_name)
        #     if not os.path.exists(bn_file_path):
        #         return
        #     bn_features = np.load(bn_file_path)
        #     selected_bn_features = bn_features[np.array(selected_indexes, dtype=int)]
        #     np.save(output_bn_path, selected_bn_features)
        #     return

        if os.path.exists(output_resnet_path) and os.path.exists(output_txt_path) and os.path.exists(output_bn_path):
            return
        
        if os.path.exists(output_resnet_path) and os.path.exists(output_txt_path) and not os.path.exists(output_bn_path):
            selected_indexes = []
            with open(output_txt_path, "r") as f:
                for line in f:
                    try:
                        idx = int(line.strip())
                        selected_indexes.append(idx)
                    except Exception as e:
                        print("DEBUG: Error parsing index:", line, e)
            bn_file_name = os.path.basename(self.video).replace('_resnet.npy', '_bn.npy')
            bn_file_path = os.path.join(os.path.dirname(self.video), bn_file_name)
            if not os.path.exists(bn_file_path):
                return
            bn_features = np.load(bn_file_path)
            selected_bn_features = bn_features[np.array(selected_indexes, dtype=int)]
            np.save(output_bn_path, selected_bn_features)
            return

        subgraphs = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]
        print("DEBUG: Number of connected subgraphs:", len(subgraphs))
        if len(subgraphs) == 0:
            return
        
        print("DEBUG: self.len_shot before division:", self.len_shot)
        self.len_shot = int(self.len_shot / len(subgraphs))
        print("DEBUG: self.len_shot after division:", self.len_shot)
        
        selected_features = []
        selected_indexes = []
        for idx, comp in enumerate(subgraphs):
            nodes_list = list(comp.nodes)
            comp_leaf_list = [node for node in nodes_list if node in leaflist]
            print(f"DEBUG: Subgraph {idx} node count:", len(nodes_list))
            print(f"DEBUG: Subgraph {idx} leaf list count:", len(comp_leaf_list))
            if comp_leaf_list:
                cn = int(len(comp_leaf_list) / 2)
                if len(comp_leaf_list) < self.len_shot:
                    init_keyshots = 0
                    end_keyshots = len(comp_leaf_list)
                else:
                    init_keyshots = max(0, cn - int(self.len_shot / 2))
                    end_keyshots = min(len(comp_leaf_list), cn + int(self.len_shot / 2))
                print("DEBUG: Keyshot range:", init_keyshots, "to", end_keyshots)
                for k in range(init_keyshots, end_keyshots):
                    key_index = comp_leaf_list[k]
                    selected_features.append(self.features_list[int(key_index)])
                    selected_indexes.append(key_index)
                    self.summ_input.content += f"{key_index}\n"
        
        selected_features_array = np.array(selected_features)
        np.save(output_resnet_path, selected_features_array)
        with open(output_txt_path, "w") as f:
            for idx in selected_indexes:
                f.write(str(idx) + "\n")
        bn_file_name = os.path.basename(self.video).replace('_resnet.npy', '_bn.npy')
        bn_file_path = os.path.join(os.path.dirname(self.video), bn_file_name)
        if not os.path.exists(bn_file_path):
            return
        bn_features = np.load(bn_file_path)
        print("DEBUG: BN features shape:", bn_features.shape)
        selected_bn_features = bn_features[np.array(selected_indexes, dtype=int)]
        np.save(output_bn_path, selected_bn_features)
        print("DEBUG: Saved new bn features to", output_bn_path)
