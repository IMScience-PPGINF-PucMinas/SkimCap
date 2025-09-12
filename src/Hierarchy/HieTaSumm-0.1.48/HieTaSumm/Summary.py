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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
class Summary:
    def __init__(self, dataset_frames, video, rate, time, hierarchy, selected_model, is_binary, percent, alpha, gen_summary_method):
        self.delta_t = rate * time

        # If input is precomputed features (full path expected)
        if video.endswith('.npy'):
            # Set output directories for features processing.
            #self.out_skim_dir = '/home/bernardop/recurrent-transformer/video_feature/rt_anet_feat/trainvalskim_dynamics_sparsed100'
            #self.out_index_dir = '/home/bernardop/recurrent-transformer/video_feature/rt_anet_feat/trainvalindexes_dynamics_sparsed100'
            self.out_skim_dir  = '/Disco01/recurrent-transformer/video_feature/rt_anet_feat/trainvalskim_area_sparsed100'
            self.out_index_dir = '/Disco01/recurrent-transformer/video_feature/rt_anet_feat/trainvalindexes_area_sparsed100'
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
            #print("DEBUG: Graph content is empty. Calling load_features...")
            # Clear previous content:
            self.input_graph_file.content = ""
            if features_list is None:
                features_list = self.model.features(self.frames_path)
                self.frame.load(self.frames_path, self.delta_t, self.input_graph_file, features_list)
            else:
                self.frame.load_features(features_list, self.input_graph_file, self.delta_t)
        else:
            pass

        #Force reprocessing for small videos
        # if self.features_list is not None and self.features_list.shape[0] < 100:
        #     print("DEBUG: Small video detected (<100 frames). Forcing full processing with no reduction.")
        #     # Remove existing in-memory higra data to force reprocessing.
        #     self.input_higra.content = ""
        
        # gen_summary_method can be a single dict or a list of dicts
        # methods = (
        #     gen_summary_method
        #     if isinstance(gen_summary_method, list)
        #     else [gen_summary_method]
        # )

        # --- multithreaded processing ---
        # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # #with ThreadPoolExecutor(max_workers=len(methods)) as executor:
        #     futures = [executor.submit(self._process_keyframes, m) for m in methods]
        #     for f in futures:
        #         f.result()  # will raise if any thread errored
        # Process gen_summary_methods in batches without concurrency
        # for i in range(0, len(methods), 10):
        #     batch = methods[i : i + 10]
        #     for m in batch:
        #         self._process_keyframes(m)

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


    # def _process_keyframes(self, gen_summary_method):
    #     # Process keyframe (or key-feature) selection.
    #     if self.input_higra.content == "":
    #         if gen_summary_method['method'] == 'n_fixed_keyframes':
    #             self.cut_number = gen_summary_method['n_keyframes'] - 1
    #             base_len = (
    #                 len(self.features_list)
    #                 if self.features_list is not None
    #                 else len(os.listdir(self.frames_path))
    #             )
    #             self.len_shot = round(base_len * (self.percent / 100)) - 1

    #         elif gen_summary_method['method'] in ['group_central_frames', 'percent_spaced']:
    #             base_len = (
    #                 len(self.features_list)
    #                 if self.features_list is not None
    #                 else len(os.listdir(self.frames_path))
    #             )
    #             self.len_shot = round(base_len * (self.percent / 100)) - 1

    #             self.cut_number = int(self.bestCutNumber(self.features_list) * (self.alpha / 100))
    #             if self.cut_number <= 2:
    #                 self.cut_number = 3

    #         if gen_summary_method['method'] == 'sequential_keyframe':
    #             self.sequential_keyframe(gen_summary_method['n_keyframes'])
    #             return

    #         tree = self.input_graph_file.read_graph_file(Files(), cut_graph=False, cut_number=0)
    #         leaflist = self.graph.compute_hierarchy(tree, self.input_higra)
    #         cuted_graph = self.graph.cut_graph(
    #             self.input_higra, self.cut_graph_file, cutNumber=self.cut_number
    #         )

    #         if gen_summary_method['method'] == 'group_central_frames':
    #             if self.features_list is not None:
    #                 self.group_central_features(cuted_graph, leaflist)
    #             else:
    #                 self.group_central_frames(cuted_graph, leaflist)
    #         else:
    #             self.get_n_frames(cuted_graph, leaflist)
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
            elif gen_summary_method['method'] in ['group_central_frames', 'percent_spaced', 'group_sparse_central_features', 'uniform_step_features']:
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
                # else:
                #     self.get_n_frames(cuted_graph, leaflist)

                elif gen_summary_method['method'] == 'group_sparse_central_features':
            # calls your already-implemented method
                    self.group_sparse_central_features(cuted_graph, leaflist)
                elif gen_summary_method['method'] == 'uniform_step_features':
                    self.uniform_step_features(cuted_graph, leaflist)

            # Instead of creating a folder on disk, we keep the summary in memory.
            # (If you need to output final results, you can later write self.summ_input.content to disk.)
        # Log final debug info:
        
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
        #print("DEBUG: bestcutnumber:", np.round(cut) - 1)
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
        #print("DEBUG: Initial len_shot:", self.len_shot)
        #print("DEBUG: Initial cut_number:", self.cut_number)
        
        base_name = os.path.splitext(os.path.basename(self.video))[0]
        
        # Define output paths for final outputs (written to disk).
        output_resnet_path = os.path.join(self.out_skim_dir, base_name + ".npy")
        output_txt_path = os.path.join(self.out_index_dir, base_name + ".txt")
        base_bn_name = base_name.replace('_resnet', '_bn')
        output_bn_path = os.path.join(self.out_skim_dir, base_bn_name + ".npy")
        
        #If the number of features is less than 100, keep them all.
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
        #print("DEBUG: Number of connected subgraphs:", len(subgraphs))
        if len(subgraphs) == 0:
            return
        
        #print("DEBUG: self.len_shot before division:", self.len_shot)
        #self.len_shot = int(self.len_shot / len(subgraphs))
        self.len_shot = max(int(self.len_shot / len(subgraphs)), 10)
        #print("DEBUG: self.len_shot after division:", self.len_shot)
        
        # ----------------------------------------------
# after you build `subgraphs`
        desired_window = max(self.len_shot, 1)      # e.g. 9
        #print("DEBUG desired window per component:", desired_window)
        # ----------------------------------------------

        selected_features, selected_indexes = [], []

        for idx, comp in enumerate(subgraphs):
            comp_leaf_list = sorted(n for n in comp.nodes if n in leaflist)
            total_leaves   = len(comp_leaf_list)
            #print(f"DEBUG comp {idx} leaves: {total_leaves}")

            if total_leaves == 0:
                continue

            # we’ll take either the whole list or exactly `desired_window`
            q = min(desired_window, total_leaves)

            # centre of the leaf list
            mid      = total_leaves // 2
            half_win = q // 2
            start    = max(0, mid - half_win)
            end      = start + q                       # end is exclusive

            # adjust if we ran past the end
            if end > total_leaves:
                end   = total_leaves
                start = end - q

            #print(f"DEBUG comp {idx} selecting indices {start}:{end} "
            #    f"(size {q})")

            for k in range(start, end):
                key_index = comp_leaf_list[k]
                selected_features.append(self.features_list[key_index])
                selected_indexes.append(key_index)
                self.summ_input.content += f"{key_index}\n"

        # for idx, comp in enumerate(subgraphs):
        #     nodes_list = list(comp.nodes)
        #     comp_leaf_list = [node for node in nodes_list if node in leaflist]
        #     print(f"DEBUG: Subgraph {idx} node count:", len(nodes_list))
        #     print(f"DEBUG: Subgraph {idx} leaf list count:", len(comp_leaf_list))
        #     if comp_leaf_list:
        #         cn = int(len(comp_leaf_list) / 2)
        #         if len(comp_leaf_list) < self.len_shot:
        #             init_keyshots = 0
        #             end_keyshots = len(comp_leaf_list)
        #         else:
        #             init_keyshots = max(0, cn - int(self.len_shot / 2))
        #             end_keyshots = min(len(comp_leaf_list), cn + int(self.len_shot / 2))
        #         print("DEBUG: Keyshot range:", init_keyshots, "to", end_keyshots)
        #         for k in range(init_keyshots, end_keyshots):
        #             key_index = comp_leaf_list[k]
        #             selected_features.append(self.features_list[int(key_index)])
        #             selected_indexes.append(key_index)
        #             self.summ_input.content += f"{key_index}\n"
            # ---------------------------------------
    # Guarantee final length == 100 (or less)
    # ---------------------------------------
        MAX_KEEP = 100
        total_available = self.features_list.shape[0]      # rows in the source array
        current_keep    = len(selected_indexes)            # what you just picked

        if total_available <= MAX_KEEP:
            # ---- 1) Video is short: ignore the earlier selection and dump everything
            selected_indexes  = list(range(total_available))
            selected_features = self.features_list.tolist()

        else:
            # ---- 2) Video is long enough to need exactly 100 outputs
            if current_keep < MAX_KEEP:
                # -- 2a) add extra indices to reach 100  ---------------------------
                need    = MAX_KEEP - current_keep
                # indices not yet chosen, in ascending order
                remaining = [i for i in range(total_available)
                            if i not in selected_indexes]

                # pick `need` evenly-spaced extras so we keep a global spread
                step = len(remaining) / need
                extra = [remaining[int(round(k * step))] for k in range(need)]

                # append in order
                for idx in extra:
                    selected_indexes.append(idx)
                    selected_features.append(self.features_list[idx])

            elif current_keep > MAX_KEEP:
                # -- 2b) trim down evenly to exactly 100  --------------------------
                step = current_keep / MAX_KEEP
                keep_mask = [int(round(k * step)) for k in range(MAX_KEEP)]
                selected_indexes  = [selected_indexes[i]  for i in keep_mask]
                selected_features = [selected_features[i] for i in keep_mask]


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
        #print("DEBUG: BN features shape:", bn_features.shape)
        selected_bn_features = bn_features[np.array(selected_indexes, dtype=int)]
        np.save(output_bn_path, selected_bn_features)
        #print("DEBUG: Saved new bn features to", output_bn_path)

        



    def group_sparse_central_features(self, graph, leaflist):
        """
        Sparse central feature selection:
        - Select k = min(total_frames * percent/100, 100) features.
        - Use bestCutNumber and alpha to compute cut_number.
        - Cut the graph into components.
        - Pick the middle leaf in each component (first if size=2, itself if size=1).
        - Trim to k selections, then save .npy and .txt outputs plus BN features.
        """
        # Determine total frames
        if self.features_list is not None:
            total = self.features_list.shape[0]
        else:
            total = len(os.listdir(self.frames_path))


        # Compute target k
        #k = min(int(total * (self.percent / 100)), 100) # USAR ESSE PRO 3
        k = 100 if total > 100 else total #USAR ESSE PRO 4


      
        cut_number = k
        # Perform hierarchy and graph cut
        tree = self.input_graph_file.read_graph_file(Files(), cut_graph=False, cut_number=0)
        cuted_graph = self.graph.cut_graph(
            self.input_higra, self.cut_graph_file, cutNumber=cut_number
        )

        # Extract connected subgraphs
        subgraphs = [
            cuted_graph.subgraph(c).copy()
            for c in nx.connected_components(cuted_graph)
        ]
        if not subgraphs:  # at least some frames
            return

        # Select one feature per component
        selected_features = []
        selected_indexes = []
        for comp in subgraphs:
            comp_leaf = [n for n in comp.nodes if n in leaflist]
            if not comp_leaf:
                continue
            size = len(comp_leaf)
            mid = size // 2 if size != 2 else 0
            key_index = comp_leaf[mid]
            selected_features.append(self.features_list[int(key_index)])
            selected_indexes.append(key_index)
            self.summ_input.content += f"{key_index}\n"

        # Trim to k if we have more
        if len(selected_features) > k:
            selected_features = selected_features[:k]
            selected_indexes = selected_indexes[:k]

        # Prepare output paths
        base = os.path.splitext(os.path.basename(self.video))[0]
        out_resnet = os.path.join(self.out_skim_dir, base + ".npy")
        out_txt = os.path.join(self.out_index_dir, base + ".txt")

        # Save features and indexes
        np.save(out_resnet, np.array(selected_features))
        with open(out_txt, "w") as f:
            for idx in selected_indexes:
                f.write(f"{idx}\n")

        # Save BN features if available
        bn_name = os.path.basename(self.video).replace('_resnet.npy', '_bn.npy')
        bn_path = os.path.join(os.path.dirname(self.video), bn_name)
        if os.path.exists(bn_path):
            bn_feats = np.load(bn_path)
            bn_selected = bn_feats[np.array(selected_indexes, dtype=int)]
            out_bn = os.path.join(self.out_skim_dir, base.replace('_resnet', '_bn') + ".npy")
            np.save(out_bn, bn_selected)

        # —— Visualization of full graph —— #
        # import matplotlib.pyplot as plt
        # # 1. Layout
        # pos = nx.spring_layout(cuted_graph, seed=42)

        # # 2. Color components
        # components = list(nx.connected_components(cuted_graph))
        # cmap = plt.cm.get_cmap('tab20', len(components))
        # node_comp = {n: ci for ci, comp in enumerate(components) for n in comp}
        # node_colors = [cmap(node_comp[n]) for n in cuted_graph.nodes()]

        # # 3. Draw nodes by component
        # nx.draw_networkx_nodes(
        #     cuted_graph, pos,
        #     node_size=100,
        #     node_color=node_colors,
        #     edgecolors='k',
        #     linewidths=0.5
        # )

        # # 4. Highlight selected leaves
        # nx.draw_networkx_nodes(
        #     cuted_graph, pos,
        #     nodelist=selected_indexes,
        #     node_size=300,
        #     node_color='none',
        #     edgecolors='red',
        #     linewidths=2.0
        # )

        # # 5. Draw edges
        # nx.draw_networkx_edges(cuted_graph, pos, alpha=0.4, edge_color='grey')

        # # 6. Optional labels on selected
        # labels = {n: str(n) for n in selected_indexes}
        # nx.draw_networkx_labels(cuted_graph, pos, labels=labels, font_size=8, font_color='red')

        # plt.axis('off')
        # plt.title(f"Components: {len(components)} · Selected: {len(selected_indexes)}")
        # plt.show()


    # def group_sparse_central_features(self, graph, leaflist):
    #         """
    #         Sparse central feature selection:
    #         - Select k = min(total_frames * percent/100, 100) features.
    #         - Use bestCutNumber and alpha to compute cut_number.
    #         - Cut the graph into components.
    #         - Pick the middle leaf in each component (first if size=2, itself if size=1).
    #         - Trim to k selections, then save .npy and .txt outputs plus BN features.
    #         """
    #         # Determine total frames
    #         if self.features_list is not None:
    #             total = self.features_list.shape[0]
    #         else:
    #             total = len(os.listdir(self.frames_path))
    #         # Compute target k
    #         k = min(int(total * (self.percent / 100)), 100)

    #         # Compute graph cut count
    #         #cut_number = int(self.bestCutNumber(self.features_list) * (self.alpha / 100))
    #         cut_number = k
    #         # if cut_number < 1:
    #         #     cut_number = 1

    #         # Perform hierarchy and graph cut
    #         tree = self.input_graph_file.read_graph_file(Files(), cut_graph=False, cut_number=0)
    #         cuted_graph = self.graph.cut_graph(
    #             self.input_higra, self.cut_graph_file, cutNumber=cut_number
    #         )

    #         # Extract connected subgraphs
    #         subgraphs = [
    #             cuted_graph.subgraph(c).copy()
    #             for c in nx.connected_components(cuted_graph)
    #         ]
    #         if not subgraphs: #ter pelo menos 10 frames
    #             return

    #         # Select one feature per component
    #         selected_features = []
    #         selected_indexes = []
    #         for comp in subgraphs:
    #             # find leaves in this component
    #             comp_leaf = [n for n in comp.nodes if n in leaflist]
    #             if not comp_leaf:
    #                 continue
    #             size = len(comp_leaf)
    #             # pick middle, or first if size=2
    #             mid = size // 2 if size != 2 else 0
    #             key_index = comp_leaf[mid]
    #             selected_features.append(self.features_list[int(key_index)])
    #             selected_indexes.append(key_index)
    #             # accumulate in summary input
    #             self.summ_input.content += f"{key_index}\n"

    #         # Trim to k if we have more
    #         if len(selected_features) > k:
    #             selected_features = selected_features[:k]
    #             selected_indexes = selected_indexes[:k]

    #         # Prepare output paths
    #         base = os.path.splitext(os.path.basename(self.video))[0]
    #         out_resnet = os.path.join(self.out_skim_dir, base + ".npy")
    #         out_txt = os.path.join(self.out_index_dir, base + ".txt")

    #         # Save features and indexes
    #         np.save(out_resnet, np.array(selected_features))
    #         with open(out_txt, "w") as f:
    #             for idx in selected_indexes:
    #                 f.write(f"{idx}\n")

    #         # Save BN features if available
    #         bn_name = os.path.basename(self.video).replace('_resnet.npy', '_bn.npy')
    #         bn_path = os.path.join(os.path.dirname(self.video), bn_name)
    #         if not os.path.exists(bn_path):
    #             return
    #         bn_feats = np.load(bn_path)
    #         bn_selected = bn_feats[np.array(selected_indexes, dtype=int)]
    #         out_bn = os.path.join(self.out_skim_dir, base.replace('_resnet', '_bn') + ".npy")
    #         np.save(out_bn, bn_selected)
    def uniform_step_features(self, graph, leaflist):
        """
        Uniform-step feature selection
        --------------------------------
        • If total ≤ 100 ⇒ keep *all* frames/features.
        • Else          ⇒ pick **exactly 100** features, evenly spaced:
              step = total_frames / 100.0
              idx  = round(step), round(2·step), …, round(100·step)
          This gives a near-perfect spread (handles non-integer steps like
          385 / 100 = 3.85 automatically).

        After selecting the indices we:
          1. Save the chosen RESNET features      →  out_skim_dir/*.npy
          2. Save the text list of indices        →  out_index_dir/*.txt
          3. If the companion *_bn.npy is present →  save the matching BN
             vectors in out_skim_dir as well.

        (All output paths follow the same conventions as your two existing
        methods, so nothing else in the pipeline has to change.)
        """
        import os, numpy as np

        # -------- work out how many frames/features we have ---------------
        total = (self.features_list.shape[0]
                 if self.features_list is not None
                 else len(os.listdir(self.frames_path)))

        if total == 0:          # safety guard
            return

        # -------- decide which 0-based indices we want --------------------
        if total <= 100:
            sel_idx = list(range(total))
        else:
            # linspace gives the most even spread & *exactly* 100 values
            sel_idx = list(np.linspace(0, total - 1, num=100, dtype=int))
            # OPTIONAL: if you prefer the *odd* indices style (1,3,5 …)
            # uncomment the 2 lines below:
            # if sel_idx[0] == 0:
            #     sel_idx = [min(i + 1, total - 1) for i in sel_idx]

        # -------- gather the feature vectors ------------------------------
        if self.features_list is None:
            # should never happen for pre-computed feature branch, but
            # included for completeness
            self.features_list = self.model.features(self.frames_path)
        selected_features = self.features_list[sel_idx]

        # -------- build output paths --------------------------------------
        base = os.path.splitext(os.path.basename(self.video))[0]
        out_resnet = os.path.join(self.out_skim_dir,  base + ".npy")
        out_txt    = os.path.join(self.out_index_dir, base + ".txt")
        out_bn     = os.path.join(
            self.out_skim_dir, base.replace("_resnet", "_bn") + ".npy"
        )
        os.makedirs(self.out_skim_dir,   exist_ok=True)
        os.makedirs(self.out_index_dir,  exist_ok=True)

        # -------- save RESNET vectors & index list ------------------------
        np.save(out_resnet, selected_features)
        with open(out_txt, "w") as f:
            for idx in sel_idx:
                f.write(f"{idx}\n")
                self.summ_input.content += f"{idx}\n"   # keep the in-mem log

        # -------- copy the matching BN vectors, if they exist -------------
        bn_source = os.path.join(
            os.path.dirname(self.video),
            os.path.basename(self.video).replace("_resnet.npy", "_bn.npy")
        )
        if os.path.exists(bn_source):
            bn = np.load(bn_source, mmap_mode="r")      # header-only read
            np.save(out_bn, bn[np.array(sel_idx, dtype=int)])
