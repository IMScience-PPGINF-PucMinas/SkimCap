from HieTaSumm import HieTaSumm
#from . import HieTaSumm
#import numpy as np
#features = np.load('/home/bernardop/Hie_Ta_Summ/HieTaSumm-0.1.48/HieTaSumm/features_directory/plastering_bn.npy')
# print("Features shape:", features.shape)
# print(features[0]) 

#HieTaSumm(video_path='/content/videos', percent=15, rate=2, time=2, alpha=25, selected_model='resnet50', summary_path='/content/skim-8-vgg', gen_summary_method={"method": "percent_spaced"}, hierarchy="watershed_hierarchy_by_area")
#HieTaSumm(video_path='/content/videos', percent=15, rate=30, time=1, alpha=25, selected_model='resnet50', summary_path='/content/skim-8-vgg', gen_summary_method={"method": "group_central_frames"}, hierarchy="watershed_hierarchy_by_area")
HieTaSumm(
        #features_path='/home/bernardop/Hie_Ta_Summ/HieTaSumm-0.1.48/HieTaSumm/features_directory',
        features_path='/home/lvcardoso/recurrent-transformer/video_feature/rt_anet_feat/trainval',
        percent=15,
        rate=30,
        time=4,
        alpha=100,
        selected_model='vgg16',
        summary_path='/home/bernardop/skim-8-vgg',
        gen_summary_method={"method": 'group_sparse_central_features'},#'group_sparse_central_features'},#"group_central_frames"},
        hierarchy="watershed_hierarchy_by_area"
    )


#(Hie_ta_Summ_venv) bernardop@LAPLACE:~/Hie_Ta_Summ/HieTaSumm-0.1.48$ sudo python -m HieTaSumm.One_Line_Function
#home/lvcardoso/recurrent-transformer/video_feature/rt_anet_feat/trainval


#ta no 200 agora

#criar um com alpha=100