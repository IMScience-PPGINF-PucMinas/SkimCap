import os

os.system("bash ./scripts/train.sh ./video_feature/trainvalindexes_nparents_central100")
os.system("bash ./scripts/train.sh ./video_feature/trainvalindexes_nparents_sparsed100")
os.system("bash ./scripts/train.sh ./video_feature/trainvalindexes_area_central100")
os.system("bash ./scripts/train.sh ./video_feature/trainvalindexes_area_sparsed100")
os.system("bash ./scripts/train.sh ./video_feature/trainvalindexes_dynamics_central100")
os.system("bash ./scripts/train.sh ./video_feature/trainvalindexes_dynamics_sparsed100")
os.system("bash ./scripts/train.sh ./video_feature/trainvalindexes_volume_central100")
os.system("bash ./scripts/train.sh ./video_feature/trainvalindexes_volume_sparsed100")
