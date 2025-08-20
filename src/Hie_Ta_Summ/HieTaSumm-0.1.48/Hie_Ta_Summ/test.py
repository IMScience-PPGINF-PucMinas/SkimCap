import os

directory = "/home/bernardop/recurrent-transformer/video_feature/rt_anet_feat/trainvalskim_200"

file_count = 0
for root, dirs, files in os.walk(directory):
    file_count += len(files)

print(f"Total number of files in '{directory}': {file_count}")
