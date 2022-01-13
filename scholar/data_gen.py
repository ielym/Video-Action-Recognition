from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types

import os

@pipeline_def
def video_pipe(file_list, sequence_length, input_size, stride, shard_id, num_shards, initial_fill):
    videos, labels = fn.readers.video(device="gpu", file_list=file_list, sequence_length=sequence_length, pad_sequences=True, stride=stride, shard_id=shard_id, num_shards=num_shards, random_shuffle=True, initial_fill=initial_fill, file_list_include_preceding_frame=False)
    videos = fn.resize(videos, resize_shorter=256, interp_type=types.INTERP_LINEAR)
    # videos = fn.crop_mirror_normalize(
    #     videos,
    #     crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
    #     crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
    #     mirror=fn.random.coin_flip(),
    #     dtype=types.FLOAT,
    #     crop=(input_size, input_size),
    #     mean=[0., 0., 0.],
    #     std=[255., 255., 255.],
    # )
    return videos, labels

def trans_fastvision_2_dali(prefix, label_path, data_dir, cache_dir):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    with open(os.path.join(cache_dir, os.path.basename(prefix)), 'w') as f:
        for line in lines:
            line = line.strip()
            file_name, category_idx = line.split()
            file_path = os.path.join(data_dir, 'videos', file_name)
            f.write(f'{file_path} {category_idx}\n')

    return os.path.join(cache_dir, os.path.basename(prefix))


def create_dataloader(prefix, data_dir, batch_size, frames, input_size, device_id, shard_id, num_shards, num_workers=1, cache='./cache'):

    stride = 8
    initial_fill = batch_size * 2

    fastvision_labels_path = os.path.join(data_dir, 'labels.txt')
    file_list_path = trans_fastvision_2_dali(prefix, fastvision_labels_path, data_dir, cache_dir=cache)

    pipe = video_pipe(file_list=file_list_path, sequence_length=frames, input_size=input_size, stride=stride, shard_id=shard_id, num_shards=num_shards, initial_fill=initial_fill, num_threads=num_workers, device_id=device_id, batch_size=batch_size)

    dataloader = DALIGenericIterator(pipelines=pipe, output_map=['frames', 'labels'])

    for batch_idx, data in enumerate(dataloader):
        frames = data[0]["frames"]
        labels = data[0]["labels"]
        print("batch {}, frames size: {}, labels : {}, device : {}".format(batch_idx, frames.size(), labels.cpu().numpy().tolist(), frames.device))
    # dataloader.reset()

    return dataloader


