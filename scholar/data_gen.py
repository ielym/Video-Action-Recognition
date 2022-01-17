from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import nvidia.dali.types as types

import os

@pipeline_def()
def video_pipe(file_list, sequence_length, input_size, stride, shard_id, num_shards, initial_fill):
    videos, labels = fn.readers.video_resize(
        device="gpu",
        resize_shorter=256,
        interp_type=types.INTERP_LINEAR,
        file_list=file_list,
        sequence_length=sequence_length,
        pad_sequences=True,
        stride=stride,
        step=stride,
        shard_id=shard_id,
        num_shards=num_shards,
        random_shuffle=True,
        initial_fill=initial_fill,
        file_list_include_preceding_frame=False,
        dtype=types.UINT8,
        prefetch_queue_depth=4,
        minibatch_size=32,
        # seed=0,
    )

    # videos = fn.resize(videos, resize_shorter=256, interp_type=types.INTERP_LINEAR)
    videos = fn.random_resized_crop(videos, size=(input_size, input_size), random_area=[1.0, 1.5], device="gpu")
    videos = fn.flip(videos, horizontal=fn.random.coin_flip(), device="gpu")
    videos = fn.transpose(videos, perm=[3, 0, 1, 2], device="gpu") # torch.Size([16, 8, 224, 224, 3])

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
    print(len(lines))
    return os.path.join(cache_dir, os.path.basename(prefix))


def create_dataloader(prefix, data_dir, batch_size, frames, input_size, device_id, shard_id, num_shards, num_workers=1, cache='./cache'):

    stride = 8
    initial_fill = batch_size * 4

    fastvision_labels_path = os.path.join(data_dir, 'labels.txt')
    file_list_path = trans_fastvision_2_dali(prefix, fastvision_labels_path, data_dir, cache_dir=cache)

    pipe = video_pipe(
        file_list=file_list_path,
        sequence_length=frames,
        input_size=input_size,
        stride=stride,
        shard_id=shard_id,
        num_shards=num_shards,
        initial_fill=initial_fill,
        num_threads=8,
        device_id=device_id,
        batch_size=batch_size,
        py_num_workers=num_workers,
        py_start_method='fork',
    )

    dataloader = DALIGenericIterator(pipelines=pipe, output_map=['frames', 'labels'])

    # alredy = set()
    # import numpy as np
    # for batch_idx, data in enumerate(dataloader):
    #     frames = data[0]["frames"]
    #     labels = data[0]["labels"]
    #     print("batch {}, frames size: {}, labels : {}, device : {}".format(batch_idx, frames.size(), labels.cpu().numpy().tolist()[0][0], frames.device))
    #     alredy.add(labels.cpu().numpy().tolist()[0][0])
    #     print(len(alredy))
        # np.save('ttt.npy', frames.cpu().numpy()[0, ...])
        # input('---')

    return dataloader


