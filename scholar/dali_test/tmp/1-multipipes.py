from nvidia.dali import pipeline_def, Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

import numpy as np

@pipeline_def
def caffe_pipeline(image_dir):
    device_id = Pipeline.current().device_id
    jpegs, labels = fn.readers.file(file_root=image_dir, name="Reader")
    images = fn.decoders.image(jpegs, device='mixed')

    # images = fn.resize(images, resize_shorter=fn.random.uniform(range=(256, 480)), interp_type=types.INTERP_LINEAR)
    images = fn.resize(images, resize_shorter=256, interp_type=types.INTERP_LINEAR)
    images = fn.crop_mirror_normalize(
                                        images,
                                        crop_pos_x=fn.random.uniform(range=(0.0, 1.0)),
                                        crop_pos_y=fn.random.uniform(range=(0.0, 1.0)),
                                        mirror=fn.random.coin_flip(),
                                        dtype=types.FLOAT,
                                        crop=(224, 224),
                                        mean=[0., 0., 0.],
                                        std=[1., 1., 1.],
                                )

    return images, labels


if __name__ == '__main__':

    image_dir = r"/app/datasets/dali"
    num_threads = 2
    gpus = [0, 1, 2, 3]  # number of GPUs
    BATCH_SIZE = 8  # batch size per GPU

    pipes = [caffe_pipeline(image_dir=image_dir, batch_size=BATCH_SIZE, num_threads=num_threads, device_id=device_id) for device_id in gpus]

    for pipe in pipes:
        pipe.build()

    dali_iter = DALIGenericIterator(pipelines=pipes, output_map=['data', 'label'], reader_name='Reader')

    for i, data in enumerate(dali_iter):
        for d in data:
            label = d["label"]
            image = d["data"]
            print(label.size(), image.size())

    print("OK")



