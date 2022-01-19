import os
import tqdm

ori_path = os.path.join(r'/app/datasets/kinetics400/train/labels.txt')

with open(ori_path, 'r') as f:
	lines = f.readlines()

ori_name_id_map = {}
for line in lines:
	line = line.strip()
	file_name, category_id = line.split()
	ori_name_id_map[file_name] = category_id

names = os.listdir(r'/app/datasets/kinetics400/train5s/videos')
with open(r'/app/datasets/kinetics400/train5s/labels.txt', 'w') as f:
	for name in tqdm.tqdm(names):
		category_id = ori_name_id_map[name.replace('.avi', '.mp4')]
		f.write(f'{name} {category_id}\n')


