import os

with open(r'./train.txt', 'r') as f:
	samples = eval(f.read())

for sample in samples:
	file_name, category_id = sample

with open(r'./dali_train.txt', 'w') as f:
	for sample in samples:
		file_name, category_id = sample
		f.write(f'{file_name} {category_id}\n')