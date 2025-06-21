import os, argparse, json, math, yaml
import numpy as np
from typing import Iterator, Tuple
from multiprocessing import Pool
from PIL import Image
from streaming import MDSWriter, StreamingDataset
from streaming.base.util import merge_index


DATASET = None
COLUMNS = {'image': 'jpeg_list', 'text': 'json'}


def init_worker() -> None:
    print(f'\nInitialised worker PID {os.getpid()}', flush=True, end='')


def get_datum(item):
    if item.get('image', None):
        image_path = [os.path.join(ARGS.root_path, ARGS.image_folder, im) for im in item['image']]
        img = [Image.open(im).convert('RGB') for im in image_path]
        sample = {"image": img, "text": item}
    else:
        assert False, f'No image found: {item}'
    return sample


def each_task(out_root: str, groups: int) -> Iterator[Tuple[str, int, int]]:
    """Yield a (subdir, start_idx, end_idx) triple for every worker."""
    num_items = len(DATASET)
    per_group = num_items // groups + 1
    for g in range(groups):
        yield (os.path.join(out_root, str(g)),
               per_group * g,
               min(per_group * (g + 1) - 1, num_items - 1))


def convert_to_mds(task: Tuple[str, int, int]) -> None:
    """Worker entry point – write one shard group."""
    sub_out_root, start, end = task
    print(f'\n{os.getpid()} → {sub_out_root} [{start}-{end}]', flush=True, end='')
    with MDSWriter(out=sub_out_root, columns=COLUMNS, size_limit='1000000kb') as out:
        for i in range(start, end + 1):
            out.write(get_datum(DATASET[i]))


def main():
    global DATASET, ARGS
    args = ARGS

    # ------------ load & filter dataset ------------
    with open(args.yaml_file, 'r') as f:
        cfg = yaml.safe_load(f)

    json_files, percents = [], []
    for d in cfg['datasets']:
        json_path, pct = d['json_path'], d['sampling_strategy']
        percents.append(pct)
        json_files.append(json_path)

    assert len(json_files) == len(percents)
    dataset = []
    for idx, (percent_str, file) in enumerate(zip(percents, json_files)):
        with open(os.path.join(file), 'r') as f:
            tmp_dataset = json.load(f)
            percent = float(percent_str.split(":")[-1][:-1]) / 100 if percent_str != 'all' else 1

            if percent < 1:
                percent_type = percent_str.split(":")[0]
                print(f'Using {percent_type.upper()} percent of the dataset, percent: {percent}')
                print(f'Original dataset size: {len(tmp_dataset)}')
                if percent_type == 'first':
                    tmp_dataset = tmp_dataset[:math.ceil(len(tmp_dataset) * percent)]
                elif percent_type == 'end':
                    tmp_dataset = tmp_dataset[math.floor(len(tmp_dataset) * (1-percent)):]
                elif percent_type == 'random':
                    np.random.shuffle(tmp_dataset)
                    tmp_dataset = tmp_dataset[:math.floor(len(tmp_dataset) * percent)]
                else:
                    raise ValueError('Invalid percent type')
            for item in tmp_dataset:
                item['source'] = file.split('/')[-1].replace('.json', '')
            if args.num_shards > 1:
                rand_ids = np.arange(len(tmp_dataset))
                random_state = np.random.RandomState(idx)
                random_state.shuffle(rand_ids)
                # Split the dataset into num_shards
                tmp_dataset = [tmp_dataset[i] for i in range(len(tmp_dataset)) if i % args.num_shards == args.shard_id]
            dataset += tmp_dataset
            print(f'Loaded {len(tmp_dataset)} samples from {file}')

    print(f'Number of samples: {len(dataset)}')
    print()

    if args.debug:
        exit()
    
    np.random.shuffle(dataset)
    
    DATASET = dataset
    print(f'Final #samples = {len(DATASET)}')

    # ------------- parallel write -----------------
    tasks = list(each_task(args.output_folder, args.processes))
    with Pool(processes=args.processes, initializer=init_worker) as pool:
        pool.map(convert_to_mds, tasks)

    merge_index(args.output_folder, keep_local=True)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default=None, help=("image_folder"))
    parser.add_argument("--yaml_file", type=str, default='/data/dataset')
    parser.add_argument("--root_path", type=str, default='/data/dataset')
    parser.add_argument("--output_folder", type=str, default='image_shard')
    parser.add_argument('--processes', default=32, type=int)
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--percents', default='all', nargs='+', type=str)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--exclude_text", type=int, default=0)
    parser.add_argument("--num_shards", type=float, default=1)
    ARGS = parser.parse_args()
    main()