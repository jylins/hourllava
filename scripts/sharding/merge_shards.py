import os
import json
from glob import glob
import argparse


def with_id(basename: str, shard_id: int) -> str:
    """Get a new basename with the given shard_id.

    Args:
        basename (str): Old basename of file.
        shard_id (int): New shard ID.

    Returns:
        str: New basename of file.
    """
    parts = basename.split('.')
    parts[1] = f'{shard_id:05}'
    return '.'.join(parts)


def merge_shard_groups(root: str) -> None:
    """Merge ephemeral sub-datasets created in parallel into one dataset.

    Args:
        root (str): Root directory.
    """
    subdirs = sorted(os.listdir(root))
    
    infos = []
    for subdir in subdirs:
        index_filename = os.path.join(root, subdir, 'index.json')
        obj = json.load(open(index_filename))
        for info in obj['shards']:
            old_basename = info['raw_data']['basename']
            new_basename = os.path.join(subdir, old_basename)
            info['raw_data']['basename'] = new_basename
            print(old_basename, '->', new_basename)
            infos.append(info)
    index_filename = os.path.join(root, 'index.json')
    obj = {
        'version': 2,
        'shards': infos,
    }
    text = json.dumps(obj, sort_keys=True)
    with open(index_filename, 'w') as out:
        out.write(text)


def main(args):
    """Main function to merge shard groups.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    root = args.root
    merge_shard_groups(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Merge shard groups into one dataset.')
    parser.add_argument('--root', type=str, required=True, help='Root directory of the dataset.')
    args = parser.parse_args()
    main(args)