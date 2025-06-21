import os
import transformers
from typing import Dict

from .collator import DataCollatorForSupervisedDataset
from .lazy_supervised_dataset import LazySupervisedDataset
from .streaming_supervised_dataset import StreamingSupervisedDataset


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def make_streaming_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args
) -> Dict:
    """Make streaming dataset and collator for supervised fine-tuning."""
    if data_args.using_local_data:
        train_dataset = StreamingSupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args,
            shuffle=True,
            batch_size=training_args.per_device_train_batch_size,
            batching_method=training_args.batching_method,
        )
    else:
        train_dataset = StreamingSupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_args.data_path,
            data_args=data_args,
            local=os.path.join(os.environ['DATA_CACHE'], 'tmp'),
            shuffle=True,
            batch_size=training_args.per_device_train_batch_size,
            batching_method=training_args.batching_method,
            cache_limit='5000gb',  # Local folder size to put the cached dataset
            shuffle_block_size=500000  # Shuffle Buffer Size
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)