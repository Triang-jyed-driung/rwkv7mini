import math
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size
        rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        if args.my_pile_version == 1:
            self.data = MMapIndexedDataset(args.data_file)
            self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
            rank_zero_info(f"Data has {self.data_size} tokens.")
        elif args.my_pile_version == 2:
            data_list = open(args.data_file, "r", encoding='utf-8').read().strip().split('\n')
            data_list = [i.strip().split(' ') for i in data_list]
            self.data = []
            self.data_size = int(data_list[-1][-1])
            rank_zero_info(f"Data has {self.data_size} chunks.")
            for d in data_list:
                data = MMapIndexedDataset(d[0])
                data_size = len(data._bin_buffer) // data._index._dtype_size
                assert (data_size - args.ctx_len) == int(d[1])
                self.data += [[int(d[-1]), int(d[1]), data]]
            # rank_zero_info(self.data)
        
        self.data_pile = None
        self.data_pile_size = 0

        if args.my_pile_stage > 0:
            self.samples_per_epoch = args.epoch_steps * args.real_bsz
            assert self.samples_per_epoch == 40320
            rank_zero_info(f"########## Pile 20b-tokenized stage {args.my_pile_stage} ##########")
            dataset_slot = self.data_size // args.ctx_len
            if args.my_pile_stage != 4:
                assert MaybeIsPrime(args.magic_prime)
                assert args.magic_prime % 3 == 2
                assert args.magic_prime / dataset_slot > 0.9 and args.magic_prime / dataset_slot <= 1

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size
        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime
        data = self.data

        if args.my_pile_stage > 0:
            ii = 1 + epoch * self.samples_per_epoch + (idx * world_size) + rank
            if data == self.data_pile:
                i = np.random.randint(0, self.data_pile_size - req_len)
            else:
                if args.my_pile_stage == 4 or ii < args.my_random_steps:
                    # cheat: pick a random spot in dataset
                    if args.my_pile_version == 1:
                        i = np.random.randint(0, self.data_size - req_len)
                    else:
                        i = np.random.randint(0, self.data_size)
                else:
                    ii = ii - args.my_random_steps
                    factor = (math.sqrt(5) - 1) / 2
                    factor = int(magic_prime * factor)
                    i = ((factor * ii * ii * ii) % magic_prime) * ctx_len
                    i = i + args.my_pile_shift
        else:
            i = np.random.randint(0, self.data_size - req_len)
        if args.my_pile_version == 1:
            dix = data.get(idx=0, offset=i, length=req_len).astype(int)
        else:
            for j in range(len(data)):
                if i < data[j][0]:
                    ii = i
                    i = (i - (data[j-1][0] if j > 0 else 0)) % data[j][1]
                    dix = data[j][2].get(idx=0, offset=i, length=req_len).astype(int)
                    break
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
