import torch
import numpy as np
from .primes import find_largest_3k_plus_2_prime

dtypes = {
    "int8": np.int8,
    "uint8": np.uint8,
    "int16": np.int16,
    "uint16": np.uint16,
    "int32": np.int32,
    "int64": np.int64,
    "": None,
}

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.bin_buffer_mmap = np.memmap(args.data_file + ".bin", mode='r', order='C')
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        if hasattr(args, 'data_dtype') and args.data_dtype != '':
            self.dtype = dtypes[args.data_dtype]
        else:
            self.dtype = np.uint16

        self.dtype_size = self.dtype().itemsize

        assert args.vocab_size <= 256 ** self.dtype_size
        assert len(self.bin_buffer) % self.dtype_size == 0

        self.data_size = len(self.bin_buffer) // self.dtype_size
        self.samples_per_epoch = args.epoch_steps * args.real_bsz
        # assert self.samples_per_epoch == 40320
        self.magic_prime = find_largest_3k_plus_2_prime(self.data_size // args.ctx_len)
        assert self.magic_prime == args.magic_prime
        self.factor = int(self.magic_prime * (5**0.5 - 1) / 2)

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        world_size = self.world_size
        ctx_len = args.ctx_len
        ii = self.real_epoch * self.samples_per_epoch + (
            idx * world_size) + self.global_rank + args.my_data_shift
        i = ((self.factor * ii * ii * ii) % self.magic_prime) * ctx_len
        np_array = np.frombuffer(self.bin_buffer, dtype=self.dtype, count=ctx_len+1, offset=i*self.dtype_size)
        x = torch.tensor(np_array[:-1], dtype=torch.long)
        y = torch.tensor(np_array[1:], dtype=torch.long)
        return x, y
