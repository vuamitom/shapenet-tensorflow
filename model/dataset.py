import os
import random

class DataSet:
    def __init__(self, path, batch_size):    
        with np.load(path) as ds:
            # ds = np.load(path)
            self.data = ds['data']
            self.labels = ds['labels']
        self.idx = 0
        self.batch_size = batch_size

    def reset_and_shuffle(self):
        permutation = np.random.permutation(self.labels.shape[0])
        self.data = self.data[permutation,:,:,:] if len(self.data.shape) == 4 else self.data[permutation,:,:]
        self.labels = self.labels[permutation]
        self.idx = 0    

    def no_batches(self):
        return math.floor(self.size() * 1.0/ self.batch_size)

    def size(self):
        return self.data.shape[0]

    def next_batch(self):
        if self.idx >= self.size():
            return None, None
        n = self.idx + self.batch_size
        n = n if n < self.size() else self.size()
        data = self.data[self.idx:n, :]
        labels = self.labels[self.idx:n, :]
        self.idx = n
        return data, labels

class ShardedDataSet:
    def __init__(self, prefix, batch_size):
        datadir = os.path.dirname(prefix)
        prefix_name = os.path.basename(prefix)
        paths = []
        for fn in os.path.lisdir(datadir):
            if fn.startswith(prefix_name):
                paths.append(os.path.join(datadir, fn))
        self.shards = paths
        self.reset_and_shuffle()
        self.data_buf = None
        self.label_buf = None

    def reset_and_shuffle(self):
        random.shuffle(self.shards)
        self.idx = 0
        self.shard_idx = 0

    def next_batch(self):
        if self.data_buf is None:
            if self.shard_idx < len(self.shards):
                cur_shard = self.shards[self.shard_idx]
                with np.load(cur_shard) as ds:
                    # ds = np.load(path)
                    self.data_buf = ds['data']
                    self.label_buf = ds['labels']
            else:
                return None

        cur_buf_len = self.data_buf.shape[0]
        n = self.idx + self.batch_size
        n = n if n < cur_buf_len else cur_buf_len
        data = self.data_buf[self.idx: n]
        labels = self.labels[self.idx: n]
        self.idx = n
        
        if self.idx >= cur_buf_len:
            self.idx = 0
            self.shard_idx += 1
            self.data_buf = None
            self.label_buf = None

        return data, labels


