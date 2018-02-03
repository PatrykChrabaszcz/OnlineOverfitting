import numpy as np


class Dataset:
    ModeZeros = 'Zeros'
    ModeSameRandom = 'SameRandom'
    ModeSameDistRandom = 'SameDistRandom'
    ModeDiffRandom = 'DiffDistRandom'

    def __init__(self, input_size=1, examples_per_class=32, seq_size=10000, mode='Zeros'):
        data = []
        labels = []
        self.seq_size = seq_size
        self.index = 0

        shape = (examples_per_class, seq_size, input_size)
        same = np.random.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32)

        for c in range(2):
            if mode == self.ModeZeros:
                data.append(np.zeros(shape=shape, dtype=np.float32))

            elif mode == self.ModeSameDistRandom:
                data.append(np.random.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32))

            elif mode == self.ModeSameRandom:
                data.append(same)

            elif mode == self.ModeDiffRandom:
                data.append(np.random.normal(loc=0.5 * c, scale=1.0, size=shape).astype(np.float32))
            else:
                raise RuntimeError('Mode %s not available' % mode)

            labels.append(np.ones((examples_per_class, seq_size), dtype=np.int) * c)

        self.data = np.concatenate(data)
        self.labels = np.concatenate(labels)

    def generate_minibatches(self, size):
        for i in range(self.seq_size // size):
            yield self.data[:, i*size: (i+1)*size, :], self.labels[:, i*size: (i+1)*size]
