import numpy as np
import tensorflow as tf

import mrcfile
import tqdm

from os.path import join


class CryoCARE_Dataset(tf.keras.utils.Sequence):
    def __init__(self, tomo_paths_odd=None, tomo_paths_even=None, mask_paths=None,
                 n_samples_per_tomo=None, extraction_shapes=None, mean=None, std=None, split_into=1,
                 sample_shape=(64, 64, 64), shuffle=True, n_normalization_samples=500, tilt_axis=None):
        self.tomo_paths_odd = tomo_paths_odd
        self.tomo_paths_even = tomo_paths_even
        self.mask_paths = mask_paths
        self.split_into = split_into
        self.n_samples_per_tomo = n_samples_per_tomo
        self.tilt_axis = tilt_axis

        if self.tilt_axis is not None:
            tilt_axis_index = ["Z", "Y", "X"].index(self.tilt_axis)
            rot_axes = [0, 1, 2]
            rot_axes.remove(tilt_axis_index)
            self.rot_axes = tuple(rot_axes)
        else:
            self.rot_axes = None

        self.extraction_shapes = extraction_shapes
        self.mean = mean
        self.std = std

        self.sample_shape = np.array(list(sample_shape))
        self.shuffle = shuffle
        #arbitrary elements for now
        self.coords = np.empty(split_into)

        self.tomos_odd = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_odd]
        self.tomos_even = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_even]
        self.n_tomos = len(self.tomo_paths_odd)

        if self.mask_paths is None:
            self.mask_paths = [None] * self.n_tomos
        
        self.create_coordinate_lists()
        print(np.shape(self.coords), 'Coords shape')
        self.length = np.empty(self.split_into, dtype= np.int64)
        #for j in range(len(self.tomo_paths_odd)):
        #    for i in range(self.split_into):
        #changes from c.shape[0] to c.shape[1] because we have added a dim for split_into partitions
        for i in range(self.split_into):
            self.length[i] = sum([c.shape[1] for c in self.coords])
        
        #assert np.all(self.length == self.length[0])
        self.indices = np.empty((self.split_into, self.length[0]), dtype = np.int64)
        print(self.length, type(self.length[0]))
        print(np.shape(self.length[0]))
        if self.shuffle:
            for i in range(self.split_into):
                self.indices[i] = np.random.permutation(self.length[i])
        else:
            for i in range(self.split_into):
                self.indices[i] = np.arange(self.length[i])
        print(np.shape(self.coords), 'coords shape')
        if self.mean == None or self.std == None:
            self.compute_mean_std(n_samples=n_normalization_samples)
        #self.split_idx = split_idx

    def save(self, path):
        np.savez(path,
                 tomo_paths_odd=self.tomo_paths_odd,
                 tomo_paths_even=self.tomo_paths_even,
                 mean=self.mean,
                 std=self.std,
                 split_into=self.split_into,
                 n_samples_per_tomo=self.n_samples_per_tomo,
                 extraction_shapes=self.extraction_shapes,
                 sample_shape=self.sample_shape,
                 shuffle=self.shuffle,
                 coords=self.coords,
                 tilt_axis=self.tilt_axis)

    @classmethod
    def load(cls, path, which_partition, split_into):
        tmp = np.load(path, allow_pickle=True)
        tomo_paths_odd = [str(p) for p in tmp['tomo_paths_odd']]
        tomo_paths_even = [str(p) for p in tmp['tomo_paths_even']]
        mean = tmp['mean']
        std = tmp['std']
        n_samples_per_tomo = tmp['n_samples_per_tomo']
        extraction_shapes = tmp['extraction_shapes']
        sample_shape = tmp['sample_shape']
        shuffle = tmp['shuffle']
        split_into = tmp['split_into']
        #slice our coords to give the ones we want
        coords = tmp['coords'] #[:, which_partition, :, :]
        if isinstance(tmp['tilt_axis'], np.ndarray):
            tilt_axis = None
        else:
            tilt_axis = tmp['tilt_axis']
        
        #make a new instance of our class having read from the numpy file
        ds = cls(tomo_paths_odd=tomo_paths_odd,
                 tomo_paths_even=tomo_paths_even,
                 mean=mean,
                 std=std,
                 split_into=split_into,
                 n_samples_per_tomo=n_samples_per_tomo, #worry about this
                 extraction_shapes=extraction_shapes,
                 sample_shape=sample_shape,
                 shuffle=shuffle,
                 tilt_axis=tilt_axis)
        #overwrite the coords
        ds.split_idx = which_partition
        ds.coords = coords
        return ds

    def compute_mean_std(self, n_samples=2000):
        samples = []
        print('Computing normalization parameters:')
        #we want to use all random halves/partitions for mean and std calculations
        # if split_into > 1
        
        for split_idx in range(self.split_into):
            for i in tqdm.trange(n_samples // self.split_into):
                x, _ = self.__getitem__(i, split_idx = split_idx)
                samples.append(x)
            
        self.mean = np.mean(samples)
        self.std = np.std(samples)
        del (samples)

    def create_coordinate_lists(self):
        coord_list = []
        
        for odd, even, es, maskfile in zip(self.tomo_paths_odd, self.tomo_paths_even, self.extraction_shapes, self.mask_paths):
            coord_list.append(self.__create_coords_for_tomo__(even, odd, es, maskfile))
        #the output from __create_coords_for_tomo__ is a stacked array or  
        #assert np.shape(coord_list[0])[1] % self.split_into == 0
        #because the coordinates are random, we can just slice
        self.coords = np.empty((len(self.tomo_paths_odd), self.split_into,  int(np.shape(coord_list[0])[0] // self.split_into), 3), dtype = np.int64)
        interval = np.shape(coord_list[0])[0] // self.split_into
        print('Interval', interval)
        for j in range(len(self.tomo_paths_odd)):
            for i in range(self.split_into):
                self.coords[j][i] = coord_list[j][i * interval : i * interval + interval, :]

    def __create_coords_for_tomo__(self, even_path, odd_path, extraction_shape, mask_path):
        even = mrcfile.mmap(even_path, mode='r')
        odd = mrcfile.mmap(odd_path, mode='r')
        
        assert even.data.shape == odd.data.shape, '{} and {} tomogram have different shapes.'.format(even_path,
                                                                                                     odd_path)
        
        # If no mask is specified, just create a one-mask
        if mask_path is None:
            mask = np.ones(even.data.shape).astype(np.bool)
        else:
            mask = mrcfile.read(mask_path).astype(np.bool)

            assert even.data.shape == mask.data.shape, '{} and {} tomogram / mask have different shapes.'.format(even_path,
                                                                                                                 mask_path)
        
        assert even.data.shape[0] > 2 * self.sample_shape[0]
        assert even.data.shape[1] > 2 * self.sample_shape[1]
        assert even.data.shape[2] > 2 * self.sample_shape[2]

        coords = self.create_random_coords(extraction_shape[0],
                                           extraction_shape[1],
                                           extraction_shape[2],
                                           mask,
                                           n_samples=self.n_samples_per_tomo)

        even.close()
        odd.close()

        return coords

    def create_random_coords(self, z, y, x, mask, n_samples):
        # Inspired by isonet preprocessing.cubes:create_cube_seeds()
        
        # Get permissible locations based on extraction_shape and sample_shape
        slices = tuple([slice(z[0],z[1]-self.sample_shape[2]),
                       slice(y[0],y[1]-self.sample_shape[1]),
                       slice(x[0],x[1]-self.sample_shape[0])])
        
        # Get intersect with mask-allowed values                       
        valid_inds = np.where(mask[slices])
        
        valid_inds = [v + s.start for s, v in zip(slices, valid_inds)]
        
        sample_inds = np.random.choice(len(valid_inds[0]),
                                       n_samples,
                                       replace=False)  #len(valid_inds[0]) < n_samples)
        #not exactly sure what the modification above will do
        rand_inds = [v[sample_inds] for v in valid_inds]
        

        return np.stack([rand_inds[0],rand_inds[1], rand_inds[2]], -1)

    def augment(self, x, y):
        if self.tilt_axis is not None:
            if self.sample_shape[0] == self.sample_shape[1] and \
                    self.sample_shape[0] == self.sample_shape[2]:
                rot_k = np.random.randint(0, 4, 1)

                x[...,0] = np.rot90(x[...,0], k=rot_k, axes=self.rot_axes)
                y[...,0] = np.rot90(y[...,0], k=rot_k, axes=self.rot_axes)


        if np.random.rand() > 0.5:
            return y, x
        else:
            return x, y

    def __len__(self):
        #might have to have this use split_idx
        return self.length

    def __getitem__(self, idx, split_idx):
        tomo_index, coord_index = idx // self.n_samples_per_tomo, idx % self.n_samples_per_tomo
        z, y, x = self.coords[tomo_index][split_idx][coord_index]

        even_subvolume = self.tomos_even[tomo_index].data[z:z + self.sample_shape[0],
                         y:y + self.sample_shape[1],
                         x:x + self.sample_shape[2]]

        odd_subvolume = self.tomos_odd[tomo_index].data[z:z + self.sample_shape[0],
                        y:y + self.sample_shape[1],
                        x:x + self.sample_shape[2]]
        return self.augment(np.array(even_subvolume)[..., np.newaxis], np.array(odd_subvolume)[..., np.newaxis])

    def __iter__(self):
        for idx in self.indices[self.split_idx]:
            yield self.__getitem__(idx, self.split_idx)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            for i in range(self.split_into):
                self.indices[i] = np.random.permutation(self.length[i])

    def close(self):
        for even, odd in zip(self.tomos_even, self.tomos_odd):
            even.close()
            odd.close()


class CryoCARE_DataModule(object):
    def __init__(self, split_into = 1):
        self.split_into = split_into
        self.train_dataset = None
        self.val_dataset = None
 
    def setup(self, tomo_paths_odd, tomo_paths_even, mask_paths, n_samples_per_tomo = 1200, validation_fraction=0.1,
              sample_shape=(64, 64, 64), tilt_axis='Y', n_normalization_samples=500):
        train_extraction_shapes = []
        val_extraction_shapes = []
        for e, o in zip(tomo_paths_even, tomo_paths_odd):
            tes, ves = self.__compute_extraction_shapes__(e, o, tilt_axis_index=['Z', 'Y', 'X'].index(tilt_axis),
                                                          sample_shape=sample_shape,
                                                          validation_fraction=validation_fraction)
            train_extraction_shapes.append(tes)
            val_extraction_shapes.append(ves)
        
        #need to use split_into here
        self.train_dataset = CryoCARE_Dataset(tomo_paths_odd=tomo_paths_odd,
                                              tomo_paths_even=tomo_paths_even,
                                              mask_paths=mask_paths,
                                              mean=None,
                                              std=None,
                                              split_into=self.split_into,
                                              n_samples_per_tomo=int(
                                                  n_samples_per_tomo * (1 - validation_fraction)),
                                              extraction_shapes=train_extraction_shapes,
                                              sample_shape=sample_shape,
                                              shuffle=True, n_normalization_samples=n_normalization_samples,
                                                  tilt_axis=tilt_axis)

        self.val_dataset = CryoCARE_Dataset(tomo_paths_odd=tomo_paths_odd,
                                            tomo_paths_even=tomo_paths_even,
                                            mask_paths=mask_paths,
                                            mean=self.train_dataset.mean,
                                            std=self.train_dataset.std,
                                            split_into=self.split_into,
                                            n_samples_per_tomo=int(n_samples_per_tomo * validation_fraction),
                                            extraction_shapes=val_extraction_shapes,
                                            sample_shape=sample_shape,
                                            shuffle=False,
                                            tilt_axis=None)

    def save(self, path):
        self.train_dataset.save(join(path,'train_data.npz'))
        self.val_dataset.save(join(path, 'val_data.npz'))

    def load(self, path, which_partition, split_into):
        self.train_dataset = CryoCARE_Dataset.load(join(path, 'train_data.npz'), which_partition, split_into)
        self.val_dataset = CryoCARE_Dataset.load(join(path, 'val_data.npz'), which_partition, split_into)
        self.which_partition = which_partition

    def __compute_extraction_shapes__(self, even_path, odd_path, tilt_axis_index, sample_shape, validation_fraction):
        even = mrcfile.mmap(even_path, mode='r')
        odd = mrcfile.mmap(odd_path, mode='r')

        assert even.data.shape == odd.data.shape, '{} and {} tomogram have different shapes.'.format(even_path,
                                                                                                     odd_path)
        assert even.data.shape[0] > 2 * sample_shape[0]
        assert even.data.shape[1] > 2 * sample_shape[1]
        assert even.data.shape[2] > 2 * sample_shape[2]

        val_cut_off = int(even.data.shape[tilt_axis_index] * (1 - validation_fraction)) - 1
        if ((even.data.shape[tilt_axis_index] - val_cut_off) < sample_shape[tilt_axis_index]) or val_cut_off < sample_shape[tilt_axis_index]:
            val_cut_off = even.data.shape[tilt_axis_index] - sample_shape[tilt_axis_index] - 1

        extraction_shape_train = [[0, even.data.shape[0]], [0, even.data.shape[1]], [0, even.data.shape[2]]]
        extraction_shape_val = [[0, even.data.shape[0]], [0, even.data.shape[1]], [0, even.data.shape[2]]]
        extraction_shape_train[tilt_axis_index] = [0, val_cut_off]
        extraction_shape_val[tilt_axis_index] = [val_cut_off, even.data.shape[tilt_axis_index]]

        return extraction_shape_train, extraction_shape_val

    def get_normalizer(self, mean, std):
        def normalize(x, y):
            x = (x - mean) / std
            y = (y - mean) / std
            return x, y

        return normalize
    
    #we assume that self.which_partition has already been set in the load() call above
    def get_train_dataset(self):
        sample_shape = self.train_dataset.sample_shape
        ds = tf.data.Dataset.from_generator(self.train_dataset.__iter__,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(
                                                tuple(sample_shape) + (1,), tuple(sample_shape) + (1,)))
        return ds.map(self.get_normalizer(self.train_dataset.mean, self.train_dataset.std)).prefetch(tf.data.experimental.AUTOTUNE).repeat()

    def get_val_dataset(self):
        sample_shape = self.val_dataset.sample_shape
        ds = tf.data.Dataset.from_generator(self.val_dataset.__iter__,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(
                                                tuple(sample_shape) + (1,), tuple(sample_shape) + (1,)))
        return ds.map(self.get_normalizer(self.train_dataset.mean, self.train_dataset.std))

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
