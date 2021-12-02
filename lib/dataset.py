from operator import le
import os
import sys
import time
import h5py
from numpy.core.numeric import True_
from numpy.core.shape_base import block
import torch
import numpy as np
import multiprocessing as mp
from torch._C import _tracer_warn_use_python
from tqdm import tqdm
from prefetch_generator import background
from collections import OrderedDict

sys.path.append(".")
from lib.config import CONF
from lib.inference_utils import mc_forward, forward, filter_points, eval_one_batch

class ScannetDataset():
    def __init__(self, phase, scene_list, num_classes=21, npoints=8192, is_weighting=True, use_multiview=False, use_color=False, use_normal=False):
        self.phase = phase
        assert phase in ["train", "val", "test"]
        self.scene_list = scene_list
        self.num_classes = num_classes
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_multiview = use_multiview
        self.use_color = use_color
        self.use_normal = use_normal
        self.chunk_data = {} # init in generate_chunks()

        self._prepare_weights()

    def _prepare_weights(self):
        self.scene_data = {}
        self.multiview_data = {}
        scene_points_list = []
        semantic_labels_list = []
        if self.use_multiview:
            multiview_database = h5py.File(CONF.MULTIVIEW, "r", libver="latest")
        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            label = scene_data[:, 10]

            # append
            scene_points_list.append(scene_data)
            semantic_labels_list.append(label)
            self.scene_data[scene_id] = scene_data

            if self.use_multiview:
                feature = multiview_database.get(scene_id)[()]
                self.multiview_data[scene_id] = feature

        if self.is_weighting:
            labelweights = np.zeros(self.num_classes)
            for seg in semantic_labels_list:
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)

    @background()
    def __getitem__(self, index):
        start = time.time()

        # load chunks
        scene_id = self.scene_list[index]
        scene_data = self.chunk_data[scene_id]
        # unpack
        point_set = scene_data[:, :3] # include xyz by default
        rgb = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]
        label = scene_data[:, 10].astype(np.int32)
        if self.use_multiview:
            feature = scene_data[:, 11:]
            point_set = np.concatenate([point_set, feature], axis=1)
        
        if self.use_color:
            point_set = np.concatenate([point_set, rgb], axis=1)

        if self.use_normal:
            point_set = np.concatenate([point_set, normal], axis=1)

        if self.phase == "train":
            point_set = self._augment(point_set)
        
        # prepare mask
        curmin = np.min(point_set, axis=0)[:3]
        curmax = np.max(point_set, axis=0)[:3]
        mask = np.sum((point_set[:, :3] >= (curmin - 0.01)) * (point_set[:, :3] <= (curmax + 0.01)), axis=1) == 3
        sample_weight = self.labelweights[label]
        sample_weight *= mask

        fetch_time = time.time() - start

        return point_set, label, sample_weight, fetch_time

    def __len__(self):
        return len(self.scene_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

        # translate the chunk center back to the original center
        coords += center
        point_set[:, :3] = coords

        return point_set

    def _translate(self, point_set):
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords

        return point_set

    def _rotate(self, point_set):
        coords = point_set[:, :3]

        # x rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        # y rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )

        # z rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

    def generate_chunks(self):
        """
            note: must be called before training
        """

        print("generate new chunks for {}...".format(self.phase))
        for scene_id in tqdm(self.scene_list):
            scene = self.scene_data[scene_id]
            semantic = scene[:, 10].astype(np.int32)
            if self.use_multiview:
                feature = self.multiview_data[scene_id]

            coordmax = np.max(scene, axis=0)[:3]
            coordmin = np.min(scene, axis=0)[:3]
            
            for _ in range(5):
                curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
                curmin = curcenter-[0.75,0.75,1.5]
                curmax = curcenter+[0.75,0.75,1.5]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = np.sum((scene[:, :3]>=(curmin-0.2))*(scene[:, :3]<=(curmax+0.2)),axis=1)==3
                cur_point_set = scene[curchoice]
                cur_semantic_seg = semantic[curchoice]
                if self.use_multiview:
                    cur_feature = feature[curchoice]

                if len(cur_semantic_seg)==0:
                    continue

                mask = np.sum((cur_point_set[:, :3]>=(curmin-0.01))*(cur_point_set[:, :3]<=(curmax+0.01)),axis=1)==3
                vidx = np.ceil((cur_point_set[mask,:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
                vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
                isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02

                if isvalid:
                    break
            
            # store chunk
            if self.use_multiview:
                chunk = np.concatenate([cur_point_set, cur_feature], axis=1)
            else:
                chunk = cur_point_set

            choices = np.random.choice(chunk.shape[0], self.npoints, replace=True)
            chunk = chunk[choices]
            self.chunk_data[scene_id] = chunk
            #self.chunk_data[scene_id] = np.load('/home/erik/TUM/Pointnet2.ScanNet/fix_inp.npy')
            #np.save('/home/erik/TUM/Pointnet2.ScanNet/fix_inp', chunk)
            
        print("done!\n")

class ScannetDatasetWholeScene():
    def __init__(self, scene_list, npoints=8192, is_weighting=True, use_color=False, use_normal=False, use_multiview=False):
        self.scene_list = scene_list
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview

        self._load_scene_file()

    def _load_scene_file(self):
        self.scene_points_list = []
        self.semantic_labels_list = []
        if self.use_multiview:
            multiview_database = h5py.File(CONF.MULTIVIEW, "r", libver="latest")
            self.multiview_data = []

        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            label = scene_data[:, 10].astype(np.int32)
            self.scene_points_list.append(scene_data)
            self.semantic_labels_list.append(label)

            if self.use_multiview:
                feature = multiview_database.get(scene_id)[()]
                self.multiview_data.append(feature)

        if self.is_weighting:
            labelweights = np.zeros(CONF.NUM_CLASSES)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(CONF.NUM_CLASSES + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(CONF.NUM_CLASSES)

    @background()
    def __getitem__(self, index):
        start = time.time()
        scene_data = self.scene_points_list[index]

        # unpack
        point_set_ini = scene_data[:, :3] # include xyz by default
        color = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]

        if self.use_color:
            point_set_ini = np.concatenate([point_set_ini, color], axis=1)

        if self.use_normal:
            point_set_ini = np.concatenate([point_set_ini, normal], axis=1)

        if self.use_multiview:
            multiview_features = self.multiview_data[index]
            point_set_ini = np.concatenate([point_set_ini, multiview_features], axis=1)

        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        coordmax = point_set_ini[:, :3].max(axis=0)
        coordmin = point_set_ini[:, :3].min(axis=0)
        xlength = 1.5
        ylength = 1.5
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()

        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin+[i*xlength, j*ylength, 0]
                curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
                mask = np.sum((point_set_ini[:, :3]>=(curmin-0.01))*(point_set_ini[:, :3]<=(curmax+0.01)), axis=1)==3
                cur_point_set = point_set_ini[mask,:]
                cur_semantic_seg = semantic_seg_ini[mask]
                if len(cur_semantic_seg) == 0:
                    continue
                
                remainder = len(cur_semantic_seg) % self.npoints
                n_splits = len(cur_semantic_seg) // self.npoints
                choice = np.random.choice(len(cur_semantic_seg), size = self.npoints - remainder, replace=True)
                cur_point_set = np.concatenate([cur_point_set, cur_point_set[choice].copy()], axis=0)
                cur_semantic_seg = np.concatenate([cur_semantic_seg, cur_semantic_seg[choice].copy()], axis=0)
                mask = np.concatenate([mask, mask[choice].copy()], axis=0)
                cur_point_sets = np.split(cur_point_set, len(cur_semantic_seg) // self.npoints, axis=0)
                cur_semantic_segs = np.split(cur_semantic_seg, len(cur_semantic_seg) // self.npoints, axis=0)
                for k in range(len(cur_point_sets)):
                    point_set = cur_point_sets[k] # Nx3
                    semantic_seg = cur_semantic_segs[k] # N
                    # if sum(mask)/float(len(mask))<0.01:
                    #     continue

                    sample_weight = self.labelweights[semantic_seg]
                    point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                    semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                    sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN

        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)

        fetch_time = time.time() - start

        return point_sets, semantic_segs, sample_weights, fetch_time

    def __len__(self):
        return len(self.scene_points_list)

class ScannetDatasetActiveLearning():
    def __init__(self, phase, scene_list, num_classes=21, npoints=8192, is_weighting=True, use_multiview=False, use_color=False, use_normal=False, points_increment=100):
        self.phase = phase
        assert phase in ["train", "val", "test"]
        self.scene_list = scene_list
        self.num_classes = num_classes
        self.points_increment = points_increment
        self.npoints = npoints
        self.is_weighting = is_weighting
        self.use_multiview = use_multiview
        self.use_color = use_color
        self.use_normal = use_normal
        self.chunk_data = {} # init in generate_chunks()

        self._init_data()
        self._prepare_weights()

    def _init_data(self):
        self.scene_data = {}
        self.selected_mask = {}
        self.multiview_data = {}
        if self.use_multiview:
            multiview_database = h5py.File(CONF.MULTIVIEW, "r", libver="latest")
        for scene_id in tqdm(self.scene_list):
            scene_data = np.load(CONF.SCANNETV2_FILE.format(scene_id))
            label = scene_data[:, 10]
            #choose initial 100 points for each scene

            coordmax = np.max(scene_data, axis=0)[:3]
            coordmin = np.min(scene_data, axis=0)[:3]
            sample_iter = 1
            while True:
                curcenter = scene_data[np.random.choice(len(label), 1)[0],:3]
                curmin = curcenter-[0.75,0.75,1.5]
                curmax = curcenter+[0.75,0.75,1.5]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = np.sum((scene_data[:, :3]>=(curmin-0.2))*(scene_data[:, :3]<=(curmax+0.2)),axis=1)==3
                cur_point_set = scene_data[curchoice]
                cur_semantic_seg = label[curchoice]
                if self.use_multiview:
                    cur_feature = multiview_database[curchoice]

                if len(cur_semantic_seg)==0:
                    continue

                mask = np.sum((cur_point_set[:, :3]>=(curmin-0.01))*(cur_point_set[:, :3]<=(curmax+0.01)),axis=1)==3
                vidx = np.ceil((cur_point_set[mask,:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
                vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
                isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02

                if (isvalid or sample_iter > 5) and cur_point_set.shape[0] >= self.points_increment:
                    break
                    
                sample_iter += 1
            
            if self.use_multiview:
                chunk = np.concatenate([cur_point_set, cur_feature], axis=1)
            else:
                chunk = cur_point_set
            # append
            choices = np.random.choice(chunk.shape[0], self.points_increment, replace=False)
            selection_mask = np.zeros(chunk.shape[0], dtype=np.bool)
            selection_mask[choices] = True
            curchoice[curchoice==True] = selection_mask.copy()
            self.scene_data[scene_id] = scene_data
            self.selected_mask[scene_id] = curchoice.copy()

            if self.use_multiview:
                feature = multiview_database.get(scene_id)[()]
                self.multiview_data[scene_id] = feature

    def _prepare_weights(self):
        if self.is_weighting:
            labelweights = np.zeros(self.num_classes)
            semantic_labels_list = []
            for scene_id in self.scene_list:
                semantic_labels_list.append(self.scene_data[scene_id][self.selected_mask[scene_id]][:, 10])
            for seg in semantic_labels_list:
                tmp,_ = np.histogram(seg,range(self.num_classes + 1))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        else:
            self.labelweights = np.ones(self.num_classes)
    
    @background()
    def __getitem__(self, index):
        start = time.time()

        # load chunks
        scene_id = self.scene_list[index]
        scene_data = self.chunk_data[scene_id]
        # unpack
        point_set = scene_data[:, :3] # include xyz by default
        rgb = scene_data[:, 3:6] / 255. # normalize the rgb values to [0, 1]
        normal = scene_data[:, 6:9]
        label = scene_data[:, 10].astype(np.int32)
        if self.use_multiview:
            feature = scene_data[:, 11:]
            point_set = np.concatenate([point_set, feature], axis=1)
        
        if self.use_color:
            point_set = np.concatenate([point_set, rgb], axis=1)

        if self.use_normal:
            point_set = np.concatenate([point_set, normal], axis=1)

        if self.phase == "train":
            point_set = self._augment(point_set)
        
        # prepare mask
        curmin = np.min(point_set, axis=0)[:3]
        curmax = np.max(point_set, axis=0)[:3]
        mask = np.sum((point_set[:, :3] >= (curmin - 0.01)) * (point_set[:, :3] <= (curmax + 0.01)), axis=1) == 3
        sample_weight = self.labelweights[label]
        sample_weight *= mask

        fetch_time = time.time() - start

        return point_set, label, sample_weight, fetch_time

    def __len__(self):
        return len(self.scene_list)

    def _augment(self, point_set):
        # translate the chunk center to the origin
        center = np.mean(point_set[:, :3], axis=0)
        coords = point_set[:, :3] - center

        p = np.random.choice(np.arange(0.01, 1.01, 0.01), size=1)[0]
        if p < 1 / 8:
            # random translation
            coords = self._translate(coords)
        elif p >= 1 / 8 and p < 2 / 8:
            # random rotation
            coords = self._rotate(coords)
        elif p >= 2 / 8 and p < 3 / 8:
            # random scaling
            coords = self._scale(coords)
        elif p >= 3 / 8 and p < 4 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
        elif p >= 4 / 8 and p < 5 / 8:
            # random translation
            coords = self._translate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 5 / 8 and p < 6 / 8:
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        elif p >= 6 / 8 and p < 7 / 8:
            # random translation
            coords = self._translate(coords)
            # random rotation
            coords = self._rotate(coords)
            # random scaling
            coords = self._scale(coords)
        else:
            # no augmentation
            pass

        # translate the chunk center back to the original center
        coords += center
        point_set[:, :3] = coords

        return point_set

    def _translate(self, point_set):
        # translation factors
        x_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        y_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]
        z_factor = np.random.choice(np.arange(-0.5, 0.501, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords += [x_factor, y_factor, z_factor]
        point_set[:, :3] = coords

        return point_set

    def _rotate(self, point_set):
        coords = point_set[:, :3]

        # x rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rx = np.array(
            [[1, 0, 0],
             [0, np.cos(theta), -np.sin(theta)],
             [0, np.sin(theta), np.cos(theta)]]
        )

        # y rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Ry = np.array(
            [[np.cos(theta), 0, np.sin(theta)],
             [0, 1, 0],
             [-np.sin(theta), 0, np.cos(theta)]]
        )

        # z rotation matrix
        theta = np.random.choice(np.arange(-5, 5.001, 0.001), size=1)[0] * 3.14 / 180 # in radians
        Rz = np.array(
            [[np.cos(theta), -np.sin(theta), 0],
             [np.sin(theta), np.cos(theta), 0],
             [0, 0, 1]]
        )

        # rotate
        R = np.matmul(np.matmul(Rz, Ry), Rx)
        coords = np.matmul(R, coords.T).T

        # dump
        point_set[:, :3] = coords

        return point_set

    def _scale(self, point_set):
        # scaling factors
        factor = np.random.choice(np.arange(0.95, 1.051, 0.001), size=1)[0]

        coords = point_set[:, :3]
        coords *= [factor, factor, factor]
        point_set[:, :3] = coords

        return point_set

    def generate_chunks(self):
        """
            note: must be called before training
        """

        print("generate new chunks for {}...".format(self.phase))
        for scene_id in tqdm(self.scene_list):
            scene = self.scene_data[scene_id][self.selected_mask[scene_id]]
            semantic = scene[:, 10].astype(np.int32)
            if self.use_multiview:
                feature = self.multiview_data[scene_id]

            coordmax = np.max(scene, axis=0)[:3]
            coordmin = np.min(scene, axis=0)[:3]
            
            for _ in range(5):
                curcenter = scene[np.random.choice(len(semantic), 1)[0],:3]
                curmin = curcenter-[0.75,0.75,1.5]
                curmax = curcenter+[0.75,0.75,1.5]
                curmin[2] = coordmin[2]
                curmax[2] = coordmax[2]
                curchoice = np.sum((scene[:, :3]>=(curmin-0.2))*(scene[:, :3]<=(curmax+0.2)),axis=1)==3
                cur_point_set = scene[curchoice]
                cur_semantic_seg = semantic[curchoice]
                if self.use_multiview:
                    cur_feature = feature[curchoice]

                if len(cur_semantic_seg)==0:
                    continue

                mask = np.sum((cur_point_set[:, :3]>=(curmin-0.01))*(cur_point_set[:, :3]<=(curmax+0.01)),axis=1)==3
                vidx = np.ceil((cur_point_set[mask,:3]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
                vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
                isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02

                if isvalid:
                    break
            
            # store chunk
            if self.use_multiview:
                chunk = np.concatenate([cur_point_set, cur_feature], axis=1)
            else:
                chunk = cur_point_set

            choices = np.random.choice(chunk.shape[0], self.npoints, replace=True)
            chunk = chunk[choices]
            self.chunk_data[scene_id] = chunk
            
        print("done!\n")

    def choose_new_points(self, heuristic="random", model=None, mc_iters=20):
        '''
        Choose new points to be annotated and added to the dataset

        :param heuristic: heuristic according to which new points are chosen, one from (mc, random, gt). Default: "random"
        :type heuristic: str
        :param model: model to use in case of mc and gt. Default: None
        :type model: torch.nn.Module
        :param mc_iters: number of Monte Carlo dropout iterations to perform. Default: 20
        :type mc_iters: int
        '''
        print("Choosing new points with {} heuristic ...".format(heuristic))
        xlength = 1.5
        ylength = 1.5
        for scene_id in tqdm(self.scene_list):
            scene_data = self.scene_data[scene_id]
            scene_data = scene_data[np.logical_not(self.selected_mask[scene_id])]
            coordmax = scene_data[:, :3].max(axis=0)
            coordmin = scene_data[:, :3].min(axis=0)
            nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/xlength).astype(np.int32)
            nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/ylength).astype(np.int32)
                
            point_set_ini = self.scene_data[scene_id][:, :3]
            color = self.scene_data[scene_id][:, 3:6] / 255.
            normal = self.scene_data[scene_id][:, 6:9]

            if self.use_color:
                point_set_ini = np.concatenate([point_set_ini, color], axis=1)

            if self.use_normal:
                point_set_ini = np.concatenate([point_set_ini, normal], axis=1)

            if self.use_multiview:
                multiview_features = self.multiview_data[scene_id]
                point_set_ini = np.concatenate([point_set_ini, multiview_features], axis=1)
            point_cubes = []
            for i in range(nsubvolume_x):
                for j in range(nsubvolume_y):
                    curmin = coordmin+[i*xlength, j*ylength, 0]
                    curmax = coordmin+[(i+1)*xlength, (j+1)*ylength, coordmax[2]-coordmin[2]]
                    mask = np.sum((point_set_ini[:, :3]>=(curmin-0.01))*(point_set_ini[:, :3]<=(curmax+0.01)), axis=1)==3
                    mask = np.logical_and(mask, np.logical_not(self.selected_mask[scene_id]))
                    cur_point_set = point_set_ini[mask,:]
                    cur_semantic_seg = self.scene_data[scene_id][mask, 10].astype(np.int32)
                    scores = np.zeros_like(cur_semantic_seg)
                    block_mask = mask.copy()
                    if len(cur_semantic_seg) == 0:
                        continue

                    remainder = len(cur_semantic_seg) % self.npoints
                    choice = np.random.choice(len(cur_semantic_seg), size = self.npoints - remainder, replace=True)
                    cur_point_set = np.concatenate([cur_point_set, cur_point_set[choice].copy()], axis=0)
                    cur_semantic_seg = np.concatenate([cur_semantic_seg, cur_semantic_seg[choice].copy()], axis=0)
                    cur_point_sets = np.split(cur_point_set, len(cur_semantic_seg) // self.npoints, axis=0)
                    cur_semantic_segs = np.split(cur_semantic_seg, len(cur_semantic_seg) // self.npoints, axis=0)
                    for k in range(len(cur_point_sets)):
                        point_set = cur_point_sets[k] # Nx3
                        mask_start, mask_end = k * self.npoints, (k + 1) * self.npoints
                        semantic_seg = cur_semantic_segs[k] # N
                        sample_weight = self.labelweights[semantic_seg]
                        with torch.no_grad():
                            point_set = np.expand_dims(point_set, 0) # 1xNx3
                            semantic_seg = np.expand_dims(semantic_seg, 0) # 1x1xN
                            sample_weight = np.expand_dims(sample_weight, 0) # 1x1xN
                            point_set_t = torch.FloatTensor(point_set)
                            targets = torch.LongTensor(semantic_seg)
                            weights = torch.FloatTensor(sample_weight)
                            coords = point_set_t[:, :, :3]
                            feats = point_set_t[:, :, 3:]
                            coords, feats, targets, weights = coords.unsqueeze(0), feats.unsqueeze(0), targets.unsqueeze(0), weights.unsqueeze(0)
                            coords, feats, targets, weights = coords.cuda(), feats.cuda(), targets.cuda(), weights.cuda()
                            if heuristic == "gt":
                                preds = forward(1, model, coords, feats)
                                coords = coords.squeeze(0).view(-1, 3).cpu().numpy()
                                preds = preds.squeeze(0).view(-1).cpu().numpy()
                                targets = targets.squeeze(0).view(-1).cpu().numpy()
                                weights = weights.squeeze(0).view(-1).cpu().numpy()
                                preds = preds[:len(mask[mask==True][mask_start:mask_end])]
                                targets = targets[:len(mask[mask==True][mask_start:mask_end])]
                                block_mask[block_mask==True][mask_start:mask_end] = preds != targets
                                scores[mask_start:mask_end] = (preds != targets).astype(float)
                            elif heuristic == "mc":
                                preds = mc_forward(1, model, coords, feats, mc_iters)
                                coords = coords.squeeze(0).view(-1, 3).cpu().numpy()
                                preds = preds.view(-1, mc_iters).cpu().numpy()
                                targets = targets.squeeze(0).view(-1).cpu().numpy()
                                weights = weights.squeeze(0).view(-1).cpu().numpy()
                                preds = preds[:len(mask[mask==True][mask_start:mask_end])]
                                scene_entropy = np.zeros(preds.shape[0])
                                for c in range(CONF.NUM_CLASSES):
                                    p = np.sum(preds == c, axis=1, dtype=np.float32) / mc_iters
                                    scene_entropy = scene_entropy - (p * np.log2(p + 1e-12))
                                scores[mask_start:mask_end] = scene_entropy.copy()
                            elif heuristic == "random":
                                block_mask[block_mask==True][mask_start:mask_end] = True

                    if heuristic == "mc":
                        slice_mask = np.zeros(len(mask[mask==True]), dtype=np.bool)
                        idx = np.argsort(-scores)[:self.points_increment]
                        slice_mask[idx] = True
                        block_mask[block_mask==True] = slice_mask.copy()
                        point_cubes.append((float(np.mean(scores[idx])), block_mask.copy()))
                    else:
                        point_cubes.append((float(np.mean(scores)), block_mask.copy()))

            if heuristic != "random":
                point_cubes_mIoU_sorted = sorted(point_cubes, key=lambda x: x[0], reverse=True)
            else:
                # instead of sorting just shuffle the points in case of the random
                idx = np.random.choice(len(point_cubes), size=len(point_cubes), replace=False)
                point_cubes_mIoU_sorted = np.array(point_cubes)[idx]
            current_choice = self.selected_mask[scene_id].copy()
            total_num_points = self.points_increment
            for mIoU, point_mask in point_cubes_mIoU_sorted:
                if total_num_points <= 0:
                    break
                cube_points = np.logical_and(point_mask, np.logical_not(current_choice))
                if cube_points.sum() >= total_num_points:
                    choice = np.random.choice(cube_points[cube_points==True].shape[0], size=total_num_points, replace=False)
                    selection_mask = np.zeros(len(cube_points[cube_points==True]), dtype=np.bool)
                    selection_mask[choice] = True
                    cube_points[cube_points==True] = selection_mask.copy()
                    current_choice = np.logical_xor(current_choice, cube_points)
                    break
                else:
                    current_choice = np.logical_xor(current_choice, cube_points)
                    total_num_points -= cube_points[cube_points==True].shape[0]
            self.selected_mask[scene_id] = current_choice.copy()
        self._prepare_weights()

def collate_random(data):
    '''
    for ScannetDataset: collate_fn=collate_random

    return: 
        coords               # torch.FloatTensor(B, N, 3)
        feats                # torch.FloatTensor(B, N, 3)
        semantic_segs        # torch.FloatTensor(B, N)
        sample_weights       # torch.FloatTensor(B, N)
        fetch_time           # float
    '''

    # load data
    (
        point_set, 
        semantic_seg, 
        sample_weight,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_set = torch.FloatTensor(point_set)
    semantic_seg = torch.LongTensor(semantic_seg)
    sample_weight = torch.FloatTensor(sample_weight)

    # split points to coords and feats
    coords = point_set[:, :, :3]
    feats = point_set[:, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_seg,      # (B, N)
        sample_weight,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch

def collate_wholescene(data):
    '''
    for ScannetDataset: collate_fn=collate_random

    return: 
        coords               # torch.FloatTensor(B, C, N, 3)
        feats                # torch.FloatTensor(B, C, N, 3)
        semantic_segs        # torch.FloatTensor(B, C, N)
        sample_weights       # torch.FloatTensor(B, C, N)
        fetch_time           # float
    '''

    # load data
    (
        point_sets, 
        semantic_segs, 
        sample_weights,
        fetch_time 
    ) = zip(*data)

    # convert to tensor
    point_sets = torch.FloatTensor(point_sets)
    semantic_segs = torch.LongTensor(semantic_segs)
    sample_weights = torch.FloatTensor(sample_weights)

    # split points to coords and feats
    coords = point_sets[:, :, :, :3]
    feats = point_sets[:, :, :, 3:]

    # pack
    batch = (
        coords,             # (B, N, 3)
        feats,              # (B, N, 3)
        semantic_segs,      # (B, N)
        sample_weights,     # (B, N)
        sum(fetch_time)          # float
    )

    return batch
