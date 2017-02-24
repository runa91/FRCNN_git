
__author__ = 'rueegnad' # derived from kitti.py

import datasets
import datasets.building
import os
from PIL import Image
import datasets.imdb
import numpy as np
import scipy.sparse
import subprocess
import cPickle
from fast_rcnn.config import cfg
import uuid
import datasets.ds_utils as ds_utils
import scipy.io as sio
from datasets.imdb import imdb

#from voc_eval import voc_eval
from bld_eval import bld_eval

# REMARK: we need to replace voc_eval and evaluate the results in an other way....




class building(datasets.imdb):
    def __init__(self, image_set, building_path=None):
        datasets.imdb.__init__(self, 'building_' + image_set)
        # example for self.name: 'building_train' (see imdb.py)
        self._image_set = image_set     # something like 'train', 'val', ...

        #look at the following two paths again ...
        self._building_path = self._get_default_path() if building_path is None \
                            else building_path
        # building_path = ... Faster-RCNN_TF/data/building_data
        #self._data_path = os.path.join(self._building_path, 'data_object_image_2')


        self._classes = ('__background__', 'building')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes))) # looks like: {'Building': 1, '__background__': 0}
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()    # something like: ['22', '23', '29']

        # Default to roidb handler
        """if cfg.IS_RPN:
            self._roidb_handler = self.gt_roidb
        else:
            self._roidb_handler = self.region_proposal_roidb"""

        self._roidb_handler = self.gt_roidb

        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}






    ####################################################################################################################
    # ---------------------------------------------------------------------------------------------------------------- #
    def _get_default_path(self):
        """
        Return the default path where KITTI is expected to be installed......?
        """
        return os.path.join(datasets.ROOT_DIR, 'data', 'building_data')

    # ---------------------------------------------------------------------------------------------------------------- #
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._building_path, self._image_set + '.txt')
        # remark: these .txt files are (at the moment) created by hand
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)

        with open(image_set_file) as f:
            image_index = [x.rstrip('\n') for x in f.readlines()]
        return image_index

    # ---------------------------------------------------------------------------------------------------------------- #

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        REMARK: a function from kitti dataset
            - index is at least 1
        """
        # building_path = ...Faster - RCNN_TF / data / building_data
        # _image_ext = '.png'
        prefix = 'chicago'
        image_path = os.path.join(self._building_path, prefix + index, prefix + index + '_wholeSatImage' + self._image_ext)
        print(image_path)
        # image_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/chicago1/chicago1_wholeSatImage.png'
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    # ---------------------------------------------------------------------------------------------------------------- #

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        #print(self._image_index)
        #print(self._image_index[i])
        return self.image_path_from_index(self._image_index[i])

    # ---------------------------------------------------------------------------------------------------------------- #

    def building_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        REMARK: a function from kitti dataset
            - index is at least 1
        """
        # building_path = ...Faster - RCNN_TF / data / building_data
        # _image_ext = '.png'
        prefix = 'chicago'
        buildings_path = os.path.join(self._building_path, prefix + str(index), prefix + str(index) + '_buildings.npy')
        # buildings_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/chicago1/chicago1_buildings.npy'
        assert os.path.exists(buildings_path), \
                'Path does not exist: {}'.format(buildings_path)
        return buildings_path

    def xy_minmax_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        REMARK: a function from kitti dataset
            - index is at least 1
        """
        # building_path = ...Faster - RCNN_TF / data / building_data
        # _image_ext = '.png'
        prefix = 'chicago'
        buildings_path = os.path.join(self._building_path, prefix + str(index), prefix + str(index) + '_xy_minmax.npy')
        # buildings_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/chicago1/chicago1_buildings.npy'
        assert os.path.exists(buildings_path), \
                'Path does not exist: {}'.format(buildings_path)
        return buildings_path

    def orientations_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        REMARK: a function from kitti dataset
            - index is at least 1
        """
        prefix = 'chicago'
        orientations_path = os.path.join(self._building_path, prefix + str(index), prefix + str(index) + '_orientation.npy')
        # buildings_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/chicago1/chicago1_buildings.npy'
        assert os.path.exists(orientations_path), \
                'Path does not exist: {}'.format(orientations_path)
        return orientations_path







    ################################
    # ---------------------------------------------------------------------------------------------------------------- #

    def _load_building_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        REMARK: this function is newly defined and replaces _load_pascal_annotation
        - process one single image
        - load data in a different way
        """

        #buildings_path = self.building_path_from_index(index)
        xy_minmax_path = self.xy_minmax_path_from_index(index)
        image_path = self.image_path_from_index(index)

        image = Image.open(image_path)
        #buildings = np.load(buildings_path)
        #num_objs = len(buildings)

        xy_minmax = np.load(xy_minmax_path)
        num_objs = len(xy_minmax)

        # new
        orientations_path = self.orientations_path_from_index(index)
        orientations_orig = np.load(orientations_path)       # contains only one value which is valid for all buildings
        #print('^^^^^^^^^^^`````````````````````````````````````````````````````````````^^^^^^^^^^^^^^^^^')
        #print(orientations_path)
        #print(orientations_orig)
        orientations = np.zeros((num_objs), dtype=np.float32)



        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area.... don't know what it should be here
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        """
        for ind, bld in enumerate(buildings):
            #x_values = bld[:,0]                                                                                        ####3
            #y_values = bld[:,1]
            x_values = bld[:,1]
            y_values = bld[:,0]
            x1 = min(x_values)
            x2 = max(x_values)
            y1 = min(y_values)
            y2 = max(y_values)"""

        for ind in range(0, num_objs):
            x1 = xy_minmax[ind][2]
            x2 = xy_minmax[ind][3]
            y1 = xy_minmax[ind][0]
            y2 = xy_minmax[ind][1]
            boxes[ind,:] = [x1, y1, x2, y2]
            gt_classes[ind] = 1
            overlaps[ind, 1] = 1.0      # what is overlaps?
            seg_areas[ind] = (x2 - x1 + 1) * (y2 - y1 + 1)

            orientations[ind] = orientations_orig     # new

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas,
                'gt_orientations' : orientations}




    # ---------------------------------------------------------------------------------------------------------------- #
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """

        #cache_file = os.path.join(self.cache_path, self.name + '_' + cfg.SUBCLS_NAME + '_gt_roidb.pkl')
        cache_file = os.path.join(self.cache_path, self.name + '_' + '_gt_roidb.pkl')
        # the path is: .../Faster-RCNN_TF/data/cache/building_train_gt_roidb.pkl
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_building_annotation(index)
                    for index in self.image_index]

        """if cfg.IS_RPN:
            # print out recall
            for i in xrange(1, self.num_classes):
                print '{}: Total number of boxes {:d}'.format(self.classes[i], self._num_boxes_all[i])
                print '{}: Number of boxes covered {:d}'.format(self.classes[i], self._num_boxes_covered[i])
                print '{}: Recall {:f}'.format(self.classes[i], float(self._num_boxes_covered[i]) / float(self._num_boxes_all[i]))"""

        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    # ---------------------------------------------------------------------------------------------------------------- #

    ############## ??
    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} ss roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        """if int(self._year) == 2007 or self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)"""
        gt_roidb = self.gt_roidb()
        ss_roidb = self._load_selective_search_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote ss roidb to {}'.format(cache_file)

        return roidb






    ############## ??
    def _load_selective_search_roidb(self, gt_roidb):
        filename = os.path.abspath(os.path.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        assert os.path.exists(filename), \
            'Selective search data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    ############## ??
    def rpn_roidb(self):
        gt_roidb = self.gt_roidb()
        rpn_roidb = self._load_rpn_roidb(gt_roidb)
        roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)

        return roidb

    ############## ??
    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']      # is set to None...?
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    ####################################################################################################################
    ########                                       NEW and not adjusted                                      ###########
    ####################################################################################################################

    def _get_bld_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        path = os.path.join(
            self._building_path,
            'a_Results',
            filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        # write the resut boxes to a file called comp4....._det_test_building.txt   ->    works
        for cls_ind, cls in enumerate(self.classes): # we have only backgroud and one more class
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_bld_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        print('------------- no boxes ----------------')
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _get_comp_id(self):     # ->    works
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _do_python_eval(self, output_dir='output'):
        """annopath = os.path.join(                                                                                        #######
            self._building_path,
            'a_Results',
            '{:s}.xml')
        imagesetfile = os.path.join(                                                                                    #######
            self._building_path,
            'a_Results',
            self._image_set + '.txt')"""
        # ammopath and imagesetfile are paths of ground truth and image information respectively ...
        imagesetfile = os.path.join(
            self._building_path,
            self._image_set + '.txt')

        cachedir = os.path.join(self._building_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        #use_07_metric = True if int(self._year) < 2010 else False
        #print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename_preds = self._get_bld_results_file_template().format(cls) # this file contains all results
            #rec, prec, ap = voc_eval(
            #    filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5,
            #    use_07_metric=False)        #use_07_metric=use_07_metric

            rec, prec, ap = bld_eval(
                self._building_path, filename_preds, imagesetfile, cachedir) # threshold could be set, is 0.5 per default


            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print(' ~~~~~~~~ finished ~~~~~~~~ ')



    # function called by fast_rcnn/test.py ...
    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        #if self.config['matlab_eval']:
        #    self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_bld_results_file_template().format(cls)
                os.remove(filename)







########################################################################################################################
#                                                         main                                                         #
########################################################################################################################

if __name__ == '__main__':
    from datasets.building import building
    d = building('trainval')
    res = d.roidb
    from IPython import embed;
    embed()




