

# This file would be used when running fast_rcnn/test.py -> as I evaluate results differently, I did not check if
# everything in here is correct. In case you'd like to use the following functions, you'd better check out voc_eval.py

import os
import cPickle
import numpy as np





########################################################################################################################

# ---------------------------------------------------------------------------------------------------------------- #
def building_path_from_index(building_path, index):
    """
    Construct an image path from the image's "index" identifier.
    REMARK: a function from kitti dataset
        - index is at least 1
    """
    # building_path = ...Faster - RCNN_TF / data / building_data
    # _image_ext = '.png'
    prefix = 'chicago'
    buildings_path = os.path.join(building_path, prefix + str(index), prefix + str(index) + '_buildings.npy')
    # buildings_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/chicago1/chicago1_buildings.npy'
    assert os.path.exists(buildings_path), \
        'Path does not exist: {}'.format(buildings_path)
    return buildings_path


def xy_minmax_path_from_index(building_path, index):
    """
    this could also be found within the array buildings_4p
    """
    prefix = 'chicago'
    buildings_path = os.path.join(building_path, prefix + str(index), prefix + str(index) + '_xy_minmax.npy')
    # buildings_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/chicago1/chicago1_xy_minmax.npy'
    assert os.path.exists(buildings_path), \
        'Path does not exist: {}'.format(buildings_path)
    return buildings_path


def orientations_path_from_index(building_path, index):
    prefix = 'chicago'
    orientations_path = os.path.join(building_path, prefix + str(index), prefix + str(index) + '_orientation.npy')
    # buildings_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/chicago1/chicago1_orientation.npy'
    assert os.path.exists(orientations_path), \
        'Path does not exist: {}'.format(orientations_path)
    return orientations_path


def building_path_4points_from_index(building_path, index):
    prefix = 'chicago'
    buildings_path = os.path.join(building_path, prefix + str(index),
                                  prefix + str(index) + '_buildings_4points.npy')
    # buildings_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/chicago1/chicago1_buildings_4points.npy'
    assert os.path.exists(buildings_path), \
        'Path does not exist: {}'.format(buildings_path)
    return buildings_path
# -------------------------------------------------------------------------------------------------------------- #



def parse_rec(single_image_buildings_4points_path, single_image_orientations_path):        #########
    """ has the same goal as parse_rec from VOC, but the structure of 'objects' is a little bit different """

    orientations_orig = np.load(single_image_orientations_path)  # contains only one value which is valid for all buildings
    buildings_4p = np.load(single_image_buildings_4points_path)

    num_objs = len(buildings_4p)


    objects = []

    for ind in range(0, num_objs):
        obj_struct = {}
        obj_struct['name'] = 'building'


        if (orientations_orig > -np.pi / 4 and orientations_orig <= np.pi / 4):
            orientations = orientations_orig
            side_bld_l1 = np.round(np.sqrt(np.sum((buildings_4p[ind][0, :] - buildings_4p[ind][1, :]) ** 2)))
            side_bld_l2 = np.round(np.sqrt(np.sum((buildings_4p[ind][1, :] - buildings_4p[ind][2, :]) ** 2)))
        elif (orientations_orig > -np.pi * 3 / 4 and orientations_orig <= -np.pi / 4):
            orientations = orientations_orig + np.pi / 2
            side_bld_l2 = np.round(np.sqrt(np.sum((buildings_4p[ind][0, :] - buildings_4p[ind][1, :]) ** 2)))
            side_bld_l1 = np.round(np.sqrt(np.sum((buildings_4p[ind][1, :] - buildings_4p[ind][2, :]) ** 2)))
        elif (orientations_orig > np.pi / 4 and orientations_orig <= np.pi * 3 / 4):
            orientations = orientations_orig - np.pi / 2
            side_bld_l2 = np.round(np.sqrt(np.sum((buildings_4p[ind][0, :] - buildings_4p[ind][1, :]) ** 2)))
            side_bld_l1 = np.round(np.sqrt(np.sum((buildings_4p[ind][1, :] - buildings_4p[ind][2, :]) ** 2)))
        elif orientations_orig <= -np.pi * 3 / 4:
            orientations = orientations_orig + np.pi
            side_bld_l1 = np.round(np.sqrt(np.sum((buildings_4p[ind][0, :] - buildings_4p[ind][1, :]) ** 2)))
            side_bld_l2 = np.round(np.sqrt(np.sum((buildings_4p[ind][1, :] - buildings_4p[ind][2, :]) ** 2)))
        elif orientations_orig > np.pi * 3 / 4:
            orientations = orientations_orig - np.pi
            side_bld_l1 = np.round(np.sqrt(np.sum((buildings_4p[ind][0, :] - buildings_4p[ind][1, :]) ** 2)))
            side_bld_l2 = np.round(np.sqrt(np.sum((buildings_4p[ind][1, :] - buildings_4p[ind][2, :]) ** 2)))
        else:
            print('error building.py orientation')

        bld_center = buildings_4p[ind][0, :] + 0.5 * (buildings_4p[ind][2, :] - buildings_4p[ind][0, :])

        x1 = min(max(bld_center[1] - 0.5 * side_bld_l1, 0), 900 - 1)  # DO NOT HARDCODE IMAGE WIDTH AND HEIGHT!!
        x2 = min(max(bld_center[1] + 0.5 * side_bld_l1, 0), 900 - 1)
        y1 = min(max(bld_center[0] - 0.5 * side_bld_l2, 0), 800 - 1)
        y2 = min(max(bld_center[0] + 0.5 * side_bld_l2, 0), 800 - 1)

        obj_struct['bbox'] = [x1, y1, x2, y2]
        #obj_struct['orientation'] = [orientations]
        objects.append(obj_struct)

    return objects



########################################################################################################################

def voc_ap(rec, prec, use_07_metric=False):             ###### check this
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    # is ap equal to average precision?
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

########################################################################################################################


def bld_eval(building_path,         #new
             detpath,       # path to file that contains all results
             imagesetfile,
             cachedir,
             classname = 'building',
             ovthresh=0.5,
             use_07_metric=False):
    # comes from voc_eval ...
    """
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations, caches the annotations in a pickle file
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation         ### check this...
        (default False)
    """
    #------------------------------------------ read ground truth -----------------------------------------------------#
    # first load gt
    if not os.path.isdir(cachedir):     #cachedir = ..../data/building_data/annotations_cache
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]     # contains the indexes of all necessary images

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}       # REMARK: recs are here not exactly the same as for VOC dataset
        for i, imagename in enumerate(imagenames):
            #single_image_building_path = building_path_from_index(building_path, imagename)
            #single_image_xy_minmax_path = xy_minmax_path_from_index(building_path, imagename)
            single_image_orientations_path = orientations_path_from_index(building_path, imagename)
            single_image_buildings_4points_path = building_path_4points_from_index(building_path, imagename)

            # recs[imagename] = parse_rec(single_image_building_path)  # building_path
            recs[imagename] = parse_rec(single_image_buildings_4points_path, single_image_orientations_path)  # building_path
            if i % 100 == 0:
                print 'Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames))
        # save
        print 'Saving cached annotations to {:s}'.format(cachefile)
        with open(cachefile, 'w') as f:
            cPickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'r') as f:
            recs = cPickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        #print(recs)
        print(imagename)
        print(recs[imagename])
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        #difficult = 0   #np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        #npos = npos + sum(~difficult)
        npos = npos + len(R)
        class_recs[imagename] = {'bbox': bbox,    #'difficult': difficult,
                                 'det': det}

    # ---------------------------------------------- read detections --------------------------------------------------#
    # read dets (similar to VOC dataset)
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

    # ---------------------------------------------- mark TPs and FPs -------------------------------------------------#
        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]        # iterate trough all possible ground truth boxes
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                #if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
         rec = -1
         prec = -1
         ap = -1

    return rec, prec, ap
