
# the goal of this file is to evaluate all test images:
# calculate tp, tn, fp and fn for each image



# this file can be run by





import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network
from PIL import Image       # NEW
import ImageDraw
import csv
from scipy.spatial import distance_matrix


CLASSES = ('__background__','building')


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf

def vis_detections(image_name, im, class_name, dets, ax, angles, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        angle = angles[i,:]

        c_orig = np.zeros((4, 2))
        c_orig[0, :] = [bbox[2], bbox[1]]
        c_orig[1, :] = [bbox[2], bbox[3]]
        c_orig[2, :] = [bbox[0], bbox[3]]
        c_orig[3, :] = [bbox[0], bbox[1]] 

        angle = - angle     # that's what I need to do in order to get correct results
        print(angle)
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)

        roi_mid_w = (bbox[2] + bbox[0])/2
        roi_mid_h = (bbox[3] + bbox[1])/2

        # corner positions for the rotated rectangle
        c_rot = np.zeros((4, 2))

        for ind in range(4):
            w = c_orig[ind,0]
            h = c_orig[ind,1]
            w_from_mid = w - roi_mid_w;
            h_from_mid = h - roi_mid_h;
            w_from_mid_rot = cos_a*w_from_mid - sin_a*h_from_mid
            h_from_mid_rot = sin_a*w_from_mid + cos_a*h_from_mid
            c_rot[ind, 0] = w_from_mid_rot + roi_mid_w
            c_rot[ind, 1] = h_from_mid_rot + roi_mid_h

        ax.add_patch(
            plt.Polygon(c_rot, True, alpha=0.4,edgecolor='red', linewidth=2)
        )

        """ax.text((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2-10,		#bbox[0], bbox[1] - 2
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')"""
        """ax.text((bbox[0]+bbox[2])/2 - 20, (bbox[1]+bbox[3])/2-2,		#bbox[0], bbox[1] - 2
                '{:.3f}'.format(score),
                bbox=dict(facecolor='blue', alpha=0.1),
                fontsize=18, color='white')"""

    ax.set_title(('image{}:    {} detections with '
                  'p({} | box) >= {:.3f}').format(image_name, class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join('/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/a_demo/', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes, angles_after_rp, rois = im_detect(sess, net, im)
    angles_before_rp = rois[:,5:6]
    #angles = angles_after_rp + angles_before_rp
    #angles = angles_before_rp    #angles_after_rp
    angles = angles_after_rp



    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    CONF_THRESH = 0.01	#0.06	#0.02	#0.2  #0.8
    NMS_THRESH = 0.3 #0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        print('boxes shape:')
        print(boxes.shape)      # has shape (300, 84), which is wrong for a two class system!!!!!!
        # print(angles.shape)
        #print(boxes)
        # #print(angles)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        angles = angles[keep,:]
    # print(keep)
    # print(dets[keep, :]
    print(len(keep))
    print(rois.shape)
    print(angles_before_rp.shape)
    print(angles_after_rp.shape)
    print(angles_before_rp[keep[0:8],:])
    print(angles_after_rp[keep[0:8],:])
    print(angles[0:8,:])
    vis_detections(image_name, im, cls, dets, ax, angles, thresh=CONF_THRESH)


#----------------------------------------------------------------------------------------------------------------------#

def calculate_score_measures(tp, tn, fp, fn):
    if tp == 0:
        precision = 0
        recall = 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    IoU = tp / (tp + fp + fn)

    return recall, precision, accuracy, f1_score, IoU



def evaluate_single_image(sess, net, conf_thresh_array, img_nr, data_path, results_path, corner_dist_thr = 20,
                          show_figures = False, save_figures = False):
    #data_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/'
    #data_path = '/scratch/rueegnad_scratch/data_rot_Hand_In'
    prefix = 'chicago'

    image_path = os.path.join(data_path, prefix + str(img_nr), prefix + str(img_nr) + '_wholeSatImage.png')
    im = cv2.imread(image_path)

    # ------------------------------------------------ ground truth ----------------------------------------------------#

    buildings_4points_path = os.path.join(data_path, prefix + str(img_nr),
                                          prefix + str(img_nr) + '_buildings_4points.npy')
    # orientation_path = os.path.join(data_path, prefix + str(img_nr), prefix + str(img_nr) + '_orientation.npy')

    buildings_4p = np.load(buildings_4points_path)
    # buildings_or = np.load(orientation_path)

    BW = 100;  # width of the image border that we cut of

    img_gt = Image.new('L', (im.shape[0], im.shape[1]), 'black')
    draw_img_gt = ImageDraw.Draw(img_gt)

    all_gt_corners = []

    if len(buildings_4p) > 0:
        for bld_nr in range(0, len(buildings_4p)):
            c_gt2 = [0, 0, 0, 0, 0, 0, 0, 0]
            for ind in range(4):
                c_gt2[2 * ind] = buildings_4p[bld_nr][ind, 1]
                c_gt2[2 * ind + 1] = buildings_4p[bld_nr][ind, 0]

                if (BW < buildings_4p[bld_nr][ind, 1] and buildings_4p[bld_nr][ind, 1] < im.shape[0] - BW) and \
                        (BW < buildings_4p[bld_nr][ind, 0] and buildings_4p[bld_nr][ind, 0] < im.shape[1] - BW):
                    all_gt_corners.append([buildings_4p[bld_nr][ind, 1], buildings_4p[bld_nr][ind, 0]])
            draw_img_gt.polygon(c_gt2, fill='white')
            # print(c_gt2)

    img_gt_arr = np.asarray(img_gt, dtype=float) / 255
    img_gt_arr_nb = img_gt_arr[BW:(img_gt_arr.shape[0] - BW), BW:(img_gt_arr.shape[1] - BW)]

    # img_gt.show()

    all_gt_corners_arr = np.asarray(all_gt_corners)

    # ------------------------------------------------ predictions ----------------------------------------------------#

    # Detect all object classes and regress object bounds
    scores, boxes, angles_after_rp, rois = im_detect(sess, net, im)

    angles = angles_after_rp

    # Visualize detections for each class
    im = im[:, :, (2, 1, 0)]

    if show_figures:
        fig, ax = plt.subplots(figsize=(12, 12))
        plt.title('image number: ' + str(img_nr))
        ax.imshow(im, aspect='equal')
        print('show image ' + str(img_nr))

    # --------------------------------------------- single predictions ------------------------------------------------#

    NMS_THRESH = 0.2  # 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background       ####
        # print('boxes shape:')
        # print(boxes.shape)
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        angles = angles[keep, :]

    n_thresh = conf_thresh_array.shape[0]

    # tp, tn, fp, fn, recall, precision, accuracy, f1_score, IoU, threshold = (np.zeros(n_thresh) for _ in xrange(10))
    tp, tn, fp, fn, cor_tp, cor_tn, cor_fp, cor_fn = (np.zeros(n_thresh) for _ in xrange(8))

    # CONF_THRESH = 0.01	#0.06	#0.02	#0.2  #0.8
    for ind_thr, CONF_THRESH in enumerate(conf_thresh_array):
        #print('ind_thr ' + str(ind_thr))

        img_pred = Image.new('L', (im.shape[0], im.shape[1]), 'black')
        draw_img_pred = ImageDraw.Draw(img_pred)

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

        all_pred_corners = []
        if len(inds) > 0:
            for i in inds:
                bbox = dets[i, :4]
                score = dets[i, -1]
                angle = angles[i, :]

                c_orig = np.zeros((4, 2))
                c_orig[0, :] = [bbox[2], bbox[1]]
                c_orig[1, :] = [bbox[2], bbox[3]]
                c_orig[2, :] = [bbox[0], bbox[3]]
                c_orig[3, :] = [bbox[0], bbox[1]]

                angle = - angle  # that's what I need to do in order to get correct results
                # print(angle)
                sin_a = np.sin(angle)
                cos_a = np.cos(angle)

                roi_mid_w = (bbox[2] + bbox[0]) / 2
                roi_mid_h = (bbox[3] + bbox[1]) / 2

                # corner positions for the rotated rectangle
                c_rot = np.zeros((4, 2))
                c_rot2 = [0, 0, 0, 0, 0, 0, 0, 0]

                for ind in range(4):
                    w = c_orig[ind, 0]
                    h = c_orig[ind, 1]
                    w_from_mid = w - roi_mid_w;
                    h_from_mid = h - roi_mid_h;
                    w_from_mid_rot = cos_a * w_from_mid - sin_a * h_from_mid
                    h_from_mid_rot = sin_a * w_from_mid + cos_a * h_from_mid
                    c_rot[ind, 0] = w_from_mid_rot + roi_mid_w
                    c_rot[ind, 1] = h_from_mid_rot + roi_mid_h

                    if (BW < c_rot[ind, 0] < im.shape[0] - BW) and (BW < c_rot[ind, 1] < im.shape[1] - BW):
                        all_pred_corners.append([c_rot[ind, 0], c_rot[ind, 1]])
                    c_rot2[ind * 2] = c_rot[ind, 0]
                    c_rot2[ind * 2 + 1] = c_rot[ind, 1]
                    # print(c_rot2)

                if show_figures and ind_thr == 1:		#ind_thr = 5
                    ax.add_patch(
                        plt.Polygon(c_rot, True, alpha=0.4, edgecolor='red', linewidth=2)
                    )
                    ax.text((bbox[0] + bbox[2]) / 2 - 20, (bbox[1] + bbox[3]) / 2 - 2,  # bbox[0], bbox[1] - 2
                            '{:.3f}'.format(score),
                            bbox=dict(facecolor='blue', alpha=0.1),
                            fontsize=18, color='white')

                draw_img_pred.polygon(c_rot2, fill='white')

            all_pred_corners_arr = np.asarray(all_pred_corners)


            # do only update cor_tp and cor_fn if there are more than 0 ground truth buildings
            if all_pred_corners_arr.shape[0] == 0:
                all_pred_corners_arr = np.zeros((0, 2))

            if all_gt_corners_arr.shape[0] > 0:
                # find the nearest gt corner for each predicted corner
                dist_mat = distance_matrix(all_pred_corners_arr, all_gt_corners_arr)
                dist_min = np.min(dist_mat, axis=1)
                dist_argmin = np.argmin(dist_mat, axis=1)

                matches = np.zeros((all_gt_corners_arr.shape[0]))
                for ind in range(0, dist_min.shape[0]):
                    if dist_min[ind] < corner_dist_thr and matches[dist_argmin[ind]] == 0:
                        cor_tp[ind_thr] += 1
                        matches[dist_argmin[ind]] = 1
                    else:
                        cor_fp[ind_thr] += 1
                cor_fn[ind_thr] = all_gt_corners_arr.shape[0] - np.sum(matches)
            else:
                cor_fp[ind_thr] = all_pred_corners_arr.shape[0]


        else:   # if there are no buildings detected
            cor_tp[ind_thr] = 0
            cor_fp[ind_thr] = 0
            cor_fn[ind_thr] = all_gt_corners_arr.shape[0]

        if ind_thr == 0:
            gt_amount_neg_samples = max(cor_fp[0], 1)       # cor_fp[0] + cor_tn[0], but cor_tn[0] is set to be zero for a threshold == 0
            #cor_tn[ind_thr] = float(max(1, 100 - all_pred_corners_arr.shape[0]))
        cor_tn[ind_thr] = gt_amount_neg_samples - cor_fp[ind_thr]
        # remark: the amount of true negative corners is not defined and even though we do here try to
        # find a solution for this 'problem', we do not count this for any evaluation!


        # show the image with plotted bounding boxes
        if show_figures and save_figures and ind_thr == 1:	#ind_thr = 5
            plt.savefig(results_path + 'res_img_nms02' + str(img_nr) + '.png')	#plt.savefig(results_path + 'res_img_' + str(img_nr) + '.png')

        img_pred_arr = np.asarray(img_pred, dtype=float) / 255
        img_pred_arr_nb = img_pred_arr[BW:(img_pred_arr.shape[0] - BW), BW:(img_pred_arr.shape[1] - BW)]

        # compare img_gt_arr and img_pred_arr
        positive = img_gt_arr_nb[img_pred_arr_nb >= 0.5]
        negative = img_gt_arr_nb[img_pred_arr_nb < 0.5]

        tp[ind_thr] = float((positive[positive >= 0.5]).shape[0])
        fp[ind_thr] = float((positive[positive < 0.5]).shape[0])
        tn[ind_thr] = float((negative[negative < 0.5]).shape[0])
        fn[ind_thr] = float((negative[negative >= 0.5]).shape[0])

        """recall[ind_thr], precision[ind_thr], accuracy[ind_thr], f1_score[ind_thr], IoU[ind_thr] = \
            calculate_score_measures(tp[ind_thr], tn[ind_thr], fp[ind_thr], fn[ind_thr])

        threshold[ind_thr] = CONF_THRESH

        print('***')
        print('CONF_THRESH: ' + str(CONF_THRESH))
        print(tp[ind_thr])
        print(tn[ind_thr])
        print(fp[ind_thr])
        print(fn[ind_thr])
        print(precision[ind_thr])
        print(recall[ind_thr])
        print(accuracy[ind_thr])
        print(f1_score[ind_thr])
        print(IoU[ind_thr])"""

    return tp, tn, fp, fn, cor_tp, cor_tn, cor_fp, cor_fn  # precision, recall, accuracy, f1_score, IoU





        #print(np.max(img_pred_arr))
        #print(img_pred_arr.shape)
        #print(img_pred_arr)
        #fig.savefig('/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/test.png')




#----------------------------------------------------------------------------------------------------------------------#





def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')
    parser.add_argument('--data', dest='data_path', help='Data path',
                        default='/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    # load model
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    #sess.run(tf.initialize_all_variables())

    print '\n\nLoaded network {:s}'.format(args.model)

    data_path = args.data_path

    # data_path = '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/data/building_data/'
    test_inds_path = os.path.join(data_path, 'test.txt')	#os.path.join(data_path, 'test.txt')

    # find indices of the test images
    test_file = open(test_inds_path, "r")
    lines = test_file.readlines()
    test_inds = []
    for ind in range(0,len(lines)):
        test_inds.append(int(lines[ind].rstrip('\n')))
    test_file.close()

    # define path of where to save the results
    path_model = args.model
    len_str = len(path_model.split('/', 30))
    name_model = path_model.split('/', 20)[len_str-1]
    results_path = path_model.split(name_model, 2)[0]
    #is usually: '/scratch/rueegnad_scratch/my_programs/Faster-RCNN_TF/output/faster_rcnn_end2end_sI/building_train/'


    thresholds = np.arange(0, 1.01, 0.01)

    tp_tot, tn_tot, fp_tot, fn_tot, recall_tot, precision_tot, accuracy_tot, f1_score_tot, IoU_tot, TPR, TNR = \
        (np.zeros(len(thresholds)) for _ in xrange(11))

    cor_tp_tot, cor_tn_tot, cor_fp_tot, cor_fn_tot, cor_recall_tot, cor_precision_tot, cor_accuracy_tot,\
    cor_f1_score_tot, cor_IoU_tot, cor_TPR, cor_TNR = (np.zeros(len(thresholds)) for _ in xrange(11))

    with open(results_path + 'results.csv', 'wb') as result_file:
        csv_writer = csv.writer(result_file)

        print('in total there are ' + str(len(test_inds)) + ' images')
        for ind_img, img_nr in enumerate(test_inds[0:10]):			#[0:600:20]	#test_inds[0:300:10]
            print('calculating results for image ' + str(img_nr) +  '      ' + str(ind_img) + '/' + str(len(test_inds)))
            CORNER_DIST_THR = 25    #20
            tp, tn, fp, fn, cor_tp, cor_tn, cor_fp, cor_fn = evaluate_single_image(sess, net, thresholds, img_nr,
                                                                                   data_path, results_path,
                                                                                   corner_dist_thr = CORNER_DIST_THR,
                                                                                   show_figures = True,
                                                                                   save_figures = False)

            csv_writer.writerow(['  '])
            csv_writer.writerow(['img_nr'] + [img_nr])
            csv_writer.writerow(['threshold'] + [x for x in thresholds])
            csv_writer.writerow(['tp'] + [x for x in tp])
            csv_writer.writerow(['tn'] + [x for x in tn])
            csv_writer.writerow(['fp'] + [x for x in fp])
            csv_writer.writerow(['fn'] + [x for x in fn])

            tp_tot = tp_tot + tp
            tn_tot = tn_tot + tn
            fp_tot = fp_tot + fp
            fn_tot = fn_tot + fn

            cor_tp_tot = cor_tp_tot + cor_tp
            cor_tn_tot = cor_tn_tot + cor_tn
            cor_fp_tot = cor_fp_tot + cor_fp
            cor_fn_tot = cor_fn_tot + cor_fn



        for ind_thr in range(0, len(thresholds)):
            recall_tot[ind_thr], precision_tot[ind_thr], accuracy_tot[ind_thr], f1_score_tot[ind_thr], IoU_tot[
                ind_thr] = \
                calculate_score_measures(tp_tot[ind_thr], tn_tot[ind_thr], fp_tot[ind_thr], fn_tot[ind_thr])
            cor_recall_tot[ind_thr], cor_precision_tot[ind_thr], cor_accuracy_tot[ind_thr], cor_f1_score_tot[ind_thr], \
            cor_IoU_tot[ind_thr] = \
                calculate_score_measures(cor_tp_tot[ind_thr], cor_tn_tot[ind_thr], cor_fp_tot[ind_thr], cor_fn_tot[ind_thr])
            # print('   ')
            # print('threshold: ' +  str(thresholds[ind_thr]))
            # print('precision: ' + str(precision_tot[ind_thr]))
            # print('recall: ' + str(recall_tot[ind_thr]))
            # print('accuracy: ' + str(accuracy_tot[ind_thr]))
            # print('f1_score: ' + str(f1_score_tot[ind_thr]))
            # print('IoU: ' + str(IoU_tot[ind_thr]))

            TPR[ind_thr] = tp_tot[ind_thr] / (tp_tot[ind_thr] + fn_tot[ind_thr])  # true positive rate
            TNR[ind_thr] = tn_tot[ind_thr] / (tn_tot[ind_thr] + fp_tot[ind_thr])  # true negative rate
            # FPR = 1-TNR   # false positive rate

            #cor_TPR[ind_thr] = cor_tp_tot[ind_thr] / (cor_tp_tot[ind_thr] + cor_fn_tot[ind_thr])  # true positive rate
            #cor_TNR[ind_thr] = cor_tn_tot[ind_thr] / (cor_tn_tot[ind_thr] + cor_fp_tot[ind_thr])  # true negative rate

            # if cor_tp_tot[ind_thr] == 0:
            #	cor_TPR[ind_thr] = 0
            # else:
            #    cor_TPR[ind_thr] = cor_tp_tot[ind_thr] / (cor_tp_tot[ind_thr] + cor_fn_tot[ind_thr])    # true positive rate
            # if cor_tn_tot[ind_thr] == 0:
            #    cor_TNR[ind_thr] = 0
            # else:
            #	cor_TNR[ind_thr] = cor_tn_tot[ind_thr] / (cor_tn_tot[ind_thr] + cor_fp_tot[ind_thr])    # true negative rate

        csv_writer.writerow(['  '])
        csv_writer.writerow(['summary'])
        csv_writer.writerow(['threshold'] + [x for x in thresholds])
        csv_writer.writerow(['tp'] + [x for x in tp_tot])
        csv_writer.writerow(['tn'] + [x for x in tn_tot])
        csv_writer.writerow(['fp'] + [x for x in fp_tot])
        csv_writer.writerow(['fn'] + [x for x in fn_tot])
        csv_writer.writerow(['precision'] + [x for x in precision_tot])
        csv_writer.writerow(['recall'] + [x for x in recall_tot])
        csv_writer.writerow(['accuracy'] + [x for x in accuracy_tot])
        csv_writer.writerow(['F1_score'] + [x for x in f1_score_tot])
        csv_writer.writerow(['IoU'] + [x for x in IoU_tot])





        # print('save results in: ' + results_path + 'results.csv')
        # for im_name in im_names:
        # print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        # print 'Demo for data/demo/{}'.format(im_name)
        # demo(sess, net, im_name)

    plt.show()
    """
    # plot recall and precision for area
    plt.figure()
    plt.plot(recall_tot, precision_tot)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title('Recall-Precision Curve')
    plt.xlim([0, 1.05])
    # plt.ylim([0, 1.05])
    plt.grid(True)
    plt.savefig(results_path + 'recall_precision.png')
    plt.show()

    # F1-score for area
    plt.figure()
    plt.plot(thresholds, f1_score_tot)
    plt.xlabel('Threshold')
    plt.ylabel('F1-score')
    #plt.title('Recall-Precision Curve')
    plt.xlim([0, 1.05])
    # plt.ylim([0, 1.05])
    plt.grid(True)
    plt.savefig(results_path + 'F1_score.png')
    plt.show()

    # ROC curve for area
    plt.figure()
    plt.plot(1 - TNR, TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC Curve')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.savefig(results_path + 'ROC.png')
    plt.show()

    #--------------------------------------
    # plot recall and precision for corners
    plt.figure()
    plt.plot(cor_recall_tot, cor_precision_tot)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title('Recall-Precision Curve')
    plt.xlim([0, 1.05])
    # plt.ylim([0, 1.05])
    plt.grid(True)
    plt.savefig(results_path + 'corners_recall_precision.png')
    plt.show()

    # F1-score for corners
    plt.figure()
    plt.plot(thresholds, cor_f1_score_tot)
    plt.xlabel('Threshold')
    plt.ylabel('F1-score')
    #plt.title('Recall-Precision Curve')
    plt.xlim([0, 1.05])
    # plt.ylim([0, 1.05])
    plt.grid(True)
    plt.savefig(results_path + 'corners_F1_score.png')
    plt.show()


    # ROC curve for corners
    plt.figure()
    plt.plot(1 - cor_TNR, cor_TPR)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('ROC Curve for Corners')
    plt.xlim([0, 1.05])
    plt.ylim([0, 1.05])
    plt.grid(True)
    plt.savefig(results_path + 'corners_ROC.png')
    plt.show()

    # tp, fp and fn for corners
    plt.figure()
    plt.plot(thresholds, tp_tot)
    plt.plot(thresholds, fp_tot)
    plt.plot(thresholds, fn_tot)
    plt.xlabel('Threshold')
    plt.ylabel('Amount of Points')
    plt.legend(['True Positives', 'False Positives', 'False Negatives'])
    #plt.title('Evaluation with Respect to Corner Points')
    plt.xlim([0, 1.05])
    plt.grid(True)
    plt.savefig(results_path + 'corners_evaluation.png')
    plt.show()
    """




