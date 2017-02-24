# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys


class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, imdb, roidb, output_dir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'
        if cfg.TRAIN.BBOX_REG:
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print 'done'

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100)

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            print(self.bbox_stds)
            print(weights_shape)
            print(orig_0.shape)
            sess.run(weights.assign(orig_0 * np.tile(self.bbox_stds, (weights_shape[0],1))))
            sess.run(biases.assign(orig_1 * self.bbox_stds + self.bbox_means))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # restore net to original state
            sess.run(weights.assign(orig_0))
            sess.run(biases.assign(orig_1))


    def train_model(self, sess, max_iters):
        """Network training loop."""

        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)
        #print('***************333331111331313131*****************')
        #print(self.roidb[0]['gt_orientations'])     # it exists!
   
        # RPN
        # classification loss
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'),[-1,2])
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0],[-1])
        # ignore_label(-1)
        rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_label,-1))),[-1,2])
        rpn_label = tf.reshape(tf.gather(rpn_label,tf.where(tf.not_equal(rpn_label,-1))),[-1])
        rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score, rpn_label))


        # bounding box regression L1 loss
        rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
        rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1],[0,2,3,1])
        rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2],[0,2,3,1])
        rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3],[0,2,3,1])
        smoothL1_sign = tf.cast(tf.less(tf.abs(tf.sub(rpn_bbox_pred, rpn_bbox_targets)),1),tf.float32)
        rpn_loss_box = tf.mul(tf.reduce_mean(tf.reduce_sum(tf.mul(rpn_bbox_outside_weights,tf.add(
                       tf.mul(tf.mul(tf.pow(tf.mul(rpn_bbox_inside_weights, tf.sub(rpn_bbox_pred, rpn_bbox_targets))*3,2),0.5),smoothL1_sign),
                       tf.mul(tf.sub(tf.abs(tf.sub(rpn_bbox_pred, rpn_bbox_targets)),0.5/9.0),tf.abs(smoothL1_sign-1)))), reduction_indices=[1,2])),10)
 
        # R-CNN
        # classification loss
        cls_score = self.net.get_output('cls_score')
        #label = tf.placeholder(tf.int32, shape=[None])
        label = tf.reshape(self.net.get_output('roi-data')[1],[-1])
        cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(cls_score, label))


        # bounding box regression L1 loss
        bbox_pred = self.net.get_output('bbox_pred')
        bbox_targets = self.net.get_output('roi-data')[2]
        bbox_inside_weights = self.net.get_output('roi-data')[3]
        bbox_outside_weights = self.net.get_output('roi-data')[4]
        loss_box = tf.reduce_mean(tf.reduce_sum(tf.mul(bbox_outside_weights,tf.mul(bbox_inside_weights, tf.abs(tf.sub(bbox_pred, bbox_targets)))), reduction_indices=[1]))


        # rotation regression L1 loss                                       # new
        orientation_pred = self.net.get_output('orientation_pred')
        orientation_targets = self.net.get_output('roi-data')[5]
        #orientation_targets = self.net.get_output('roi-data/gt_orientations_new')  # why does this not work?
        #loss_orientation = tf.reduce_mean(tf.abs(tf.sub(orientation_pred, orientation_targets)))
        label_as_float = tf.cast(label, tf.float32)
        #sub = tf.abs(tf.sub(orientation_pred, orientation_targets))
        sub = tf.square(tf.sub(orientation_pred, orientation_targets))
        mult = tf.mul(label_as_float, tf.transpose(sub))
        #loss_orientation = tf.reduce_mean(mult)
        loss_orientation = tf.div(tf.reduce_sum(mult), tf.reduce_sum(label_as_float))


        #loss_orientation = tf.reduce_mean(tf.abs(tf.mul(label_as_float, tf.sub(orientation_pred, orientation_targets))))


        print('................. loss orientation .....................')
        print(tf.sub(orientation_pred, orientation_targets))
        print(orientation_pred)
        print(orientation_targets)
        print(loss_orientation)
        print(label)
        print(cls_score)


        """print('...................................................................................................')
        print(self.net.get_output('roi-data'))
        print(orientation_pred)
        orientation_targets = self.net.get_output('gt_orientations')  #????
        print(orientation_targets)"""




        loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box + 0.1*loss_orientation       # was 0.3
        #loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box


        # optimizer
        lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
        momentum = cfg.TRAIN.MOMENTUM
        train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss)

        # iintialize variables
        sess.run(tf.initialize_all_variables())
        if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, True)

        last_snapshot_iter = -1
        timer = Timer()
        for iter in range(max_iters):
            # learning rate
            if iter >= cfg.TRAIN.STEPSIZE:
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE * cfg.TRAIN.GAMMA))
            else:
                sess.run(tf.assign(lr, cfg.TRAIN.LEARNING_RATE))

            # get one batch
            blobs = data_layer.forward()
            #print('******************************************************************')
            #print(blobs['gt_orientations'])

            # Make one SGD update
            #feed_dict={self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5, \
            #             self.net.gt_boxes: blobs['gt_boxes']}
            feed_dict = {self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob: 0.5,
                         self.net.gt_boxes: blobs['gt_boxes'], self.net.gt_orientations: blobs['gt_orientations']}

            timer.tic()

            #rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, loss_orientation, _ = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, loss_orientation, train_op], feed_dict=feed_dict)
            rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, loss_or, _ = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, loss_orientation, train_op], feed_dict=feed_dict)

            """#my_labels, or_pred, loss_or, diff_orient = sess.run([label_as_float, orientation_pred, loss_orientation, diff_orient], feed_dict=feed_dict)
            my_labels, or_pred, loss_or, mult_pred, sub_pred  = sess.run([label_as_float, orientation_pred, loss_orientation, mult, sub], feed_dict=feed_dict)

            print(' ..................... my labels ....................')
            #print(my_labels)
            #print(or_pred)
            print(loss_or)
            print(sub_pred)
            print(mult_pred)
            print 'loss orientation: %.4f' %(loss_or)"""


            timer.toc()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d,total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, loss_or: %.4f, lr: %f' % \
                      (iter + 1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value + loss_or,
                       rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, loss_or, lr.eval())
                """print 'iter: %d / %d,total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,
                         rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr.eval())"""

                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    """if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        imdb.append_flipped_images()
        print 'done'"""                                     # think about including this flipping operation again....

    print 'Preparing training data...'
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
            """print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            print(imdb.image_index)
            print(len(imdb.image_index))        # is twice as long as it should be!! <- due to flipping!
            print('&&&&&&&&&&&&&&&&&&&&&&&')"""

    else:
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb


def get_data_layer(roidb, num_classes):
    """return a data layer."""
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer


def train_net(network, imdb, roidb, output_dir, pretrained_model=None, max_iters=40000):
    """Train a Fast R-CNN network."""

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sw = SolverWrapper(sess, network, imdb, roidb, output_dir, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'
