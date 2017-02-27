# FRCNN_git



### 



### Requirements for Software and Hardware

The requirements are the same as for the original tensorflow Faster R-CNN implementation (see [Faster R-CNN tensorflow](https://github.com/smallcorgi/Faster-RCNN_TF)). 


### Installation 

1. Clone the Faster R-CNN repository
  ```Shell
  # Clone the git repository
  git clone --recursive https://github.com/runa91/FRCNN_git.git
  ```

2. Build Cython modules
    ```Shell
    cd $FRCNN_ROOT/lib
    make
    ```

3. In case you'd like to train a model, download a pretrained VGG model here:
[Pretrained VGG](https://polybox.ethz.ch/index.php/s/oitt4w7HRWNxDmY)
store it as follows: 
$FRCNN_ROOT/data/pretrained_model/VGG_imagenet.npy

 
4. A model trained on buildings can be found at the same location:
[Pretrained VGG](https://polybox.ethz.ch/index.php/s/oitt4w7HRWNxDmY)
store it as follows:
$FRCNN_ROOT/output/faster_rcnn_end2end_sI/building_train/ VGGnet_fast_rcnn_iter_60000.ckpt


### Demo

1. I've included a few test images, you are now able to run a demonstration:
	```Shell
	cd $FRCN_ROOT/tools
	python2.7 building_evaluation_git.py --model "$FRCNN_ROOT/output/faster_rcnn_end2end_sI/building_train/VGGnet_fast_rcnn_iter_60000.ckpt" --data "$FRCNN_ROOT/data/building_data/"
	```

### Training

1. Create your own data set and add a new class similar to my 'building' class to the folder $FRCNN_ROOT/lib/datasets/

You may also have a look at $FRCNN_ROOT/changes_wrt_orig_frcnn.odt for more information.

2. Train a model:
    ```Shell
    cd $FRCNN_ROOT
    ./experiments/scripts/faster_rcnn_end2end_new.sh 0 VGG16 building
    ```

### References

[Faster R-CNN paper](https://papers.nips.cc/paper/5638-faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.pdf)

[Faster R-CNN tensorflow](https://github.com/smallcorgi/Faster-RCNN_TF)



