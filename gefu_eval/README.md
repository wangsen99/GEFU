<h2 align="center">
  <b>Generalized Enhancement For Understanding (GEFU)</b>
</h2>

## Enhancment
After enhancing all the dark images, you need to put them into the corresponding folders according to the task. The final file format should be:
```shell
gefu_eval
├── classification
│   ├── checkpoints
│   │   ├── model_best.pt
│   ├── data
│   │   ├── gefu_cls # your enhanced images
│   ├── ...
├── detection
│   ├── checkpoints
│   │   ├── yolov3.pth
│   ├── data
│   │   ├── DARK_FACE
│   │   │   ├── gefu_det # your enhanced images
│   │   │   ├── label
│   │   │   ├── main
│   │   │   ├── val
│   │   │   ├── xml
│   ├── ...
├── segmentation
│   ├── checkpoints
│   │   ├── best_weights.pth.tar
│   ├── data
│   │   ├── bdd100k-night
│   │   │   ├── images
│   │   │   │   ├── gefu_seg # your enhanced images
│   │   │   │   ├── test
│   │   │   ├── labels
│   ├── ...
```

## Classification
We porvide test data in [Classification](https://huggingface.co/wangsen99/GEFU/tree/main/Classification) from [CODaN](https://github.com/Attila94/CODaN), and baseline model.

Get the results by running `python eval.py --checkpoint checkpoints/model_best.pt`.

## Segmentation
We porvide test data in [Segmentation](https://huggingface.co/wangsen99/GEFU/tree/main/Segmentation) from [BDD100k-night](https://github.com/wangsen99/FDLNet), and baseline model.

Get the results by running `python eval.py --weight checkpoints/best_weights.pth.tar --save_path './' --save` with visual results.

## Detection
This part is a little complicate, our test code is based on [MAET](https://github.com/cuiziteng/ICCV_MAET), we porvide test data in [Detection](https://huggingface.co/wangsen99/GEFU/tree/main/Detection), and baseline model.

First, you need to prepare a new enviroment:

```shell
conda create -n face python=3.7 -y
conda activate face
cd detection
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
pip install mmcv-full==1.1.6 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html
pip install matplotlib opencv-python Pillow tqdm scipy dataclasses future
pip install yapf==0.40.1 cython==0.29.33
pip install -e .
```

Then, get the results by running `python tools/test.py configs/YOLO/yolov3_darkface.py checkpoints/yolov3.pth --eval mAP --show-dir ./results/` with visual results.