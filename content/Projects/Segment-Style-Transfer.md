



pip install -r requirements.txt


### 1. Training neural style 

first run python root/fast-neural-style-pytorch/download_coco_and_vgg.py 


it will download coco dataset in 
root/fast-neural-style-pytorch/dataset
and vgg pretrained in 
root/fast-neural-style-pytorch/models

then 
put the desired style image in the root/fast-neural-style-pythorch/images 

and in the train.py, change STYLE_IMAGE_PATH to match your file


then run sh ./script/train_style.sh

it will create checkpoints in the root/fast-neural-style-pytorch/models

and the final model in root/fast-neural-style-pytorch/transforms
in the name of the style name
i.e. if trained with jojo.png.  the final model name will be saved as -> jojo.pth







