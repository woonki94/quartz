
first 

pip install -r requirements.txt

recommended , H100 GPU with memory 40GB


### 1. Preprocess dataset
	1. Download enron data from https://www.cs.cmu.edu/~enron/ -> ./preprocessing/datasets
	2. Download Pii data from https://www.kaggle.com/datasets/verracodeguacas/ai4privacy-pii?select=pii-masking-43k -> ./preprocessing/datasets
	3. Wiki103 via library
	4. openwebtext via library


#### 1) Data for GPT-2 Nano
	1. Wiki-text 103 (General)
	2. openwebtext 5G (General)
	3. Enron Email (Privacy)
	4. PII from kaggle (Privacy)

#### 2)Data for Pre-trained GPT-2 Large
	1. Enron Email (Privacy)
	2. PII from kaggle(Privacy)


make dir
preprocess/cleaned_dataset
preprocess/merged_dataset

run sh scripts/run_preprocess.sh

then we get 
/preprocess/cleaned_dataset/

cleaned_email.txt
cleaned_pii.txt
cleaned_openwebtext_5gb.txt
cleaned_wikitext103.txt

Then run sh scripts/run_merge_dataset.sh

it will run merge_dataset_large.py and merge_dataset_nano.py

since large we are doing finetuning, we only need enron email and pii data
and nano from scratch we also need some general text data like openwebtext and wikitext + enron and pii


then we get, 
preprocess/merged_dataset/merged_large.txt
preprocess/merged_dataset/merged_nano.txt


now preprocessing is done



### 2. Train  Models


if you want wandb log put your wandb keys on

root/wandbkey.env
inside    WANDB_API_KEY=yourkey



#### 1. GPT -Large

make folder, root/chkpt and root/GPT-Large/data/tokenized_data

for plain model-> run python ./GPT2-Large/GPT2_finetune.py,, will stack checkpoints on root/chkpt/large/plain

for dp-sgd attached model->  run python ./GPT2-Large/GPT2_opacus_finetune.py,, will stack checkpoints on root/chkpt/large/dp_sgd




#### 2. GPT-Nano

make dir './GPT2-Nano/data'

and Run python GPT2-Nano/prepare.py 

It will make a vocab and data  in ./GPT2-Nano/data/customtext

the generated files are
train.bin
val.bin
meta.pkl


now run python GPT2-Nano/train.py. for plain
and python GPT2-Nano/train_DP_SGD.py for dp-sgd defense applied

it will make checkpoints on

root/chkpt/nano/plain/
root/chkpt/nano/dp-sgd/



Now we are done training 4 models...

or you can download chkpt of ours

and put it in that folder


### 4. Try Extraction 
1. extraction_nano.py -> log file in categorization/Empty input cases (shell script)
2. categorization.py -> 

### 5. Measure Accuracy

for nano gpt - we measure accuracy by building prompt - reference set of files from wiki103 the dataset which we trained for nano

for large gpt- we measure accuracy by building prompt- reference set of files from the enron email and pii data which we used to finetune gpt large.


first run python  ./measure_accuracy/generate_prset

it will create 
root/measure_accuracy/ref/prompts.txt<- gpt nano
root/measure_accuracy/ref/references.txt<- gpt nano
root/measure_accuracy/ref/prompts_pii.txt <- as discussed used for gpt large
root/measure_accuracy/ref/references_pii.txt <- gpt large


then we generate the text file using each models using the prompts we made previously

run sh root/scripts/generate_output.sh

It will create 
root/measure_accuracy/outputs_from_model/large/output_dp.txt
root/measure_accuracy/outputs_from_model/large/output_plain.txt
root/measure_accuracy/outputs_from_model/nano/output_dp.txt
root/measure_accuracy/outputs_from_model/nano/output_plain.txt




now 
to measure accuracy for nano
run     python root/measure_accuracy/measure_accuracy.py --size nano 


for large
 python root/measure_accuracy/measure_accuracy.py --size large

