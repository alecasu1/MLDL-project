
from imports import *
from functions import *

# Download the AWS and Ego4D CLIs, then download the annotations locally
%%bash

# Set up the AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip -o awscliv2.zip >/dev/null
sudo ./aws/install >/dev/null 2>&1
aws configure set aws_access_key_id "$AWS_ACCESS_KEY_ID" && aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY"
rm "awscliv2.zip"
# Set up the Ego4D CLI
!pip install ego4d
# Download the Ego4D Annotations to ego4d_data/
!ego4d --output_directory="/content/ego4d_data" --datasets annotations omnivore_video_swinl_fp16 --benchmarks nlq -y  --version v1
# Ensure we have downloaded the files correctly
!ls /content/ego4d_data/v1/annotations | grep nlq
!ls /content/ego4d_data/v1/omnivore_video_swinl_fp16 | wc -l
%%bash
git clone https://github.com/EGO4D/episodic-memory
cd episodic-memory
git pull
git checkout nlq_fixes_and_fp16_support
ann_data = json.load(open("/content/ego4d_data/v1/annotations/nlq_train.json"))
print(ann_data["version"])
print(ann_data["date"])
print(ann_data["description"])
print(ann_data["manifest"])
#print(ann_data["videos"])
# please see https://ego4d-data.org/docs/data/annotations-schemas/

anns = []
for vid in ann_data["videos"]:
    for clip in vid["clips"]:
        for ann in clip["annotations"]:
            for query in ann["language_queries"]:
                anns.append({
                    "id":vid["video_uid"],
                    "query_start_time_sec": clip["video_start_sec"],
                    "query_end_time_sec": clip["video_end_sec"],
                    "query_response_start_time_sec": query["video_start_sec"],
                    "query_response_end_time_sec": query["video_end_sec"],
                    "query_template": query.get("template", None),
                    "query": query.get("query", None),
                })
ann_data_ego4d = json.load(open("/content/ego4d_data/ego4d.json"))
anns_ego4d= []
for vid in ann_data_ego4d["videos"]:
    for item in vid["scenarios"]:
      if "Cooking" in item  or "commuting" in item:
        anns_ego4d.append({"id":vid["video_uid"]})

df = pd.DataFrame(anns_ego4d)
match_count = 0
unique_ann = set()
for item in anns:
  unique_ann.add(item["id"])
for item in anns_ego4d:
    if item['id'] in unique_ann:
        match_count += 1
print(f"Percentage of seek scenarios in dataset nlq_train : {match_count/len(unique_ann)*100:.2f}%")
print(anns[0])
print(anns[1])
num_queries = len(anns)
relative_query_sizes = np.array([
    (a["query_response_end_time_sec"] - a["query_response_start_time_sec"]) / (a["query_end_time_sec"] - a["query_start_time_sec"])
    for a in anns
])
query_sizes = np.array([
    (a["query_response_end_time_sec"] - a["query_response_start_time_sec"])
    for a in anns
])
clip_sizes = np.array([
    (a["query_end_time_sec"] - a["query_start_time_sec"])
    for a in anns
])
query_sizes.max(), query_sizes.min(), query_sizes.std()
clip_sizes.max(), clip_sizes.min(), clip_sizes.std()
# less than or equal to 4 frames => 9% of training data
(query_sizes <= 4/30).sum() / len(relative_query_sizes)
plt.rcParams["figure.figsize"] = (16, 9)
plt.hist(relative_query_sizes[relative_query_sizes < 0.2], density=True, bins=128)
plt.show()
plt.hist(relative_query_sizes[relative_query_sizes > 0.2], density=True, bins=128)
plt.show()
clip_sizes.mean(), clip_sizes.max(), clip_sizes.min(), clip_sizes.std(), np.median(clip_sizes)
plt.hist(clip_sizes)
plt.show()
with open("vars.sh", "w") as out_f:
  out_f.write("""
export NAME=VSLNet_omnivore_video_fp16
export TASK_NAME=nlq_official_v1_$NAME
export BASE_DIR=data/dataset/nlq_official_v1_$NAME
export FEATURE_BASE_DIR=data/features/nlq_official_v1_$NAME/
export FEATURE_DIR=$FEATURE_BASE_DIR/video_features
export MODEL_BASE_DIR=/content/nlq_official_v1/checkpoints/

cd episodic-memory/NLQ/VSLNet
"""
  )
%%bash

source vars.sh

echo $FEATURE_BASE_DIR
mkdir -p $FEATURE_BASE_DIR
ln -s /content/ego4d_data/v1/omnivore_video_swinl_fp16 $FEATURE_DIR
%%bash
%%capture

source vars.sh
pip install nltk submitit torch torchaudio torchvision tqdm transformers tensorboard Pillow terminaltables
%%bash

source vars.sh

python utils/prepare_ego4d_dataset.py \
    --input_train_split /content/ego4d_data/v1/annotations/nlq_train.json \
    --input_val_split /content/ego4d_data/v1/annotations/nlq_val.json \
    --input_test_split /content/ego4d_data/v1/annotations/nlq_test_unannotated.json \
    --video_feature_read_path $FEATURE_DIR \
    --clip_feature_save_path $FEATURE_BASE_DIR/official \
    --output_save_path $BASE_DIR
# Load the TensorBoard notebook extension
%load_ext tensorboard
!mkdir -p /content/episodic-memory/NLQ/VSLNet/runs/
%%bash

source vars.sh

export DATALOADER_WORKERS=1
export NUM_WORKERS=2
export VAL_JSON_PATH="/content/ego4d_data/v1/annotations/nlq_val.json"

export BATCH_SIZE=32
export DIM=128
export NUM_EPOCH=10
export MAX_POS_LEN=128
export INIT_LR=0.0025

export TB_LOG_NAME="${NAME}_bs${BATCH_SIZE}_dim${DIM}_epoch${NUM_EPOCH}_ilr${INIT_LR}"

python main.py \
    --task $TASK_NAME \
    --predictor bert \
    --dim $DIM \
    --mode train \
    --video_feature_dim 1536 \
    --max_pos_len $MAX_POS_LEN \
    --init_lr $INIT_LR \
    --epochs $NUM_EPOCH \
    --batch_size $BATCH_SIZE \
    --fv official \
    --num_workers $NUM_WORKERS \
    --data_loader_workers $DATALOADER_WORKERS \
    --model_dir $MODEL_BASE_DIR/$NAME \
    --eval_gt_json $VAL_JSON_PATH \
    --log_to_tensorboard $TB_LOG_NAME \
    --tb_log_freq 5 \
    --remove_empty_queries_from train
%%bash

source vars.sh

echo $FEATURE_BASE_DIR
mkdir -p $FEATURE_BASE_DIR
ln -s /content/ego4d_data/v1/omnivore_video_swinl_fp16 $FEATURE_DIR
%%bash
%%capture

source vars.sh
pip install nltk submitit torch torchaudio torchvision tqdm transformers tensorboard Pillow terminaltables
%%bash

source vars.sh

python utils/prepare_ego4d_dataset.py \
    --input_train_split /content/ego4d_data/v1/annotations/nlq_train.json \
    --input_val_split /content/ego4d_data/v1/annotations/nlq_val.json \
    --input_test_split /content/ego4d_data/v1/annotations/nlq_test_unannotated.json \
    --video_feature_read_path $FEATURE_DIR \
    --clip_feature_save_path $FEATURE_BASE_DIR/official \
    --output_save_path $BASE_DIR
%%bash
source vars.sh

export DATALOADER_WORKERS=1
export NUM_WORKERS=2
export VAL_JSON_PATH="/content/ego4d_data/v1/annotations/nlq_val.json"

export BATCH_SIZE=32
export DIM=128
export NUM_EPOCH=10
export MAX_POS_LEN=128
export INIT_LR=0.0025

export TB_LOG_NAME="${NAME}_bs${BATCH_SIZE}_dim${DIM}_epoch${NUM_EPOCH}_ilr${INIT_LR}"

python main.py \
    --task $TASK_NAME \
    --predictor bert \
    --dim $DIM \
    --mode train \
    --video_feature_dim 1536 \
    --max_pos_len $MAX_POS_LEN \
    --init_lr $INIT_LR \
    --epochs $NUM_EPOCH \
    --batch_size $BATCH_SIZE \
    --fv official \
    --num_workers $NUM_WORKERS \
    --data_loader_workers $DATALOADER_WORKERS \
    --model_dir $MODEL_BASE_DIR/$NAME \
    --eval_gt_json $VAL_JSON_PATH \
    --log_to_tensorboard $TB_LOG_NAME \
    --tb_log_freq 5 \
    --remove_empty_queries_from train

# ID del file Google Drive (estratto dal link condiviso)
file_id = "1U318S34jw3uNnsURJ1T40YwsSuK5_-RJ"#1U318S34jw3uNnsURJ1T40YwsSuK5_-RJ"
# Scarica il file
downloaded = drive.CreateFile({'id': file_id})
downloaded.GetContentFile('egovlp_fp16.tar.gz')

# Percorso del file vars.sh
file_path = 'vars.sh'

# Nome della variabile da modificare e il nuovo valore
var_name = 'NAME'
new_value = 'VSLNet_egovlp_fp16'
change_var_sh(file_path,var_name,new_value)
# Nome della variabile da modificare e il nuovo valore
var_name = 'MODEL_BASE_DIR'
new_value = '/content/storage/ego4d/v2'
change_var_sh(file_path,var_name,new_value)
%%bash

source vars.sh

echo $FEATURE_BASE_DIR
mkdir -p $FEATURE_BASE_DIR
ln -s /content/ego4d_data/v1/egovlp_fp16 $FEATURE_DIR
%%bash
%%capture

source vars.sh
pip install nltk submitit torch torchaudio torchvision tqdm transformers tensorboard Pillow terminaltables
%%bash

source vars.sh

python utils/prepare_ego4d_dataset.py \
    --input_train_split /content/ego4d_data/v1/annotations/nlq_train.json \
    --input_val_split /content/ego4d_data/v1/annotations/nlq_val.json \
    --input_test_split /content/ego4d_data/v1/annotations/nlq_test_unannotated.json \
    --video_feature_read_path $FEATURE_DIR \
    --clip_feature_save_path $FEATURE_BASE_DIR/official \
    --output_save_path $BASE_DIR
%%bash

# Source the environment variables from vars.sh
source vars.sh

export NAME=EgoVLP
export DATALOADER_WORKERS=1
export NUM_WORKERS=2
export VAL_JSON_PATH="/content/ego4d_data/v1/annotations/nlq_val.json"

export BATCH_SIZE=64
export DIM=128
export NUM_EPOCH=10
export MAX_POS_LEN=128
export INIT_LR=0.0025

export TB_LOG_NAME="${NAME}_bs${BATCH_SIZE}_dim${DIM}_epoch${NUM_EPOCH}_ilr${INIT_LR}"

# Run the Python script with the specified hyperparameters
python main.py \
    --task $TASK_NAME \
    --predictor bert \
    --dim $DIM \
    --mode train \
    --video_feature_dim 256 \
    --max_pos_len $MAX_POS_LEN \
    --init_lr $INIT_LR \
    --epochs $NUM_EPOCH \
    --batch_size $BATCH_SIZE \
    --fv official \
    --num_workers $NUM_WORKERS \
    --data_loader_workers $DATALOADER_WORKERS \
    --model_dir $MODEL_BASE_DIR/$NAME \
    --eval_gt_json $VAL_JSON_PATH \
    --log_to_tensorboard $TB_LOG_NAME \
    --tb_log_freq 5 \
    --remove_empty_queries_from train
# Percorso del file vars.sh
file_path = 'vars.sh'

# Nome della variabile da modificare e il nuovo valore
var_name = 'NAME'
new_value = 'VSLBase_egovlp_fp16'
change_var_sh(file_path,var_name,new_value)
change_layers(new_class_content)

%%bash

source vars.sh

echo $FEATURE_BASE_DIR
mkdir -p $FEATURE_BASE_DIR
ln -s /content/ego4d_data/v1/egovlp_fp16 $FEATURE_DIR
%%bash
%%capture

source vars.sh
pip install nltk submitit torch torchaudio torchvision tqdm transformers tensorboard Pillow terminaltables
%%bash

source vars.sh

python utils/prepare_ego4d_dataset.py \
    --input_train_split /content/ego4d_data/v1/annotations/nlq_train.json \
    --input_val_split /content/ego4d_data/v1/annotations/nlq_val.json \
    --input_test_split /content/ego4d_data/v1/annotations/nlq_test_unannotated.json \
    --video_feature_read_path $FEATURE_DIR \
    --clip_feature_save_path $FEATURE_BASE_DIR/official \
    --output_save_path $BASE_DIR
%%bash

source vars.sh

export DATALOADER_WORKERS=1
export NUM_WORKERS=2
export VAL_JSON_PATH="/content/ego4d_data/v1/annotations/nlq_val.json"

export BATCH_SIZE=32
export DIM=128
export NUM_EPOCH=10
export MAX_POS_LEN=128
export INIT_LR=0.0025

export TB_LOG_NAME="${NAME}_bs${BATCH_SIZE}_dim${DIM}_epoch${NUM_EPOCH}_ilr${INIT_LR}"

python main.py \
    --task $TASK_NAME \
    --predictor bert \
    --dim $DIM \
    --mode train \
    --video_feature_dim 256 \
    --max_pos_len $MAX_POS_LEN \
    --init_lr $INIT_LR \
    --epochs $NUM_EPOCH \
    --batch_size $BATCH_SIZE \
    --fv official \
    --num_workers $NUM_WORKERS \
    --data_loader_workers $DATALOADER_WORKERS \
    --model_dir $MODEL_BASE_DIR/$NAME \
    --eval_gt_json $VAL_JSON_PATH \
    --log_to_tensorboard $TB_LOG_NAME \
    --tb_log_freq 5 \
    --remove_empty_queries_from train
%tensorboard --logdir /content/episodic-memory/NLQ/VSLNet/runs/
change_layers(prec_class_content)
with open("vars.sh", "w") as out_f:
  out_f.write("""
export NAME=2D-TAN_omnivore_video_fp16
export TASK_NAME=nlq_official_v1_$NAME
export BASE_DIR=data/dataset/nlq_official_v1_$NAME
export FEATURE_BASE_DIR=data/features/nlq_official_v1_$NAME/
export FEATURE_DIR=$FEATURE_BASE_DIR/video_features
export MODEL_BASE_DIR=/content/nlq_official_v1/checkpoints/

cd /content/episodic-memory/NLQ/2D-TAN
"""
  )
%%bash

source vars.sh

echo $FEATURE_BASE_DIR
mkdir -p $FEATURE_BASE_DIR
ln -s /content/ego4d_data/v1/omnivore_video_swinl_fp16 $FEATURE_DIR
%%bash
%%capture

source vars.sh
pip install nltk submitit torch torchaudio torchvision tqdm transformers tensorboard Pillow terminaltables
%%bash

source vars.sh

python utils/prepare_ego4d_dataset.py \
    --input_train_split /content/ego4d_data/v1/annotations/nlq_train.json \
    --input_val_split /content/ego4d_data/v1/annotations/nlq_val.json \
    --input_test_split /content/ego4d_data/v1/annotations/nlq_test_unannotated.json \
    --video_feature_read_path $FEATURE_DIR \
    --clip_feature_save_path $FEATURE_BASE_DIR/official \
    --output_save_path $BASE_DIR
%%bash
git clone https://github.com/EGO4D/episodic-memory
cd episodic-memory
git pull
git checkout nlq_fixes_and_fp16_support
%%bash

source vars.sh

export workers = "${config["WORKERS"]}"
export dataDir = "${config["DATA_DIR"]}"
export modelDir = "${config["MODEL_DIR"]}"
export logDir = "${config["LOG_DIR"]}"
export verbose = "${True}"
export tag = "${"shown"}"



source vars.sh

export DATALOADER_WORKERS=1
export NUM_WORKERS=2
export VAL_JSON_PATH="/content/ego4d_data/v1/annotations/nlq_val.json"

export BATCH_SIZE=32
export DIM=128
export NUM_EPOCH=10
export MAX_POS_LEN=128
!pip install terminaltables
%%bash

source vars.sh

python moment_localization/train.py --cfg  experiments/ego4d/2D-TAN-40x40-K9L4-pool-window-std-sf.yaml --verbose
%%bash

source vars.sh

export WORKERS=0
export DATADIR="./data/Ego4D_clip"
export MODELDIR="./checkpoints"
export LOGDIR="./log"
export DEBUG="mode"
export TAG="shown"



python train.py \
    --task $TASK_NAME \
    --workers $WORKERS \
    --dataDir $DATADIR \
    --modelDir $MODELDIR \
    --logDir $LOGDIR \

    --tag $TAG \
    --debug $DEBUG \
