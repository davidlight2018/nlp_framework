TASK_NAME="daily2014"
CURRENT_DIR=$(pwd)
export BERT_BASE_DIR=$CURRENT_DIR/models/pretrained_bert/bert-base-chinese
export DATASET_DIR=$CURRENT_DIR/datasets/$TASK_NAME
export OUTPUT_DIR=$CURRENT_DIR/outputs
export EXEC_FILE=$CURRENT_DIR/main/main.py

python "$EXEC_FILE" \
  --model_type=bert \
  --model_name_or_path="$BERT_BASE_DIR" \
  --task_name=$TASK_NAME \
  --device=cuda \
  --do_train \
  --do_eval \
  --do_lower_case \
  --markup="bio" \
  --data_dir="$DATASET_DIR" \
  --train_max_seq_length=256 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=16 \
  --per_gpu_eval_batch_size=16 \
  --learning_rate=3e-5 \
  --num_train_epochs=5.0 \
  --output_dir="$OUTPUT_DIR"/$TASK_NAME/ \
  --overwrite_output_dir \
  --seed=42
