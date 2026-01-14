# Assurez-vous d'être dans C:\Users\thoma\Documents\GitHub\SensorLLM
$env:PYTHONPATH = "."

python sensorllm/train/train.py `
    --model_name_or_path "Qwen/Qwen2.5-3B" `
    --pt_encoder_backbone_ckpt "amazon/chronos-t5-small" `
    --data_path "data/train/mhealth_train_data_stage1.pkl" `
    --qa_path "data/train/mhealth_train_qa_stage1.json" `
    --dataset "mhealth" `
    --output_dir "./output/sensorllm_stage1" `
    --tokenize_method "StanNormalizeUniformBins" `
    --max_steps 500 `
    --per_device_train_batch_size 2


$env:PYTHONPATH = "."

python sensorllm/train/train.py `
    --model_name_or_path "Qwen/Qwen2.5-3B" `
    --pt_encoder_backbone_ckpt "amazon/chronos-t5-small" `
    --data_path "data/train/mhealth_train_data_stage1.pkl" `
    --qa_path "data/train/mhealth_train_qa_stage1.json" `
    --dataset "mhealth" `
    --output_dir "./output/sensorllm_stage1" `
    --tokenize_method "StanNormalizeUniformBins" `
    --max_steps 500 `
    --per_device_train_batch_size 2 `
    --learning_rate 2e-5 `
    --do_train True `
    --do_eval True



# Configuration de l'environnement
$env:PYTHONPATH = "."
$env:HF_TOKEN = "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"

torchrun --nproc_per_node=1 sensorllm/train/train_mem.py `
    --model_name_or_path "Qwen/Qwen2.5-3B" `
    --pt_encoder_backbone_ckpt "amazon/chronos-t5-small" `
    --tokenize_method "StanNormalizeUniformBins" `
    --dataset "mhealth" `
    --data_path "data/train/mhealth_train_data_stage1.pkl" `
    --eval_data_path "data/train/mhealth_train_data_stage1.pkl" `
    --qa_path "data/train/mhealth_train_qa_stage1.json" `
    --eval_qa_path "data/train/mhealth_train_qa_stage1.json" `
    --output_dir "./output/sensorllm_stage1" `
    --model_max_length 2048 `
    --num_train_epochs 3 `
    --per_device_train_batch_size 2 `
    --per_device_eval_batch_size 2 `
    --evaluation_strategy "steps" `
    --save_strategy "steps" `
    --save_steps 100 `
    --eval_steps 100 `
    --learning_rate 2e-3 `
    --weight_decay 0.0 `
    --warmup_ratio 0.03 `
    --lr_scheduler_type "cosine" `
    --logging_steps 1 `
    --gradient_checkpointing True `
    --save_total_limit 1 `
    --bf16 True `
    --fix_llm True `
    --fix_ts_encoder True `
    --model_type "CasualLM" `
    --load_best_model_at_end True



$env:PYTHONPATH = "."
# Votre token est déjà configuré dans votre session, pas besoin de le remettre

python sensorllm/train/train_mem.py `
    --model_name_or_path "Qwen/Qwen2.5-3B" `
    --pt_encoder_backbone_ckpt "amazon/chronos-t5-large" `
    --tokenize_method "StanNormalizeUniformBins" `
    --dataset "mhealth" `
    --data_path "data/train/mhealth_train_data_stage1.pkl" `
    --eval_data_path "data/train/mhealth_train_data_stage1.pkl" `
    --qa_path "data/train/mhealth_train_qa_stage1.json" `
    --eval_qa_path "data/train/mhealth_train_qa_stage1.json" `
    --output_dir "./output/sensorllm_stage1" `
    --model_max_length 2048 `
    --num_train_epochs 3 `
    --per_device_train_batch_size 1 `
    --per_device_eval_batch_size 1 `
    --evaluation_strategy "steps" `
    --save_strategy "steps" `
    --save_steps 100 `
    --eval_steps 100 `
    --learning_rate 2e-3 `
    --weight_decay 0.0 `
    --warmup_ratio 0.03 `
    --lr_scheduler_type "cosine" `
    --logging_steps 1 `
    --gradient_checkpointing True `
    --save_total_limit 1 `
    --bf16 False `
    --fix_llm True `
    --fix_ts_encoder True `
    --model_type "CasualLM" `
    --load_best_model_at_end True



python sensorllm/train/train_mem.py `
    --model_name_or_path "Qwen/Qwen2.5-3B" `
    --pt_encoder_backbone_ckpt "amazon/chronos-t5-large" `
    --tokenize_method "StanNormalizeUniformBins" `
    --dataset "mhealth" `
    --data_path "data/train/mhealth_subset_data.pkl" `
    --eval_data_path "data/train/mhealth_subset_data.pkl" `
    --qa_path "data/train/mhealth_subset_qa.json" `
    --eval_qa_path "data/train/mhealth_subset_qa.json" `
    --output_dir "./output/sensorllm_stage1" `
    --model_max_length 2048 `
    --num_train_epochs 3 `
    --per_device_train_batch_size 1 `
    --per_device_eval_batch_size 1 `
    --evaluation_strategy "steps" `
    --save_strategy "steps" `
    --save_steps 100 `
    --eval_steps 100 `
    --learning_rate 2e-3 `
    --weight_decay 0.0 `
    --warmup_ratio 0.03 `
    --lr_scheduler_type "cosine" `
    --logging_steps 1 `
    --gradient_checkpointing True `
    --save_total_limit 1 `
    --bf16 False `
    --fix_llm True `
    --fix_ts_encoder True `
    --model_type "CasualLM" `
    --load_best_model_at_end True


python sensorllm/train/train.py `
    --model_name_or_path "Qwen/Qwen2.5-3B" `
    --pt_encoder_backbone_ckpt "amazon/chronos-t5-large" `
    --data_path "data/train/mhealth_subset_data.pkl" `
    --qa_path "data/train/mhealth_subset_qa.json" `
    --dataset "mhealth" `
    --output_dir "./output/sensorllm_test" `
    --num_train_epochs 1 `
    --max_steps 50 `
    --per_device_train_batch_size 1 `
    --fix_llm True `
    --fix_ts_encoder True `
    --bf16 False