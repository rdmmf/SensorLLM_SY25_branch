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