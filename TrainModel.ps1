$env:PYTHONPATH = "."

python sensorllm/train/train.py `
     --model_name_or_path "Qwen/Qwen2.5-3B" `
     --pt_encoder_backbone_ckpt "amazon/chronos-t5-small" `
     --qa_path "./data/mhealth_stage1_qa.json" `
     --data_path "./whole_data/train/" `
     --dataset "mhealth" `
     --output_dir "./output/sensorllm_stage1" `
     --tokenize_method "StanNormalizeUniformBins"