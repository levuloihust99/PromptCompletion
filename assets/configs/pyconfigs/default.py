model_size = "base"

spiece_model = "assets/tokenizer/spiece/vien.model"
train_data_path = "data/prompts/split/train/bytedataset"
valid_data_path = "data/prompts/split/valid/bytedataset"

# training config
output_dir = "assets/outputs"
do_train = True
do_eval = True
learning_rate = 4e-3
num_train_epochs = 10
warmup_ratio = 0.0
warmup_steps = 0
weight_decay = 0
per_device_train_batch_size = 2
per_device_eval_batch_size = 2
logging_dir = "assets/logs"
group_by_length = False
save_strategy = "steps"
save_steps = 100
evaluation_strategy = "steps"
eval_steps = 100
save_total_limit = 2
fp16 = False
gradient_accumulation_steps = 1
log_level = "info"
logging_steps = 50
logging_first_step = True
max_grad_norm = 2.0
label_smoothing_factor = 0.1
report_to = ["tensorboard"]
load_best_model_at_end = True
metric_for_best_model = "acc"
greater_is_better = True
predict_with_generate = False
resume_from_checkpoint = None
remove_unused_columns = False
generation_max_length = 100
generation_num_beams = 1
input_name = "prompt"
output_name = "completion"
data_seed = None
max_input_len = None
max_output_len = None

# model config
d_model = 384
d_kv = 48
d_ff = 1536
num_layers = 4
num_decoder_layers = 4
num_heads = 8

# data config
input_transform = None # [json_sequentialize]
output_transform = None # [json_sequentialize]
