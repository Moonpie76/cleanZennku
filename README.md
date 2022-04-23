# cleanZennku
nohup python create_pre_train_data.py --train_corpus ../datasets/chr1.txt --output_dir ../datasets/Cor1 --bert_model bert-base-cased --max_ngram_in_sequence 800 --max_seq_len 403&

CUDA_VISIBLE_DEVICES=0 nohup python run_pre_train.py --pregenerated_data ../datasets/Cor1/ --output_dir ../models/chr1_lr3e-6 --scratch --bert_model bert-base-cased --train_batch_size 8 --epochs 1 --learning_rate 3e-6&

有validate：
CUDA_VISIBLE_DEVICES=0 nohup python run_sequence_level_classification_doval.py --data_dir test_validate/ --bert_model ../models/chr1_lr3e-6/zen0311192942_epoch_0 --task_name thucnews --do_train --max_seq_length 103 --output_dir test_val_model/ --train_batch_size 1 --num_train_epochs 10 --early_stop_steps 150 --save_steps 50 --learning_rate 5e-6 > testval.out
其中early_stop_steps是超过目前最好表现多少步后停止训练

无validate：
CUDA_VISIBLE_DEVICES=0 nohup python run_sequence_level_classification.py --data_dir dnabert_data/ --bert_model ../models/chr1_lr3e-6_range_0.01/zen0317072716_epoch_0/ --task_name thucnews --do_train --max_seq_length 202 --output_dir ../models/fintune/chr1_lr3e-6_range_0.01/lr5e-6/ --train_batch_size 8 --num_train_epochs 15 --save_steps 100 --learning_rate 5e-6&
