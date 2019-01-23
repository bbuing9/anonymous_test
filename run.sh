# nia 6

#CUDA_VISIBLE_DEVICES=0 python3 train_boundary.py --model ResNet18_100 --lr 0.1 --name 'icml_submission_test_no' --num_search 5 --beta 0.1 --num_min 25 --gamma 0.5 --warm 180 --epoch 200 --decay 1e-4 --alpha 2.0 --num_imb 24
CUDA_VISIBLE_DEVICES=1 python3 train_boundary.py --model ResNet18_100 --lr 0.1 --name 'icml_submission_test_over' -o --num_search 5 --beta 0.1 --num_min 25 --gamma 0.5 --warm 180 --epoch 200 --decay 1e-4 --alpha 2.0 --num_imb 24

