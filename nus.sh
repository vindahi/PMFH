set -e
bits=(128)

for i in ${bits[*]}; do
  CUDA_VISIBLE_DEVICES=1 python main.py --bit $i \
                                                     --dataset nuswide \
                                                     --rounds 15 \
                                                     --batch_size 64 \
                                                     --classes 21 \
                                                     --text_dim 1000 \
                                                     --mlpdrop 0.7 \

done


# set -e
# bits=(16 32 64 128)

# for i in ${bits[*]}; do
#   CUDA_VISIBLE_DEVICES=1 python main.py --bit $i \
#                                                      --dataset nuswide \
#                                                      --rounds 55 \
#                                                      --batch_size 1024 \
#                                                      --classes 10 \

# done


# set -e
# alpha=(0.000001 0.0001 0.01 1 100)
# for i in ${alpha[*]}; do
#   CUDA_VISIBLE_DEVICES=1 python main.py --bit 64 \
#                                                      --dataset nuswide \
#                                                      --rounds 15 \
#                                                      --batch_size 64\
#                                                      --classes 21 \
#                                                      --text_dim 1000 \
#                                                      --bb $i \

# done