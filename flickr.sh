set -e
bits=(16 32 64 128)
for i in ${bits[*]}; do
 CUDA_VISIBLE_DEVICES=0 python main.py --bit $i \
                                                    --dataset flickr \
                                                    --rounds 20 \
                                                    --batch_size 64\
                                                    --classes 24 \
                                                    --mlpdrop 0.1 \

done




# set -e
# learning_rate=(0.0000001 0.000001 0.00001)
# for i in ${learning_rate[*]}; do
#   CUDA_VISIBLE_DEVICES=1 python main.py --bit 32 \
#                                                      --dataset flickr \
#                                                      --rounds 13 \
#                                                      --batch_size 64\
#                                                      --classes 24 \
#                                                      --learning_rate $i \

# done





# set -e
# bits=(16 32 64 128)
# drops=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# for i in ${bits[*]}; do
# for j in ${drops[*]}; do
#   CUDA_VISIBLE_DEVICES=0 python main.py --bit $i \
#                                                      --dataset flickr \
#                                                      --rounds 25 \
#                                                      --batch_size 1024\
#                                                      --classes 24 \
#                                                      --dropout $j \

# done
# done



# set -e
# alpha=(0.000001 0.0001 0.01 1 100)
# for i in ${alpha[*]}; do
#   CUDA_VISIBLE_DEVICES=1 python main.py --bit 64 \
#                                                      --dataset flickr \
#                                                      --rounds 15 \
#                                                      --batch_size 64\
#                                                      --classes 24 \
#                                                      --ff $i \

# done


