set -e
bits=(16 32 64 128)
for i in ${bits[*]}; do
 CUDA_VISIBLE_DEVICES=0 python main.py --bit $i \
                                                    --dataset flickr \
                                                    --rounds 15 \
                                                    --batch_size 64\
                                                    --classes 24 \

done

