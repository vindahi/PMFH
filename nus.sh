set -e
bits=(128)

for i in ${bits[*]}; do
  CUDA_VISIBLE_DEVICES=1 python main.py --bit $i \
                                                     --dataset nuswide \
                                                     --rounds 15 \
                                                     --batch_size 64 \
                                                     --classes 21 \

done
