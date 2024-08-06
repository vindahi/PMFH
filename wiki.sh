set -e
bits=(16 32 64 128)
bits=(32 64)

for i in ${bits[*]}; do
  CUDA_VISIBLE_DEVICES=0 python main.py --bit $i \
                                                     --dataset wiki \
                                                     --rounds 20 \
                                                     --batch_size 64 \
                                                     --classes 10 \
                                                     --image_dim 128 \
                                                     --text_dim 10 \

done



