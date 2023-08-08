DATASET="dblp"
N_WAY=(5 10)
K_SHOT=(3 5)
AUGMENTATION_METHOD=("subgraph" "drop_feature")
AUGMENTATION_PARAMETER=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for WAY in ${N_WAY[*]}
do
  for SHOT in ${K_SHOT[*]}
  do
    for METHOD in ${AUGMENTATION_METHOD[*]}
    do
      for PARAMETER in ${AUGMENTATION_PARAMETER[*]}
      do
        LOG_FILE="./${DATASET}-${WAY}way${SHOT}shot['${METHOD}'][${PARAMETER}].log"
        if [ -f "$LOG_FILE" ]; then
          echo "${LOG_FILE} exist";
          continue
        fi
        a=CUDA_VISIBLE_DEVICES=1 python3 main_augmentation.py --dataset $DATASET --way $WAY --shot $SHOT --augmentation_method $METHOD --augmentation_parameter $PARAMETER
      done
    done
  done
done