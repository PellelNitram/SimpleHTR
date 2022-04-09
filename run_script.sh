cd src

OUTDIR=/home/martin/data/xournalpp_ml/simplehtr/1_outdir
rm -r ${OUTDIR}

cp -r /home/martin/data/xournalpp_ml/worddetectornn/2_output_data/new/image0 ${OUTDIR}

for p in $(ls ${OUTDIR}); do
  IMG_FILE=${OUTDIR}/${p}/pic.jpg
  OUTFILE=${OUTDIR}/${p}/prediction.json

  python infer_martin.py \
      --img_file ${IMG_FILE} \
      --model-dir /home/martin/data/xournalpp_ml/simplehtr/0_trained_model/word_model \
      --outfile ${OUTFILE}

done
