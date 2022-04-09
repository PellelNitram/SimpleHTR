cd src

# python main.py --img_file ../data/word.png # Example 1
# python main.py --img_file ../data/line.png # Example 2

python infer_martin.py \
    --img_file /home/martin/data/xournalpp_ml/worddetectornn/2_output_data/old/10.jpg \
    --model-dir /home/martin/data/xournalpp_ml/simplehtr/0_trained_model/word_model \
    --outdir /home/martin/data/xournalpp_ml/simplehtr/1_outdir

#     --img_file /home/martin/Development/WordDetectorNN/data/test/outdir/10.jpg
# E.g. 'retro' is slightly miss classified; is that maybe because of a too small aabb box?



# Accept model dir:
    #model_dir = '../model/' # TODO: Tweak this here!
    #model_dir = '/home/martin/data/xournalpp_ml/simplehtr/0_trained_model/word_model/'
    #model_dir = '/home/martin/data/xournalpp_ml/simplehtr/0_trained_model/line_model/'
# Save result of inference somewhere
