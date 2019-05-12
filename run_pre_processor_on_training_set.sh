sudo docker run -v $PWD:/medium-facenet-tutorial \
-e PYTHONPATH=$PYTHONPATH:/medium-facenet-tutorial \
-it colemurray/medium-facenet-tutorial python3 /medium-facenet-tutorial/pre_process.py \
--input-dir /medium-facenet-tutorial/raw_training_set \
--output-dir /medium-facenet-tutorial/pre_processed_training_set \
--crop-dim 180
