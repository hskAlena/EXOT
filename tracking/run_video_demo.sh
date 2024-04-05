# We only support manually setting the bounding box of first frame and save the results in debug directory.
# We plan to release a colab for running your own video demo in the future.
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir .

#########-------------Annotation--------------------############

#python tracking/video_demo.py exotst_tracker baseline_mix_sim --track_format run_video_robot \
#  --modelname exot_st2 --ckpt_name baseline_mix_sim/EXOTST_ep0060.pth.tar --neg_thres 0.27 --version h\
#  --epsilon 0.01 --avg_thres 10 --save_dir data/robot-data/data_RGB/final_test.txt ##0.4 \test_seq2.txt  \

#python tracking/video_demo.py exotst_tracker baseline_mix_hs --track_format run_video_robot \
#  --modelname exot_st2 --ckpt_name baseline_mix_hs/EXOTST1_ep0060.pth.tar --neg_thres 1.25 --version h\
#  --epsilon 0.005 --avg_thres 10 --save_dir data/robot-data/data_RGB/final_test.txt ##0.4 \test_seq2.txt  \

#python tracking/video_demo.py exotst_tracker baseline_mix_enc --track_format run_video_robot \
#  --modelname exot_st2 --ckpt_name baseline_mix_enc/EXOTST_ep0040.pth.tar --neg_thres 0.45 --version h\
#  --epsilon 0.005 --avg_thres 10 --save_dir data/robot-data/data_RGB/final_test.txt ##0.4 \test_seq2.txt  \
################################################################################################

#python tracking/video_demo.py exotst_tracker baseline_mix_lowdim --track_format run_video_robot \
#  --modelname exot_merge --ckpt_name baseline_mix_lowdim/EXOTST_ep0105.pth.tar --neg_thres 0.05 --version h\
#  --epsilon 0.005 --avg_thres 10 --save_dir data/robot-data/data_RGB/final_test.txt ##0.4 \test_seq2.txt  \

#python tracking/video_demo.py exotst_tracker baseline_mix_enc --track_format run_video_robot \
#  --modelname exot_merge --ckpt_name baseline_mix_enc/EXOTST_ep0040.pth.tar --neg_thres 0.45 --version h\
#  --epsilon 0.005 --avg_thres 10 --save_dir data/robot-data/data_RGB/final_test.txt ##0.4 \test_seq2.txt  \

###########################################################################################
#python tracking/video_demo.py exotst_tracker baseline_mix_lowdim --track_format run_video_robot \
#  --modelname exot_st2 --ckpt_name baseline_mix_lowdim/EXOTST_ep0069.pth.tar --neg_thres 0.1 --version h\
#  --epsilon 0.005 --avg_thres 10 --save_dir data/robot-data/data_RGB/final_test.txt ##0.4 \test_seq2.txt  \

####################################################################################################
# python tracking/video_demo.py stark_st baseline_mix --track_format run_stark_robot \
#   --modelname stark_st2 --ckpt_name STARKST_epoch=49-v1.pth.tar --avg_thres 5

# python tracking/video_demo.py exotst_tracker baseline_mix_lowdim --track_format run_video_annot_odin \
#  --modelname exot_merge --ckpt_name baseline_mix_lowdim/EXOTST_ep0105.pth.tar --neg_thres 0.26 --version h\
#  --epsilon 0.005 --avg_thres 10 --save_dir data/robot-data/data_RGB/final_test.txt ##0.4 \test_seq2.txt  \

#python tracking/video_demo.py exotst_tracker baseline_mix_sim --track_format run_video_annot_odin \
#--modelname exot_st2 --ckpt_name baseline_mix_sim/EXOTST_ep0060.pth.tar --neg_thres 0.4 \
#--epsilon 0.005 --avg_thres 5 --save_dir data/robot-data/data_RGB/final_test.txt #test_seq2.txt  \

# python tracking/video_demo.py stark_st baseline_mix --track_format run_video_annot_stark \
# --modelname stark_st2 --ckpt_name STARKST_epoch=49-v1.pth.tar --neg_thres 0.4 \
# --epsilon 0.005 --avg_thres 20 --save_dir data/robot-data/data_RGB/final_test.txt #test_seq2.txt  \

###################################################################################################
# python tracking/video_demo.py exotst_tracker cos_mix_lowdim --track_format run_video_annot_odin \
#  --modelname exot_merge --ckpt_name cos_mix_lowdim/EXOTST_epoch=280.pth.tar --neg_thres 0.1 --version h\
#  --epsilon 0.4 --avg_thres 10 --save_dir data/TREK-150/test_seq2.txt ##0.4 \test_seq2.txt  \

#  python tracking/video_demo.py exotst_tracker baseline_mix_sim --track_format run_video_save_jpg \
#  --modelname exot_st2 --ckpt_name baseline_mix_sim/EXOTST_ep0060.pth.tar --neg_thres 0.54 --version h\
#  --epsilon 0.01 --avg_thres 20 --save_dir data/robot-data/data_RGB/test_seq2_0501.txt ##0.4 \test_seq2.txt  \

# python tracking/video_demo.py exotst_tracker cos_mix_lowdim --track_format run_video_annot_odin \
#  --modelname exot_st2 --ckpt_name cos_mix_lowdim/EXOTST1_ep0050.pth.tar --neg_thres 0.05 --version h\
#  --epsilon 0.005 --avg_thres 10 --save_dir data/TREK-150/test_seq2.txt #data/TREK-150/test_seq2.txt ##0.4 \test_seq2.txt  \

# python tracking/video_demo.py exotst_tracker cos_mix_lowdim --track_format run_video_annot_odin \
#  --modelname exot_st2 --ckpt_name cos_mix_lowdim/exotst1_250v1/EXOTST_epoch=71.pth.tar --neg_thres 0.05 --version h\
#  --epsilon 0.005 --avg_thres 10 --save_dir data/TREK-150/test_seq2.txt 

python tracking/video_demo.py stark_st baseline_mix --track_format run_video_annot_stark \
--modelname stark_st2 --ckpt_name baseline_mix/STARKST_epoch=49-v1.pth.tar --neg_thres 0.4 \
--epsilon 0.005 --avg_thres 20 --save_dir data/robot-data/data_RGB/test_seq2_0501.txt #test_seq2.txt  \