# self_validata_prostate12_d3_unet
采用prostate12数据验证baseline方法的效果，验证处理过程的有效性；
对数据进行偏移场矫正、resample、归一化之后，划分训练测试数据之后，直接运行train_prostate（切块前划分val）/train_porstate_all（切块后划分val）
