params = dict()
params['params'] = 'task backbone arch modality data_norm_type in_channel real_neg_num real_num_by_class gen_neg_num gen_num_by_class'
# contrast-enhanced CT
params['values'] = ('seg', 'TransUNet', 'R50-ViT-B_16', 'CE', 'imagenet', 3, 99999, 99999, 0, 0)
params['values'] = ('seg', 'TransUNet', 'R50-ViT-B_16', 'CE', 'imagenet', 3, 0, 0, 10000, 10000)
# params['values'] = ('seg', 'unet', '', 'CE', 'gray', 1, 99999, 99999, 0, 0)
# params['values'] = ('seg', 'unet', '', 'CE', 'gray', 1, 0, 0, 10000, 10000)
# # plain CT
# params['values'] = ('seg', 'TransUNet', 'R50-ViT-B_16','PL', 'imagenet', 3, 99999, 99999, 0, 0)
# params['values'] = ('seg', 'TransUNet', 'R50-ViT-B_16','PL', 'imagenet', 3, 0, 0, 10000, 10000)
# params['values'] = ('seg', 'unet', '', 'PL', 'gray', 1, 99999, 99999, 0, 0)
# params['values'] = ('seg', 'unet', '', 'PL', 'gray', 1, 0, 0, 10000, 10000)
