params = dict()
params['params'] = 'task modality shadow_model'
# contrast-enhanced CT
params['values'] = ('MIA', 'CE', 'nnUNet')
params['values'] = ('MIA', 'CE', 'TransUNet')
params['values'] = ('MIA', 'CE', 'unet')
# plain CT
params['values'] = ('MIA', 'PL', 'nnUNet')
params['values'] = ('MIA', 'PL', 'TransUNet')
params['values'] = ('MIA', 'PL', 'unet')
