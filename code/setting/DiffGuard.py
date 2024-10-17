params = dict()    
params['params'] = 'task model method_name modality condition in_channel out_channel noise_channel'
# contrast-enhanced CT, five mediastinal neoplasm subtypes
params['values'] = ('DiffGuard', 'diffusion', 'DiffGuard', 'CE', 'saliency_tumor', 2, 2, 2)
# contrast-enhanced CT, normal control
params['values'] = ('DiffGuard', 'diffusion', 'DiffGuard', 'CE', 'none', 1, 1, 1)
# plain CT, five mediastinal neoplasm subtypes
params['values'] = ('DiffGuard', 'diffusion', 'DiffGuard', 'PL', 'saliency_tumor', 2, 2, 2)
# plain CT, normal control
params['values'] = ('DiffGuard', 'diffusion', 'DiffGuard', 'PL', 'none', 1, 1, 1)

