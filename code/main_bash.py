import datetime
import setting
from options.base_options import TrainOptions
from train import train
from util.basic import *

def getTime():
    timeNow = datetime.datetime.now().strftime('%b%d_%H-%M')
    return timeNow

def genetate_code(tmp_code, tmp_text, code_list, text_list, params_list, value_matrix):
    tmp_params = params_list.split()
    p_code = tmp_code
    p_text = tmp_text
    for i in range(len(tmp_params)):
        p_code = p_code + ' --' + tmp_params[i] + ' ' + str(value_matrix[i])
        p_text = p_text + ',' + tmp_params[i] + '=' + str(value_matrix[i])
    code_list.append(p_code)
    text_list.append(getTime() + '_' + p_text)

if __name__ == "__main__":
    # Choose the setting
    setting_name = 'DiffGuard'
    setting_name = 'segmentation'
    setting_name = 'reconstruction'
    setting_name = 'membership_inference_attack'
    param_setting = setting.get_param_setting(setting_name)

    # Set available GPU IDs
    gpu_list = '0'
    gpu_num = len(gpu_list.split(','))

    add_info = setting_name
    code_list = []
    text_list = []

    params_list = param_setting.params['params']
    value_matrix = param_setting.params['values']
    tmp_params = params_list.split()
    kwargs = {tmp_params[i]: value_matrix[i] for i in range(len(tmp_params))}
    option = TrainOptions(**kwargs)

    start_code = ''
    genetate_code(start_code,add_info,code_list,text_list,params_list,value_matrix)
    option.parser.set_defaults(name=text_list[0])
    tmp_gpus = ''
    for j in range(gpu_num):
        tmp_gpus += str(j) + ','
    tmp_gpus = tmp_gpus[:-1]
    option.parser.set_defaults(gpu_ids=tmp_gpus)
    opt = option.parse()
    train(opt)
