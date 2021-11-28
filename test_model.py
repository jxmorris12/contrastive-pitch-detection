import torch


# from models import Bytedance_Regress_pedal_Notes
# m = Bytedance_Regress_pedal_Notes(88, 'softmax', tiny=True).cuda()

# from models import CREPE
# m = CREPE(model='tiny', num_output_nodes=88, out_activation='sigmoid', load_pretrained=True).cuda()

from models import S4Model
m = S4Model(d_output=88, out_activation='sigmoid').cuda()

x = torch.rand((16,16000))
output = m(x.cuda())
breakpoint()

from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters", "%"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        total_params += parameter.numel()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param_num = parameter.numel()
        table.add_row([name, param_num, f'{param_num/total_params*100}'])
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
count_parameters(m)
