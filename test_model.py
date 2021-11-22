import torch
from models import CREPE, Bytedance_Regress_pedal_Notes

m = Bytedance_Regress_pedal_Notes(88, 'softmax', tiny=False).cuda()
# m = CREPE(model='tiny', num_output_nodes=88, out_activation='sigmoid', load_pretrained=True).cuda()
x = torch.rand((16,16000))
print('input.shape:', x.shape)
output = m(x.cuda())
print('output.shape:', output.shape)
# breakpoint()

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
