from dataloader import dataset_load_funcs

f = dataset_load_funcs['nsynth_keyboard_train']
l = f()
print(len(l))
