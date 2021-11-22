from dataloader import dataset_load_funcs
from generator import AudioDataGenerator

# for split in {'train', 'test', 'valid'}:
#    f = dataset_load_funcs[f'nsynth_keyboard_{split}']
#    l = f()
#    print(len(l))

gen = AudioDataGenerator(
    [], 1024, 6,
    randomize_train_frame_offsets=False,
    batch_size=4,
    augmenter=None,
    sample_rate=16000,
    num_fake_nsynth_chords=1000,
)

x,y = gen[0]
breakpoint()