from dataloader import dataset_load_funcs
from generator import AudioDataGenerator

# for split in {'train', 'test', 'valid'}:
#    f = dataset_load_funcs[f'nsynth_keyboard_{split}']
#    l = f()
#    print(len(l))

frame_length = 1024
batch_size = 64
max_polyphony = 6
gen = AudioDataGenerator(
    [], frame_length, max_polyphony,
    randomize_train_frame_offsets=False,
    batch_size=batch_size,
    augmenter=None,
    sample_rate=16000,
    num_fake_nsynth_chords=10000,
)

x,y = gen[0]
breakpoint()