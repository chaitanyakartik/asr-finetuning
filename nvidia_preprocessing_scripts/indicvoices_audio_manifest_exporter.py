from datasets import Audio, load_dataset
import soundfile as sf
import os,json

LANG="malayalam"
DATASET_SPLIT="train"
manifest_filepath=f"/datasets/indicvoices-nemo/indicvoices_wav_files_{LANG}_{DATASET_SPLIT}.json"
# Load the dataset
dataset = load_dataset(path=f"/datasets/indicvoices-malayalam/IndicVoices/{LANG}",split=DATASET_SPLIT,num_proc=36)
# dataset = load_dataset(path=f"/datasets/indicvoices-malayalam/IndicVoices",name=f"{LANG}",split=DATASET_SPLIT)
dataset = dataset.cast_column("audio_filepath",Audio(16000, mono=True,decode=False))
# print(dataset)

# num_threads = num_threads = min(32, (os.cpu_count() or 1) + 4)
# dataset = dataset.decode(num_threads=num_threads)
# Create a directory to save WAV files
OUTPUT_DIR=f"/datasets/indicvoices-nemo/indicvoices_wav_files_{LANG}_{DATASET_SPLIT}"
os.makedirs(OUTPUT_DIR, exist_ok=True)
import librosa

# Function to save audio files
# def save_audio(example):
    
#         
#     return example

# Map the function with batching and num_workers

# dataset = dataset.map(
# save_audio,
# batched=False,  # Set to True if you want to process in batches
# num_proc=122  # Adjust the number of workers as needed
# )
# dataset = dataset.decode(False)
# ds_iter = iter(dataset)
import tqdm
with open(manifest_filepath, 'w') as manifest_f:
    for idx, example in enumerate(tqdm.tqdm(dataset, desc=f'Processing')):
        try:
            wav_file_path = f"{OUTPUT_DIR}/{example['audio_filepath']['path']}".split('.flac')[0]+".wav" 
            if os.path.exists(wav_file_path):
                # print('The file exists!')
                _=6 # kept for debugging
            else:
                # _=6
                # print(example)
                print('The file does not exist.')
                audio_array = example['audio_filepath']['array']  # Adjust the key based on your dataset
                # Use a unique identifier
                sf.write(wav_file_path, audio_array, 16000)  # Adjust if necessary
                

            
            example['audio_filepath'] = wav_file_path# f"{OUTPUT_DIR}/{example['audio_filepath']['path']}" 
            # example['duration'] = librosa.get_duration(y=example['audio']['array'], sr=example['audio']['sampling_rate'])
            # example['text'] = example['text']
            # del example['audio_filepath']['array']
            if example['duration']>0:
                manifest_f.write(f"{json.dumps(example, ensure_ascii=True)}\n")
        except Exception as e:
            print(e,f"{example['audio_filepath']['path']}")
            pass
            # exit()


print("Export completed! ",manifest_filepath)