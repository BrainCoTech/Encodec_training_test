import os
import pandas as pd
import torch
import torchaudio
import random


class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, csv_input_file, csv_output_file, audio_dir, transform=None, tensor_cut=0, fixed_length=None):
        self.audio_input_labels = pd.read_csv(csv_input_file)
        self.audio_output_labels = pd.read_csv(csv_output_file)
        self.audio_dir = audio_dir
        self.transform = transform
        self.fixed_length = fixed_length
        self.tensor_cut = tensor_cut

    def __len__(self):
        if self.fixed_length:
            return self.fixed_length
        return len(self.audio_input_labels)

    def __getitem__(self, idx):
        input_audio_path = os.path.join(self.audio_dir, self.audio_input_labels.iloc[idx, 10])
        output_audio_path = os.path.join(self.audio_dir, self.audio_output_labels.iloc[idx, 10])
        input_waveform, input_sample_rate = torchaudio.load(input_audio_path)
        output_waveform, output_sample_rate = torchaudio.load(output_audio_path)
        if self.transform:
            input_waveform = self.transform(input_waveform)
            output_waveform = self.transform(output_waveform)

        if self.tensor_cut > 0:
            if input_waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, input_waveform.size()[1]-self.tensor_cut-1)
                input_waveform = input_waveform[:, start:start+self.tensor_cut]
            if output_waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, output_waveform.size()[1]-self.tensor_cut-1)
                output_waveform = output_waveform[:, start:start+self.tensor_cut]
        #return input_waveform, input_sample_rate
        wav_data = {
            "input": input_waveform,
            "input_sr": input_sample_rate,
            "output": output_waveform,
            "output_sr": output_sample_rate
        }
        # print(wav_data.shape)
        return wav_data
        #return {
        #    "input": input_waveform,
        #    "input_sr": input_sample_rate,
        #    "output": output_waveform,
        #    "output_sr": output_sample_rate
        #}

