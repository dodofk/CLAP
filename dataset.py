import numpy as np
import pandas as pd
import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import rnn
from sklearn import preprocessing
import hydra
from omegaconf import DictConfig
from transformers import BertTokenizer
from typing import Tuple, Dict, List


class FluentSpeechDATASET(Dataset):
    def __init__(
            self,
            cfg: DictConfig,
            split: str = "train",
    ):
        assert split in ['train', 'test', 'valid'], 'Invalid Split'
        self.cfg = cfg
        self.data_root = cfg.data_folder
        self.df = pd.read_csv(os.path.join(hydra.utils.get_original_cwd(), self.data_root, f'data/{split}_data.csv'))
        self.df['intent'] = self.df[['action', 'object', 'location']].apply('-'.join, axis=1)
        self.intent_encoder = preprocessing.LabelEncoder()
        self.intent_encoder.fit(self.df['intent'])
        self.df['intent_label'] = self.intent_encoder.transform(self.df['intent'])
        self.labels_set = set(self.df['intent_label'])
        self.labels2index = dict()

        for label in self.labels_set:
            idx = np.where(self.df['intent_label'] == label)[0][0]
            self.labels2index[label] = idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = dict()
        waveform, mfcc, intent, transcription = self.load_audio(index)
        item['waveform'] = waveform
        item['mfcc'] = mfcc
        item['intent'] = intent
        item['transcription'] = transcription
        return item

    def load_audio(self, idx):
        df_row = self.df.iloc[idx]
        filename = os.path.join(hydra.utils.get_original_cwd(), self.data_root, df_row['path'])
        waveform, _ = torchaudio.load(filename, channels_first=True)
        params = {
            "num_mel_bins": 40,
            "num_ceps": 13,
            "high_freq": -400,
        }
        mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **params)
        intent = df_row['intent_label']
        transcription = df_row['transcription']
        return waveform.squeeze(), mfcc, intent, transcription

    def labels_list(self):
        return self.intent_encoder.classes_


class CustomLibriSpeech(torchaudio.datasets.LIBRISPEECH):
    """
    Create A Custom Dataset Modify from torchaudio.datasets.LIBRISpeech
    need to get the mfcc of audio, and use tokenizer on text
    """
    def __init__(
            self,
            cfg: DictConfig,
            split: str = "train-clean-100",
    ):
        super().__init__(root=hydra.utils.get_original_cwd()+"/"+cfg.data_folder, url=split, download=cfg.download)
        self.cfg = cfg
        self.tokenizer = BertTokenizer.from_pretrained(cfg.tokenizer)

    def load_librispeech_item(
            self,
            fileid: str,
            path: str,
            ext_audio: str,
            ext_txt: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        speaker_id, chapter_id, utterance_id = fileid.split("-")

        file_text = speaker_id + "-" + chapter_id + ext_txt
        file_text = os.path.join(path, speaker_id, chapter_id, file_text)

        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + ext_audio
        file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

        # Load audio and get mfcc
        waveform, sample_rate = torchaudio.load(file_audio)
        params = {
            "num_mel_bins": self.cfg.num_mel_bins,
            "num_ceps": self.cfg.num_ceps,
            "high_freq": self.cfg.high_freq,
        }
        mfcc = torchaudio.compliance.kaldi.mfcc(waveform, **params)

        # Load text
        with open(file_text) as ft:
            for line in ft:
                fileid_text, transcript = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                raise FileNotFoundError("Translation not found for " + fileid_audio)
        encode_transcript = self.tokenizer.encode_plus(
            transcript,
            add_special_tokens=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        return (
            waveform,
            mfcc,
            encode_transcript['input_ids'].squeeze(),
            transcript,
        )

    def __getitem__(self, n: int) -> Dict:
        fileid = self._walker[n]
        item = dict()
        waveform, mfcc, encoded_text, _ = self.load_librispeech_item(fileid, self._path, self._ext_audio, self._ext_txt)
        item['mfcc'] = mfcc
        item['text'] = encoded_text
        item['waveform'] = waveform
        return item


def default_fsc_collate(inputs: List) -> Dict:
    waveforms = [data['waveform'] for data in inputs]
    intents = [data['intent'] for data in inputs]
    transcriptions = [data['transcription'] for data in inputs]
    mfccs = [data['mfcc'] for data in inputs]
    padded_mfccs = rnn.pad_sequence(mfccs, batch_first=True)
    padded_waveforms = rnn.pad_sequence(waveforms, batch_first=True)

    return {
        'waveform': padded_waveforms,
        'mfcc': padded_mfccs,
        'intent': torch.tensor(intents),
        'transcription': transcriptions,
    }


def default_librispeech_collate(inputs: List) -> Dict:
    padded_mfcc = rnn.pad_sequence([data['mfcc'] for data in inputs], batch_first=True)
    padded_text = rnn.pad_sequence([data['text'] for data in inputs])
    padded_waveform = rnn.pad_sequence([data['waveform'].T for data in inputs])

    return{
        "mfcc": padded_mfcc,
        "text": padded_text.T,
        "waveform": padded_waveform.squeeze().T,
    }


def build_loaders(
        cfg: DictConfig,
        split: str = None,
):
    dataset = cfg.dataset
    assert dataset in ['fsc', 'librispeech'], 'Available Dataset: fsc, librispeech'
    if dataset == 'fsc':
        assert split in ['train', 'test', 'valid'], 'Invalid Split'
        dataset = FluentSpeechDATASET(
            cfg=cfg,
            split=split,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=default_fsc_collate,
            shuffle=True if split == "train" else False,
        )
        return dataloader
    elif dataset == 'librispeech':
        assert split in ['train-clean-100', 'dev-clean', 'test-clean'], 'Invalid Split'
        dataset = CustomLibriSpeech(
            cfg,
            split=split,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            collate_fn=default_librispeech_collate,
            shuffle=True if "train" in split else False,
            drop_last=True,
        )
        return dataloader





