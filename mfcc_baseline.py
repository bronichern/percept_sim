import argparse
import os
from librosa.sequence import dtw as lib_dtw

import librosa
import pandas as pd
import numpy as np

SR = 16000
columns = ["speaker1_word_wid", "speaker2_word_wid", "group", "DTW"]
rows = []

def kr_df(kr_mfa_path):
    '''
    kr_mfa_path: path to the Korean MFA output
    returns: dataframe containing the information about the files
    '''
    files_kr = []

    for spkr in os.listdir(kr_mfa_path):
        if '.' in spkr[0]:
            continue
        if not os.path.exists(os.path.join(kr_mfa_path, spkr, 'word_split')):
            print('path doesnt exist ', os.path.join(
                kr_mfa_path, spkr, 'word_split'), "\t...skipping")
            continue
        files_kr.extend([os.path.join(kr_mfa_path, spkr, 'word_split', f)
                         for f in os.listdir(os.path.join(kr_mfa_path, spkr, 'word_split'))])

    df = pd.DataFrame(files_kr, columns=['file_path'])
    df['l1'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("_")[1][0])
    df['sent_id'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("sent")[1].split("_")[-2])
    df['word_id'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("word")[1].split("_")[0])
    df['word_txt'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("_")[-1].split('.wav')[0])
    df['speaker'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("_")[1])
    return df


def cmn_df(mandarin_data_path, spanish_data_path, task):
    '''
    mandarin_data_path: path to the mandarin data
    spanish_data_path: path to the spanish data
    task: task to run the baseline on (HT1/HT2/DHR/LPP)
    '''
    files = []

    for db in [mandarin_data_path, spanish_data_path]:
        for dir in os.listdir(db):
            # avoid NWS task and use only sentences in english
            if '.' in dir[0] or 'CMN_CMN' in dir or 'SHS_SPA' in dir or 'NWS' in dir \
                    or not os.path.isdir(os.path.join(db, dir)) or task not in dir:
                continue
            if not os.path.exists(os.path.join(db, dir, 'word_split')):
                print('path doesnt exist ', os.path.join(
                    db, dir, 'word_split'), '\t...skipping')
                continue
            files.extend([os.path.join(db, dir, 'word_split', f)
                          for f in os.listdir(os.path.join(db, dir, 'word_split'))])

    assert len(files) > 0, "No files found in the path"

    df = pd.DataFrame(files, columns=['file_path'])
    df['l1'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("_")[3])
    df['sent_lng'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("_")[4])
    df['sent_type'] = df['file_path'].apply(
        lambda r: "_".join(os.path.basename(r).split("_")[3:5]))
    df['sent_id'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("sent")[1].split("_")[0])
    df['word_id'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("word")[1].split("_")[0])
    df['word_txt'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("_")[-1].split('.wav')[0])
    df['exp_type'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("_")[5])
    df['sent'] = df['file_path'].apply(lambda r: "_".join(
        os.path.basename(r).split("sent")[1].split("_")[1:-1]))
    df['speaker'] = df['file_path'].apply(
        lambda r: os.path.basename(r).split("_")[1])
    return df


def compute_speakers_distance(file1, file2, n_fft, hop, n_bins, group):
    '''
    file1: df row of a speaker
    file2: df row of a speaker
    n_fft: number of fft bins
    hop: hop length
    n_bins: number of mfcc bins
    group: 'BT' if between groups comparison, 'EN' if its within L1 comparison
    returns: distance between the two speakers
    '''
    audio1, sr1 = librosa.load(file1.file_path, sr=SR)
    audio2, sr2 = librosa.load(file2.file_path, sr=SR)
    if len(audio1) == 0:
        print("Skipping Empty audio: ", file1.file_path)
        return -1
    if len(audio2) == 0:
        print("Skipping Empty audio: ", file2.file_path)
        return -1

    mfcc1 = librosa.feature.mfcc(
        y=audio1, sr=SR, n_fft=n_fft, hop_length=hop, n_mfcc=n_bins)
    mfcc1 = (mfcc1 - mfcc1.mean()) / mfcc1.std()
    mfcc2 = librosa.feature.mfcc(
        y=audio2, sr=SR, n_fft=n_fft, hop_length=hop, n_mfcc=n_bins)
    mfcc2 = (mfcc2 - mfcc2.mean()) / mfcc2.std()

    dist = lib_dtw(mfcc2, mfcc1)[
        0][-1, -1] if (group == 'BT' and file1.l1 == 'ENG') else lib_dtw(mfcc1, mfcc2)[0][-1, -1]
    return dist


def dist_cmn(mandarin_data_path, spanish_data_path, task, n_fft, n_bins, hop, save_path):
    '''
    Compute the DTW distance between the Mandarin/Spanish and English speakers based on MFCC features.
    Since DTW isn't symmetric, a key is constructed for each pair, to maintain comparison order.
    mandarin_data_path: path to the mandarin data
    spanish_data_path: path to the spanish data
    task: task to run the baseline on (HT1/HT2/DHR/LPP)
    n_fft: number of fft bins
    n_bins: number of mfcc bins
    hop: hop length
    save_path: path to save the distance csv file
    '''
    print("Running baseline with the mandarin dataset with task, ", task)
    keys = []
    df = cmn_df(mandarin_data_path, spanish_data_path, task)
    for id1, file1 in df.iterrows():
        for id2, file2 in df[(df.sent_id == file1.sent_id) & (df.speaker != file1.speaker)
                             & (file1.sent_lng == df.sent_lng) & (file1.exp_type == df.exp_type)
                             & ((df.l1 == 'ENG') | (df.l1 == file1.l1)) & (file1.word_id == df.word_id)].iterrows():
            if file1.speaker == file2.speaker or file1.sent_lng != file2.sent_lng or file1.exp_type != file2.exp_type \
                    or (file1.l1 != 'ENG' and file2.l1 != 'ENG' and file1.l1 != file2.l1):
                continue
            if float(file1.speaker) < float(file2.speaker):
                pair_key = f"{file1.speaker}_{file1.word_id}_{file1.sent_id}_{file2.speaker}"
            else:
                pair_key = f"{file2.speaker}_{file2.word_id}_{file2.sent_id}_{file1.speaker}"
            if pair_key in keys:
                continue
            w1, w2 = file1.word_id, file2.word_id
            if w1 != w2:
                continue
            group = file1.sent_type if file1.l1 == file2.l1 else 'BT'

            dist = compute_speakers_distance(
                file1, file2, n_fft, n_bins, hop, group)
            if dist == -1:
                continue

            if group == 'BT' and file1.l1 == 'ENG':
                rows.append([f"{file2.speaker}_{w2}_{file2.word_txt}_{file2.sent_id}",
                             f"{file1.speaker}_{w1}_{file1.word_txt}_{file1.sent_id}", group, dist])
            else:
                rows.append([f"{file1.speaker}_{w1}_{file1.word_txt}_{file1.sent_id}",
                             f"{file2.speaker}_{w2}_{file2.word_txt}_{file2.sent_id}", group, dist])
            keys.append(pair_key)
    dist_df = pd.DataFrame(rows, columns=columns)
    dist_df.to_csv(os.path.join(
        save_path, f'cmn_shs_mfcc_baseline_{task.lower()}.csv'))


def dist_kr(kr_mfa_path, n_fft, n_bins, hop, save_path):
    '''
    Compute the DTW distance between the Korean and English speakers based on MFCC features.
    Since DTW isn't symmetric, a key is constructed for each pair, to maintain comparison order.
    kr_mfa_path: path to the Korean MFA output
    n_fft: number of fft bins
    n_bins: number of mfcc bins
    hop: hop length
    save_path: path to save the distance csv file
    '''
    print("Running baseline with the Korean dataset")
    df = kr_df(kr_mfa_path)
    kspl = df[df.l1 != 'E'].speaker.unique()
    kspl.sort()
    espl = df[df.l1 == 'E'].speaker.unique()
    espl.sort()
    spkrs_rnk = {s: i for i, s in enumerate(np.concatenate((kspl, espl)))}
    keys = []
    for id1, file1 in df.iterrows():
        for id2, file2 in df[(df.sent_id == file1.sent_id)
                             & (df.speaker != file1.speaker) & (file1.word_id == df.word_id)].iterrows():
            group = file1.l1 if file1.l1 == file2.l1 else 'BT'
            if file1.speaker == file2.speaker:
                continue

            if spkrs_rnk[file1.speaker] < spkrs_rnk[file2.speaker]:
                pair_key = f"{file1.speaker}_{file1.word_id}_{file1.sent_id}_{file2.speaker}"
            else:
                pair_key = f"{file2.speaker}_{file2.word_id}_{file2.sent_id}_{file1.speaker}"
            if pair_key in keys:
                continue
            w1, w2 = file1.word_id, file2.word_id
            if w1 != w2:
                continue

            if file1.word_txt != file2.word_txt:
                print("ERROR: Invalid comparison between different words.")
                print(os.path.basename(file1.file_path),
                      " XXX ", os.path.basename(file2.file_path))
                print("skiping...")

            dist = compute_speakers_distance(file1, file2, n_fft, n_bins, hop, group)
            if dist == -1:
                continue

            if group == 'BT' and file1.l1 == 'ENG':
                rows.append([f"{file2.speaker}_{w2}_{file2.word_txt}_{file2.sent_id}",
                             f"{file1.speaker}_{w1}_{file1.word_txt}_{file1.sent_id}", group, dist])
            else:
                rows.append([f"{file1.speaker}_{w1}_{file1.word_txt}_{file1.sent_id}",
                             f"{file2.speaker}_{w2}_{file2.word_txt}_{file2.sent_id}", group, dist])
            keys.append(pair_key)
    dist_df = pd.DataFrame(rows, columns=columns)
    dist_df.to_csv(os.path.join(
        save_path, f'kr_mfcc_baseline.csv'))


parser = argparse.ArgumentParser(
    description='Parser for DTW baseline evaluation', )
parser.add_argument('--kr', action='store_true',
                    help='run korean dataset analysis')
parser.add_argument('--task', type=str,
                    help='reading task for the mandarin dataset')
parser.add_argument('--n_fft', type=int, default=int(0.02 * 16000))
parser.add_argument('--hop', type=int, default=int((0.01 * 16000) / 2))
parser.add_argument('--cmn_data_path', type=str,
                    help="path of the mandarin dataset")
parser.add_argument('--spn_data_path', type=str,
                    help="path of the spanish dataset")
parser.add_argument('--kr_mfa_path', type=str,
                    help="path of the korean dataset")
parser.add_argument('--save_path', type=str, required=True,
                    help="Path to directory in which the csv will be saved")
parser.add_argument('--n_bins', type=int, default=80)
args = parser.parse_args()

if __name__ == '__main__':
    if args.kr:
        if args.kr_mfa_path == "":
            print("Please provide the path to the Korean dataset")
            exit()
        dist_kr(args.kr_mfa_path, args.n_fft,
                args.n_bins, args.hop, args.save_path)
    else:
        if args.cmn_data_path is None or args.spn_data_path is None:
            print("Please provide the path to the Mandarin and Spanish dataset")
            exit()
        if args.task is None:
            print("Please provide a reading task to evaluate")
            exit()
        dist_cmn(args.cmn_data_path, args.spn_data_path, args.task,
                 args.n_fft, args.n_bins, args.hop, args.save_path)
