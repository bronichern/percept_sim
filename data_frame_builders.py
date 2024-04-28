
import re
from typing import List
import os
import numpy as np
import pandas as pd

CMN = 0
KR = 1
SPN = 2

def load_speakers_metadata(feats_file:str, tsv_name:str):
    '''
    Loads time indexes for each speaker and their names
    feats_file: path to the .npy file containing the features
    tsv_name: name of the tsv file containing the speaker names
    '''
    speakers_len = open(feats_file.replace('.npy', '.len'), 'r').read().split('\n')
    speakers_len = np.array([int(f) for f in speakers_len if f != ''])

    wav_paths = re.split('\t\d\n|\n', open("tsvs/"+tsv_name, 'r').read())
    names = np.array([f.split(".")[0] for f in wav_paths[1:] if f != ""])
    return speakers_len, names


def create_df(data_group:int, feats, speaker_len:List[int], names:List[str]):
    '''
    Creates a dataframe with the features and the speaker information. Each row is a time step of a speaker.
    data_group: CMN, KR, SPN
    speaker_len: list of the number of time steps for each speaker
    names: list of speaker names
    '''

    hubert_feature_columns = [f"val {i}" for i in range(feats.shape[1])]
    df = pd.DataFrame(feats, columns=hubert_feature_columns)
    df['index'] = df.index
    # match speaker to rows
    time_index = {i: speaker_len[i] for i in range(len(speaker_len))}
    com_time_index = {i: sum(speaker_len[:i]) for i in range(len(speaker_len))}
    df_speaker_count = pd.Series(time_index)
    df_speaker_count = df_speaker_count.reindex(df_speaker_count.index.repeat(
        df_speaker_count.to_numpy())).rename_axis('speaker_id').reset_index()
    df['speaker_id'] = df_speaker_count['speaker_id']
    df['speaker_len'] = df['speaker_id'].apply(lambda row: speaker_len[row])
    df['com_sum'] = df['speaker_id'].apply(lambda i: com_time_index[i])
    df['speaker'] = df['speaker_id'].apply(lambda i: names[i])

    assert len(df_speaker_count) == len(df)

    if data_group == CMN:
        df['path'] = df['speaker'].str.split('/').str[0]
        df['speaker'] = df['speaker'].str.split('/').str[1]
    elif data_group == SPN:
        df['path'] = df['speaker'].str.split('/').str[0]
        df['speaker'] = df['speaker'].str.split('/').str[-1]

    assert len(df.loc[df['speaker'] == -1]) == 0
    df_subset = df.copy()
    data_group = df_subset[hubert_feature_columns].values
    return data_group, df_subset, hubert_feature_columns


def data_group_name(data_group):
    '''
    Returns the name of the data group
    '''
    return 'CMN' if data_group == CMN else 'KR' if data_group == KR else 'SPN'
     

def build_kr_df(args, portions=4):
    '''
    Load the Korean data and create a dataframe with the features and speaker information.
    '''
    feat_path_data = args.feat_path
    tsv_name_data = args.tsv_name

    feats_file_path = os.path.join(
        feat_path_data, f"kr_layer{args.layer}", tsv_name_data.replace('.tsv', '_0_1.npy'))

    feats = np.load(feats_file_path)
    speakers_len, names = load_speakers_metadata(feats_file_path, tsv_name_data)
    if not args.run_all_data and args.project:
        snts = list(set([f.split("_")[2] for f in names]))
        splt_len = int(len(snts)/portions)
        sub_portion = (args.portion-1) * splt_len
        curr_snts = snts[sub_portion: sub_portion +
                            splt_len] if args.portion < portions else snts[sub_portion:]
        idxs = [i for i, f in enumerate(names) if f.split("_")[2] in curr_snts]

        feat_idxs = []
        for i, id in enumerate(idxs):
            # treat each speaker individ. and load his idxs - up tohim is start id and speakers_len[id] is length
            feat_idxs.extend(range(sum(speakers_len[:id]), sum(speakers_len[:id]) + speakers_len[id]))
        speakers_len = speakers_len[idxs]
        names = names[idxs]
        feats = feats[feat_idxs]

    data_subset, df_subset, hubert_feature_columns = create_df(args.data_group, feats, speakers_len, names)
    df_subset['idx'] = df_subset.index
    df_subset['speaker_main'] = df_subset['speaker'].apply(lambda i: "_".join(i.split(
        "_")[1:2]))
    df_subset['speaker_par'] = df_subset['speaker'].apply(lambda i: "_".join(i.split(
        "_")[2:3]))
    df_subset['sentLNG'] = df_subset['speaker_par'].apply(lambda i: i[:2])
    df_subset['sent_cross_unq'] = df_subset['sentLNG']+ df_subset['speaker_par']
    return data_subset, df_subset, speakers_len, hubert_feature_columns


def build_cmn_df(args):
    '''
    Load the CMN data and create a dataframe with the features and speaker information.
    '''
    files_to_load = []
    feat_path_data = args.feat_path
    tsv_name_data = args.tsv_name

    feat_file_task = 'ht' if args.reading_task == '' or 'ht' in args.reading_task.lower(
    ) else args.reading_task.lower()
    with open('tsvs/'+tsv_name_data, 'r') as f:
        data_path = f.read().split('\n')

    if args.data_group == CMN:
        all_paths = [data_path[0]]+list(set([os.path.join(data_path[0], f.split('/')[
                                        0]) for f in data_path[1:] if args.reading_task in f]))
        assert len(all_paths) == 4, all_paths
        print(all_paths)
        feats_file_data = os.path.join(
            feat_path_data, f"{data_group_name(args.data_group)}_{feat_file_task}_layer{args.layer}",
            tsv_name_data.replace('.tsv', '_0_1.npy'))

    else:
        all_paths = [data_path[0]]+list(set([os.path.join(data_path[0], f.split('/')[0], f.split(
            '/')[1]) for f in data_path[1:] if args.reading_task in f and 'ENG_ENG' not in f]))
        all_paths += list(set([os.path.join(data_path[0], f.split('/')[0])
                            for f in data_path[1:] if args.reading_task in f and 'ENG_ENG' in f]))
    
        assert len(all_paths) == 4

    for ap in all_paths[1:]:
        files_to_load += [os.path.join(ap, f)
                            for f in os.listdir(ap) if '.wav' in f and 'sent' in f]
    
    feats = np.load(feats_file_data)
    speakers_len, names = load_speakers_metadata(feats_file_data, tsv_name_data)
    
    if not args.run_all_data and args.project:
        idxs = [i for i, f in enumerate(names) if os.path.join(
            all_paths[0], f+'.wav') in files_to_load and f'sent{args.portion}' in f]
        name2idx = {f.split('/')[1]: i for i, f in enumerate(names) if (os.path.join(
            all_paths[0], f+'.wav') in files_to_load) and f'sent{args.portion}' in f}
    
        feat_idxs = []
        for i, id in enumerate(idxs):
            # treat each speaker individ. and load his idxs - up tohim is start id and speakers_len[id] is length
            feat_idxs.extend(range(sum(speakers_len[:id]), sum(speakers_len[:id]) + speakers_len[id]))
        speakers_len = speakers_len[idxs]
        names = names[idxs]
        feats = feats[feat_idxs]

    data_subset, df_subset, cols = create_df(
        args.data_group, feats, speakers_len, names)
    df_subset['idx'] = df_subset.index
    
    df_subset.path = all_paths[0]+df_subset.path
    df_subset['speaker_main'] = df_subset['speaker'].apply(
        lambda i: "_".join(i.split("_")[1:4]))
    df_subset['speaker_par'] = df_subset['speaker'].apply(
        lambda i: "_".join(i.split("_")[3:6]))
    df_subset['speaker_LNG'] = df_subset['speaker'].apply(
        lambda i: "_".join(i.split("_")[3:4]))
    df_subset['rng'] = df_subset['speaker'].apply(lambda r: name2idx[r])
    df_subset['sentid'] = df_subset['speaker'].apply(lambda i:i.split("_")[-1].split("sent")[1])
    df_subset['sent_cross_unq'] =  df_subset['speaker'].apply(lambda i:"_".join(i.split("_")[4:5])) + df_subset.sentid
    return data_subset, df_subset, speakers_len, cols

