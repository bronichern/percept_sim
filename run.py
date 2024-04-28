import os
import numpy as np
import pandas as pd
from distances import mut_normalize_sequences, librosa_dtw
import argparse
from projections import *
from data_frame_builders import build_cmn_df, build_kr_df, CMN, KR, SPN 


TSNE_COL=['tsne-1', 'tsne-2', 'tsne-3']

def measure_dist(filename, sub, speakers, feature_columns, data_group, verbose=False):
    normalize = True
    normalize_by_len = True
    normalization_func = mut_normalize_sequences
    dist_df = pd.DataFrame(columns=["speaker1", "speaker2", "group", "part", "sent"])
    feature_columns = feature_columns[0]
    distance_func, func_name = librosa_dtw, 'Distance'

    if data_group == KR:
        reading_task = ""
    elif data_group in [CMN, SPN]:
        reading_task = speakers[0].split("_")[5]

    for sp_id, speaker1 in enumerate(speakers[:-1]):
        for speaker2 in speakers[sp_id+1:]:
            if data_group == KR:
                spkr1_sent_lng = speaker1.split("_")[2]
                spkr2_sent_lng = speaker2.split("_")[2]

                speaker1_l1 = speaker1.split("_")[1][:1]
                speaker2_l1 = speaker2.split("_")[1][:1]
            elif data_group in [CMN, SPN]:
                speaker1_l1 = speaker1.split("_")[3][:2]
                speaker2_l1 = speaker2.split("_")[3][:2]
                spkr1_sent_lng = speaker1.split("_")[4][:2]
                spkr2_sent_lng = speaker2.split("_")[4][:2]
                speaker1_sentence = speaker1.split("_")[-1]
                speaker2_sentence = speaker2.split("_")[-1]
                speaker1_reading_task = speaker1.split("_")[5]
                speaker2_reading_task = speaker2.split("_")[5]

                if speaker1_sentence != speaker2_sentence or speaker1_reading_task != speaker2_reading_task:
                    continue
            group = 'BT' if speaker1_l1[0] != speaker2_l1[0] else speaker1_l1[0]
            if spkr1_sent_lng != spkr2_sent_lng:
                continue
            if verbose:
                print(f"Measuring distance for spkr {speaker1} and {speaker2}")

            sent_id = speaker1_sentence if data_group in [CMN, SPN] else speaker1.split("_")[2]
            dist_row = {"speaker1": speaker1, "speaker2": speaker2,
                        "group": group, "part": reading_task, "sent": sent_id}

            features_speaker1 = sub[(sub.speaker == speaker1)][feature_columns].to_numpy()
            features_speaker2 = sub[(sub.speaker == speaker2)][feature_columns].to_numpy()

            features_speaker1, features_speaker2 = normalization_func(features_speaker1, features_speaker2, normalize)
            distance = distance_func(features_speaker1, features_speaker2)

            if type(distance) == tuple and len(distance) > 0:
                distance = distance[0]

            if normalize_by_len:
                distance = distance/(len(features_speaker1)+len(features_speaker2))

            dist_row[f"{func_name}"] = distance

            dist_df = pd.concat([dist_df, pd.DataFrame([dist_row])], ignore_index=True)

    out_name = f"{filename}.csv"
    dist_df.to_csv(out_name)


def fill_projection_in_df(df_subset,ids, tsne_results):
    '''
    Fill the t-SNE results into the dataframe for each available dimension.
    '''
    df_subset.loc[ids,('tsne-1')] = tsne_results[:,0]
    df_subset.loc[ids,('tsne-2')] = tsne_results[:,1]
    if tsne_results.shape[1] == 3:
        df_subset.loc[ids,('tsne-3')] = tsne_results[:,2]
    return df_subset

def construct_filename(args, projection_type):
    # create output file name based on the argments
    portion_str = f"_portion{args.portion}" if args.project else ""
    if args.data_group in [CMN,SPN]:
        task = args.reading_task.lower()
        if not os.path.exists(os.path.join(args.output_path, f"{task}")):
            os.mkdir(os.path.join(args.output_path, f"{task}"))
        filename = os.path.join(
            args.output_path, f"{args.data}_{projection_type}_layer{args.layer}_{task}{portion_str}")
    else:
        filename = os.path.join(args.output_path, 
                                f"{args.data}_{projection_type}_layer{args.layer}{portion_str}")
    return filename

def project_samples(args, data_subset):
    '''
    Project the samples using the specified method.
    '''
    if not args.pca and not args.umap:
        projection_result = tsne(data_subset,
                                init=args.projection_init_method,
                                early_exaggeration=args.early_exagg,
                                lr=args.tsne_lr,
                                n_comp=args.num_projection_components,
                                perplexity=args.tsne_perplexity,
                                iters=args.tsne_iter,
                                seed=args.seed,)
    elif args.pca:
        projection_result = kpca(data_subset,
                        n_comp=args.num_projection_components,
                        kernel=args.kernel,
                        random_state=args.seed,
                        degree=args.degree,
                    )
    elif args.umap:
        projection_result = umap(
            data_subset,
            n_comp=args.num_projection_components,
            n_neighbors=args.umap_nei,
            min_dist=args.min_umap_dist,
            init=args.umap_init,
            random_state=args.seed,
            densmap=args.densmap,
            dens_lambda=args.d_lamb,
            dens_frac=args.d_frac,
        )
    return projection_result

def main(args):
    '''
    Main function to run the distance evaluation.
    '''
    use_dim_reduction = args.project
    projection_type = 'full_dim' if not use_dim_reduction else ('tsne' if not args.umap and not args.pca else ('umap' if args.umap else 'kpca'))
    if args.data_group == KR:
        data_subset, df, speakers_len, hubert_feature_columns = build_kr_df(args)
    else:
        data_subset, df, speakers_len, hubert_feature_columns = build_cmn_df(args)

    df['new_id'] = list(range(df.shape[0]))

    filename = construct_filename(args, projection_type)

    if args.verbose:
        print(
            f"Running distnace evaluation for layer {args.layer} with projection {use_dim_reduction} and projection type {projection_type}")
    print("File will be saved to ", filename)
    
    ids = np.array(df.index)       
    if use_dim_reduction:
        projection_result = project_samples(args, data_subset)
        df = fill_projection_in_df(df,ids,projection_result)
    print(df.columns)
    feature_columns = hubert_feature_columns if not args.project else [f"tsne-{i+1}" for i in range(projection_result.shape[1])],
    measure_dist(filename, df, df.speaker.unique(), feature_columns, args.data_group, args.verbose)


parser = argparse.ArgumentParser(description='Parser for distance evaluation',)
parser.add_argument('--layer', type=int, required=True, help='HuBERT Layer to use for the distance evaluation')
parser.add_argument('--project', action='store_true', help='Project the data to lower dimensions')
parser.add_argument('--data', type=str, required=True, help='Type of data group: cmn, kr')
parser.add_argument('--projection_init_method', type=str, default='pca', help='Initialization method for t-SNE and Umap')
parser.add_argument("--tsne_iter", type=int, default=1000, help="Number of iterations for t-SNE")
parser.add_argument("--early_exagg", type=float, default=12.0, help="Early exaggeration for t-SNE")
parser.add_argument("--tsne_perplexity", type=int, default=50, help="Perplexity for t-SNE")
parser.add_argument('--portion', type=int, default=1, help='Portion (an integer between 1-9) of the data to use - only relevant if projecting')
parser.add_argument("--tsne_lr", type=str, default="auto", help="Learning rate for t-SNE")

parser.add_argument("--pca", action="store_true", help="Use Kernel-PCA for dimensionality reduction")
parser.add_argument("--kernel", type=str, default="rbf", help="Kernel to use for Kernel-PCA")
parser.add_argument("--degree", type=int, default=3, help="Degree of the polynomial kernel for Kernel-PCA")

parser.add_argument("--umap", action="store_true", help="Use UMAP for dimensionality reduction")
parser.add_argument("--umap_nei", type=int, default=15, help="Number of neighbors for UMAP")
parser.add_argument("--min_umap_dist", type=float, default=0.1, help="Minimum distance for UMAP")
parser.add_argument("--umap_init", type=str, default="spectral", help="Initialization method for UMAP")
parser.add_argument("--densmap", action="store_true", help="Use densmap for UMAP")
parser.add_argument("--d_lamb", type=float, default=0.0, help="Lambda for densmap")
parser.add_argument("--d_frac", type=float, default=0.3, help="Fraction for densmap")

parser.add_argument('--file_pref', type=str, default="", help="Prefix for output file name")
parser.add_argument('--tsv_name', type=str, required=True, help="Name of the tsv file")
parser.add_argument('--output_path', type=str, required=True,
                    help="Path where distance outputs will be stored (csv format)")
parser.add_argument('--feat_path', type=str, required=True,
                    help="Path to where Hubert's .npy feature files are stored")
parser.add_argument('--reading_task', type=str, default=None, help="Reading task for the data: HT1 or HT2 (not relevant for Korean)")
parser.add_argument('--verbose', action='store_true', help='Print verbose output')
parser.add_argument('--run_all_data', action='store_true')
parser.add_argument('--num_projection_components', type=int, default=3, help='Number of components to project to')
parser.add_argument("--seed", type=int, default=0, help="Seed for reproducibility")

args = parser.parse_args()

if __name__ == '__main__':
    if args.project:
        print(f'evaluating distance with {args.num_projection_components} dimensions')
    else:
        print('evaluating distance with all dimensions')
    
    if args.tsne_lr != "auto":
        args.tsne_lr = float(args.tsne_lr)

    if args.data.lower() == 'cmn':
        args.data_group = CMN
    elif args.data.lower() == 'kr':
        args.data_group = KR
    elif args.data.lower() == 'spn':
        args.data_group = SPN
    else:
        raise Exception(
            f"Invalid data passed {args.data} | Possible values are 'cmn', 'spn' or 'kr'")
    
    if args.data_group != KR and args.reading_task is None and not args.project:
        raise Exception(
            f"For ALLSTAR dataset you must specify reading_task and portion")
    main(args)
