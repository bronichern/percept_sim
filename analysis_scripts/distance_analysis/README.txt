These files provide the statistical analysis for Chernyak et al.

DistanceCalc_Main.R: R script for producing distance measures for different groups of L1 speakers, using either the t-SNE projection or Full Dimensionality of the HuBERT encoding. Yields as output distance_ratio_L1vL2_Main.csv, which provides 95% bootstrapped confidence intervals for the ratio of L2 English speakers to L1 English speakers.

intelligibility_by_distance.R: R script that calculates beta regressions for intelligibility data. Generates intelligibility_by_distance_results.csv, which provides the statistics for perceptual similarity space distance alone; intelligibility_by_all_results.csv, which provides statistics for models including acoustic measures in addition to perceptual similarity space distance; and intelligibility_by_all_results_scaled.csv, which provides statistics for models in which the predictors are normalized.

DistanceCalc_Other.R, intelligibility_by_distance_Other.R: Same structure as scripts above, but operating over (a) UMAP and (b) KPCA projections.

DistanceCalc_Korean_optim.R,intelligibility_by_distance_Korean_optim_test.R: Same structure as above, but examining Chinese and Spanish Heritage talkers after optimizing parameters for the Korean data.

intelligibility_by_distance_betareg.R: Calculates, at transformer layer 12, fixed-effect-only beta regression models with vs. without perceptual similarity space distance as a predictor.

intelligibility_filenames.csv: Filenames of acoustic files used to gather intelligibility data (a subset of the full set of acoustic files analyzed in DistanceCalc_Main.R).

CMN_KOR_SHS_lvl_p02.csv: Acoustic measures for each talker, calculated using Praat scripts.

intelligibilityScores.txt: Intelligibility data for the talkers.