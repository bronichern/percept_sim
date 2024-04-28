# Calculate distances within each group (L1 vs. L2)
# Calculate 95% confidence interval for ratio of L2 / L1 distances

library (tidyverse)

# initialize data frame for storing results
results_frame <- data.frame(analysis_method_label = "dummy")

for (analysis_method in c("Parameters_search","UMAP","KPCA"))
{
kr_files <- list.files(path=paste("./",analysis_method,"/KR",sep=""),full.names = TRUE)
# for each file
for (i in 1:length(kr_files)){

  # for each language 
  # CMN = Chinese Mandarin
  # SHS = Spanish Heritage Speakers
  # KR = Korean
  for (lang in c("CMN","SHS","KR")){
    if (lang == "KR"){
      # read in data
      distance_data <- read.csv(kr_files[i])
      # Get L2 data
      L2_pairs_data <- distance_data %>% filter (grp == "K_EN") %>% 
        # separate out speaker ID
        separate_wider_delim(speaker1, names=c(NA,"Speaker_1_ID",NA,NA),delim="_") %>%
        separate_wider_delim(speaker2, names=c(NA,"Speaker_2_ID",NA,NA),delim="_")
      
      # Get L1 data
      L1_pairs_data <- distance_data %>% filter (grp == "E") %>% 
        # separate out speaker ID
        separate_wider_delim(speaker1, names=c(NA,"Speaker_1_ID",NA,NA),delim="_") %>%
        separate_wider_delim(speaker2, names=c(NA,"Speaker_2_ID",NA,NA),delim="_")
    }
    else { #CMN, SHS data
      # get file lists for 2 halves of sentence data
      # (corresponding to split on SpeechBox https://speechbox.linguistics.northwestern.edu/#!/home)
      ht1_files <- list.files(path=paste("./",analysis_method,"/",lang,"/HT1",sep=""),full.names = TRUE)
      ht2_files <- list.files(path=paste("./",analysis_method,"/",lang,"/HT2",sep=""),full.names = TRUE)
      
      # load distance data
      distance_data <- rbind(read.csv(ht1_files[i]),read.csv(ht2_files[i]))
      # Get L2 data
      L2_pairs_data <- distance_data %>% filter (grp == substring(lang,1,1)) %>% 
        # separate out speaker ID for both speakers. Only use cases where speakers are producing English
        separate_wider_delim(speaker1, names=c(NA,"Speaker_1_ID","Gender","Language_Background","Language_Produced",NA,NA),delim="_") %>%
        filter (Language_Produced == "ENG") %>% 
        unite(col="Speaker_1_ID",c("Speaker_1_ID", "Gender","Language_Background","Language_Produced"),remove=TRUE) %>%
        separate_wider_delim(speaker2, names=c(NA,"Speaker_2_ID","Gender","Language_Background","Language_Produced",NA,NA),delim="_") %>%
        filter (Language_Produced == "ENG") %>% 
        unite(col="Speaker_2_ID",c("Speaker_2_ID", "Gender","Language_Background"),"Language_Produced",remove=TRUE)
      # Get L1 data
      L1_pairs_data <- distance_data %>% filter (grp == "E") %>% 
        # separate out speaker ID for both speakers. Retain information about language background so L1, L2 talkers don't get mixed up
        separate_wider_delim(speaker1, names=c(NA,"Speaker_1_ID","Gender","Language_Background",NA,NA,NA),delim="_") %>%
        unite(col="Speaker_1_ID",c("Speaker_1_ID", "Gender","Language_Background"),remove=TRUE) %>%
        separate_wider_delim(speaker2, names=c(NA,"Speaker_2_ID","Gender","Language_Background",NA,NA,NA),delim="_") %>%
        unite(col="Speaker_2_ID",c("Speaker_2_ID", "Gender","Language_Background"),remove=TRUE)
    }
    
    # Calculate mean distance for pairs of L2 speakers for current language
    L2_pairs_mean <- L2_pairs_data  %>% 
      # create speaker pair, ensuring consistent ordering of speaker number
      mutate (speaker_pair = ifelse(Speaker_1_ID < Speaker_2_ID, paste(Speaker_1_ID,Speaker_2_ID),paste(Speaker_2_ID,Speaker_1_ID))) %>%
      # for each speaker pair, calculate the mean distance
      group_by (speaker_pair) %>% summarise(mean_distance = mean(Distance))
    
    # Calculate mean distance for pairs of L1 speakers using same procedure as above
    L1_pairs_mean <- L1_pairs_data %>% 
      mutate (speaker_pair = ifelse(Speaker_1_ID < Speaker_2_ID, paste(Speaker_1_ID,Speaker_2_ID),paste(Speaker_2_ID,Speaker_1_ID))) %>%
      group_by (speaker_pair) %>% summarise(mean_distance = mean(Distance))
  
    # use bootstrap resampling to calculate 95% CIs
    # initial vector to store results
    boot_sample_ratios <- c()
    # for 1000 replicates
    for (j in 1:1000){
      # re-sample L2 pair means, L1 pair means with replacement, and recalculate ration
      boot_sample_ratios <- append (boot_sample_ratios,
                                    mean(sample(L2_pairs_mean$mean_distance,replace=T))/
                                    mean(sample(L1_pairs_mean$mean_distance,replace=T)))
    }
  
    # store results, extracting layer information from filename, and saving 2.5% and 97.5% percentile of bootstrap distribution
    results_row <- data.frame(analysis_method_label = c(analysis_method),
                              filename = c(substring(kr_files[i],first=27,last=str_length(kr_files[i])-5)), 
                              lang=lang,L1_L2_mean = mean(L2_pairs_mean$mean_distance)/mean(L1_pairs_mean$mean_distance),
                                L1_L2_lower = sort(boot_sample_ratios)[25],
                                L1_L2_upper = sort(boot_sample_ratios)[975])
    
    # if results_frame has just been initialized, replace with current row, otherwise append
    if (results_frame$analysis_method_label[1] == "dummy"){
      results_frame <- results_row
      } else
        results_frame <- rbind(results_frame,results_row)
    
    # Save results
    write.csv(results_frame,"distance_ratio_L1vL2_Other.csv",row.names=F)
  } # for each language
  }# for each file

} # for each analysis method