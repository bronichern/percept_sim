
library (tidyverse)
library (betareg)

# Read in intelligibility data
intelligibility <- read.delim(file="intelligibilityScores.txt",as.is=T)
# Harmonize subject IDs across files
intelligibility <- intelligibility %>% mutate (Subject_1_ID = sub("_M_","_",Subject_1_ID),
                                               Subject_1_ID = sub("_F_","_",Subject_1_ID))


# Read in acoustic parameters
talker_acoustics <- read.csv(file="CMN_KOR_SHS_lvl_p02.csv")

# Harmonize subject IDs across files
talker_acoustics <- talker_acoustics %>% mutate (Subject_1_ID = str_remove(talker,"_ENG"),
                                                 Subject_1_ID = sub("_M_","_",Subject_1_ID),
                                                 Subject_1_ID = sub("_F_","_",Subject_1_ID))

# read in DTW distances over MFCCs data
mfcc_results_frame <- data.frame(mean_distance_dtw=-999)
# for each lang
for (lang in c("CMN","SHS","KR")){
  #Extract MFC files
  if (lang == "KR"){
    # read in and separate out subject info
    distance_data <- read.csv("./MFCC_DTW_baseline/KR/KR_MFCC_BASELINE.csv") %>% separate(speaker1_word_wid,into=c("Subject_1_ID","N1","Word","N2"),sep="_") %>% 
      separate(speaker2_word_wid,into=c("Subject_2_ID","N3","Word2","N4"),sep="_")
    # Get L2 speaker IDs
    summary_L2 <- distance_data %>% filter (group == "K_EN") %>% 
      summarise (talkers1 = unique(Subject_1_ID),talkers2=unique(Subject_2_ID))
    L2_talkers <- unique(c(summary_L2$talkers1,summary_L2$talkers2))
    
  } else {
    # read in and separate out subject info
    distance_data <- rbind (read.csv("./MFCC_DTW_baseline/CMN_SHS/HT1/cmn_shs_mfcc_baseline_ht1.csv"),
                            read.csv("./MFCC_DTW_baseline/CMN_SHS/HT2/cmn_shs_mfcc_baseline_ht2.csv")) %>% 
      separate(speaker1_word_wid,into=c("Subject_1_ID","N1","Word","N2"),sep="_") %>% 
      separate(speaker2_word_wid,into=c("Subject_2_ID","N3","Word2","N4"),sep="_") 
    # Get L2 speaker IDs
    summary_L2 <- distance_data %>% filter (group == paste(lang,"ENG",sep="_")) %>%
      reframe (talkers1 = unique(Subject_1_ID),talkers2=unique(Subject_2_ID))
    L2_talkers <- unique(c(summary_L2$talkers1,summary_L2$talkers2))
  }
  # get relevant subset of data for distance between L2 and L1
  between_data <- distance_data %>% 
    filter (group=="BT",(Subject_1_ID %in% L2_talkers) | (Subject_2_ID %in% L2_talkers))  
  # code one column with L2 subject
  between_data$L2_Subject_ID <- Map (function(x,y) ifelse(x %in% L2_talkers,x,y),between_data$Subject_1_ID,between_data$Subject_2_ID)
  # get DTW distance for each talker
  byTalker <- between_data %>% 
    group_by(L2_Subject_ID) %>% summarise (mean_distance_dtw = mean (Distance))
  # Harmonize Subject IDs for merging with intelligibility data
  if (lang == "KR"){
    byTalker$L2_Subject_ID <- paste("KEI",byTalker$L2_Subject_ID,sep="_")
  } else {
    byTalker$L2_Subject_ID <- paste("ALL",byTalker$L2_Subject_ID,lang,sep="_")
  }
  # rename subject column for merging
  colnames(byTalker)[1] <- "Subject_1_ID"
  # store the data
  # if this is the first language, overwrite dummy variable
  if (mfcc_results_frame$mean_distance_dtw[1] == -999){
    mfcc_results_frame <- byTalker
  } else
    # otherwise append to end of results
    mfcc_results_frame <- rbind(mfcc_results_frame,byTalker)
} # for each language

# merge DTW and acoustics
talker_acoustics <- merge(talker_acoustics,mfcc_results_frame)

# Merge acoustics and intelligibility data
intelligibility <- merge(intelligibility,talker_acoustics)

# center acoustic values, finding first and last columns of acoustic variables
intelligibility[,match("Sum.of..nsyll",colnames(intelligibility)):match("mean_distance_dtw",colnames(intelligibility))] <- scale(intelligibility[,match("Sum.of..nsyll",colnames(intelligibility)):match("mean_distance_dtw",colnames(intelligibility))],center=T,scale=F)
# center SNR
intelligibility$center_SNR <- scale(intelligibility$SNR, center=T,scale=F)

# created new dataframe with scaled variables
scaled_intelligibility <- intelligibility
scaled_intelligibility[,match("Sum.of..nsyll",colnames(scaled_intelligibility)):match("mean_distance_dtw",colnames(scaled_intelligibility))] <- scale(scaled_intelligibility[,match("Sum.of..nsyll",colnames(scaled_intelligibility)):match("mean_distance_dtw",colnames(scaled_intelligibility))],center=T,scale=T)
scaled_intelligibility$center_SNR <- scale(intelligibility$SNR, center=T,scale=T)

# read in list of sentences used in intelligibility testing for each language
intelligibility_filenames <- read.csv("intelligibility_filenames.csv")

# initialize storage of beta regression results
beta_results <- data.frame(layer = c(-999))
regression_results <- data.frame(beta_weight = c(-999))
# for each dimensionality
d <- "TSNE"
  kr_files <- list.files(path=paste("./",d,"/KR",sep=""),full.names = TRUE)
# for each layer
  i <- 4
  
  # Match the sentences used in the distance-from-L1 calculation so that they are the same as those used
  # in intelligibility testing
  # initialize frame for storing intelligibility data
  talker_frame <- data.frame(mean_distance=c(-999))
  # for each lang
  for (lang in c("CMN","SHS","KR")){
    
    # Process Korean files
    if (lang == "KR"){
      distance_data <- read.csv(kr_files[i])
      
      # select between-group (L2 vs. L1) comparisons 
      # convert filename conventions to match acoustic analysis
      # (distance data has _cnv after each name, acoustics does not)
      # even though using Subject 2 (containing Korean participant IDs),
      # use Subject 1 for consistency across all speaker groups and intelligibility data file
      harmonized_distance_data <- distance_data %>% filter(grp=="BT") %>% 
        mutate( allsstar_file = str_remove(speaker2,"_cnv")) %>%
        separate(speaker2,into=c("Subject_1_ID","sentence"),sep="_EN")  
      
      # For distance analysis, select only those observations that are in Korean traditional acoustic analysis data
      harmonized_distance_data_match <- harmonized_distance_data %>% filter (allsstar_file %in% intelligibility_filenames$allsstar_file[intelligibility_filenames$language_background==lang])
      
      # Get mean distance 
      byTalkerDistanceData <- harmonized_distance_data_match %>% group_by(Subject_1_ID) %>% summarise (mean_distance = mean(Distance))
    } else {
      # Processes Chinese or Spanish Data
      # retrieve file list
      ht1_files <- list.files(path=paste("./",d,"/",lang,"/HT1",sep=""),full.names = TRUE)
      ht2_files <- list.files(path=paste("./",d,"/",lang,"/HT2",sep=""),full.names = TRUE)
      # Read in distance data
      distance_data <- rbind(read.csv(ht1_files[i]),read.csv(ht2_files[i]))
      
      # get filenames used in intelligibility testing for this group; remove the attempt number from the filename
      rel_filenames <- intelligibility_filenames %>% filter (language_background == lang) %>%
        mutate (allsstar_file = str_replace(allsstar_file,"_0[123]_","_"))

            
      # select between-group (L2 vs. L1) comparisons for current language
      harmonized_distance_data <- distance_data %>% filter(grp=="BT")
      
      if (lang == "SHS"){
        harmonized_distance_data <- harmonized_distance_data %>% 
          # harmonize sentence numbers with acoustic data
          # distance sentences are odd numbers 1-119, to match with acoustic should be 1-60
          # sentence number preceded by S rather than sent
          separate_wider_delim(speaker1, names=c("ALL","speaker_number","speaker_gender","speaker_L1","speech_language","HT_set","sentenceID"),delim="_",cols_remove=FALSE) %>%
          mutate(sentenceNumber = as.numeric(str_sub(sentenceID,start=5)) %/% 2 +1,
                 sentenceIDcorrected = str_c("S",str_pad(as.character(sentenceNumber),3,pad="0"))) %>%
          unite(col='allsstar_file',c("ALL","speaker_number","speaker_gender","speaker_L1","speech_language","HT_set","sentenceIDcorrected"),remove=FALSE) %>%
          # format subject identifier for merger with intelligibility data
          unite(col='Subject_1_ID',c("ALL","speaker_number","speaker_L1"))
      } else { #for CMN data
        harmonized_distance_data <- harmonized_distance_data %>% 
          # harmonize sentence numbers with acoustic data
          # distance sentences are odd numbers 1-119, to match with acoustic should be 1-60 for HT1, 61-120 for HT2
          # sentence number preceded by S rather than sent
          separate_wider_delim(speaker1, names=c("ALL","speaker_number","speaker_gender","speaker_L1","speech_language","HT_set","sentenceID"),delim="_",cols_remove=FALSE) %>%
          mutate(sentenceNumber60 = as.numeric(str_sub(sentenceID,start=5)) %/% 2 +1,
                 sentenceNumber = ifelse(HT_set == "HT1",sentenceNumber60,sentenceNumber60+60),
                 sentenceIDcorrected = str_c("S",str_pad(as.character(sentenceNumber),3,pad="0"))) %>%
          unite(col='allsstar_file',c("ALL","speaker_number","speaker_gender","speaker_L1","speech_language","HT_set","sentenceIDcorrected"),remove=FALSE) %>%
          # format subject identifier for merging with intelligibility data
          unite(col='Subject_1_ID',c("ALL","speaker_number","speaker_L1"))
      }
       
      harmonized_distance_data_match <- harmonized_distance_data %>%  filter (allsstar_file %in% rel_filenames$allsstar_file)
      
      # Get mean distance
      byTalkerDistanceData <- harmonized_distance_data_match %>% group_by(Subject_1_ID) %>% summarise (mean_distance = mean (Distance))
    }
    if (talker_frame$mean_distance[1] == -999){
      talker_frame <- byTalkerDistanceData
    } else
      talker_frame <- rbind(talker_frame,byTalkerDistanceData)
  } # for each language
  

  # merge in intelligibility data
  talker_frame_intelligibility <- merge(talker_frame,intelligibility)
  talker_frame_scaled_intelligibility <- merge(talker_frame,scaled_intelligibility)
  
  # center distance values
  talker_frame_intelligibility$center_mean_distance <- scale(talker_frame_intelligibility$mean_distance, center=T, scale=F)
  # scale distance values
  talker_frame_scaled_intelligibility$center_mean_distance <- scale(talker_frame_scaled_intelligibility$mean_distance, center=T, scale=T)
  
  # build beta regression with all factors

  betareg_acoustics <- betareg(intelligibility~center_SNR+mean_distance_dtw+Sum.of..npause+Average.of..articRate.nsyll.phonationtime.+Average.of..f0Mean.Hz.+Average.of..f0CoefVar+dispersionMean.Bark.+sylReduction.nsyll.orthSyll.,data=talker_frame_intelligibility,link = "logit")
  summary(betareg_acoustics)
  # Pseudo R-squared: 0.484
  
  betareg_acoustics_distance <- betareg(intelligibility~center_SNR+center_mean_distance+mean_distance_dtw+Sum.of..npause+Average.of..articRate.nsyll.phonationtime.+Average.of..f0Mean.Hz.+Average.of..f0CoefVar+dispersionMean.Bark.+sylReduction.nsyll.orthSyll.,data=talker_frame_intelligibility,link = "logit")
  summary(betareg_acoustics_distance)
  #Pseudo R-squared: 0.7232
  