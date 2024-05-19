
library (tidyverse)
library (glmmTMB)

# Read in intelligibility data
intelligibility <- read.delim(file="intelligibilityScores.txt",as.is=T)
# Harmonize subject IDs across files
intelligibility <- intelligibility %>% mutate (Subject_1_ID = sub("_M_","_",Subject_1_ID),
                                               Subject_1_ID = sub("_F_","_",Subject_1_ID))



intelligibility$center_SNR <- scale(intelligibility$SNR, center=T,scale=F)

# read in list of sentences used in intelligibility testing for each language
intelligibility_filenames <- read.csv("intelligibility_filenames.csv")

# initialize storage of beta regression results
beta_results <- data.frame(filename = "dummy")


  kr_files <- list.files(path="./Parameters_search/KR",full.names = TRUE)
# for each file
for (i in 1: length(kr_files)){
  
  # Match the sentences used in the distance-from-L1 calculation so that they are the same as those used
  # in intelligibility testing
  # initialize frame for storing intelligibility data
  talker_frame <- data.frame(mean_distance=c(-999))
  # for each lang
  lang <- "KR"
    
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
    } 
       
    if (talker_frame$mean_distance[1] == -999){
      talker_frame <- byTalkerDistanceData
    } else
      talker_frame <- rbind(talker_frame,byTalkerDistanceData)
  
  

  # merge in intelligibility data
  talker_frame_intelligibility <- merge(talker_frame,intelligibility)
  
  # center distance values
  talker_frame_intelligibility$center_mean_distance <- scale(talker_frame_intelligibility$mean_distance, center=T, scale=F)
  # calculate Beta Regression
  betareg_distance <- glmmTMB(intelligibility~center_SNR+center_mean_distance+(1|Subject_1_ID),data=talker_frame_intelligibility,family=beta_family(link="logit"))
  sig_calc <- anova(betareg_distance,update(betareg_distance,.~.-center_mean_distance)) 
  
  #store results, extracting results for distance
  if (beta_results$filename[1] == "dummy"){
    beta_results <- data.frame(filename = c(substring(kr_files[i],first=27,last=str_length(kr_files[i])-5)),
                               beta_intelligibility=summary(betareg_distance)$coefficients$cond[3,1],
                               se_beta_intelligibility = summary(betareg_distance)$coefficients$cond[3,2],
                               chi_sq_beta_intelligibility = sig_calc$Chisq[2],
                               p_chi_sq_beta_intelligibility = sig_calc$"Pr(>Chisq)"[2])
  } else
    beta_results <- rbind(beta_results,data.frame(filename = c(substring(kr_files[i],first=27,last=str_length(kr_files[i])-4)),
                                                  beta_intelligibility=summary(betareg_distance)$coefficients$cond[3,1],
                                                  se_beta_intelligibility = summary(betareg_distance)$coefficients$cond[3,2],
                                                  chi_sq_beta_intelligibility = sig_calc$Chisq[2],
                                                  p_chi_sq_beta_intelligibility = sig_calc$"Pr(>Chisq)"[2]))
  
  # Save results
  write.csv(beta_results,"intelligibility_by_distance_results_Korean_optim_train.csv",row.names=F)

} # for each filename
 
