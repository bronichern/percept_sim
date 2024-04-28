##############################################################################################################
## Calculate Vowel Dispersion in F1 x F2 space (Bark) on Concatenated DJW wav+textgrids                     ##
##                                                                                                          ##
##  * Gets first 2 DJW parameters plus vowel dispersion(Bark)                                               ##                               
##  * Runs on concatenated files so that dispersion is calculated from overall centroid in the F1xF2 space  ##
##  * txt file with 4 parameters: nsyll, npause, vowel dispersion                                           ##
##                                                                                                          ##
##                                                                                                          ##
## IMPORTANT NOTE                                                                                           ##
##  * Concatenation must be run before this script                                                          ##
##                                                                                                          ##
## Ann Bradlow, January 2023                                                                                ##
##                                                                                                          ##
## Updated March 27, 2024 by Ann Bradlow and Seung-Eun Kim                                                  ##
##  + Report number of intra-sentence pauses (nIntrasentencePause)and total number of pauses(nAllPause).    ##
##    (Prior version only reported number of intra-sentence pauses.)                                        ##                                                
##                                                                                                          ##
##############################################################################################################

# Initialize directory list
form Calculate Vowel Dispersion in F1 x F2 space (Bark) on Concatenated DJW wav+textgrids
   comment Directory with concatenated wav and TextGrid files:
   text directory
endform

# Create list of files to work with 
Create Strings as file list... list 'directory$'/*.wav
select Strings list
Rename... wavlist

Create Strings as file list... list 'directory$'/*.TextGrid
select Strings list
Rename... textgridlist

printline Calculate Vowel Dispersion in F1 x F2 space (Bark) on Concatenated DJW wav+textgrids
printline filename, nsyll, nIntrasentencePause, nAllPause, dispersionMean(Bark)

# read files 
select Strings wavlist
numberOfFiles = Get number of strings

for ifile from 1 to numberOfFiles
   select Strings wavlist
   wavfileName$ = Get string... ifile
   Read from file... 'directory$'/'wavfileName$'
   soundid = selected("Sound")

   select Strings textgridlist
   tgfileName$ = Get string... ifile
   Read from file... 'directory$'/'tgfileName$'
   textgridid = selected("TextGrid")

   # fill array with time points (tier 2 is DJW point tier for acoustic syllable peaks)
   select textgridid
   numpeaks = Get number of points: 2
   for j from 1 to numpeaks
      t'j' = Get time of point: 2, j
   endfor 

   # Get F1, F2 statistics at peaks
   # make "times" vector 
   times# = zero# (numpeaks)
   for k from 1 to numpeaks
      times#[k] = t'k'
   endfor
    
   # make "f1" and "f2" vectors and fill with values at peaks (Hertz)
   select 'soundid'
   To Formant (burg)... 0.01 5 5500 0.025 50
   formantidHz = selected("Formant")
   f1hz# = zero# (numpeaks)
   f2hz# = zero# (numpeaks)
   for l from 1 to numpeaks   
      f1hz#[l] = Get value at time: 1,times#[l],"hertz", "linear"
      f2hz#[l] = Get value at time: 2,times#[l],"hertz", "linear"
    endfor
    
   # make "f1" and "f2" vectors and fill with values at peaks (Bark)
   select 'soundid'
   To Formant (burg)... 0.01 5 5500 0.025 50
   formantidBk = selected("Formant")
   f1bk# = zero# (numpeaks)
   f2bk# = zero# (numpeaks)
   for m from 1 to numpeaks   
      f1bk#[m] = Get value at time: 1,times#[m],"bark", "linear"
      f2bk#[m] = Get value at time: 2,times#[m],"bark", "linear"
   endfor

   meanF1hz = mean(f1hz#)
   meanF2hz = mean(f2hz#)
   meanF1bk = mean(f1bk#)
   meanF2bk = mean(f2bk#)

   # Calculate vowel dispersion
   # make "dispersion" vector
   dispersion# = zero# (numpeaks)
    
   # fill "dispersion" vector with distance of each vowel to f1-f2 mean
   for n from 1 to numpeaks
      dispersion#[n] = sqrt(((f1bk#[n]-meanF1bk)*(f1bk#[n]-meanF1bk)) + ((f2bk#[n]-meanF2bk)*(f2bk#[n]-meanF2bk)))
   endfor
   dispersionMean = mean(dispersion#)
   
   # Get additional parameters from DJW script 
   # Tier 3 is DJW interval tier for silence or sounding
   # Tier 2 is DJW point tier for acoustic syllable peaks
   # Tier 1 is sentence tier from concatenation

   # get pauses (silences)
   select textgridid
   nsentences = Get number of intervals: 1
   silencetierid = Extract tier... 3
   silencetableid = Down to TableOfReal... sounding
   nsounding = Get number of rows
   
   ## Pauses within sentences only
   nIntrasentPauses = 'nsounding'-'nsentences'
   
   ## All pauses, i.e., between and within sentences
   select textgridid
   silencetierid = Extract tier... 3
   Get starting points... silent
   npoints = Get number of points
   if npoints == 0
	nAllPauses = 0
   else
	select textgridid
	silencetierid = Extract tier... 3
	silencetableid = Down to TableOfReal... silent
    	nAllPauses = Get number of rows
   endif	

   printline 'wavfileName$', 'numpeaks', 'nIntrasentPauses', 'nAllPauses', 'dispersionMean:3'
 
   # clean up
   select all
   minus Strings wavlist
   minus Strings textgridlist
   Remove

endfor 

printline Done!