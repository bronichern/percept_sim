#########################################################################################################
## Concatenate wav+textgrids                                                                           ##
##                                                                                                     ##
## IMPORTANT NOTES                                                                                     ##
##  * Works on a top-level directory with sub-directories that contain wav and textgrid files          ##
##  * No pause between files                                                                           ##
##  * Will resamples to 22050 Hz if needed (all files must be same sampling rate for concatenation )   ##                                                            ##
##  * TextGrids must already exist                                                                     ##
##                                                                                                     ##
## Ann Bradlow, January 2023                                                                           ##
#########################################################################################################

# Initialize directory list
form Concatenate DJW wav+textgrids and Calculate Vowel Dispersion
   comment Top-level directory with directories of wav and TextGrid files for concatenation):
   text toplevelDir 
   comment Target directory for concatenated wav and TextGrid files:
   text targetDir 
endform

# Create list of directories to work with 
Create Strings as directory list: "directoryList", toplevelDir$
numberOfFolders = Get number of strings


for ifolder to numberOfFolders
   
   # read and concatenate .wav 
   select Strings directoryList
   setName$ = Get string... ifolder
   directory$ = toplevelDir$ + "/" + setName$ + "/"
  
   Create Strings as file list... list 'directory$'/*.wav
   numberOfFiles = Get number of strings

   for ifile to numberOfFiles
      select Strings list
      sound$ = Get string... ifile
      Read from file... 'directory$''sound$'
      
      # resample to 22050 if necessary      
      srOrig = Get sampling frequency
      if (srOrig <> 22050)
	 current = selected("Sound")
         resample = Resample: 22050, 50
         printline resampled, 'sound$'       
         select current
         Remove
      endif
   endfor  

   select all 
   minus Strings list
   minus Strings directoryList
   Concatenate recoverably

   # Rename concatenated wav files
   select Sound chain
   Rename... concatenated
   select TextGrid chain
   Rename... concatenated
   select Strings list
   Rename... wavlist

   # clean up to prepare for DJW TextGrid concatenation
   select all 
   minus Strings wavlist
   minus Sound concatenated
   minus TextGrid concatenated
   minus Strings directoryList
   Remove

   # read and concatenate .textgrid files
   Create Strings as file list... list 'directory$'/*.TextGrid
   numberOfFiles = Get number of strings

   for ifile to numberOfFiles
      select Strings list
      textgrid$ = Get string... ifile
      Read from file... 'directory$''textgrid$'
   endfor  

   select Strings list
   Rename... textgridlist

   select all 
   minus Strings wavlist
   minus Strings textgridlist
   minus Sound concatenated
   minus TextGrid concatenated
   minus Strings directoryList
   Concatenate

   # Merge concatenation TextGrid and DJW TextGrids
   select TextGrid concatenated
   plus TextGrid chain
   Merge

   # Write concatenated files (wav and textgrid)
   select Sound concatenated
   Save as WAV file: targetDir$ + "/" + setName$ + "_cat.wav"
   select TextGrid merged
   Write to text file: targetDir$ + "/" + setName$ + "_cat.TextGrid"

   # Clean up
   select all 
   minus Strings directoryList
   Remove

endfor

printline Done!
