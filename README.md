# Extracellular electrophysiology analysis for tetrode data


## Overview
Analysis for in vivo electrophysiology recordings saved in open ephys format. 

The current pipeline runs on a linux computer and uses MountainSort 3 to automatically sort data. The analysis is set up to minimize user interaction as much as possible, and analyses are initiated by copying files to a designated computer. Sorted clusteres are automatically curated and output plots of spatial firing properties are generated. Spatial scores (grid score, HD score, speed score) are calculated and saved in pickle (pandas) data frames.

The main script (control_sorting_analysis.py) monitors a designated folder (nolanlab/to_sort/recordings) on the computer, and calls all processing scripts if users put recordings in this folder (and added a copied.txt file as well to indicate that copying is complete).
Another option is to add a text file with a list of folders on the server that the script will copy when the 'priority' sorting folder is empty.


(1) OpenEphys continuous files are converted to mda format (in Python) both tetrode by tetrode (4 files) and all 16 channels together into one mda file. The folder structure required by MountainSort (MS) is created in this step, dead channels are removed.

(2) MountainSort (MS) is called (via a shell script written by the previous Python step) to perform spike sorting in the mda files, and saves the results in the local folder

(3) Post-processing is done. This makes plots of firing fields, light stimulation plots depending on the data, and saves the output on the lab's server based on a parameter file that's saved by the user in the original recording folder.

Please see more detailed documentation in the /documentation folder.

### Example output figures
![image](https://user-images.githubusercontent.com/16649631/123976239-e806cd80-d9b5-11eb-839b-28c86352e201.png)


## How to contribute
Please submit an issue to discuss.

## Contributors

[<img src="https://avatars.githubusercontent.com/u/16649631?v=4" width="100" height="100">](https://github.com/klaragerlei)
[<img src="https://avatars.githubusercontent.com/u/6987144?v=4" width="100" height="100">](https://github.com/stennant)
[<img src="https://avatars.githubusercontent.com/u/28258157?v=4" width="100" height="100">](https://github.com/HDClark94)
[<img src="https://avatars.githubusercontent.com/u/3406709?v=4" width="100" height="100">](https://github.com/teristam)
[<img src="https://avatars.githubusercontent.com/u/8053216?v=4" width="100" height="100">](https://github.com/4iar)
[<img src="https://avatars.githubusercontent.com/u/46969515?v=4" width="100" height="100">](https://github.com/JesPass)
[<img src="https://avatars.githubusercontent.com/u/20047754?v=4" width="100" height="100">](https://github.com/TizzyAnastasia)
[<img src="https://avatars.githubusercontent.com/u/37214499?v=4" width="100" height="100">](https://github.com/BriVandrey)
[<img src="https://avatars.githubusercontent.com/u/6878017?v=4" width="100" height="100">](https://github.com/vzickus)
