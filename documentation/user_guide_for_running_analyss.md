# User guide for running analysis pipeline
## **High priority sorting**
1. Acquire data in Open Ephys, and save in openephys format, and upload it to the server
2. Connect to the sorting computer using an SSH connection
3. Copy the whole recording folder including paramters.txt and dead_channels.txt and any movement information to nolanlab/to_sort/recordings

### parameters.txt
This should be added to every recording folder before the analysis is done. The first row should have the session type, which is either vr or openfield. The second line should have the location on the server starting from your name, so for example:
> openfield

> Klara/Open_field_opto_tagging_p038/M3_2018-03-13_10-44-37_of

### dead_channels.txt
This should only be made if the recording contains dead channels. Each channel id for a dead channel (1-16) should be in a new line. So for example if 1 and 3 are dead, dead_channels.txt should have
> 1

> 3

4. When the folder is fully copied, **put copied.txt in the folder** (so the script knows it's ready for sorting)

Do not ever put more than 10 folders in this folder. The sorting computer has 250GB of space, which is used for temporary sorting files in addition to your files stored here. Please always check how many folders others put in there.

## **Low priority sorting**
1. Acquire data in Open Ephys, and save in openephys format, and upload it to the server
2. Create a text file with any name and in each line put the end of the server path to a folder (same format as parameters file).
3. Copy this text file to the sorting computer using an SSH connection to nolanlab/to_sort/downtime_sort
These folders will be copied to the sorting computer one by one whenever nolanlab/to_sort/recordings is empty

Your results will be uploaded to the server based on the path you gave in the parameters file.
The sorting will be logged in sorting_log.txt that will be put in your recording folder on the server if possible. If your recording crashes, the folder name will be added to crashlist.txt that is located in nolanlab/to_sort/crashlist.txt
