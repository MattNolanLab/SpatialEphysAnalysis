## Recording types
The analysis pipeline currently supports 4 recording session types: open field, vr, opto and sleep. 

### Open field
A typical open field recording has data from a mouse that explored a box for about 20-30 minutes. The analysis pipeline is flexible enough to support various box shapes and sizes (for example in an associative memory task).
Example data from open field session:


| open field exploration     | associative memory task |
| ----------- | ----------- |
| ![image](https://user-images.githubusercontent.com/16649631/119966116-e4d98580-bfa2-11eb-80b9-b859457b254b.png) | ![image](https://user-images.githubusercontent.com/16649631/119965779-8d3b1a00-bfa2-11eb-9a71-42ee364c345c.png)      |

### VR
VR recordings contain data from the virtual reality based linear location estimation task developed in the lab. 
This is a published task, please see Tennant et al 2018 (doi: 10.1016/j.celrep.2018.01.005.)
![image](https://user-images.githubusercontent.com/16649631/119966564-57e2fc00-bfa3-11eb-9be8-815f6f1bde0a.png)

### Sleep
Sleep recordings are performed in the home cage placed inside the open field arena. Because of the lid covering the home cage, these recordings often miss most of the motion tracking data. Sleep recordings can be longer than an hour.
![image](https://user-images.githubusercontent.com/16649631/119967041-dd66ac00-bfa3-11eb-936f-bbf40726ad06.png)

### Opto-tagging
Opto-tagging sessions are typically short (~10 minutes) and are collected while the mouse is in the home cage inside the open field. Most of these recordings include some baseline data, a series of stimulation pulses.
The light pulses are saved on this channel and correspond to the .continuous ephys data files: '100_ADC3.continuous'

| last few minutes of opto channel     | zoomed in on a pulse |
| ----------- | ----------- |
| ![image](https://user-images.githubusercontent.com/16649631/119967683-88776580-bfa4-11eb-89ef-04d333ad01c8.png) | ![image](https://user-images.githubusercontent.com/16649631/119967839-ae046f00-bfa4-11eb-8625-4bf1e251609e.png)|

Opto sessions have their own metadata file called opto_parameters.csv that contain the number of pulses, pulse durations and pulse intensities for each series of pulses sent. This is important to save because the LED intesity (manually set on the LED driver) does not get saved anywhere otherwise. If the metadata is missing the script will try to figure it out and assume 100% intensity.
For example this is a series of 200 pulses at 100% intensity. Each pulse is 3ms long followed by another series of 300 pulses at 25% intensity with 5ms durations
> pulses,intensities,duration
> 
> 200,100,3
> 
> 300, 25, 5

Open field or sleep session can contain opto data (typically at the end of the recording). In these cases, the opto metadata should be added too.


## How to set the session type
The recording type can be specified in the paramters.txt file by putting 'openfield', 'vr', 'sleep' or 'opto' in the first line.

Example parmeters file:

> opto
>
> Klara/CA1_to_deep_MEC_in_vivo/Duplicate_depth/M3_2021-05-24_14-45-35_opto

