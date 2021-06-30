## Why sort together
In some experiments we are interested in the activity of neurons across behaviours. If the recordings are sorted separately, the output cluster will not match. To address this, we can concatenate the raw ephys data so the spike sorter uses the data from all recordings for the sorting. We can then split the output results based on timestamps and run post-soritng analyses on the recordings separately.

## Analysis workflow
![pipeline_sleep_opto2](https://user-images.githubusercontent.com/16649631/119969079-0720d280-bfa6-11eb-87e3-9ea520598bd1.png)

## How to run analysis
You only need to call the pipeline for one of the recordings and list all the recordings you want it to be combined with in th parameters.txt file
For example, this open field recording will be combined with an 'opto' and a 'sleep' recording.

> openfield
> 
> Klara/CA1_to_deep_MEC_in_vivo/M1_2021-03-10_14-10-09_of
> 
> paired=Klara/CA1_to_deep_MEC_in_vivo/M1_2021-03-10_14-41-25_opto,Klara/CA1_to_deep_MEC_in_vivo/M1_2021-03-10_14-57-39_sleep
