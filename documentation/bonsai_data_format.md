
## Bonsai output file

For open field recordings, Bonsai saves the position of two beads on the headstage of the animal and the intensity of an LED used to synchronize the position data in Bonsai with the electrophysiology data in OpenEphys.

the csv file saved by Bonsai contains the following information in each line:
- date of recording
- 'T'
- exact time of given line
- x position of left side bead on headstage
- y position of left side bead on headstage
- x position of right side bead on headstage
- y position of right side bead on headstage
- intensity of sync LED

example line:
2018-03-06T15:34:39.8242304+00:00 106.0175 134.4123 114.1396 148.1054 1713 
