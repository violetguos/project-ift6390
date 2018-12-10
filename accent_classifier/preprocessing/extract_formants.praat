# This Praat script takes as input the name of a .wav file
# and the name of an output file, and extracts the formants
# for each voiced frame of the wav to the output text file.

form Test command line calls
	sentence First_text
	sentence Second_text
endform

sound = Read from file: first_text$
selectObject: sound
startTime = Get start time
endTime = Get end time
numberOfTimeSteps = (endTime - startTime) / 0.05

# Loop through all the demarcated segments
for step to numberOfTimeSteps
	tmin = startTime + (step - 1) * 0.05
	tmax = tmin + 0.05
	midpoint = (tmax+tmin)/2

	#extraction du part (split du son)
	sound_part = Extract part: tmin, tmax, "rectangular", 1, "yes"
    selectObject: sound_part
    this_part$ = selected$("Sound")    

    # Get energy (intensity, in dB; this is already logged)
	energy = Get intensity (dB)

	# Get the number of zero crossings in the frame
	do ("To PointProcess (zeroes)...", 1, "yes", "no")
	zcr = Get number of points

	# DO NOT REMOVE: this 'undoes' the To PointProcess... above;
	# changing will break!
	selectObject: sound
    sound_part = Extract part: tmin, tmax, "rectangular", 1, "yes"    

    # If energy is >= -9.0, and 1 < zcr < 45, we assume it's voiced
    voiced = 0
	if energy > -9.0
        if 1 < zcr
            if zcr < 45
                # Voiced! Get the formants.
                #appendInfoLine: "Voiced"
                voiced = 1
                To Formant (burg)... 0.05 5 5000 0.025 50
                select Formant 'this_part$'
                last_formant$ = selected$ ("Formant", -1)
                selectObject: "Formant "+ last_formant$
                f1 = Get value at time... 1 midpoint Hertz Linear
                f2 = Get value at time... 2 midpoint Hertz Linear
                f3 = Get value at time... 3 midpoint Hertz Linear
            endif
        endif
    endif    

    # If not voiced, just assume all formants to be 0
    if voiced == 0
        f1 = 0
        f2 = 0
        f3 = 0
    endif    

    # Write f1, f2, f3 to file
    appendFileLine: second_text$, f1, ",", f2 , ",", f3

    # DO NOT REMOVE: this reselects the original sound for the next
    # pass of the loop; changing will break!
    selectObject: sound
endfor

select all
Remove