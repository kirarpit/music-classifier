for genre in ./genres/*
do
	echo $genre;
	cd $genre;
	for f in *.au
		do echo $f;
		#sox $f -e signed-integer ../../wavs/${f%%.au}.wav;
		sox $f -n spectrogram -Y 200 -X 100 -m -r -o ../../spectrograms/${f%%.au}.png;
	done
	cd ../..;
done
