for genre in *
do
	echo $genre;
	cd $genre;
	for f in *.au
		do echo $f;
		#sox $f -e signed-integer ${f%%.au}.wav;
		sox $f -n spectrogram -Y 200 -X 50 -m -r -o ../spectrograms/${f%%.au}.png;
	done
	cd ..;
done