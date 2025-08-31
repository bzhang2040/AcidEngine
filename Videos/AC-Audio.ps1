$dateString = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

$outputFolder = "AcidEngine_$dateString"
$outputFileName = "AcidEngine_$dateString.mp4"

ffmpeg -r 60 -start_number 10 -i ../Frames/ebin%05d.png -ss 0.0 -i ../Untitled.mp3 -r 60 -map 0:v -map 1:a -shortest -c:v libx264 -preset fast -b:v 20M -pix_fmt yuv420p -c:a aac -ac 2 -b:a 128k -tune zerolatency -r 60 $outputFileName
