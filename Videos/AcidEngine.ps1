$dateString = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

$outputFolder = "AcidEngine_$dateString"
$outputFileName = "AcidEngine_$dateString.mp4"

ffmpeg -r 60 -start_number 10 -i ../Frames/ebin%05d.png -r 60 -c:v libx264 -preset fast -b:v 20M -pix_fmt yuv420p -tune zerolatency -r 60 $outputFileName

New-Item -Path $outputFolder -ItemType Directory

Copy-Item -Path "../Shaders/*" -Destination $outputFolder


$content = Get-Content -Path "../Frames/START_FRAME.txt"

Copy-Item -Path $outputFileName -Destination "../VegasClips/$($content).mp4"