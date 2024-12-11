$DownloadUrl = "https://www.kaggle.com/api/v1/datasets/download/prithwirajmitra/covid-face-mask-detection-dataset"
$OutputFile = "archive.zip"
$ExtractPath = (Get-Location).Path # + "\Dataset"
Write-Host "START DOWNLOADING..."
Invoke-WebRequest -Uri $DownloadUrl -OutFile $OutputFile
Write-Host "Done 👍"

# Unzip the downloaded file
Write-Host "Unzipping the archive..."
Expand-Archive -Path $OutputFile -DestinationPath $ExtractPath -Force
Rename-Item -Path "New Masks Dataset" -NewName "Dataset"
Write-Host "Unzip completed! Files extracted to $ExtractPath."

Remove-Item $OutputFile