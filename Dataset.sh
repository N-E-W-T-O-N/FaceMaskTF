#!/bin/bash
curl -L -o archive.zip\
  https://www.kaggle.com/api/v1/datasets/download/prithwirajmitra/covid-face-mask-detection-dataset


ExtractPath="$(pwd)/Dataset"

echo "START DOWNLOADING..."
wget --no-verbose --output-document="archive.zip" "$DownloadUrl"
echo "Done 👍"

# Unzip the downloaded file
echo "Unzipping the archive..."
unzip -o "archive.zip" -d "$ExtractPath"
echo "Unzip completed! Files extracted to $ExtractPath."

rm "archive.zip"  