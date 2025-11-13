import gdown
import os

if not os.path.exists("artifacts/model.pt"):
    print("Downloading artifacts from Google Drive...")
    gdown.download_folder(
        id="YOUR_FOLDER_ID_HERE",
        output="artifacts/",
        quiet=False
    )
    print("Artifacts downloaded!")
