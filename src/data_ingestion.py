import gdown
import os

# Ingest data from Google Drive
url = "https://drive.google.com/uc?id=" # Replace with your Google Drive ID
output = "Logistics data.pdf"  # The local filename

# Folder where the file would be saved
save_folder = "data"  

# Ensure the folder exists
os.makedirs(save_folder, exist_ok=True)

# Full path to save the file
output = os.path.join(save_folder, "Logistics_data.pdf")

# Download the file
gdown.download(url, output, quiet=False)

print(f"File downloaded successfully as {output}")