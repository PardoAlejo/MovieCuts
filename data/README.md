# Data

**PLEASE READ!**

If you are interested in the data fill this [Google form](https://forms.gle/FUNnZ8wpYRCspTJq9), and in a couple of days I'll send directly to you the  links to download the data. 

## Data Download

After receiving the instruction via email, you can download the data from the web interface. The other option is to use the provided script `moviecuts_downloader.py`, you can use it as follows:

```bash 
python moviecuts_downloader.py --moviecuts_path {PATH_TO_DOWNLOAD_MOVIECUTS} --download_link {LINK} --password {PASSWORD} 
```

LINK and PASSWORD are the same provided in the email's instructions.

The script has several options:
- `--zip_file_index`: To download a specific zip file, if not provided the script will download one by one.
- `--download_and_unzip`: If provided the script will download and unzip the data afterwards.
- `--unzip`: If provided the script will attempt to unzip the data, not that it will check that all zip files are in the folder, if not, it will throw an error.

## Videos and Annotations:

*VIDEOS*: To request access to the videos, please fill up [this form](https://forms.gle/FUNnZ8wpYRCspTJq9), agree with all the terms and you will receive and email with a link to access the data.

After receiving the link, please download each one of the zip files (the zip file is partionioed acrross 10 zip files). You can also use the script above to do this process by passing the option --unzip.

After all the files are downloaded (the have to be 12 of them), run the following to combine the files into a single zip:

` zip -s 0 moviecuts.zip --out moviecuts_single_file.zip `

Then you can simply unzip the folder and place it under data:

`unzip moviecuts_single.zip -d ./data/ `
