
import urllib.request
import os
from tqdm import tqdm

class MyProgressBar():
    def __init__(self, filename):
        self.pbar = None
        self.filename = filename

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size, unit='iB', unit_scale=True)
            self.pbar.set_description(f"Downloading {self.filename}...")
            self.pbar.refresh()  # to show immediately the update

        self.pbar.update(block_size)



import uuid
from google_measurement_protocol import event, report

class OwnCloudDownloader():
    def __init__(self, LocalDirectory, OwnCloudServer):
        self.LocalDirectory = LocalDirectory
        self.OwnCloudServer = OwnCloudServer

        self.client_id = uuid.uuid4()

    def downloadFile(self, path_local, path_owncloud, user=None, password=None, verbose=True):
        # return 0: successfully downloaded
        # return 1: HTTPError
        # return 2: unsupported error
        # return 3: file already exist locally
        # return 4: password is None
        # return 5: user is None

        if password is None:
            print(f"password required for {path_local}")
            return 4
        if user is None:
            return 5

        if user is not None or password is not None:  
            # update Password
             
            password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_mgr.add_password(
                None, self.OwnCloudServer, user, password)
            handler = urllib.request.HTTPBasicAuthHandler(
                password_mgr)
            opener = urllib.request.build_opener(handler)
            urllib.request.install_opener(opener)

        if os.path.exists(path_local): # check existence
            if verbose:
                print(f"{path_local} already exists")
            return 2

        try:
            try:
                os.makedirs(os.path.dirname(path_local), exist_ok=True)
                urllib.request.urlretrieve(
                    path_owncloud, path_local, MyProgressBar(path_local))

            except urllib.error.HTTPError as identifier:
                print(identifier)
                return 1
        except:
            os.remove(path_local)
            raise
            return 2

        # record googleanalytics event
        data = event('download', os.path.basename(path_owncloud))
        report('UA-99166333-3', self.client_id, data)

        return 0

# https://exrcsdrive.kaust.edu.sa/exrcsdrive/index.php/s/ijkfNOmBWv9uuaz 
    
class MovieCutsDownloader(OwnCloudDownloader):
    def __init__(self, LocalDirectory,
                 OwnCloudServer="https://exrcsdrive.kaust.edu.sa/exrcsdrive/public.php/webdav/"):
        super(MovieCutsDownloader, self).__init__(
            LocalDirectory, OwnCloudServer)
        self.password = None
        self.zip_files = ["moviecuts.zip", 
                          "moviecuts.z01",
                          "moviecuts.z02",
                          "moviecuts.z03",
                          "moviecuts.z04",
                          "moviecuts.z05",
                          "moviecuts.z06",
                          "moviecuts.z07",
                          "moviecuts.z08",
                          "moviecuts.z09",
                          "moviecuts.z10",
                          "moviecuts.z11",]
        
    def downloadZipFiles(self, zip_file_index=None, verbose=True):
        if zip_file_index is None:
            download_list = self.zip_files
        else:
            download_list = [self.zip_files[zip_file_index]]

        print(f'Files to download: {download_list}')
        for zip_file in download_list:
            print(f'Downloading {zip_file}...')
            res = self.downloadFile(path_local=os.path.join(self.LocalDirectory, zip_file),
                                    path_owncloud=os.path.join(self.OwnCloudServer, zip_file).replace(
                                        ' ', '%20').replace('\\', '/'),
                                    user="ijkfNOmBWv9uuaz",
                                    password=self.password,
                                    verbose=verbose)

    def download_and_unzip(self):
        self.downloadZipFiles()
        self.unzip_moviecuts()
        
    def unzip_moviecuts(self):
        missing_files = self.check_integrity()
        while missing_files:
            for zip_file, idx in missing_files:
                self.downloadZipFiles(zip_file_index=idx)      
        os.system(f"zip -s 0 {self.LocalDirectory}/moviecuts.zip --out {self.LocalDirectory}/moviecuts_single_file.zip")
        os.system(f"unzip {self.LocalDirectory}/moviecuts_single_file.zip -d {self.LocalDirectory}")
    
    def check_integrity(self):
        """
        Check that all sel.zipfiles are present in self.LocalDirectory
        """
        missing_files = []
        for idx, zip_file in enumerate(self.zip_files):
            if not os.path.exists(os.path.join(self.LocalDirectory, zip_file)):
                missing_files.append(zip_file, idx)
        return missing_files
            
if __name__ == "__main__":

    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    # Load the arguments
    parser = ArgumentParser(description='MovieCuts Downloader',
                            formatter_class=ArgumentDefaultsHelpFormatter) 

    parser.add_argument('--moviecuts_path',   required=True,
                        type=str, help='Path to the local folder to download MovieCuts')
    parser.add_argument('--password',   required=False,
                        type=str, help='Password to download the data')
    parser.add_argument('--zip_file_index',   required=False, default=None, type=int,
                        help='Index of the zip file to download (0: moviecuts.zip, 1: moviecuts.z01, ...)')
    parser.add_argument('--download_and_unzip',   required=False, action='store_true',)
    args = parser.parse_args()

    Downloader = MovieCutsDownloader(args.moviecuts_path)
    Downloader.password = args.password
    if args.download_and_unzip:
        Downloader.download_and_unzip()
    else:
        Downloader.downloadZipFiles(zip_file_index=args.zip_file_index, verbose=True)
   