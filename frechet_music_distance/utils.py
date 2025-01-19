from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm

KB = 1024
MB = 1024 * KB


def download_file(url: str, destination: Union[str, Path], verbose: bool = True, chunk_size: int = 10 * MB) -> None:
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("content-length", 0))

            if verbose:
                progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if verbose:
                        progress_bar.update(len(chunk))
                    file.write(chunk)

            if verbose:
                progress_bar.close()

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file from url: {url}. Error: {e}")
