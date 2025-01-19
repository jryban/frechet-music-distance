from abc import ABC, abstractmethod
from typing import Union, Callable, Iterable, Any, Optional
from pathlib import Path
from functools import reduce
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool as ProcessPool
from tqdm import tqdm


class DataLoader(ABC):

    def __init__(self, supported_extensions: tuple[str], verbose: bool = True):
        self.verbose = verbose
        self.supported_extensions = supported_extensions

    @abstractmethod
    def load_file(self, filepath: Union[str, Path]) -> Any:
        pass

    def load_dataset(self, dataset: Union[str, Path]):
        if self.verbose:
            print(f"Loading files from {dataset}")
        file_paths = self.get_file_paths(dataset)
        return self._load_files(file_paths)
    
    def load_dataset_async(self, dataset: Union[str, Path]):
        if self.verbose:
            print(f"Loading files from {dataset}")
        file_paths = self.get_file_paths(dataset)
        return self._load_files_async(file_paths)

    def get_file_paths(self, dataset_path: Union[str, Path]) -> Iterable[str]:
        dataset_path = Path(dataset_path)
        file_paths = reduce(
            lambda acc, arr: acc + arr,
            [[str(f) for f in dataset_path.rglob(f'**/*{file_ext}')] for file_ext in self.supported_extensions]
        )
        return file_paths
    
    def _load_files(self, file_paths: Iterable[str]) -> Iterable[Any]:
        results = []

        pbar = tqdm(total=len(file_paths), disable=(not self.verbose))

        for filepath in file_paths:
            res = self.load_file(filepath)
            results.append(res)
            pbar.update()
   

        return results

    def _load_files_async(self, file_paths: Iterable[str]) -> Iterable[Any]:
        task_results = []

        pool = ProcessPool()
        pbar = tqdm(total=len(file_paths), disable=(not self.verbose))

        for filepath in file_paths:
            res = pool.apply_async(
                self.load_file,
                args=(filepath,),
                callback=lambda *args, **kwargs: pbar.update(),
            )
            task_results.append(res)
        pool.close()
        pool.join()

        return [task.get() for task in task_results]
    
    def _validate_file(self, filepath: Union[str, Path]) -> None:
        ext = Path(filepath).suffix
        if ext not in self.supported_extensions:
            raise ValueError(f"{self} supports the following extensions: {self.supported_extensions}, but got: {ext}")
        
    def get_dataset_ext(self, dataset_path: Union[str, Path]) -> Optional[str]:
        for file in Path(dataset_path).rglob("*"):
            if file.suffix in self.supported_extensions:
                return file.suffix
        return None
