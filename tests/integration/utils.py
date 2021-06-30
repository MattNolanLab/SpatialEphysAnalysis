import subprocess
import shutil
from pathlib import Path
from functools import wraps
from typing import Callable


def is_running_in_docker() -> bool:
    """Returns true if the script is being ran in a Docker container"""
    return Path('/.dockerenv').exists()


def require_docker(func: Callable) -> Callable:
    """A decorator that throws an EnvironmentError if the wrapped function is not called from a Docker environment"""
    @wraps(func)
    def wrap(*args, **kwargs) -> Callable:
        if is_running_in_docker():
            return func(*args, **kwargs)
        else:
            raise EnvironmentError('These integration test functions are potentially destructive and should only be run in a temporary Docker container/CI')
    return wrap


@require_docker
def set_up(raw_output_path: str, raw_base_sorting_path: str, data_url: str, data_password: str) -> None:
    """Fetches the integration test data and sets up the test folders"""
    output_path = Path(raw_output_path)
    base_sorting_path = Path(raw_base_sorting_path)

    # If the specific analysis folder already exists then it means we might be overwriting actual data somehow
    # in a non temporary environment so we'll explicitly unset exists_ok.
    output_path.mkdir(parents=True, exist_ok=False)

    sorting_log_txt = output_path / 'sorting_log.txt'
    sorting_log_txt.touch()

    # Create a fresh sorting folder just in case (1) previous tests have somehow interfered with the sorting files
    # or (2) an failed analysis was not deleted (which would hang up the current test)
    try:
        shutil.rmtree(base_sorting_path)
    except FileNotFoundError:
        pass

    base_sorting_path.mkdir(parents=True)

    recordings_path = base_sorting_path / 'recordings'
    recordings_path.mkdir()

    sorting_files_source = Path('./sorting_files')  # Sorting files are root of the repo, which should also be our current working directory
    sorting_files_destination = base_sorting_path / 'sorting_files'
    shutil.copytree(sorting_files_source, sorting_files_destination)

    try:
        # With the folders set up, we're ready to fetch the test data.
        # This cryptic shell command downloads data (curl), decrypts it, and unzips it (tar). We don't have to worry about creating
        # intermediary files because the pipes (| characters) pass the data between each command using the magic of linux.
        subprocess.check_call(
            f'curl {data_url} | gpg --batch --passphrase {data_password} -d | tar xvzf - -C {recordings_path}',
            shell=True)
    except subprocess.CalledProcessError as ex:
        raise Exception(f'Encountered an error while fetching the integration test data') from ex

    env = {
        'SERVER_PATH_FIRST_HALF': '/ActiveProjects/',
        'SINGLE_RUN': 'true',
        'HEATMAP_CONCURRENCY': '1',
    }

    try:
        subprocess.check_call('bash -lc "conda activate env && python3 control_sorting_analysis.py"', env=env,
                              shell=True)
    except subprocess.CalledProcessError as ex:
        raise Exception('Failed to successfully run the sorting pipeline') from ex


@require_docker
def tear_down(raw_output_path: str) -> None:
    """Perform cleanup operations

    This copies result files to the artifacts directory so we can look at them later
    """
    output_path = Path(raw_output_path)
    data_dest = Path('artifacts') / output_path.name
    shutil.copytree(output_path, data_dest)
