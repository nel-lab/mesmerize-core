import os
import pandas as pd
from mesmerize_core import create_batch, load_batch, CaimanDataFrameExtensions, CaimanSeriesExtensions,\
    set_parent_data_path, get_parent_data_path, get_full_data_path
from mesmerize_core.batch_utils import DATAFRAME_COLUMNS, COMPUTE_BACKEND_SUBPROCESS
from uuid import uuid4
from typing import *
import pytest
import requests
from tqdm import tqdm
from .params import test_params
from uuid import UUID
from pathlib import Path
import shutil


tmp_dir = Path(os.path.dirname(os.path.abspath(__file__)), 'tmp')
vid_dir = Path(os.path.dirname(os.path.abspath(__file__)), 'videos')

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(vid_dir, exist_ok=True)


def get_tmp_filename():
    return os.path.join(tmp_dir, f'{uuid4()}.pickle')


def clear_tmp():
    if 'MESMERIZE_KEEP_TEST_DATA' in os.environ.keys():
        if os.environ['MESMERIZE_KEEP_TEST_DATA'] == '1':
            return

    shutil.rmtree(tmp_dir)
    shutil.rmtree(vid_dir)


def get_datafile(fname: str):
    local_path = Path(os.path.join(vid_dir, f'{fname}.tif'))
    if local_path.is_file():
        return local_path
    else:
        return download_data(fname)


def download_data(fname: str):
    """
    Download the large network files from Zenodo
    """
    url = {
        'mcorr': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/demoMovie.tif',
        'cnmf': None,
        'cnmfe': 'https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/data_endoscope.tif',
    }.get(fname)

    print(f'Downloading test data from: {url}')

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    path = os.path.join(vid_dir, f'{fname}.tif')

    with open(path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        raise ConnectionError("Couldn't download test test")

    return Path(path)


def teardown_module():
    clear_tmp()


def _create_tmp_batch() -> Tuple[pd.DataFrame, str]:
    fname = get_tmp_filename()
    df = create_batch(fname)

    return df, fname


def test_create_batch():
    df, fname = _create_tmp_batch()

    for c in DATAFRAME_COLUMNS:
        assert c in df.columns

    # test that existing batch is not overwritten
    with pytest.raises(FileExistsError):
        create_batch(fname)


def test_mcorr():
    set_parent_data_path(vid_dir)
    algo = 'mcorr'
    df, batch_path = _create_tmp_batch()
    print(f"Testing mcorr")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        name=f'test-{algo}',
        input_movie_path=input_movie_path,
        params=test_params[algo]
    )

    assert df.iloc[-1]['algo'] == algo
    assert df.iloc[-1]['name'] == f'test-{algo}'
    assert df.iloc[-1]['params'] == test_params[algo]
    assert df.iloc[-1]['outputs'] is None
    try:
        UUID(df.iloc[-1]['uuid'])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert get_full_data_path(df.iloc[-1]['input_movie_path']) == \
                  vid_dir.joinpath(f"{algo}.tif") == \
                  vid_dir.joinpath(df.iloc[-1]['input_movie_path'])

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None)
    process.wait()

    df = load_batch(batch_path)
    print(df)
    print(df.iloc[-1]['outputs']['traceback'])
    assert df.iloc[-1]['outputs']['success'] is True
    assert df.iloc[-1]['outputs']['traceback'] is None

    # test to check mmap output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['mcorr-output-path']
                        ) == \
        get_full_data_path(df.iloc[-1]['outputs']['mcorr-output-path']
                           )== \
        vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap')

    # test to check mean-projection output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['mean-projection-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['mean-projection-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_mean_projection.npy')

    # test to check std-projection output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['std-projection-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['std-projection-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_std_projection.npy')

    # test to check max-projection output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['max-projection-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['max-projection-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_max_projection.npy')

    # test to check correlation image output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['corr-img-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['corr-img-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_cn.npy')


def test_cnmf():
    set_parent_data_path(vid_dir)
    algo = 'mcorr'
    df, batch_path = _create_tmp_batch()
    print(f"Testing mcorr")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        name=f'test-{algo}',
        input_movie_path=input_movie_path,
        params=test_params[algo]
    )

    assert df.iloc[-1]['algo'] == algo
    assert df.iloc[-1]['name'] == f'test-{algo}'
    assert df.iloc[-1]['params'] == test_params[algo]
    assert df.iloc[-1]['outputs'] is None
    try:
        UUID(df.iloc[-1]['uuid'])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert vid_dir.joinpath(df.iloc[-1]['input_movie_path']) == \
           vid_dir.joinpath(f'{algo}.tif')

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None)
    process.wait()

    df = load_batch(batch_path)
    print(df)
    print(df.iloc[-1]['outputs']['traceback'])
    assert df.iloc[-1]['outputs']['success'] is True
    assert df.iloc[-1]['outputs']['traceback'] is None

    assert vid_dir.joinpath(df.iloc[-1]['outputs']['mcorr-output-path']
                        ) == \
        get_full_data_path(df.iloc[-1]['outputs']['mcorr-output-path']
                           )== \
        vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap')

    algo = 'cnmf'
    print("Testing cnmf")
    input_movie_path = vid_dir.joinpath(df.iloc[-1]['outputs']['mcorr-output-path'])
    df.caiman.add_item(
        algo=algo,
        name=f'test-{algo}',
        input_movie_path=input_movie_path,
        params=test_params[algo]
    )

    assert df.iloc[-1]['algo'] == algo
    assert df.iloc[-1]['name'] == f'test-{algo}'
    assert df.iloc[-1]['params'] == test_params[algo]
    assert df.iloc[-1]['outputs'] is None
    try:
        UUID(df.iloc[-1]['uuid'])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")
    print('df input path:', df.iloc[-1]['input_movie_path'])
    assert vid_dir.joinpath(df.iloc[-1]['input_movie_path']) == input_movie_path

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None)
    process.wait()

    df = load_batch(batch_path)
    print(df)
    # Confirm output path is as expected
    assert df.iloc[-1]['outputs']['success'] is True
    assert df.iloc[-1]['outputs']['traceback'] is None
    

    assert vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}.hdf5') == \
        get_full_data_path(df.iloc[-1]['outputs']['cnmf-hdf5-path']) == \
        vid_dir.joinpath(df.iloc[-1]['outputs']['cnmf-hdf5-path'])

    # test to check mmap output path
    assert vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}_cnmf-memmap__d1_60_d2_80_d3_1_order_C_frames_2000_.mmap') == \
           get_full_data_path(df.iloc[-1]['outputs']['cnmf-memmap-path']) == \
          vid_dir.joinpath(df.iloc[-1]['outputs']['cnmf-memmap-path']
                       )

    # test to check mean-projection output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['mean-projection-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['mean-projection-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_mean_projection.npy')

    # test to check std-projection output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['std-projection-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['std-projection-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_std_projection.npy')

    # test to check max-projection output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['max-projection-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['max-projection-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_max_projection.npy')

    # test to check correlation image output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['corr-img-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['corr-img-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_cn.npy')


def test_cnmfe():
    # Test if pnr and cn alone work
    set_parent_data_path(vid_dir)
    algo = 'cnmfe'
    param_name = 'cnmfe_partial'
    df, batch_path = _create_tmp_batch()
    print(f"testing cnmfe - partial")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        name=f'test-{algo}',
        input_movie_path=input_movie_path,
        params=test_params[param_name]
    )

    assert df.iloc[-1]['algo'] == algo
    assert df.iloc[-1]['name'] == f'test-{algo}'
    assert df.iloc[-1]['params'] == test_params[param_name]
    assert df.iloc[-1]['outputs'] is None
    try:
        UUID(df.iloc[-1]['uuid'])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert vid_dir.joinpath(df.iloc[-1]['input_movie_path']) == \
           vid_dir.joinpath(f'{algo}.tif')

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None)
    process.wait()

    df = load_batch(batch_path)
    # Confirm output path is as expected
    assert df.iloc[-1]['outputs']['success'] is True
    assert df.iloc[-1]['outputs']['traceback'] is None
    assert vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}_cnmf-memmap__d1_128_d2_128_d3_1_order_C_frames_1000_.mmap') == \
           get_full_data_path(df.iloc[-1]['outputs']['cnmf-memmap-path'])
    assert vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_pn.npy') == \
           get_full_data_path(df.iloc[-1]['outputs']['pnr-image-path'])
    assert vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_cn.npy') == \
           get_full_data_path(df.iloc[-1]['outputs']['corr-img-path'])

    # Test if running full cnmfe works
    algo = 'cnmfe'
    param_name = 'cnmfe_full'
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
    df.caiman.add_item(
         algo=algo,
         name=f'test-{algo}',
         input_movie_path=input_movie_path,
         params=test_params[param_name]
     )

    assert df.iloc[-1]['algo'] == algo
    assert df.iloc[-1]['name'] == f'test-{algo}'
    assert df.iloc[-1]['params'] == test_params[param_name]
    assert df.iloc[-1]['outputs'] is None
    try:
        UUID(df.iloc[-1]['uuid'])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert vid_dir.joinpath(df.iloc[-1]['input_movie_path']) == \
           vid_dir.joinpath(f'{algo}.tif')

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None)
    process.wait()

    df = load_batch(batch_path)
    # Confirm output path is as expected
    assert df.iloc[-1]['outputs']['success'] is True
    assert df.iloc[-1]['outputs']['traceback'] is None
    assert vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}_cnmf-memmap__d1_128_d2_128_d3_1_order_C_frames_1000_.mmap') == \
        get_full_data_path(df.iloc[-1]['outputs']['cnmf-memmap-path'])
    assert vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}.hdf5') == \
        get_full_data_path(df.iloc[-1]['outputs']['cnmf-hdf5-path'])
    
    # test to check mmap output path
    assert vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}_cnmf-memmap__d1_128_d2_128_d3_1_order_C_frames_1000_.mmap') == \
           get_full_data_path(df.iloc[-1]['outputs']['cnmf-memmap-path']) == \
          vid_dir.joinpath(df.iloc[-1]['outputs']['cnmf-memmap-path']
                       )

    # test to check mean-projection output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['mean-projection-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['mean-projection-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_mean_projection.npy')

    # test to check std-projection output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['std-projection-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['std-projection-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_std_projection.npy')

    # test to check max-projection output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['max-projection-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['max-projection-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_max_projection.npy')

    # test to check correlation image output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['corr-img-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['corr-img-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_cn.npy')
    
    # test to check pnr image output path
    assert vid_dir.joinpath(df.iloc[-1]['outputs']['pnr-image-path']) == \
        get_full_data_path(df.iloc[-1]['outputs']['pnr-image-path']) == \
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_pn.npy')


def test_remove_item():
    set_parent_data_path(vid_dir)
    algo = 'mcorr'
    df, batch_path = _create_tmp_batch()
    print(f"Testing mcorr")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        name=f'test-{algo}',
        input_movie_path=input_movie_path,
        params=test_params[algo]
    )

    assert df.iloc[-1]['algo'] == algo
    assert df.iloc[-1]['name'] == f'test-{algo}'
    assert df.iloc[-1]['params'] == test_params[algo]
    assert df.iloc[-1]['outputs'] is None
    try:
        UUID(df.iloc[-1]['uuid'])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert get_full_data_path(df.iloc[-1]['input_movie_path']) == \
                  vid_dir.joinpath(f"{algo}.tif") == \
                  vid_dir.joinpath(df.iloc[-1]['input_movie_path'])

    df.caiman.add_item(
        algo=algo,
        name=f'test1-{algo}',
        input_movie_path=input_movie_path,
        params=test_params[algo]
    )

    assert df.iloc[-1]['algo'] == algo
    assert df.iloc[-1]['name'] == f'test1-{algo}'
    assert df.iloc[-1]['params'] == test_params[algo]
    assert df.iloc[-1]['outputs'] is None
    try:
        UUID(df.iloc[-1]['uuid'])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert get_full_data_path(df.iloc[-1]['input_movie_path']) == \
                  vid_dir.joinpath(f"{algo}.tif") == \
                  vid_dir.joinpath(df.iloc[-1]['input_movie_path'])
    # Check removing specific rows works
    assert df.iloc[0]['name'] == f'test-{algo}'
    assert df.iloc[1]['name'] == f'test1-{algo}'
    assert df.empty == False
    df.caiman.remove_item(index = 1)
    assert df.iloc[0]['name'] == f'test-{algo}'
    assert df.isin([f'test1-{algo}']).any().any() == False
    assert df.empty == False
    df.caiman.remove_item(index=0)
    assert df.isin([f'test-{algo}']).any().any() == False
    assert df.isin([f'test1-{algo}']).any().any() == False
    assert df.empty == True
