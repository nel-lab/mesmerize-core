import os

from caiman.utils.utils import load_dict_from_hdf5
from caiman.source_extraction.cnmf import cnmf
import numpy.testing
import pandas as pd
from mesmerize_core import (
    create_batch,
    load_batch,
    CaimanDataFrameExtensions,
    CaimanSeriesExtensions,
    set_parent_data_path,
    get_parent_data_path,
    get_full_data_path,
)
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
from zipfile import ZipFile
from hashlib import sha1

tmp_dir = Path(os.path.dirname(os.path.abspath(__file__)), "tmp")
vid_dir = Path(os.path.dirname(os.path.abspath(__file__)), "videos")
ground_truths_dir = Path(os.path.dirname(os.path.abspath(__file__)), "ground_truths")
ground_truths_file = Path(
    os.path.dirname(os.path.abspath(__file__)), "ground_truths.zip"
)

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(vid_dir, exist_ok=True)
os.makedirs(ground_truths_dir, exist_ok=True)


def _download_ground_truths():
    print(f"Downloading ground truths")
    url = f"https://zenodo.org/record/6592084/files/ground_truths.zip"

    # basically from https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests/37573701
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

    with open(ground_truths_file, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    ZipFile(ground_truths_file).extractall(ground_truths_dir.parent)


if len(list(ground_truths_dir.iterdir())) == 0:
    _download_ground_truths()

elif "DOWNLOAD_GROUND_TRUTHS" in os.environ.keys():
    if os.environ["DOWNLOAD_GROUND_TRUTHS"] == "1":
        _download_ground_truths()


def get_tmp_filename():
    return os.path.join(tmp_dir, f"{uuid4()}.pickle")


def clear_tmp():
    if "MESMERIZE_KEEP_TEST_DATA" in os.environ.keys():
        if os.environ["MESMERIZE_KEEP_TEST_DATA"] == "1":
            return

    shutil.rmtree(tmp_dir)
    shutil.rmtree(vid_dir)


def get_datafile(fname: str):
    local_path = Path(os.path.join(vid_dir, f"{fname}.tif"))
    if local_path.is_file():
        return local_path
    else:
        return download_data(fname)


def download_data(fname: str):
    """
    Download the large network files from Zenodo
    """
    url = {
        "mcorr": "https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/demoMovie.tif",
        "cnmf": None,
        "cnmfe": "https://caiman.flatironinstitute.org/~neuro/caiman_downloadables/data_endoscope.tif",
    }.get(fname)

    print(f"Downloading test data from: {url}")

    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    path = os.path.join(vid_dir, f"{fname}.tif")

    with open(path, "wb") as file:
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
    algo = "mcorr"
    df, batch_path = _create_tmp_batch()
    print(f"Testing mcorr")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert (
        get_full_data_path(df.iloc[-1]["input_movie_path"])
        == vid_dir.joinpath(f"{algo}.tif")
        == vid_dir.joinpath(df.iloc[-1]["input_movie_path"])
    )

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None,
    )
    process.wait()

    df = load_batch(batch_path)
    print(df)
    print(df.iloc[-1]["outputs"]["traceback"])
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    # test to check mmap output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap'
        )
    )

    # test to check mean-projection output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["mean-projection-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["mean-projection-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_mean_projection.npy')
    )

    # test to check std-projection output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["std-projection-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["std-projection-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_std_projection.npy')
    )

    # test to check max-projection output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["max-projection-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["max-projection-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_max_projection.npy')
    )

    # test to check correlation image output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["corr-img-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["corr-img-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_cn.npy')
    )

    # test to check mcorr get_output_path()
    assert df.iloc[-1].mcorr.get_output_path() == vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap'
    )

    # test to check mcorr get_output()
    mcorr_output = df.iloc[-1].mcorr.get_output()
    mcorr_output_actual = numpy.load(
        ground_truths_dir.joinpath("mcorr", "mcorr_output.npy")
    )
    numpy.testing.assert_array_equal(mcorr_output, mcorr_output_actual)

    # test to check caiman get_input_movie_path()
    assert df.iloc[-1].caiman.get_input_movie_path() == get_full_data_path(
        df.iloc[0]["input_movie_path"]
    )

    # test to check caiman get_correlation_img()
    mcorr_corr_img = df.iloc[-1].caiman.get_correlation_image()
    mcorr_corr_img_actual = numpy.load(
        ground_truths_dir.joinpath("mcorr", "mcorr_correlation_img.npy")
    )
    numpy.testing.assert_array_equal(mcorr_corr_img, mcorr_corr_img_actual)

    # test to check caiman get_projection("mean")
    mcorr_mean = df.iloc[-1].caiman.get_projection("mean")
    mcorr_mean_actual = numpy.load(
        ground_truths_dir.joinpath("mcorr", "mcorr_mean.npy")
    )
    numpy.testing.assert_array_equal(mcorr_mean, mcorr_mean_actual)

    # test to check caiman get_projection("std")
    mcorr_std = df.iloc[-1].caiman.get_projection("std")
    mcorr_std_actual = numpy.load(ground_truths_dir.joinpath("mcorr", "mcorr_std.npy"))
    numpy.testing.assert_array_equal(mcorr_std, mcorr_std_actual)

    # test to check caiman get_projection("max")
    mcorr_max = df.iloc[-1].caiman.get_projection("max")
    mcorr_max_actual = numpy.load(ground_truths_dir.joinpath("mcorr", "mcorr_max.npy"))
    numpy.testing.assert_array_equal(mcorr_max, mcorr_max_actual)


def test_cnmf():
    set_parent_data_path(vid_dir)
    algo = "mcorr"
    df, batch_path = _create_tmp_batch()
    print(f"Testing mcorr")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert vid_dir.joinpath(df.iloc[-1]["input_movie_path"]) == vid_dir.joinpath(
        f"{algo}.tif"
    )

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None,
    )
    process.wait()

    df = load_batch(batch_path)
    print(df)
    print(df.iloc[-1]["outputs"]["traceback"])
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000_.mmap'
        )
    )

    algo = "cnmf"
    print("Testing cnmf")
    input_movie_path = vid_dir.joinpath(df.iloc[-1]["outputs"]["mcorr-output-path"])
    df.caiman.add_item(
        algo=algo,
        name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")
    print("df input path:", df.iloc[-1]["input_movie_path"])
    assert vid_dir.joinpath(df.iloc[-1]["input_movie_path"]) == input_movie_path

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None,
    )
    process.wait()

    df = load_batch(batch_path)
    print(df)
    # Confirm output path is as expected
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    assert (
        vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}.hdf5')
        == get_full_data_path(df.iloc[-1]["outputs"]["cnmf-hdf5-path"])
        == vid_dir.joinpath(df.iloc[-1]["outputs"]["cnmf-hdf5-path"])
    )

    # test to check mmap output path
    assert (
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_cnmf-memmap__d1_60_d2_80_d3_1_order_C_frames_2000_.mmap'
        )
        == get_full_data_path(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
        == vid_dir.joinpath(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
    )

    # test to check mean-projection output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["mean-projection-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["mean-projection-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_mean_projection.npy')
    )

    # test to check std-projection output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["std-projection-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["std-projection-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_std_projection.npy')
    )

    # test to check max-projection output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["max-projection-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["max-projection-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_max_projection.npy')
    )

    # test to check correlation image output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["corr-img-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["corr-img-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_cn.npy')
    )

    print("testing cnmf.get_cnmf_memmap()")
    # test to check cnmf get_cnmf_memmap()
    cnmf_mmap_output = df.iloc[-1].cnmf.get_cnmf_memmap()
    cnmf_mmap_output_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "cnmf_output_mmap.npy")
    )
    numpy.testing.assert_array_equal(cnmf_mmap_output, cnmf_mmap_output_actual)

    print("testing cnmf.get_input_memmap()")
    # test to check cnmf get_input_memmap()
    cnmf_input_mmap = df.iloc[-1].cnmf.get_input_memmap()
    cnmf_input_mmap_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "cnmf_input_mmap.npy")
    )
    numpy.testing.assert_array_equal(cnmf_input_mmap, cnmf_input_mmap_actual)
    # cnmf input memmap from mcorr output should also equal mcorr output
    mcorr_output = df.iloc[-2].mcorr.get_output()
    numpy.testing.assert_array_equal(cnmf_input_mmap, mcorr_output)

    # test to check cnmf get_output_path()
    assert df.iloc[-1].cnmf.get_output_path() == vid_dir.joinpath(
        df.iloc[-1]["outputs"]["cnmf-hdf5-path"]
    )

    # test to check cnmf get_output()
    assert isinstance(df.iloc[-1].cnmf.get_output(), cnmf.CNMF)
    # this doesn't work because some keys in the hdf5 file are
    # not always identical, like the path to the mmap file
    # assert sha1(open(df.iloc[1].cnmf.get_output_path(), "rb").read()).hexdigest() == sha1(open(ground_truths_dir.joinpath('cnmf', 'cnmf_output.hdf5'), "rb").read()).hexdigest()

    # test to check cnmf get_spatial_masks()
    cnmf_spatial_masks = df.iloc[-1].cnmf.get_spatial_masks()
    cnmf_spatial_masks_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "spatial_masks.npy")
    )
    numpy.testing.assert_array_equal(cnmf_spatial_masks, cnmf_spatial_masks_actual)

    # test to check get_spatial_contours()
    cnmf_spatial_contours_contours = df.iloc[-1].cnmf.get_spatial_contours()[0]
    cnmf_spatial_contours_coms = df.iloc[-1].cnmf.get_spatial_contours()[1]
    cnmf_spatial_contours_contours_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "spatial_contours_contours.npy"),
        allow_pickle=True,
    )
    cnmf_spatial_contours_coms_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "spatial_contours_coms.npy"),
        allow_pickle=True,
    )
    for contour, actual_contour in zip(
        cnmf_spatial_contours_contours, cnmf_spatial_contours_contours_actual
    ):
        numpy.testing.assert_allclose(contour, actual_contour, rtol=1e-2, atol=1e-10)
    for com, actual_com in zip(
        cnmf_spatial_contours_coms, cnmf_spatial_contours_coms_actual
    ):
        numpy.testing.assert_allclose(com, actual_com, rtol=1e-2, atol=1e-10)

    # test to check get_temporal_components()
    cnmf_temporal_components = df.iloc[-1].cnmf.get_temporal_components()
    cnmf_temporal_components_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "temporal_components.npy")
    )
    numpy.testing.assert_allclose(
        cnmf_temporal_components, cnmf_temporal_components_actual, rtol=1e-2, atol=1e-10
    )

    # test to check get_reconstructed_movie()
    cnmf_reconstructed_movie = df.iloc[-1].cnmf.get_reconstructed_movie()
    cnmf_reconstructed_movie_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "reconstructed_movie.npy")
    )
    numpy.testing.assert_allclose(
        cnmf_reconstructed_movie, cnmf_reconstructed_movie_actual, rtol=1e-2, atol=1e-10
    )

    # test to check caiman get_input_movie_path()
    assert df.iloc[-1].caiman.get_input_movie_path() == get_full_data_path(
        df.iloc[-1]["input_movie_path"]
    )

    # test to check caiman get_correlation_img()
    cnmf_corr_img = df.iloc[-1].caiman.get_correlation_image()
    cnmf_corr_img_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "cnmf_correlation_img.npy")
    )
    numpy.testing.assert_array_equal(cnmf_corr_img, cnmf_corr_img_actual)

    # test to check caiman get_projection("mean")
    cnmf_mean = df.iloc[-1].caiman.get_projection("mean")
    cnmf_mean_actual = numpy.load(ground_truths_dir.joinpath("cnmf", "cnmf_mean.npy"))
    numpy.testing.assert_array_equal(cnmf_mean, cnmf_mean_actual)

    # test to check caiman get_projection("std")
    cnmf_std = df.iloc[-1].caiman.get_projection("std")
    cnmf_std_actual = numpy.load(ground_truths_dir.joinpath("cnmf", "cnmf_std.npy"))
    numpy.testing.assert_array_equal(cnmf_std, cnmf_std_actual)

    # test to check caiman get_projection("max")
    cnmf_max = df.iloc[-1].caiman.get_projection("std")
    cnmf_max_actual = numpy.load(ground_truths_dir.joinpath("cnmf", "cnmf_std.npy"))
    numpy.testing.assert_array_equal(cnmf_max, cnmf_max_actual)


def test_cnmfe():
    set_parent_data_path(vid_dir)
    df, batch_path = _create_tmp_batch()
    print(f"Testing mcorr")
    input_movie_path = get_datafile("cnmfe")
    print(input_movie_path)
    df.caiman.add_item(
        algo="mcorr",
        name=f"test-cnmfe-mcorr",
        input_movie_path=input_movie_path,
        params=test_params["mcorr"],
    )
    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None,
    )
    process.wait()

    df = load_batch(batch_path)

    # Test if pnr and cn alone work
    algo = "cnmfe"
    param_name = "cnmfe_partial"
    print(f"testing cnmfe - partial")
    input_movie_path = df.iloc[0].mcorr.get_output_path()
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[param_name],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[param_name]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert (
        vid_dir.joinpath(df.iloc[-1]["input_movie_path"])
        == vid_dir.joinpath(df.iloc[0].mcorr.get_output_path())
        == get_full_data_path(input_movie_path)
    )

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None,
    )
    process.wait()

    df = load_batch(batch_path)
    # Confirm output path is as expected
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None
    assert vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}_cnmf-memmap__d1_128_d2_128_d3_1_order_C_frames_1000_.mmap'
    ) == get_full_data_path(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
    assert vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_pn.npy') == get_full_data_path(
        df.iloc[-1]["outputs"]["pnr-image-path"]
    )
    assert vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_cn.npy') == get_full_data_path(
        df.iloc[-1]["outputs"]["corr-img-path"]
    )

    # extension tests - partial

    # test to check caiman get_correlation_image()
    corr_img = df.iloc[-1].caiman.get_correlation_image()
    corr_img_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_partial", "cnmfe_partial_correlation_img.npy")
    )
    numpy.testing.assert_allclose(corr_img, corr_img_actual, rtol=1e-1, atol=1e-10)

    # test to check caiman get_pnr_image()
    pnr_image = df.iloc[-1].caiman.get_pnr_image()
    pnr_image_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_partial", "cnmfe_partial_pnr_img.npy")
    )
    numpy.testing.assert_allclose(pnr_image, pnr_image_actual, rtol=1e2, atol=1e-10)

    # Test if running full cnmfe works
    algo = "cnmfe"
    param_name = "cnmfe_full"
    input_movie_path = df.iloc[0].mcorr.get_output_path()
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[param_name],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[param_name]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert vid_dir.joinpath(df.iloc[-1]["input_movie_path"]) == \
           vid_dir.joinpath(df.iloc[0].mcorr.get_output_path()) ==\
           get_full_data_path(input_movie_path)

    process = df.iloc[-1].caiman.run(
        batch_path=df.paths.get_batch_path(),
        backend=COMPUTE_BACKEND_SUBPROCESS,
        callbacks_finished=None,
    )
    process.wait()

    df = load_batch(batch_path)
    # Confirm output path is as expected
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None
    assert vid_dir.joinpath(
        f'{df.iloc[-1]["uuid"]}_cnmf-memmap__d1_128_d2_128_d3_1_order_C_frames_1000_.mmap'
    ) == get_full_data_path(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
    assert vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}.hdf5') == get_full_data_path(
        df.iloc[-1]["outputs"]["cnmf-hdf5-path"]
    )

    # test to check mmap output path
    assert (
        vid_dir.joinpath(
            f'{df.iloc[-1]["uuid"]}_cnmf-memmap__d1_128_d2_128_d3_1_order_C_frames_1000_.mmap'
        )
        == get_full_data_path(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
        == vid_dir.joinpath(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
    )

    # test to check mean-projection output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["mean-projection-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["mean-projection-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_mean_projection.npy')
    )

    # test to check std-projection output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["std-projection-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["std-projection-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_std_projection.npy')
    )

    # test to check max-projection output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["max-projection-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["max-projection-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_max_projection.npy')
    )

    # test to check correlation image output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["corr-img-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["corr-img-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_cn.npy')
    )

    # test to check pnr image output path
    assert (
        vid_dir.joinpath(df.iloc[-1]["outputs"]["pnr-image-path"])
        == get_full_data_path(df.iloc[-1]["outputs"]["pnr-image-path"])
        == vid_dir.joinpath(f'{df.iloc[-1]["uuid"]}_pn.npy')
    )

    # extension tests - full

    # test to check cnmf get_cnmf_memmap()
    cnmfe_mmap_output = df.iloc[-1].cnmf.get_cnmf_memmap()
    cnmfe_mmap_output_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_full_output_mmap.npy")
    )
    numpy.testing.assert_array_equal(cnmfe_mmap_output, cnmfe_mmap_output_actual)

    # test to check cnmf get_input_memmap()
    cnmfe_input_mmap = df.iloc[-1].cnmf.get_input_memmap()
    cnmfe_input_mmap_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_full_input_mmap.npy")
    )
    numpy.testing.assert_array_equal(cnmfe_input_mmap, cnmfe_input_mmap_actual)
    # cnmf input memmap from mcorr output should also equal mcorr output
    mcorr_output = df.iloc[0].mcorr.get_output()
    numpy.testing.assert_array_equal(cnmfe_input_mmap, mcorr_output)

    # test to check cnmf get_output_path()
    assert df.iloc[-1].cnmf.get_output_path() == vid_dir.joinpath(
        df.iloc[-1]["outputs"]["cnmf-hdf5-path"]
    )

    # test to check cnmf get_output()
    assert isinstance(df.iloc[-1].cnmf.get_output(), cnmf.CNMF)
    # this doesn't work because some keys in the hdf5 file are
    # not always identical, like the path to the mmap file
    # assert sha1(open(df.iloc[1].cnmf.get_output_path(), "rb").read()).hexdigest() == sha1(open(ground_truths_dir.joinpath('cnmf', 'cnmf_output.hdf5'), "rb").read()).hexdigest()

    # test to check cnmf get_spatial_masks()
    cnmfe_spatial_masks = df.iloc[-1].cnmf.get_spatial_masks()
    cnmfe_spatial_masks_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_spatial_masks.npy")
    )
    numpy.testing.assert_array_equal(cnmfe_spatial_masks, cnmfe_spatial_masks_actual)

    # test to check get_spatial_contours()
    cnmfe_spatial_contours_contours = df.iloc[-1].cnmf.get_spatial_contours()[0]
    cnmfe_spatial_contours_coms = df.iloc[-1].cnmf.get_spatial_contours()[1]
    cnmfe_spatial_contours_contours_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_spatial_contours_contours.npy"),
        allow_pickle=True,
    )
    cnmfe_spatial_contours_coms_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_spatial_contours_coms.npy"),
        allow_pickle=True,
    )
    for contour, actual_contour in zip(
            cnmfe_spatial_contours_contours, cnmfe_spatial_contours_contours_actual
    ):
        numpy.testing.assert_allclose(contour, actual_contour, rtol=1e-2, atol=1e-10)
    for com, actual_com in zip(
            cnmfe_spatial_contours_coms, cnmfe_spatial_contours_coms_actual
    ):
        numpy.testing.assert_allclose(com, actual_com, rtol=1e-2, atol=1e-10)

    # test to check get_temporal_components()
    cnmfe_temporal_components = df.iloc[-1].cnmf.get_temporal_components()
    cnmfe_temporal_components_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_temporal_components.npy")
    )
    numpy.testing.assert_allclose(
        cnmfe_temporal_components, cnmfe_temporal_components_actual, rtol=1e2, atol=1e-10
    )

    # test to check get_reconstructed_movie()
    cnmfe_reconstructed_movie = df.iloc[-1].cnmf.get_reconstructed_movie()
    cnmfe_reconstructed_movie_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_reconstructed_movie.npy")
    )
    numpy.testing.assert_allclose(
        cnmfe_reconstructed_movie, cnmfe_reconstructed_movie_actual, rtol=1e-2, atol=1e-10
    )


def test_remove_item():
    set_parent_data_path(vid_dir)
    algo = "mcorr"
    df, batch_path = _create_tmp_batch()
    print(f"Testing mcorr")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert (
        get_full_data_path(df.iloc[-1]["input_movie_path"])
        == vid_dir.joinpath(f"{algo}.tif")
        == vid_dir.joinpath(df.iloc[-1]["input_movie_path"])
    )

    df.caiman.add_item(
        algo=algo,
        name=f"test1-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["name"] == f"test1-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert (
        get_full_data_path(df.iloc[-1]["input_movie_path"])
        == vid_dir.joinpath(f"{algo}.tif")
        == vid_dir.joinpath(df.iloc[-1]["input_movie_path"])
    )
    # Check removing specific rows works
    assert df.iloc[0]["name"] == f"test-{algo}"
    assert df.iloc[1]["name"] == f"test1-{algo}"
    assert df.empty == False
    df.caiman.remove_item(index=1)
    assert df.iloc[0]["name"] == f"test-{algo}"
    assert df.isin([f"test1-{algo}"]).any().any() == False
    assert df.empty == False
    df.caiman.remove_item(index=0)
    assert df.isin([f"test-{algo}"]).any().any() == False
    assert df.isin([f"test1-{algo}"]).any().any() == False
    assert df.empty == True
