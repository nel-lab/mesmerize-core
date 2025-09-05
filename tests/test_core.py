import os

import numpy as np
from caiman.utils.utils import load_dict_from_hdf5
from caiman.source_extraction.cnmf.cnmf import CNMF
from caiman.base.rois import extract_binary_masks_from_structural_channel
import numpy.testing
import pandas as pd
from mesmerize_core import (
    create_batch,
    load_batch,
    CaimanDataFrameExtensions,
    CaimanSeriesExtensions,
    set_parent_raw_data_path,
)
from mesmerize_core.batch_utils import (
    DATAFRAME_COLUMNS,
    COMPUTE_BACKEND_SUBPROCESS,
    get_full_raw_data_path,
)
from mesmerize_core.utils import IS_WINDOWS
from uuid import uuid4
import pytest
import requests
from tqdm import tqdm
from .params import test_params
from uuid import UUID
from pathlib import Path
import shutil
from zipfile import ZipFile
from pprint import pprint
from mesmerize_core.caiman_extensions import cnmf
import time
import tifffile
from copy import deepcopy

# don't call "resolve" on these - want to make sure we can handle non-canonical paths correctly
testdata_dir = Path(os.path.dirname(os.path.abspath(__file__)), "test data")
tmp_dir = testdata_dir / "tmp"
vid_dir = testdata_dir / "videos"
seed_dir = testdata_dir / "seeds"
ground_truths_dir = testdata_dir / "ground_truths"
ground_truths_file = testdata_dir / "ground_truths.zip"

os.makedirs(tmp_dir, exist_ok=True)
os.makedirs(vid_dir, exist_ok=True)
os.makedirs(seed_dir, exist_ok=True)
os.makedirs(ground_truths_dir, exist_ok=True)


def _download_ground_truths():
    print(f"Downloading ground truths")
    url = f"https://zenodo.org/records/17059175/files/ground_truths.zip"

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
    # add a $ (legal on both UNIX and Windows) to ensure we are escaping it correctly
    return os.path.join(tmp_dir, f"{uuid4()}$test.pickle")


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
    }[fname]

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
    if IS_WINDOWS:  # because windows is stupid with permissions
        return
    clear_tmp()


def _create_tmp_batch() -> tuple[pd.DataFrame, str]:
    fname = get_tmp_filename()
    df = create_batch(fname)

    return df, fname


def make_test_seed(input_data: np.ndarray):
    """Function call used to create Ain for testing"""
    mean_proj = np.mean(input_data, axis=0)
    return extract_binary_masks_from_structural_channel(mean_proj, gSig=3)[0]


def test_create_batch():
    df, fname = _create_tmp_batch()

    for c in DATAFRAME_COLUMNS:
        assert c in df.columns

    # test that existing batch is not overwritten
    with pytest.raises(FileExistsError):
        create_batch(fname)


def test_mcorr():
    set_parent_raw_data_path(vid_dir)
    algo = "mcorr"
    df, batch_path = _create_tmp_batch()

    batch_path = Path(batch_path)
    batch_dir = batch_path.parent

    print(f"Testing mcorr")
    input_movie_path = get_datafile(algo)
    print(input_movie_path)
    df.caiman.add_item(
        algo=algo,
        item_name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["item_name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    # test that batch path is propagated to pd.Series
    assert (
        df.attrs["batch_path"].resolve()
        == df.paths.get_batch_path()
        == df.iloc[-1].paths.get_batch_path()
        == df.iloc[-1].attrs["batch_path"].resolve()
    )

    # test that path resolve works for parent_raw_dir
    rel_input_movie_path = input_movie_path.relative_to(vid_dir)
    assert (
        df.paths.resolve(rel_input_movie_path)
        == df.iloc[-1].paths.resolve(rel_input_movie_path)
        == input_movie_path.resolve()
    )
    # test that path splitting works for parent_raw_dir
    vid_dir_canon = vid_dir.resolve()
    split = (vid_dir_canon, input_movie_path.resolve().relative_to(vid_dir_canon))
    assert (
        df.paths.split(input_movie_path)
        == df.iloc[-1].paths.split(input_movie_path)
        == split
    )
    # test that the input_movie_path in the DataFrame rows are relative
    assert Path(df.iloc[-1]["input_movie_path"]) == split[1]

    assert (
        get_full_raw_data_path(df.iloc[-1]["input_movie_path"])
        == vid_dir_canon.joinpath(f"{algo}.tif")
        == vid_dir_canon.joinpath(df.iloc[-1]["input_movie_path"])
        == df.paths.resolve(df.iloc[-1]["input_movie_path"])
    )

    process = df.iloc[-1].caiman.run()
    # process.wait()

    df = load_batch(batch_path)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    pprint(df.iloc[-1]["outputs"], width=-1)
    print(df.iloc[-1]["outputs"]["traceback"])
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    # test that path resolve works for batch_dir
    mcorr_memmap_path = batch_dir.joinpath(
        str(df.iloc[-1]["uuid"]),
        f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000.mmap',
    )
    rel_mcorr_memmap_path = mcorr_memmap_path.relative_to(batch_dir)
    mcorr_memmap_path_canon = mcorr_memmap_path.resolve()

    assert (
        df.paths.resolve(rel_mcorr_memmap_path)
        == df.iloc[-1].paths.resolve(rel_mcorr_memmap_path)
        == mcorr_memmap_path_canon
    )
    # test that path splitting works for batch_dir
    batch_dir_canon = batch_dir.resolve()
    split = (batch_dir_canon, mcorr_memmap_path_canon.relative_to(batch_dir_canon))
    assert (
        df.paths.split(mcorr_memmap_path)
        == df.iloc[-1].paths.split(mcorr_memmap_path)
        == split
    )

    assert (
        input_movie_path.resolve()
        == df.iloc[-1].caiman.get_input_movie_path()
        == df.paths.resolve(df.iloc[-1]["input_movie_path"])
    )

    # test to check mmap output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]),
            f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000.mmap',
        )
    )

    # test to check shifts output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["shifts"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["shifts"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_shifts.npy'
        )
    )

    # test to check mean-projection output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["mean-projection-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["mean-projection-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_mean_projection.npy'
        )
    )

    # test to check std-projection output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["std-projection-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["std-projection-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_std_projection.npy'
        )
    )

    # test to check max-projection output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["max-projection-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["max-projection-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_max_projection.npy'
        )
    )

    # test to check correlation image output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["corr-img-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["corr-img-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_cn.npy'
        )
    )

    # test to check mcorr get_output_path()
    assert (
        df.iloc[-1].mcorr.get_output_path()
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]),
            f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000.mmap',
        )
        == df.paths.resolve(df.iloc[-1]["outputs"]["mcorr-output-path"])
    )

    # test to check mcorr get_output()
    mcorr_output = df.iloc[-1].mcorr.get_output()
    mcorr_output_actual = numpy.load(
        ground_truths_dir.joinpath("mcorr", "mcorr_output.npy")
    )
    numpy.testing.assert_array_equal(mcorr_output, mcorr_output_actual)

    # test to check mcorr get_shifts()
    mcorr_shifts = df.iloc[-1].mcorr.get_shifts(
        pw_rigid=test_params[algo]["main"]["pw_rigid"]
    )
    mcorr_shifts_actual = numpy.load(
        ground_truths_dir.joinpath("mcorr", "mcorr_shifts.npy")
    )
    numpy.testing.assert_array_equal(mcorr_shifts, mcorr_shifts_actual)

    # test to check caiman get_input_movie_path()
    assert df.iloc[-1].caiman.get_input_movie_path() == get_full_raw_data_path(
        df.iloc[0]["input_movie_path"]
    )

    # test to check caiman get_correlation_img()
    mcorr_corr_img = df.iloc[-1].caiman.get_corr_image()
    mcorr_corr_img_actual = numpy.load(
        ground_truths_dir.joinpath("mcorr", "mcorr_correlation_img.npy")
    )
    numpy.testing.assert_allclose(
        mcorr_corr_img, mcorr_corr_img_actual, rtol=1e-2, atol=1e-10
    )

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
    set_parent_raw_data_path(vid_dir)
    algo = "mcorr"

    df, batch_path = _create_tmp_batch()

    batch_path = Path(batch_path)
    batch_dir = batch_path.parent
    batch_dir_canon = batch_dir.resolve()

    input_movie_path = get_datafile(algo)
    print(input_movie_path)

    df.caiman.add_item(
        algo=algo,
        item_name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["item_name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    vid_dir_canon = vid_dir.resolve()
    assert vid_dir_canon.joinpath(
        df.iloc[-1]["input_movie_path"]
    ) == vid_dir_canon.joinpath(f"{algo}.tif")

    process = df.iloc[-1].caiman.run()
    # process.wait()

    df = load_batch(batch_path)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    pprint(df.iloc[-1]["outputs"], width=-1)
    print(df.iloc[-1]["outputs"]["traceback"])
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]),
            f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000.mmap',
        )
    )

    algo = "cnmf"
    print("Testing cnmf")
    input_movie_path = df.iloc[-1].mcorr.get_output_path()
    df.caiman.add_item(
        algo=algo,
        item_name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["item_name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")
    print("cnmf input_movie_path:", df.iloc[-1]["input_movie_path"])
    assert batch_dir_canon.joinpath(df.iloc[-1]["input_movie_path"]) == input_movie_path

    process = df.iloc[-1].caiman.run()
    # process.wait()

    df = load_batch(batch_path)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    pprint(df.iloc[-1]["outputs"], width=-1)
    print(df.iloc[-1]["outputs"]["traceback"])

    # Confirm output path is as expected
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    assert (
        input_movie_path
        == df.iloc[-1].caiman.get_input_movie_path()
        == df.paths.resolve(df.iloc[-1]["input_movie_path"])
    )

    assert (
        batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}.hdf5'
        )
        == df.paths.resolve(df.iloc[-1]["outputs"]["cnmf-hdf5-path"])
        == batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["cnmf-hdf5-path"])
    )

    # test to check mmap output path
    assert (
        batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]),
            f'{df.iloc[-1]["uuid"]}_cnmf-memmap_d1_60_d2_80_d3_1_order_C_frames_2000.mmap',
        )
        == df.paths.resolve(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
        == batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
    )

    # test to check mean-projection output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["mean-projection-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["mean-projection-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_mean_projection.npy'
        )
    )

    # test to check std-projection output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["std-projection-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["std-projection-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_std_projection.npy'
        )
    )

    # test to check max-projection output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["max-projection-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["max-projection-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_max_projection.npy'
        )
    )

    # test to check correlation image output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["corr-img-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["corr-img-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_cn.npy'
        )
    )

    print("testing cnmf.get_cnmf_memmap()")
    # test to check cnmf get_cnmf_memmap()
    cnmf_mmap_output = df.iloc[-1].cnmf.get_cnmf_memmap()
    cnmf_mmap_output_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "cnmf_output_mmap.npy")
    )
    numpy.testing.assert_array_equal(cnmf_mmap_output, cnmf_mmap_output_actual)

    print("testing caiman.get_input_movie() for cnmf")
    cnmf_input_mmap = df.iloc[-1].caiman.get_input_movie()
    cnmf_input_mmap_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "cnmf_input_mmap.npy")
    )
    numpy.testing.assert_array_equal(cnmf_input_mmap, cnmf_input_mmap_actual)
    # cnmf input memmap from mcorr output should also equal mcorr output
    mcorr_output = df.iloc[-2].mcorr.get_output()
    numpy.testing.assert_array_equal(cnmf_input_mmap, mcorr_output)

    # test to check cnmf get_output_path()
    assert (
        df.iloc[-1].cnmf.get_output_path()
        == batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["cnmf-hdf5-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["cnmf-hdf5-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}.hdf5'
        )
    )

    # test to check cnmf get_output()
    assert isinstance(df.iloc[-1].cnmf.get_output(), CNMF)
    # this doesn't work because some keys in the hdf5 file are
    # not always identical, like the path to the mmap file
    # assert sha1(open(df.iloc[1].cnmf.get_output_path(), "rb").read()).hexdigest() == sha1(open(ground_truths_dir.joinpath('cnmf', 'cnmf_output.hdf5'), "rb").read()).hexdigest()

    # test to check cnmf get_masks()
    cnmf_spatial_masks = df.iloc[-1].cnmf.get_masks("good")
    cnmf_spatial_masks_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "spatial_masks.npy")
    )
    numpy.testing.assert_array_equal(cnmf_spatial_masks, cnmf_spatial_masks_actual)

    # test to check get_contours()
    cnmf_spatial_contours_contours = df.iloc[-1].cnmf.get_contours("good")[0]
    cnmf_spatial_contours_coms = df.iloc[-1].cnmf.get_contours("good")[1]
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

    # test to check get_temporal()
    cnmf_temporal_components = df.iloc[-1].cnmf.get_temporal("good")
    cnmf_temporal_components_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "temporal_components.npy")
    )
    numpy.testing.assert_allclose(
        cnmf_temporal_components, cnmf_temporal_components_actual, rtol=1e-2, atol=1e-10
    )

    # test to check get_rcm()
    cnmf_reconstructed_movie_AouterC = df.iloc[-1].cnmf.get_rcm("all")
    cnmf_reconstructed_movie_AouterC_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "reconstructed_movie_new.npy")
    )
    numpy.testing.assert_allclose(
        cnmf_reconstructed_movie_AouterC.as_numpy(),
        cnmf_reconstructed_movie_AouterC_actual,
        rtol=1e-1,
        atol=1e-10,
    )

    # test that get_item is working properly for LazyArrays
    for i in np.random.randint(
        10, cnmf_reconstructed_movie_AouterC_actual.shape[0] - 11, size=10
    ):
        numpy.testing.assert_allclose(
            cnmf_reconstructed_movie_AouterC[i],
            cnmf_reconstructed_movie_AouterC_actual[i],
            rtol=1e-1,
            atol=1e-10,
        )
    for i in np.random.randint(
        10, cnmf_reconstructed_movie_AouterC_actual.shape[0] - 11, size=10
    ):
        numpy.testing.assert_allclose(
            cnmf_reconstructed_movie_AouterC[i - 5 : i + 5],
            cnmf_reconstructed_movie_AouterC_actual[i - 5 : i + 5],
            rtol=1e-1,
            atol=1e-10,
        )

    # test to check get_rcb()
    cnmf_reconstructed_background = df.iloc[-1].cnmf.get_rcb()
    cnmf_reconstructed_background_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "reconstructed_background.npy")
    )
    numpy.testing.assert_allclose(
        cnmf_reconstructed_background.as_numpy(),
        cnmf_reconstructed_background_actual,
        rtol=1e-2,
        atol=1e-10,
    )

    # test to check get_residuals()
    cnmf_residuals = df.iloc[-1].cnmf.get_residuals()
    # I think something is wrong with the residuals groundtruth file
    # cnmf_residuals_actual = numpy.load(ground_truths_dir.joinpath("cnmf", "residuals.npy"))
    numpy.testing.assert_allclose(
        cnmf_residuals.as_numpy(),
        df.iloc[-1].caiman.get_input_movie()
        - cnmf_reconstructed_movie_AouterC_actual
        - cnmf_reconstructed_background_actual,
        rtol=1e2,
        atol=1e-5,
    )

    # test to check caiman get_input_movie_path(), should be output of previous mcorr
    assert (
        df.iloc[-1].caiman.get_input_movie_path()
        == df.paths.resolve(df.iloc[-1]["input_movie_path"])
        == batch_dir_canon.joinpath(df.iloc[-1]["input_movie_path"])
    )

    # test to check caiman get_correlation_img()
    cnmf_corr_img = df.iloc[-1].caiman.get_corr_image()
    cnmf_corr_img_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "cnmf_correlation_img.npy")
    )
    numpy.testing.assert_allclose(
        cnmf_corr_img, cnmf_corr_img_actual, rtol=1e-5, atol=1e-5
    )

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

    # test to check passing optional ixs components to various functions
    ixs_components = numpy.array([1, 3, 5, 2])

    # test to check ixs components for cnmf.get_masks()
    ixs_spatial_masks = df.iloc[-1].cnmf.get_masks(ixs_components)
    ixs_spatial_masks_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "cnmf_ixs", "ixs_spatial_masks.npy"),
        allow_pickle=True,
    )
    numpy.testing.assert_array_equal(ixs_spatial_masks, ixs_spatial_masks_actual)

    # test to check ixs components for cnmf.get_contours()
    ixs_contours_contours = df.iloc[-1].cnmf.get_contours(ixs_components)[0]
    ixs_contours_contours_actual = numpy.load(
        ground_truths_dir.joinpath(
            "cnmf", "cnmf_ixs", "ixs_spatial_contours_contours.npy"
        ),
        allow_pickle=True,
    )
    ixs_contours_coms = df.iloc[-1].cnmf.get_contours(ixs_components)[1]
    ixs_contours_coms_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "cnmf_ixs", "ixs_spatial_contours_coms.npy"),
        allow_pickle=True,
    )
    for contour, actual_contour in zip(
        ixs_contours_contours, ixs_contours_contours_actual
    ):
        numpy.testing.assert_allclose(contour, actual_contour, rtol=1e-2, atol=1e-10)
    for com, actual_com in zip(ixs_contours_coms, ixs_contours_coms_actual):
        numpy.testing.assert_allclose(com, actual_com, rtol=1e-2, atol=1e-10)

    # test to check ixs components for cnmf.get_temporal()
    ixs_temporal_components = df.iloc[-1].cnmf.get_temporal(ixs_components)
    ixs_temporal_components_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf", "cnmf_ixs", "ixs_temporal_components.npy"),
        allow_pickle=True,
    )
    numpy.testing.assert_allclose(
        ixs_temporal_components, ixs_temporal_components_actual, rtol=1e-2, atol=1e-10
    )


def test_cnmfe():
    set_parent_raw_data_path(vid_dir)

    df, batch_path = _create_tmp_batch()

    batch_path = Path(batch_path)
    batch_dir = batch_path.parent
    batch_dir_canon = batch_dir.resolve()

    input_movie_path = get_datafile("cnmfe")
    print(input_movie_path)
    df.caiman.add_item(
        algo="mcorr",
        item_name=f"test-cnmfe-mcorr",
        input_movie_path=input_movie_path,
        params=test_params["mcorr"],
    )
    process = df.iloc[-1].caiman.run()
    # process.wait()

    df = load_batch(batch_path)

    # Test if running full cnmfe works
    print("testing cnmfe")
    algo = "cnmfe"
    param_name = "cnmfe_full"
    input_movie_path = df.iloc[0].mcorr.get_output_path()
    print(input_movie_path)

    df.caiman.add_item(
        algo=algo,
        item_name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[param_name],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["item_name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[param_name]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["input_movie_path"])
        == batch_dir_canon.joinpath(df.iloc[0].mcorr.get_output_path())
        == df.paths.resolve(df.iloc[-1]["input_movie_path"])
    )

    process = df.iloc[-1].caiman.run()
    # process.wait()

    df = load_batch(batch_path)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    pprint(df.iloc[-1]["outputs"], width=-1)
    print(df.iloc[-1]["outputs"]["traceback"])

    # Confirm output path is as expected
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    assert (
        input_movie_path
        == df.iloc[-1].caiman.get_input_movie_path()
        == df.paths.resolve(df.iloc[-1]["input_movie_path"])
    )

    assert batch_dir_canon.joinpath(
        str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}.hdf5'
    ) == df.paths.resolve(df.iloc[-1]["outputs"]["cnmf-hdf5-path"])

    # test to check mmap output path
    assert (
        batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]),
            f'{df.iloc[-1]["uuid"]}_cnmf-memmap_d1_128_d2_128_d3_1_order_C_frames_1000.mmap',
        )
        == df.paths.resolve(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
        == batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["cnmf-memmap-path"])
    )

    # test to check mean-projection output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["mean-projection-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["mean-projection-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_mean_projection.npy'
        )
    )

    # test to check std-projection output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["std-projection-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["std-projection-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_std_projection.npy'
        )
    )

    # test to check max-projection output path
    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["max-projection-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["max-projection-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]), f'{df.iloc[-1]["uuid"]}_max_projection.npy'
        )
    )

    # extension tests - full

    # test to check cnmf get_cnmf_memmap()
    cnmfe_mmap_output = df.iloc[-1].cnmf.get_cnmf_memmap()
    cnmfe_mmap_output_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_full_output_mmap.npy")
    )
    numpy.testing.assert_array_equal(cnmfe_mmap_output, cnmfe_mmap_output_actual)

    # test to check input memmap paths
    cnmfe_input_mmap = df.iloc[-1].caiman.get_input_movie()
    cnmfe_input_mmap_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_full_input_mmap.npy")
    )
    numpy.testing.assert_array_equal(cnmfe_input_mmap, cnmfe_input_mmap_actual)
    # cnmf input memmap from mcorr output should also equal mcorr output
    mcorr_output = df.iloc[0].mcorr.get_output()
    numpy.testing.assert_array_equal(cnmfe_input_mmap, mcorr_output)

    # test to check cnmf get_output_path()
    assert (
        df.iloc[-1].cnmf.get_output_path()
        == batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["cnmf-hdf5-path"])
        == df.iloc[-1].paths.resolve(df.iloc[-1]["outputs"]["cnmf-hdf5-path"])
    )

    # test to check cnmf get_output()
    assert isinstance(df.iloc[-1].cnmf.get_output(), CNMF)
    # this doesn't work because some keys in the hdf5 file are
    # not always identical, like the path to the mmap file
    # assert sha1(open(df.iloc[1].cnmf.get_output_path(), "rb").read()).hexdigest() == sha1(open(ground_truths_dir.joinpath('cnmf', 'cnmf_output.hdf5'), "rb").read()).hexdigest()

    # test to check cnmf get_masks()
    cnmfe_spatial_masks = df.iloc[-1].cnmf.get_masks("good")
    cnmfe_spatial_masks_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_spatial_masks.npy")
    )
    numpy.testing.assert_array_equal(cnmfe_spatial_masks, cnmfe_spatial_masks_actual)

    # test to check get_contours()
    cnmfe_spatial_contours_contours = df.iloc[-1].cnmf.get_contours("good")[0]
    cnmfe_spatial_contours_coms = df.iloc[-1].cnmf.get_contours("good")[1]
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

    # test to check get_temporal()
    cnmfe_temporal_components = df.iloc[-1].cnmf.get_temporal("good")
    cnmfe_temporal_components_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_temporal_components.npy")
    )
    numpy.testing.assert_allclose(
        cnmfe_temporal_components,
        cnmfe_temporal_components_actual,
        rtol=1e1,
        atol=1e-1,
    )

    # test to check get_rcm()
    cnmfe_reconstructed_movie_AouterC = df.iloc[-1].cnmf.get_rcm("all")
    cnmfe_reconstructed_movie_AouterC_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_reconstructed_movie_new.npy")
    )
    numpy.testing.assert_allclose(
        cnmfe_reconstructed_movie_AouterC.as_numpy(),
        cnmfe_reconstructed_movie_AouterC_actual,
        rtol=1e2,
        atol=1e-1,
    )

    # test to check get_rcb()
    cnmfe_reconstructed_background = df.iloc[-1].cnmf.get_rcb()
    cnmfe_reconstructed_background_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_reconstructed_background.npy")
    )
    numpy.testing.assert_allclose(
        cnmfe_reconstructed_background.as_numpy(),
        cnmfe_reconstructed_background_actual,
        rtol=1e-2,
        atol=1e-10,
    )

    # test to check get_residuals()
    cnmfe_residuals = df.iloc[-1].cnmf.get_residuals()
    # something wrong with residuals groundtruth file, maybe it was not created with proper Y - (A * C) - (b * f)
    # cnmfe_residuals_actual = numpy.load(ground_truths_dir.joinpath("cnmfe_full", "cnmfe_residuals.npy"))
    numpy.testing.assert_allclose(
        cnmfe_residuals.as_numpy(),
        df.iloc[-1].caiman.get_input_movie()
        - cnmfe_reconstructed_movie_AouterC_actual
        - cnmfe_reconstructed_background_actual,
        rtol=1e2,
        atol=1e-1,
    )

    # test to check passing optional ixs components to various functions
    ixs_components = numpy.array([1, 4, 7, 3])

    # test to check ixs components for cnmf.get_masks()
    ixs_spatial_masks = df.iloc[-1].cnmf.get_masks(ixs_components)
    ixs_spatial_masks_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_full", "cnmfe_ixs", "ixs_spatial_masks.npy"),
        allow_pickle=True,
    )
    numpy.testing.assert_array_equal(ixs_spatial_masks, ixs_spatial_masks_actual)

    # test to check ixs components for cnmf.get_contours()
    ixs_contours_contours = df.iloc[-1].cnmf.get_contours(ixs_components)[0]
    ixs_contours_contours_actual = numpy.load(
        ground_truths_dir.joinpath(
            "cnmfe_full", "cnmfe_ixs", "ixs_spatial_contours_contours.npy"
        ),
        allow_pickle=True,
    )
    ixs_contours_coms = df.iloc[-1].cnmf.get_contours(ixs_components)[1]
    ixs_contours_coms_actual = numpy.load(
        ground_truths_dir.joinpath(
            "cnmfe_full", "cnmfe_ixs", "ixs_spatial_contours_coms.npy"
        ),
        allow_pickle=True,
    )
    for contour, actual_contour in zip(
        ixs_contours_contours, ixs_contours_contours_actual
    ):
        numpy.testing.assert_allclose(contour, actual_contour, rtol=1e-2, atol=1e-10)
    for com, actual_com in zip(ixs_contours_coms, ixs_contours_coms_actual):
        numpy.testing.assert_allclose(com, actual_com, rtol=1e-2, atol=1e-10)

    # test to check ixs components for cnmf.get_temporal()
    ixs_temporal_components = df.iloc[-1].cnmf.get_temporal(ixs_components)
    ixs_temporal_components_actual = numpy.load(
        ground_truths_dir.joinpath(
            "cnmfe_full", "cnmfe_ixs", "ixs_temporal_components.npy"
        ),
        allow_pickle=True,
    )
    numpy.testing.assert_allclose(
        ixs_temporal_components, ixs_temporal_components_actual, rtol=1e1, atol=1e-5
    )


def test_remove_item():
    set_parent_raw_data_path(vid_dir)
    algo = "mcorr"
    df, batch_path = _create_tmp_batch()
    input_movie_path = get_datafile(algo)

    # make small version of movie for quick testing
    movie = tifffile.imread(input_movie_path)
    small_movie_path = input_movie_path.parent.joinpath("small_movie.tif")
    tifffile.imwrite(small_movie_path, movie[:1001])

    print(input_movie_path)
    df.caiman.add_item(
        algo="mcorr",
        item_name=f"test0",
        input_movie_path=small_movie_path,
        params=test_params["mcorr"],
    )

    df.caiman.add_item(
        algo="mcorr",
        item_name=f"test1",
        input_movie_path=small_movie_path,
        params=test_params["mcorr"],
    )

    df.caiman.add_item(
        algo="cnmf",
        item_name=f"test2",
        input_movie_path=small_movie_path,
        params=test_params["cnmf"],
    )

    diff_params = deepcopy(test_params["cnmf"])
    diff_params["main"]["gSig"] = (6, 6)
    diff_params["main"]["merge_thr"] = 0.85
    df.caiman.add_item(
        algo="cnmf",
        item_name=f"test3",
        input_movie_path=small_movie_path,
        params=diff_params,
    )

    for i, r in df.iterrows():
        proc = r.caiman.run()
        # proc.wait()

    df = load_batch(df.paths.get_batch_path())

    # make sure we can get mcorr movie output of 0th and 1st indices
    path0 = df.iloc[0].mcorr.get_output_path()
    path0_input = df.iloc[0].caiman.get_input_movie_path()
    assert path0.exists()
    df.iloc[0].mcorr.get_output()
    # index 1
    path1 = df.iloc[1].mcorr.get_output_path()
    path1_input = df.iloc[1].caiman.get_input_movie_path()
    assert path1.exists()
    df.iloc[1].mcorr.get_output()

    # make sure we can get cnmf output of 2nd and 3rd indices
    path2 = df.iloc[2].cnmf.get_output_path()
    assert path2.exists()
    df.iloc[2].cnmf.get_output()
    data2 = df.iloc[2].cnmf.get_temporal()
    path3 = df.iloc[3].cnmf.get_output_path()
    assert path3.exists()
    df.iloc[3].cnmf.get_output()
    data3 = df.iloc[3].cnmf.get_temporal()

    # remove index 1
    df.caiman.remove_item(index=1, remove_data=True)
    assert path1.exists() == False
    assert df.isin([f"test1"]).any().any() == False
    # input movie path should be unaffected
    assert path1_input.exists()

    # shouldn't affect data at other indices
    assert path0.exists()
    assert df.iloc[0]["item_name"] == f"test0"
    assert df.iloc[1]["item_name"] == f"test2"
    assert df.iloc[2]["item_name"] == f"test3"
    assert df.iloc[0].mcorr.get_output_path().exists()
    df.iloc[0].mcorr.get_output()
    assert df.iloc[1].cnmf.get_output_path().exists()
    df.iloc[1].cnmf.get_output()
    # check that the earlier data from index 2, now index 1, is equal
    np.testing.assert_array_equal(data2, df.iloc[1].cnmf.get_temporal())
    assert df.iloc[2].cnmf.get_output_path().exists()
    df.iloc[2].cnmf.get_output()
    # check that the earlier data from index 3, now index 2, is equal
    np.testing.assert_array_equal(data3, df.iloc[2].cnmf.get_temporal())
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_array_equal,
        data2,
        df.iloc[2].cnmf.get_temporal(),
    )


def test_cache():
    print("*** Testing cache ***")
    cnmf.cnmf_cache.clear_cache()

    set_parent_raw_data_path(vid_dir)
    algo = "mcorr"

    df, batch_path = _create_tmp_batch()

    batch_path = Path(batch_path)
    batch_dir = batch_path.parent
    batch_dir_canon = batch_dir.resolve()

    input_movie_path = get_datafile(algo)
    print(input_movie_path)

    df.caiman.add_item(
        algo=algo,
        item_name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["item_name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    vid_dir_canon = vid_dir.resolve()
    assert vid_dir_canon.joinpath(
        df.iloc[-1]["input_movie_path"]
    ) == vid_dir_canon.joinpath(f"{algo}.tif")

    process = df.iloc[-1].caiman.run()
    # process.wait()

    df = load_batch(batch_path)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    pprint(df.iloc[-1]["outputs"], width=-1)
    print(df.iloc[-1]["outputs"]["traceback"])
    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == df.paths.resolve(df.iloc[-1]["outputs"]["mcorr-output-path"])
        == batch_dir_canon.joinpath(
            str(df.iloc[-1]["uuid"]),
            f'{df.iloc[-1]["uuid"]}-mcorr_els__d1_60_d2_80_d3_1_order_F_frames_2000.mmap',
        )
    )

    algo = "cnmf"
    input_movie_path = df.iloc[-1].mcorr.get_output_path()
    df.caiman.add_item(
        algo=algo,
        item_name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["item_name"] == f"test-{algo}"
    assert df.iloc[-1]["params"] == test_params[algo]
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")
    print("cnmf input_movie_path:", df.iloc[-1]["input_movie_path"])
    assert batch_dir_canon.joinpath(df.iloc[-1]["input_movie_path"]) == input_movie_path

    process = df.iloc[-1].caiman.run()
    # process.wait()

    df = load_batch(batch_path)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    pprint(df.iloc[-1]["outputs"], width=-1)
    print(df.iloc[-1]["outputs"]["traceback"])

    # test that cache values are returned when calls are made to same function

    # testing that cache size limits work
    cnmf.cnmf_cache.set_maxsize("1M")
    cnmf_output = df.iloc[-1].cnmf.get_output()
    hex_get_output = hex(id(cnmf_output))
    cache = cnmf.cnmf_cache.get_cache()
    hex1 = hex(id(cache[cache["function"] == "get_output"]["return_val"].item()))
    # assert(hex(id(df.iloc[-1].cnmf.get_output(copy=False))) == hex1)
    # assert(hex_get_output != hex1)
    time_stamp1 = cache[cache["function"] == "get_output"]["time_stamp"].item()
    df.iloc[-1].cnmf.get_temporal("good")
    df.iloc[-1].cnmf.get_contours("good")
    df.iloc[-1].cnmf.get_masks("good")
    df.iloc[-1].cnmf.get_temporal(np.arange(7))
    df.iloc[-1].cnmf.get_temporal(np.arange(8))
    df.iloc[-1].cnmf.get_temporal(np.arange(9))
    df.iloc[-1].cnmf.get_temporal(np.arange(6))
    df.iloc[-1].cnmf.get_temporal(np.arange(5))
    df.iloc[-1].cnmf.get_temporal(np.arange(4))
    df.iloc[-1].cnmf.get_temporal(np.arange(3))
    df.iloc[-1].cnmf.get_masks(np.arange(8))
    df.iloc[-1].cnmf.get_masks(np.arange(9))
    df.iloc[-1].cnmf.get_masks(np.arange(7))
    df.iloc[-1].cnmf.get_masks(np.arange(6))
    df.iloc[-1].cnmf.get_masks(np.arange(5))
    df.iloc[-1].cnmf.get_masks(np.arange(4))
    df.iloc[-1].cnmf.get_masks(np.arange(3))
    time_stamp2 = cache[cache["function"] == "get_output"]["time_stamp"].item()
    hex2 = hex(id(cache[cache["function"] == "get_output"]["return_val"].item()))
    assert cache[cache["function"] == "get_output"].index.size == 1
    # after adding enough items for cache to exceed max size, cache should remove least recently used items until
    # size is back under max
    assert len(cnmf.cnmf_cache.get_cache().index) == 17
    # the time stamp to get_output the second time should be greater than the original time
    # stamp because the cached item is being returned and therefore will have been accessed more recently
    assert time_stamp2 > time_stamp1
    # the hex id of the item in the cache when get_output is first called
    # should be the same hex id of the item in the cache when get_output is called again
    assert hex1 == hex2

    # test clear_cache()
    cnmf.cnmf_cache.clear_cache()
    assert len(cnmf.cnmf_cache.get_cache().index) == 0

    # checking that cache is cleared, checking speed at which item is returned
    start = time.time()
    df.iloc[-1].cnmf.get_output()
    end = time.time()
    assert len(cnmf.cnmf_cache.get_cache().index) == 1

    # second call to item now added to cache, time to return item should be must faster than before because item has
    # now been cached
    start2 = time.time()
    df.iloc[-1].cnmf.get_output()
    end2 = time.time()
    assert end2 - start2 < end - start

    # testing clear_cache() again, length of dataframe should be zero
    cnmf.cnmf_cache.clear_cache()
    assert len(cnmf.cnmf_cache.get_cache().index) == 0

    # test setting maxsize as 0, should effectively disable the cache...additionally, time to return an item called
    # twice should roughly be the same because item is not being stored in the cache
    # cache length should remain zero throughout calls to extension functions
    cnmf.cnmf_cache.set_maxsize(0)
    start = time.time()
    df.iloc[-1].cnmf.get_output()
    end = time.time()
    assert len(cnmf.cnmf_cache.get_cache().index) == 0

    start2 = time.time()
    df.iloc[-1].cnmf.get_output()
    end2 = time.time()
    assert len(cnmf.cnmf_cache.get_cache().index) == 0
    assert abs((end - start) - (end2 - start2)) < 0.1

    # test to check that separate cache items are being returned for different batch items
    # must add another item to the batch, running cnmfe

    input_movie_path = get_datafile("cnmfe")
    print(input_movie_path)
    df.caiman.add_item(
        algo="mcorr",
        item_name=f"test-cnmfe-mcorr",
        input_movie_path=input_movie_path,
        params=test_params["mcorr"],
    )
    process = df.iloc[-1].caiman.run()
    # process.wait()

    df = load_batch(batch_path)

    algo = "cnmfe"
    param_name = "cnmfe_full"
    input_movie_path = df.iloc[-1].mcorr.get_output_path()
    print(input_movie_path)

    df.caiman.add_item(
        algo=algo,
        item_name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[param_name],
    )

    process = df.iloc[-1].caiman.run()
    # process.wait()

    df = load_batch(batch_path)

    cnmf.cnmf_cache.set_maxsize("1M")

    df.iloc[1].cnmf.get_output()  # cnmf output
    df.iloc[-1].cnmf.get_output()  # cnmfe output

    cache = cnmf.cnmf_cache.get_cache()

    # checking that both outputs from different batch items are added to the cache
    assert len(cache.index) == 2

    # checking that the uuid of each outputs from the different batch items are not the same
    assert cache.iloc[-1]["uuid"] != cache.iloc[-2]["uuid"]

    # checking that the uuid of the output in the cache is the correct uuid of the batch item in the df
    assert cache.iloc[-1]["uuid"] == df.iloc[-1]["uuid"]

    # call get output from cnmf, check that it is the most recent thing called in the cache
    time.sleep(0.01)  # make absolutely sure the times aren't identical
    df.iloc[1].cnmf.get_output()
    cnmf_uuid = df.iloc[1]["uuid"]
    cache_sorted = cache.sort_values(by=["time_stamp"], ascending=True)
    print("Cache sorted from oldest to newest call:")
    print(cache_sorted)
    print("Call times:")
    for _, row in cache_sorted.iterrows():
        print(f"{row['time_stamp']}", end=", ")
    print("")

    most_recently_called = cache_sorted.iloc[-1]
    cache_uuid = most_recently_called["uuid"]
    assert cnmf_uuid == cache_uuid

    # check to make sure by certain params that it is cnmf vs cnmfe
    output = df.iloc[1].cnmf.get_output()
    assert output.params.patch["low_rank_background"] == True
    output2 = df.iloc[-1].cnmf.get_output()
    assert output2.params.patch["low_rank_background"] == False

    # test for copy
    # if return_copy=True, then hex id of calls to the same function should be false
    output = df.iloc[1].cnmf.get_output()
    assert hex(id(output)) != hex(
        id(cache.sort_values(by=["time_stamp"], ascending=True).iloc[-1])
    )
    # if return_copy=False, then hex id of calls to the same function should be true
    output = df.iloc[1].cnmf.get_output(return_copy=False)
    output2 = df.iloc[1].cnmf.get_output(return_copy=False)
    assert hex(id(output)) == hex(id(output2))
    assert hex(id(cnmf.cnmf_cache.get_cache().iloc[-1]["return_val"])) == hex(
        id(output)
    )


def test_seeded_cnmf():
    """Test seeded CNNF (Ain)"""
    set_parent_raw_data_path(vid_dir)
    algo = "mcorr"

    df, batch_path = _create_tmp_batch()

    batch_path = Path(batch_path)
    batch_dir = batch_path.parent
    batch_dir_canon = batch_dir.resolve()

    input_movie_path = get_datafile(algo)
    print(input_movie_path)

    df.caiman.add_item(
        algo=algo,
        item_name=f"test-{algo}",
        input_movie_path=input_movie_path,
        params=test_params[algo],
    )

    df.iloc[-1].caiman.run()
    df = load_batch(batch_path)

    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    # make seed
    mcorr_output = df.iloc[-1].mcorr.get_output()
    seed = make_test_seed(mcorr_output)
    seed_path = seed_dir / "Ain_cnmf.npy"
    np.save(seed_path, seed)

    algo = "cnmf"
    print("Testing seeded cnmf")
    input_movie_path = df.iloc[-1].mcorr.get_output_path()
    seeded_params = {
        **test_params[algo],
        "Ain_path": seed_path,
        "refit": False
    }

    df.caiman.add_item(
        algo=algo,
        item_name=f"test-seeded-{algo}",
        input_movie_path=input_movie_path,
        params=seeded_params
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["item_name"] == f"test-seeded-{algo}"
    assert df.iloc[-1]["params"] == seeded_params
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")
    print("cnmf input_movie_path:", df.iloc[-1]["input_movie_path"])
    assert batch_dir_canon.joinpath(df.iloc[-1]["input_movie_path"]) == input_movie_path

    df.iloc[-1].caiman.run()

    df = load_batch(batch_path)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    pprint(df.iloc[-1]["outputs"], width=-1)
    print(df.iloc[-1]["outputs"]["traceback"])

    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    # test to check cnmf get_masks()
    cnmf_spatial_masks = df.iloc[-1].cnmf.get_masks("good")
    cnmf_spatial_masks_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf_seeded", "spatial_masks.npy")
    )
    numpy.testing.assert_array_equal(cnmf_spatial_masks, cnmf_spatial_masks_actual)

    # test to check get_temporal()
    cnmf_temporal_components = df.iloc[-1].cnmf.get_temporal("good")
    cnmf_temporal_components_actual = numpy.load(
        ground_truths_dir.joinpath("cnmf_seeded", "temporal_components.npy")
    )
    numpy.testing.assert_allclose(
        cnmf_temporal_components, cnmf_temporal_components_actual, rtol=1e-2, atol=1e-10
    )


def test_seeded_cnmfe():
    set_parent_raw_data_path(vid_dir)

    df, batch_path = _create_tmp_batch()

    batch_path = Path(batch_path)
    batch_dir = batch_path.parent
    batch_dir_canon = batch_dir.resolve()

    input_movie_path = get_datafile("cnmfe")
    print(input_movie_path)
    df.caiman.add_item(
        algo="mcorr",
        item_name="test-cnmfe-mcorr",
        input_movie_path=input_movie_path,
        params=test_params["mcorr"],
    )
    df.iloc[-1].caiman.run()

    df = load_batch(batch_path)

    # Test if running seeded cnmfe works
    # this seed is actually trash for CNMFE but just see if it's consistent
    mcorr_output = df.iloc[-1].mcorr.get_output()
    seed = make_test_seed(mcorr_output)
    seed_path = seed_dir / "Ain_cnmfe.npy"
    np.save(seed_path, seed)

    print("testing seeded cnmfe")
    algo = "cnmfe"
    param_name = "cnmfe_full"
    input_movie_path = df.iloc[0].mcorr.get_output_path()
    print(input_movie_path)
    seeded_params = {
        **test_params[param_name],
        "Ain_path": seed_path,
        "refit": False
    }

    df.caiman.add_item(
        algo=algo,
        item_name=f"test-seeded-{algo}",
        input_movie_path=input_movie_path,
        params=seeded_params,
    )

    assert df.iloc[-1]["algo"] == algo
    assert df.iloc[-1]["item_name"] == f"test-seeded-{algo}"
    assert df.iloc[-1]["params"] == seeded_params
    assert df.iloc[-1]["outputs"] is None
    try:
        UUID(df.iloc[-1]["uuid"])
    except:
        pytest.fail("Something wrong with setting UUID for batch items")

    assert (
        batch_dir_canon.joinpath(df.iloc[-1]["input_movie_path"])
        == batch_dir_canon.joinpath(df.iloc[0].mcorr.get_output_path())
        == df.paths.resolve(df.iloc[-1]["input_movie_path"])
    )

    df.iloc[-1].caiman.run()
    df = load_batch(batch_path)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df)

    pprint(df.iloc[-1]["outputs"], width=-1)
    print(df.iloc[-1]["outputs"]["traceback"])

    assert df.iloc[-1]["outputs"]["success"] is True
    assert df.iloc[-1]["outputs"]["traceback"] is None

    # test to check cnmf get_masks()
    cnmf_spatial_masks = df.iloc[-1].cnmf.get_masks("good")
    cnmf_spatial_masks_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_seeded", "spatial_masks.npy")
    )
    numpy.testing.assert_array_equal(cnmf_spatial_masks, cnmf_spatial_masks_actual)

    # test to check get_temporal()
    cnmf_temporal_components = df.iloc[-1].cnmf.get_temporal("good")
    cnmf_temporal_components_actual = numpy.load(
        ground_truths_dir.joinpath("cnmfe_seeded", "temporal_components.npy")
    )
    numpy.testing.assert_allclose(
        cnmf_temporal_components, cnmf_temporal_components_actual, rtol=1e-2, atol=1e-4
    )