"""Performs CNMF in a separate process"""
import click
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.params import CNMFParams
import psutil
import numpy as np
import pandas as pd
import traceback
from pathlib import Path
from shutil import move as move_file
import os
import time
from datetime import datetime

# prevent circular import
if __name__ in ["__main__", "__mp_main__"]:  # when running in subprocess
    from mesmerize_core import set_parent_raw_data_path, load_batch
    from mesmerize_core.utils import IS_WINDOWS
else:  # when running with local backend
    from ..batch_utils import set_parent_raw_data_path, load_batch
    from ..utils import IS_WINDOWS


def run_algo(batch_path, uuid, data_path: str = None):
    algo_start = time.time()
    set_parent_raw_data_path(data_path)

    df = load_batch(batch_path)
    item = df[df["uuid"] == uuid].squeeze()

    input_movie_path = item["input_movie_path"]
    # resolve full path
    input_movie_path = str(df.paths.resolve(input_movie_path))

    # make output dir
    output_dir = Path(batch_path).parent.joinpath(str(uuid)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = item["params"]
    print(
        f"************************************************************************\n\n"
        f"Starting CNMF item:\n{item}\nWith params:{params}"
    )

    # adapted from current demo notebook
    if "MESMERIZE_N_PROCESSES" in os.environ.keys():
        try:
            n_processes = int(os.environ["MESMERIZE_N_PROCESSES"])
        except:
            n_processes = psutil.cpu_count() - 1
    else:
        n_processes = psutil.cpu_count() - 1
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend="local", n_processes=n_processes, single_thread=False
    )

    # merge cnmf and eval kwargs into one dict
    cnmf_params = CNMFParams(params_dict=params["main"])
    # Run CNMF, denote boolean 'success' if CNMF completes w/out error
    try:
        fname_new = cm.save_memmap(
            [input_movie_path], base_name=f"{uuid}_cnmf-memmap_", order="C", dview=dview
        )

        print("making memmap")

        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order="F")

        proj_paths = dict()
        for proj_type in ["mean", "std", "max"]:
            p_img = getattr(np, f"nan{proj_type}")(images, axis=0)
            proj_paths[proj_type] = output_dir.joinpath(
                f"{uuid}_{proj_type}_projection.npy"
            )
            np.save(str(proj_paths[proj_type]), p_img)

        # in fname new load in memmap order C
        cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend="local", n_processes=None, single_thread=False
        )

        print("performing CNMF")
        cnm = cnmf.CNMF(n_processes, params=cnmf_params, dview=dview)

        print("fitting images")
        cnm = cnm.fit(images)
        #
        if "refit" in params.keys():
            if params["refit"] is True:
                print("refitting")
                cnm = cnm.refit(images, dview=dview)

        print("performing eval")
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

        output_path = output_dir.joinpath(f"{uuid}.hdf5")

        cnm.save(str(output_path))

        Cn = cm.local_correlations(images.transpose(1, 2, 0))
        Cn[np.isnan(Cn)] = 0

        corr_img_path = output_dir.joinpath(f"{uuid}_cn.npy")
        np.save(str(corr_img_path), Cn, allow_pickle=False)

        # output dict for dataframe row (pd.Series)
        d = dict()

        cnmf_memmap_path = output_dir.joinpath(Path(fname_new).name)
        if IS_WINDOWS:
            Yr._mmap.close()  # accessing private attr but windows is annoying otherwise
        move_file(fname_new, cnmf_memmap_path)

        cnmf_hdf5_path = output_path.relative_to(output_dir.parent)
        cnmf_memmap_path = cnmf_memmap_path.relative_to(output_dir.parent)
        corr_img_path = corr_img_path.relative_to(output_dir.parent)
        for proj_type in proj_paths.keys():
            d[f"{proj_type}-projection-path"] = proj_paths[proj_type].relative_to(
                output_dir.parent
            )

        d.update(
            {
                "cnmf-hdf5-path": cnmf_hdf5_path,
                "cnmf-memmap-path": cnmf_memmap_path,
                "corr-img-path": corr_img_path,
                "success": True,
                "traceback": None,
            }
        )

    except:
        d = {"success": False, "traceback": traceback.format_exc()}

    cm.stop_server(dview=dview)

    # Add dictionary to output column of series
    df.loc[df["uuid"] == uuid, "outputs"] = [d]
    # Add ran timestamp to ran_time column of series
    df.loc[df["uuid"] == uuid, "ran_time"] = datetime.now().isoformat(timespec="seconds", sep="T")
    df.loc[df["uuid"] == uuid, "algo_duration"] = str(round(time.time() - algo_start, 2)) + " sec"
    # save dataframe to disc
    df.to_pickle(batch_path)


@click.command()
@click.option("--batch-path", type=str)
@click.option("--uuid", type=str)
@click.option("--data-path")
def main(batch_path, uuid, data_path: str = None):
    run_algo(batch_path, uuid, data_path)


if __name__ == "__main__":
    main()
