import traceback
import click
import caiman as cm
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import MotionCorrect
from caiman.summary_images import local_correlations_movie_offline
import psutil
import pandas as pd
import os
from pathlib import Path
import numpy as np
from shutil import move as move_file
import time
from datetime import datetime
from filelock import SoftFileLock, Timeout


# prevent circular import
if __name__ in ["__main__", "__mp_main__"]:  # when running in subprocess
    from mesmerize_core import set_parent_raw_data_path, load_batch
else:  # when running with local backend
    from ..batch_utils import set_parent_raw_data_path, load_batch


def run_algo(batch_path, uuid, data_path: str = None):
    algo_start = time.time()
    set_parent_raw_data_path(data_path)

    batch_path = Path(batch_path)
    df = load_batch(batch_path)

    item = df[df["uuid"] == uuid].squeeze()
    # resolve full path
    input_movie_path = str(df.paths.resolve(item["input_movie_path"]))

    # because caiman doesn't let you specify filename to save memmap files
    # create dir with uuid as the dir item_name
    output_dir = Path(batch_path).parent.joinpath(str(uuid))
    caiman_temp_dir = str(output_dir)
    os.makedirs(caiman_temp_dir, exist_ok=True)
    os.environ["CAIMAN_TEMP"] = caiman_temp_dir
    os.environ["CAIMAN_NEW_TEMPFILE"] = "True"

    params = item["params"]

    # adapted from current demo notebook
    if "MESMERIZE_N_PROCESSES" in os.environ.keys():
        try:
            n_processes = int(os.environ["MESMERIZE_N_PROCESSES"])
        except:
            n_processes = psutil.cpu_count() - 1
    else:
        n_processes = psutil.cpu_count() - 1

    print("starting mc")
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend="local", n_processes=n_processes, single_thread=False
    )

    rel_params = dict(params["main"])
    opts = CNMFParams(params_dict=rel_params)
    # Run MC, denote boolean 'success' if MC completes w/out error
    try:
        # Run MC
        fnames = [input_movie_path]
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group("motion"))
        mc.motion_correct(save_movie=True)

        # find path to mmap file
        memmap_output_path_temp = df.paths.resolve(mc.mmap_file[0])

        # filename to move the output back to data dir
        mcorr_memmap_path = output_dir.joinpath(
            f"{uuid}-{memmap_output_path_temp.name}"
        )

        # move the output file
        move_file(memmap_output_path_temp, mcorr_memmap_path)

        print("mc finished successfully!")

        print("computing projections")
        Yr, dims, T = cm.load_memmap(str(mcorr_memmap_path))
        images = np.reshape(Yr.T, [T] + list(dims), order="F")

        proj_paths = dict()
        for proj_type in ["mean", "std", "max"]:
            p_img = getattr(np, f"nan{proj_type}")(images, axis=0)
            proj_paths[proj_type] = output_dir.joinpath(
                f"{uuid}_{proj_type}_projection.npy"
            )
            np.save(str(proj_paths[proj_type]), p_img)

        print("Computing correlation image")
        Cns = local_correlations_movie_offline(
            [str(mcorr_memmap_path)],
            remove_baseline=True,
            window=1000,
            stride=1000,
            winSize_baseline=100,
            quantil_min_baseline=10,
            dview=dview,
        )
        Cn = Cns.max(axis=0)
        Cn[np.isnan(Cn)] = 0
        cn_path = output_dir.joinpath(f"{uuid}_cn.npy")
        np.save(str(cn_path), Cn, allow_pickle=False)

        # output dict for pandas series for dataframe row
        d = dict()

        print("finished computing correlation image")

        # Compute shifts
        if params["main"]["pw_rigid"] == True:
            x_shifts = mc.x_shifts_els
            y_shifts = mc.y_shifts_els
            shifts = [x_shifts, y_shifts]
            shift_path = output_dir.joinpath(f"{uuid}_shifts.npy")
            np.save(str(shift_path), shifts)
        else:
            shifts = mc.shifts_rig
            shift_path = output_dir.joinpath(f"{uuid}_shifts.npy")
            np.save(str(shift_path), shifts)

        # relative paths
        cn_path = cn_path.relative_to(output_dir.parent)
        mcorr_memmap_path = mcorr_memmap_path.relative_to(output_dir.parent)
        shift_path = shift_path.relative_to(output_dir.parent)
        for proj_type in proj_paths.keys():
            d[f"{proj_type}-projection-path"] = proj_paths[proj_type].relative_to(
                output_dir.parent
            )

        d.update(
            {
                "mcorr-output-path": mcorr_memmap_path,
                "corr-img-path": cn_path,
                "shifts": shift_path,
                "success": True,
                "traceback": None,
            }
        )

    except:
        d = {"success": False, "traceback": traceback.format_exc()}
        print("mc failed, stored traceback in output")

    cm.stop_server(dview=dview)

    # lock batch file while writing back results
    batch_lock = SoftFileLock(batch_path + '.lock', timeout=30)
    try:
        with batch_lock:
            df = load_batch(batch_path)

            # Add dictionary to output column of series
            df.loc[df["uuid"] == uuid, "outputs"] = [d]
            # Add ran timestamp to ran_time column of series
            df.loc[df["uuid"] == uuid, "ran_time"] = datetime.now().isoformat(timespec="seconds", sep="T")
            df.loc[df["uuid"] == uuid, "algo_duration"] = str(round(time.time() - algo_start, 2)) + " sec"
            # Save DataFrame to disk
            df.to_pickle(batch_path)
    except Timeout:
        # Print a message with details in lieu of writing to the batch file
        msg = f"Batch file could not be written to within {batch_lock.timeout} seconds."
        if d["success"]:
            msg += f"\nRun succeeded; results are in {output_dir}."
        else:
            msg += f"Run failed. Traceback:\n"
            msg += d["traceback"]

        raise RuntimeError(msg)


@click.command()
@click.option("--batch-path", type=str)
@click.option("--uuid", type=str)
@click.option("--data-path", type=str)
def main(batch_path, uuid, data_path: str = None):
    run_algo(batch_path, uuid, data_path)


if __name__ == "__main__":
    main()
