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


# prevent circular import
if __name__ == "__main__":
    from mesmerize_core import set_parent_data_path, get_full_data_path


@click.command()
@click.option("--batch-path", type=str)
@click.option("--uuid", type=str)
@click.option("--data-path", type=str)
def main(batch_path, uuid, data_path: str = None):
    df = pd.read_pickle(batch_path)
    item = df[df["uuid"] == uuid].squeeze()

    input_movie_path = item["input_movie_path"]

    set_parent_data_path(data_path)
    input_movie_path = str(get_full_data_path(input_movie_path))

    # because caiman doesn't let you specify filename to save memmap files
    # create dir with uuid as the dir name
    caiman_temp_dir = str(Path(input_movie_path).parent.joinpath(str(uuid)))
    os.makedirs(caiman_temp_dir, exist_ok=True)
    os.environ["CAIMAN_TEMP"] = caiman_temp_dir
    os.environ["CAIMAN_NEW_TEMPFILE"] = "True"

    params = item['params']

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

    rel_params = dict(params["mcorr_kwargs"])
    opts = CNMFParams(params_dict=rel_params)
    # Run MC, denote boolean 'success' if MC completes w/out error
    try:
        # Run MC
        fnames = [input_movie_path]
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True)

        # Find path to mmap file
        output_path = Path(mc.mmap_file[0])
        if data_path is not None:
            output_path = Path(output_path).relative_to(data_path)

        print("mc finished successfully!")

        print("computing projections")
        Yr, dims, T = cm.load_memmap(str(get_full_data_path(output_path)))
        images = np.reshape(Yr.T, [T] + list(dims), order="F")

        proj_paths = dict()
        for proj_type in ["mean", "std", "max"]:
            p_img = getattr(np, f"nan{proj_type}")(images, axis=0)
            proj_paths[proj_type] = Path(input_movie_path).parent.joinpath(
                f"{uuid}_{proj_type}_projection.npy"
            )
            np.save(str(proj_paths[proj_type]), p_img)

        print("Computing correlation image")
        Cns = local_correlations_movie_offline(
            [mc.mmap_file[0]],
            remove_baseline=True,
            window=1000,
            stride=1000,
            winSize_baseline=100,
            quantil_min_baseline=10,
            dview=dview,
        )
        Cn = Cns.max(axis=0)
        Cn[np.isnan(Cn)] = 0
        cn_path = Path(input_movie_path).parent.joinpath(f"{uuid}_cn.npy")
        np.save(str(cn_path), Cn, allow_pickle=False)

        # output dict for pandas series for dataframe row
        d = dict()

        print("finished computing correlation image")

        # Compute shifts
        if params["mcorr_kwargs"]["pw_rigid"] == True:
            x_shifts = mc.x_shifts_els
            y_shifts = mc.y_shifts_els
            shifts = [x_shifts, y_shifts]
            shift_path = Path(input_movie_path).parent.joinpath(f"{uuid}_shifts.npy")
            np.save(str(shift_path), shifts)
        else:
            shifts = mc.shifts_rig
            shift_path = Path(input_movie_path).parent.joinpath(f"{uuid}_shifts.npy")
            np.save(str(shift_path), shifts)

        # filename to move the output back to data dir
        mcorr_memmap = Path(input_movie_path).parent.joinpath(
            f"{uuid}_{output_path.stem}.mmap"
        )

        # move the output file
        move_file(get_full_data_path(output_path), mcorr_memmap)
        os.removedirs(caiman_temp_dir)

        if data_path is not None:
            cn_path = cn_path.relative_to(data_path)
            mcorr_memmap = get_full_data_path(mcorr_memmap).relative_to(data_path)
            shift_path = shift_path.relative_to(data_path)
            for proj_type in proj_paths.keys():
                d[f"{proj_type}-projection-path"] = proj_paths[proj_type].relative_to(
                    data_path
                )
        else:
            cn_path = cn_path
            mcorr_memmap = get_full_data_path(mcorr_memmap)
            shift_path = shift_path.resolve()

        d.update(
            {
                "mcorr-output-path": mcorr_memmap,
                "corr-img-path": cn_path,
                "shifts": shift_path,
                "success": True,
                "traceback": None,
            }
        )

    except:
        d = {"success": False, "traceback": traceback.format_exc()}
        print("mc failed, stored traceback in output")

    print(d)
    # Add dictionary to output column of series
    df.loc[df["uuid"] == uuid, "outputs"] = [d]
    # Save DataFrame to disk
    df.to_pickle(batch_path)


if __name__ == "__main__":
    main()
