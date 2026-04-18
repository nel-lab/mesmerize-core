import click
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.motion_correction import MotionCorrect
import os
from pathlib import Path, PurePosixPath
import numpy as np
from shutil import move as move_file
import time
import traceback
from typing import Optional

# prevent circular import
if __name__ in ["__main__", "__mp_main__"]:  # when running in subprocess
    from mesmerize_core import set_parent_raw_data_path, load_batch
    from mesmerize_core.algorithms._utils import (
        ensure_server,
        save_projections_parallel,
        save_correlation_parallel,
        setup_logging,
    )
else:  # when running with local backend
    from ..batch_utils import set_parent_raw_data_path, load_batch
    from ._utils import (
        ensure_server,
        save_projections_parallel,
        save_correlation_parallel,
        setup_logging,
    )


def run_algo(batch_path, uuid, data_path: Optional[str] = None, dview=None, log_level=None):
    algo_start = time.time()

    if log_level is not None:
        setup_logging(log_level)
    
    if data_path is not None:
        set_parent_raw_data_path(data_path)

    batch_path = Path(batch_path)
    df = load_batch(batch_path)

    item = df.caiman.uloc(uuid)
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

    with ensure_server(dview) as (dview, n_processes):
        print("starting mc")

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

            proj_paths = save_projections_parallel(
                uuid=uuid,
                movie_path=mcorr_memmap_path,
                output_dir=output_dir,
                dview=dview,
            )

            print("Computing correlation image")
            cn_path = save_correlation_parallel(
                uuid=uuid,
                movie_path=mcorr_memmap_path,
                output_dir=output_dir,
                dview=dview,
            )

            print("finished computing correlation image")

            # Compute shifts
            if opts.motion["pw_rigid"] == True:
                x_shifts = mc.x_shifts_els
                y_shifts = mc.y_shifts_els
                shifts = [x_shifts, y_shifts]
                if hasattr(mc, "z_shifts_els"):
                    shifts.append(mc.z_shifts_els)
                shift_path = output_dir.joinpath(f"{uuid}_shifts.npy")
                np.save(str(shift_path), shifts)
            else:
                shifts = mc.shifts_rig
                shift_path = output_dir.joinpath(f"{uuid}_shifts.npy")
                np.save(str(shift_path), shifts)

            # output dict for pandas series for dataframe row
            d = dict()

            # save paths as relative path strings with forward slashes
            cn_path = str(PurePosixPath(cn_path.relative_to(output_dir.parent)))
            mcorr_memmap_path = str(
                PurePosixPath(mcorr_memmap_path.relative_to(output_dir.parent))
            )
            shift_path = str(PurePosixPath(shift_path.relative_to(output_dir.parent)))
            for proj_type in proj_paths.keys():
                d[f"{proj_type}-projection-path"] = str(
                    PurePosixPath(proj_paths[proj_type].relative_to(output_dir.parent))
                )

            d.update(
                {
                    "mcorr-output-path": mcorr_memmap_path,
                    "corr-img-path": cn_path,
                    "shifts": shift_path,
                    "border": mc.border_to_0,
                    "success": True,
                    "traceback": None,
                }
            )

        except:
            d = {"success": False, "traceback": traceback.format_exc()}
            print("mc failed, stored traceback in output")

    runtime = round(time.time() - algo_start, 2)
    df.caiman.update_item_with_results(uuid, d, runtime)


@click.command()
@click.option("--batch-path", type=str)
@click.option("--uuid", type=str)
@click.option("--data-path", default=None)
@click.option("--log-level", type=int, default=None)
def main(batch_path, uuid, data_path, log_level):
    run_algo(batch_path, uuid, data_path, log_level=log_level)


if __name__ == "__main__":
    main()
