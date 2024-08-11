import asyncio
import click
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.params import CNMFParams
import psutil
import traceback
from pathlib import Path, PurePosixPath
from shutil import move as move_file
import os
import time

if __name__ in ["__main__", "__mp_main__"]:  # when running in subprocess
    from mesmerize_core import set_parent_raw_data_path, load_batch
    from mesmerize_core.utils import IS_WINDOWS
    from mesmerize_core.algorithms._utils import ensure_server
else:  # when running with local backend
    from ..batch_utils import set_parent_raw_data_path, load_batch
    from ..utils import IS_WINDOWS
    from ._utils import ensure_server


def run_algo(batch_path, uuid, data_path: str = None, dview=None):
    asyncio.run(run_algo_async(batch_path, uuid, data_path=data_path, dview=dview))

async def run_algo_async(batch_path, uuid, data_path: str = None, dview=None):
    algo_start = time.time()
    set_parent_raw_data_path(data_path)

    df = load_batch(batch_path)
    item = df.caiman.uloc(uuid)

    input_movie_path = item["input_movie_path"]
    # resolve full path
    input_movie_path = str(df.paths.resolve(input_movie_path))

    output_dir = Path(batch_path).parent.joinpath(str(uuid)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = item["params"]
    print("cnmfe params:", params)

    with ensure_server(dview) as (dview, n_processes):
        try:
            fname_new = cm.save_memmap(
                [input_movie_path], base_name=f"{uuid}_cnmf-memmap_", order="C", dview=dview
            )

            print("making memmap")
            Yr, dims, T = cm.load_memmap(fname_new)
            images = np.reshape(Yr.T, [T] + list(dims), order="F")

            # TODO: if projections already exist from mcorr we don't
            #  need to waste compute time re-computing them here
            proj_paths = dict()
            for proj_type in ["mean", "std", "max"]:
                p_img = getattr(np, f"nan{proj_type}")(images, axis=0)
                proj_paths[proj_type] = output_dir.joinpath(
                    f"{uuid}_{proj_type}_projection.npy"
                )
                np.save(str(proj_paths[proj_type]), p_img)

            d = dict()  # for output

            # force the CNMFE params
            cnmfe_params_dict = {
                "method_init": "corr_pnr",
                "n_processes": n_processes,
                "only_init": True,  # for 1p
                "center_psf": True,  # for 1p
                "normalize_init": False,  # for 1p
            }

            params_dict = {**cnmfe_params_dict, **params["main"]}

            cnmfe_params_dict = CNMFParams(params_dict=params_dict)
            cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, params=cnmfe_params_dict)
            print("Performing CNMFE")
            cnm.fit(images)
            print("evaluating components")
            cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

            cnmf_hdf5_path = output_dir.joinpath(f"{uuid}.hdf5")
            cnm.save(str(cnmf_hdf5_path))

            # save output paths to outputs dict
            d["cnmf-hdf5-path"] = cnmf_hdf5_path.relative_to(output_dir.parent)

            for proj_type in proj_paths.keys():
                d[f"{proj_type}-projection-path"] = proj_paths[proj_type].relative_to(
                    output_dir.parent
                )

            cnmf_memmap_path = output_dir.joinpath(Path(fname_new).name)
            if IS_WINDOWS:
                Yr._mmap.close()  # accessing private attr but windows is annoying otherwise
            move_file(fname_new, cnmf_memmap_path)

            # save path as relative path strings with forward slashes
            cnmfe_memmap_path = str(
                PurePosixPath(cnmf_memmap_path.relative_to(output_dir.parent))
            )

            d.update(
                {
                    "cnmf-memmap-path": cnmfe_memmap_path,
                    "success": True,
                    "traceback": None,
                }
            )

        except:
            d = {"success": False, "traceback": traceback.format_exc()}

    runtime = round(time.time() - algo_start, 2)
    df.caiman.update_item_with_results(uuid, d, runtime)


@click.command()
@click.option("--batch-path", type=str)
@click.option("--uuid", type=str)
@click.option("--data-path")
def main(batch_path, uuid, data_path: str = None):
    run_algo(batch_path, uuid, data_path)


if __name__ == "__main__":
    main()
