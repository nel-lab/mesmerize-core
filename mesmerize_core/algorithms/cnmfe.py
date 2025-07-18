import click
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.params import CNMFParams
from caiman.paths import decode_mmap_filename_dict
from caiman.base.movies import get_file_size
import traceback
from pathlib import Path, PurePosixPath
from shutil import move as move_file
import time

if __name__ in ["__main__", "__mp_main__"]:  # when running in subprocess
    from mesmerize_core import set_parent_raw_data_path, load_batch
    from mesmerize_core.utils import IS_WINDOWS
    from mesmerize_core.algorithms._utils import (
        ensure_server,
        save_projections_parallel,
        save_c_order_mmap_parallel,
        estimate_n_pixels_per_process
    )
else:  # when running with local backend
    from ..batch_utils import set_parent_raw_data_path, load_batch
    from ..utils import IS_WINDOWS
    from ._utils import ensure_server, save_projections_parallel, save_c_order_mmap_parallel, estimate_n_pixels_per_process


def run_algo(batch_path, uuid, data_path: str = None, dview=None):
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

            # only re-save memmap if necessary
            save_new_mmap = True
            if Path(input_movie_path).suffix == ".mmap":
                mmap_info = decode_mmap_filename_dict(input_movie_path)
                save_new_mmap = "order" not in mmap_info or mmap_info["order"] != "C"

            if save_new_mmap:
                print("making memmap")
                dims, T = get_file_size(input_movie_path, var_name_hdf5=cnmfe_params_dict.data['var_name_hdf5'])
                assert isinstance(T, int)
                print('Movie dims:', dims)
                print('N frames:', T)
                print('N processes:', n_processes)
                print('N pixels per process:', chunk_size := estimate_n_pixels_per_process(n_processes, T, dims))
                print('Columns per chunk:', max(chunk_size // dims[0], 1))
                breakpoint()
                fname_new = save_c_order_mmap_parallel(
                    input_movie_path,
                    base_name=f"{uuid}_cnmf-memmap_",
                    dview=dview,
                    var_name_hdf5=cnmfe_params_dict.data['var_name_hdf5']
                )
                cnmf_memmap_path = output_dir.joinpath(Path(fname_new).name)
                move_file(fname_new, cnmf_memmap_path)
            else:
                cnmf_memmap_path = Path(input_movie_path)

            Yr, dims, T = cm.load_memmap(str(cnmf_memmap_path))
            images = np.reshape(Yr.T, [T] + list(dims), order="F")

            # TODO: if projections already exist from mcorr we don't
            #  need to waste compute time re-computing them here
            proj_paths = save_projections_parallel(
                uuid=uuid, movie_path=cnmf_memmap_path, output_dir=output_dir, dview=dview
            )

            d = dict()  # for output

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

            if IS_WINDOWS:
                Yr._mmap.close()  # accessing private attr but windows is annoying otherwise

            # save path as relative path strings with forward slashes
            cnmfe_memmap_path = str(
                PurePosixPath(df.paths.split(cnmf_memmap_path)[1])
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
