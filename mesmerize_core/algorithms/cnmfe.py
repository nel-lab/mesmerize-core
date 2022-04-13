import pathlib
import click
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf.params import CNMFParams
import psutil
import pandas as pd
import pickle
import traceback
from napari.viewer import Viewer
from pathlib import Path

if __name__ == '__main__':
    from mesmerize_napari.core import set_parent_data_path, get_full_data_path

@click.command()
@click.option('--batch-path', type=str)
@click.option('--uuid', type=str)
@click.option('--data-path')
def main(batch_path, uuid, data_path: str = None):
    df = pd.read_pickle(batch_path)
    item = df[df['uuid'] == uuid].squeeze()

    input_movie_path = item['input_movie_path']
    set_parent_data_path(data_path)
    input_movie_path = str(get_full_data_path(input_movie_path))

    params = item['params']
    print("cnmfe params:", params)

    #adapted from current demo notebook
    n_processes = psutil.cpu_count() - 1
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
     )

    try:
        fname_new = cm.save_memmap(
            [input_movie_path],
            base_name=f'{uuid}_cnmf-memmap_',
            order='C',
            dview=dview
        )

        print('making memmap')
        gSig = params['cnmfe_kwargs']['gSig'][0]

        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        mean_projection_path = str(Path(input_movie_path).parent.joinpath(f'{uuid}_mean_projection.npy'))
        std_projection_path = str(Path(input_movie_path).parent.joinpath(f'{uuid}_std_projection.npy'))
        max_projection_path = str(Path(input_movie_path).parent.joinpath(f'{uuid}_max_projection.npy'))
        np.save(mean_projection_path, np.mean(images, axis=0))
        np.save(std_projection_path, np.std(images, axis=0))
        np.save(max_projection_path, np.max(images, axis=0))

        downsample_ratio = params['downsample_ratio']
        # in fname new load in memmap order C

        cn_filter, pnr = cm.summary_images.correlation_pnr(
            images[::downsample_ratio], swap_dim=False, gSig=gSig
        )

        pnr_output_path = str(Path(input_movie_path).parent.joinpath(f"{uuid}_pn.npy").resolve())
        cn_output_path = str(Path(input_movie_path).parent.joinpath(f"{uuid}_cn.npy").resolve())

        np.save(str(cn_output_path), cn_filter, allow_pickle=False)
        np.save(str(pnr_output_path), pnr, allow_pickle=False)

        d = dict()  # for output

        if params['do_cnmfe']:
            cnmfe_params_dict = \
                {
                    "method_init": 'corr_pnr',
                    "n_processes": n_processes,
                    "only_init": True,    # for 1p
                    "center_psf": True,         # for 1p
                    "normalize_init": False     # for 1p
                }
            tot = {**cnmfe_params_dict, **params['cnmfe_kwargs']}
            cnmfe_params_dict = CNMFParams(params_dict=tot)
            cnm = cnmf.CNMF(
                n_processes=n_processes,
                dview=dview,
                params=cnmfe_params_dict
            )
            print("Performing CNMFE")
            cnm = cnm.fit(images)
            print("evaluating components")
            cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

            output_path = str(Path(input_movie_path).parent.joinpath(f"{uuid}.hdf5").resolve())
            cnm.save(output_path)

            if data_path is not None:
                cnmf_hdf5_path = Path(output_path).relative_to(data_path)
            else:
                cnmf_hdf5_path = output_path

            d.update(
                {
                    "cnmf-hdf5-path": cnmf_hdf5_path,
                }
            )

        if data_path is not None:
            cnmfe_memmap_path = Path(fname_new).relative_to(data_path)
            cn_output_path = Path(cn_output_path).relative_to(data_path)
            pnr_output_path = Path(pnr_output_path).relative_to(data_path)
        else:
            cnmfe_memmap_path = fname_new

        d.update(
            {
                "cnmf-memmap-path": cnmfe_memmap_path,
                "corr-img-path": cn_output_path,
                "pnr-image-path": pnr_output_path,
                "mean-projection-path": mean_projection_path,
                "std-projection-path": std_projection_path,
                "max-projection-path": max_projection_path,
                "success": True,
                "traceback": None
            }
        )

        print(d)

    except:
        d = {"success": False, "traceback": traceback.format_exc()}

    # Add dictionary to output column of series
    df.loc[df['uuid'] == uuid, 'outputs'] = [d]
    # save dataframe to disc
    df.to_pickle(batch_path)


if __name__ == "__main__":
    main()