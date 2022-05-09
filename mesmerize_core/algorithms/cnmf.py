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

# prevent circular import
if __name__ == '__main__':
    from mesmerize_core import set_parent_data_path, get_full_data_path


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
    print("cnmf params", params)

    # adapted from current demo notebook
    n_processes = psutil.cpu_count() - 1
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
    )

    # merge cnmf and eval kwargs into one dict
    c = dict(params['cnmf_kwargs'])
    e = dict(params['eval_kwargs'])
    tot = {**c, **e}
    cnmf_params = CNMFParams(params_dict=tot)
    # Run CNMF, denote boolean 'success' if CNMF completes w/out error
    try:
        fname_new = cm.save_memmap(
            [input_movie_path],
            base_name=f'{uuid}_cnmf-memmap_',
            order='C',
            dview=dview
        )

        print('making memmap')

        Yr, dims, T = cm.load_memmap(fname_new)
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        paths = []
        for proj_type in ['mean', 'std', 'max']:
            p_img = getattr(np, f'nan{proj_type}')(images, axis=0)
            np.save(str(Path(input_movie_path).parent.joinpath(f'{uuid}_{proj_type}.npy')), p_img)
            paths.append(str(Path(input_movie_path).parent.joinpath(f'{uuid}_{proj_type}.npy')))


        # in fname new load in memmap order C

        cm.stop_server(dview=dview)
        c, dview, n_processes = cm.cluster.setup_cluster(
            backend='local',
            n_processes=None,
            single_thread=False
        )

        print("performing CNMF")
        cnm = cnmf.CNMF(
            n_processes,
            params=cnmf_params,
            dview=dview
        )

        print("fitting images")
        cnm = cnm.fit(images)
        #
        if params['refit'] is True:
            print('refitting')
            cnm = cnm.refit(images, dview=dview)

        print("Eval")
        cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

        output_path = str(get_full_data_path(input_movie_path).parent.joinpath(f"{uuid}.hdf5").resolve())

        cnm.save(output_path)

        Cn = cm.local_correlations(images.transpose(1, 2, 0))
        Cn[np.isnan(Cn)] = 0

        corr_img_path = Path(input_movie_path).parent.joinpath(f'{uuid}_cn.npy').resolve()
        np.save(str(corr_img_path), Cn, allow_pickle=False)

        if data_path is not None:
            cnmf_hdf5_path = Path(output_path).relative_to(data_path)
            cnmf_memmap_path = Path(fname_new).relative_to(data_path)
            corr_img_path = corr_img_path.relative_to(data_path)
        else:
            cnmf_hdf5_path = output_path
            cnmf_memmap_path = fname_new

        d = dict()
        d.update(
            {
                "cnmf-hdf5-path": cnmf_hdf5_path,
                "cnmf-memmap-path": cnmf_memmap_path,
                "corr-img-path": corr_img_path,
                "mean-projection-path": paths[0],
                "std-projection-path": paths[1],
                "max-projection-path": paths[2],
                "success": True,
                "traceback": None
            }
        )

    except:
        d = {"success": False, "traceback": traceback.format_exc()}
    
    print(f"Final output dict:\n{d}")
    
    # Add dictionary to output column of series
    df.loc[df['uuid'] == uuid, 'outputs'] = [d]
    # save dataframe to disc
    df.to_pickle(batch_path)


if __name__ == "__main__":
    main()
