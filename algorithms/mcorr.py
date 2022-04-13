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


# prevent circular import
if __name__ == '__main__':
    from mesmerize_napari.core.batch_utils import set_parent_data_path, get_full_data_path


@click.command()
@click.option('--batch-path', type=str)
@click.option('--uuid', type=str)
@click.option('--data-path', type=str)
def main(batch_path, uuid, data_path: str = None):
    df = pd.read_pickle(batch_path)
    item = df[df['uuid'] == uuid].squeeze()

    input_movie_path = item['input_movie_path']

    set_parent_data_path(data_path)
    input_movie_path = str(get_full_data_path(input_movie_path))

    params = item['params']

    # adapted from current demo notebook
    if 'MESMERIZE_N_PROCESSES' in os.environ.keys():
        try:
            n_processes = int(os.environ["MESMERIZE_N_PROCESSES"])
        except:
            n_processes = psutil.cpu_count() - 1
    else:
        n_processes = psutil.cpu_count() - 1

    print("starting mc")
    # Start cluster for parallel processing
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local',
        n_processes=n_processes,
        single_thread=False
    )

    rel_params = dict(params['mcorr_kwargs'])
    opts = CNMFParams(params_dict=rel_params)
    # Run MC, denote boolean 'success' if MC completes w/out error
    try:
        # Run MC
        fnames = [input_movie_path]
        mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
        mc.motion_correct(save_movie=True, base_name_prefix=uuid)

        # Find path to mmap file
        output_path = Path(mc.mmap_file[0])
        if data_path is not None:
            output_path = Path(output_path).relative_to(data_path)

        print("mc finished successfully!")

        print("computing projections")
        Yr, dims, T = cm.load_memmap(str(get_full_data_path(output_path)))
        images = np.reshape(Yr.T, [T] + list(dims), order='F')

        paths=[]
        for proj_type in ['mean', 'std', 'max']:
            p_img = getattr(np, f"nan{proj_type}")(images, axis=0)
            np.save(str(Path(input_movie_path).parent.joinpath(f"{uuid}_{proj_type}.npy")), p_img)
            paths.append(str(Path(input_movie_path).parent.joinpath(f"{uuid}_{proj_type}.npy")))


        print("Computing correlation image")
        Cns = local_correlations_movie_offline([mc.mmap_file[0]],
                                               remove_baseline=True, window=1000, stride=1000,
                                               winSize_baseline=100, quantil_min_baseline=10,
                                               dview=dview)
        Cn = Cns.max(axis=0)
        Cn[np.isnan(Cn)] = 0
        cn_path = str(Path(input_movie_path).parent.joinpath(f'{uuid}_cn.npy'))
        np.save(cn_path, Cn, allow_pickle=False)

        if data_path is not None:
            cn_path = Path(cn_path).relative_to(data_path)

        print("finished computing correlation image")

        # Compute shifts
        if params['mcorr_kwargs']['pw_rigid'] == True:
            x_shifts = mc.x_shifts_els
            y_shifts = mc.y_shifts_els
            shifts = [x_shifts, y_shifts]
            shift_path = str(Path(input_movie_path).parent.joinpath(f"{uuid}_shifts.npy"))
            np.save(shift_path, shifts)
        else:
            shifts = mc.shifts_rig
            shift_path = str(Path(input_movie_path).parent.joinpath(f"{uuid}_shifts.npy"))
            np.save(shift_path, shifts)

        d = dict()
        d.update(
            {
                "mcorr-output-path": output_path,
                "corr-img-path": cn_path,
                "mean-projection-path": paths[0],
                "std-projection-path": paths[1],
                "max-projection-path": paths[2],
                "shifts": shift_path,
                "success": True,
                "traceback": None
            }
        )
    except:
        d = {"success": False, "traceback": traceback.format_exc()}
        print("mc failed, stored traceback in output")

    print(d)
    # Add dictionary to output column of series
    df.loc[df['uuid'] == uuid, 'outputs'] = [d]
    # Save DataFrame to disk
    df.to_pickle(batch_path)


if __name__ == "__main__":
    main()
