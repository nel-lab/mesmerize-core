test_params = {
    "mcorr": {
        "main": {
            "max_shifts": [24, 24],
            "strides": [48, 48],
            "overlaps": [24, 24],
            "max_deviation_rigid": 3,
            "border_nan": "copy",
            "pw_rigid": True,
            "gSig_filt": None,
        },
        "optional_outputs":
            {
            "corr_image":
                {
                    "remove_baseline": True,
                    "window": 1000,
                    "stride": 1000,
                    "winSize_baseline": 100,
                    "quantil_min_baseline": 10,
                },
            },
    },
    "cnmf": {
        "main": {
            "p": 2,
            "nb": 1,
            # raises error: no parameter 'merge_thresh' found
            # 'merge_thresh': 0.7,
            "rf": None,
            "stride": 30,
            "K": 10,
            "gSig": [5, 5],
            "ssub": 1,
            "tsub": 1,
            "method_init": "greedy_roi",
            "min_SNR": 2.50,
            "rval_thr": 0.8,
            "use_cnn": True,
            "min_cnn_thr": 0.8,
            "cnn_lowest": 0.1,
            "decay_time": 1,
        },
        "refit": True,
    },
    "cnmfe_full": {
        "do_cnmfe": True,
        "keep_memmap": True,
        "main": {
            "gSig": (10, 10),
            "gSiz": (41, 41),
            "p": 1,
            "min_corr": 0.89,
            "min_pnr": 4,
            "rf": 50,
            "stride": 30,
            "gnb": 1,
            "nb_patch": 1,
            "K": 10,
            "ssub": 1,
            "tsub": 1,
            "ring_size_factor": 1.5,
            "merge_thresh": 0.7,
            "low_rank_background": False,
            "method_deconvolution": "oasis",
            "update_background_components": True,
            "del_duplicates": True,
        },
        "downsample_ratio": 1,
    },
    "cnmfe_partial": {
        "do_cnmfe": False,
        "keep_memmap": True,
        "main": {
            "gSig": (10, 10),
        },
        "downsample_ratio": 1,
    },
}
