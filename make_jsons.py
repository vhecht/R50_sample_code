import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from csv import DictReader

import argparse
from collections import namedtuple

# define named tuple with fields for model upload
ModelPaths = namedtuple(
    'ModelPaths', [
        'chrombpnet_no_bias_h5',
        'chrombpnet_no_bias_tar',
        'chrombpnet_h5',
        'chrombpnet_tar',
        'bias_scaled_h5',
        'bias_scaled_tar']
)

# define named tuple for logs
LogPaths = namedtuple(
    'LogPaths', [
        'chrombpnet_args_json',
        'chrombpnet_data_params_tsv',
        'chrombpnet_model_params_tsv',
        'chrombpnet_log',
        'chrombpnet_log_batch',
        'h5_to_tar_log_chrombpnet_no_bias',
        'h5_to_tar_log_chrombpnet',
        'h5_to_tar_log_bias_scaled'
    ]
)

TrainTestValPaths = namedtuple(
    'TrainTestValPaths', [
        'negatives_with_summit_bed_gz',
        'peaks_trainingset',
        'peaks_valset',
        'peaks_testset',
        'nonpeaks_trainingset',
        'nonpeaks_valset',
        'nonpeaks_testset',
    ]
)


def validate_file_paths(upload_dict: Dict) -> Dict:
    """Use this to traverse all the levels of the filepaths dictionary

    Args:
        upload_dict: dictionary to be converted to json
    Returns:
        None
    """
    # need to make copy of dict to iterate over; otherwise, loop will
    # throw 'DictionaryHasChanged' error
    upload_dict_iterator = upload_dict.copy()

    for key, value in upload_dict_iterator.items():
        if isinstance(value, dict):
            validate_file_paths(value)
        else:
            if not isinstance(value, list):
                continue
            if key == 'bam files':
                continue
            # drop items that are None
            value = [item for item in value if item[0] != 'None']
            upload_dict[key] = value
            # if no items remain in list, then drop that key
            if not value:
                print(f"Deleting {key}")
                del upload_dict[key]
                continue
            # file paths are stored as the first element in a list of lists
            for item in value:
                filepath = Path(item[0])
                if filepath.suffix == ".tar":
                    assert filepath.is_dir(), f"{filepath} does not exist!"
                else:
                    assert filepath.is_file(), f"{filepath} does not exist!"

    return upload_dict


def construct_model_upload_json(
        bias_model_encid: str,
        experiment: str,
        bam_to_experiment: Dict,
        assay: str,
        observed_signal_profile_bigwig: str,
        readme_filepath: str,
        train_test_val_readme: str,
        input_regions: str,
        model_paths: List[ModelPaths],
        log_paths: List[LogPaths],
        train_test_val_paths: List[TrainTestValPaths],
        splits_json: List[str],
        outfile: Path
) -> None:
    """
    Create json for uploading models. If any files are missing, will throw an error

    Args:
        bias_model_encid: ENCID for matching bias model. DNase should be ENCSR283TME, ENCSR880CUB, or ENCSR146KFX
        experiment: ENCSR###ABC
        bam_to_experiment: dictionary with key = experiment, value = list of associated bam files
        assay: DNASE or ATAC
        observed_signal_profile_bigwig: bigwig generated from bam in preprocessing
        readme_filepath: location of models.README
        train_test_val_readme: location of train test val README
        input_regions: bed.gz with peak regions used for training
        model_paths: list of ModelPaths objects containing all paths for model files to be uploaded
        log_paths: list of LogPaths objects containing all paths for log files to be uploaded
        train_test_val_paths: list of TrainTestValPaths containing all paths for train test vals to be uploaded
        splits_json: list of locations of json files with splits
        outfile: where to save jsons
    """
    # initialize dictionary, populate straightforward fields
    upload_metadata: Dict = {
        "upload bias": "false",
        "bias model encid": bias_model_encid,
        "experiment": experiment,
        "bam files": bam_to_experiment[experiment].split(' '),
        "assay": assay,
        "observed signal profile bigWig": observed_signal_profile_bigwig,
        "models tar": {
            "file.paths": [
                [
                    readme_filepath, "README.md"
                ]
            ],
        },
        "training and test regions tar": {
            "file.paths": [
                [
                    train_test_val_readme, "README.md"
                ],
                [
                    input_regions, f"peaks.all_input_regions.{experiment}.bed.gz"
                ]
            ]
        }
    }

    for fold_num in range(0, 5):
        cur_model = model_paths[fold_num]
        cur_logs = log_paths[fold_num]
        cur_train_test_val = train_test_val_paths[fold_num]
        upload_metadata['models tar'][f'fold_{fold_num}']: Dict = {
            "file.paths": [
                [
                    str(cur_model.chrombpnet_no_bias_h5),
                    f"model.chrombpnet_nobias.fold_{fold_num}.{experiment}.h5"
                ],
                [
                    str(cur_model.chrombpnet_h5),
                    f"model.chrombpnet.fold_{fold_num}.{experiment}.h5"
                ],
                [
                    str(cur_model.bias_scaled_h5),
                    f"model.bias_scaled.fold_{fold_num}.{experiment}.h5"
                ],
                [
                    str(cur_model.chrombpnet_no_bias_tar),
                    f"model.chrombpnet_nobias.fold_{fold_num}.{experiment}.tar"
                ],
                [
                    str(cur_model.chrombpnet_tar),
                    f"model.chrombpnet.fold_{fold_num}.{experiment}.tar"
                ],
                [
                    str(cur_model.bias_scaled_tar),
                    f"model.bias_scaled.fold_{fold_num}.{experiment}.tar"
                ]
            ],
            f"logs.models.fold_{fold_num}.{experiment}": {
                "file.paths": [
                    [
                        str(cur_logs.h5_to_tar_log_chrombpnet_no_bias),
                        f"logfile.modelling.fold_{fold_num}.{experiment}.chrombpnet_no_bias_formatting.stdout.txt"
                    ],
                    [
                        str(cur_logs.h5_to_tar_log_chrombpnet),
                        f"logfile.modelling.fold_{fold_num}.{experiment}.chrombpnet_formatting.stdout.txt"
                    ],
                    [
                        str(cur_logs.h5_to_tar_log_bias_scaled),
                        f"logfile.modelling.fold_{fold_num}.{experiment}.bias_formatting.stdout.txt"
                    ],
                    [
                        str(cur_logs.chrombpnet_args_json),
                        f"logfile.modelling.fold_{fold_num}.{experiment}.args.json"
                    ],
                    [
                        str(cur_logs.chrombpnet_data_params_tsv),
                        f"logfile.modelling.fold_{fold_num}.{experiment}.chrombpnet_data_params.tsv"
                    ],
                    [
                        str(cur_logs.chrombpnet_model_params_tsv),
                        f"logfile.modelling.fold_{fold_num}.{experiment}.chrombpnet_model_params.tsv"
                    ],
                    [
                        str(cur_logs.chrombpnet_log),
                        f"logfile.modelling.fold_{fold_num}.{experiment}.epoch_loss.csv"
                    ],
                    [
                        str(cur_logs.chrombpnet_log_batch),
                        f"logfile.modelling.fold_{fold_num}.{experiment}.batch_loss.tsv"
                    ]
                ]
            }
        }

        upload_metadata['training and test regions tar'][f'fold_{fold_num}']: Dict = {
            "file.paths": [
                [
                    str(splits_json[fold_num]), f"cv_params.fold_{fold_num}.json"
                ],
                [
                    str(cur_train_test_val.negatives_with_summit_bed_gz),
                    f"nonpeaks.{fold_num}_input_regions.{experiment}.bed.gz"
                ],
                [
                    str(cur_train_test_val.peaks_trainingset),
                    f"peaks.trainingset.fold_{fold_num}.{experiment}.bed.gz"
                ],
                [
                    str(cur_train_test_val.peaks_valset),
                    f"peaks.validationset.fold_{fold_num}.{experiment}.bed.gz"
                ],
                [
                    str(cur_train_test_val.peaks_testset),
                    f"peaks.testset.fold_{fold_num}.{experiment}.bed.gz"
                ],
                [
                    str(cur_train_test_val.nonpeaks_trainingset),
                    f"nonpeaks.trainingset.fold_{fold_num}.{experiment}.bed.gz"
                ],
                [
                    str(cur_train_test_val.nonpeaks_valset),
                    f"nonpeaks.validationset.fold_{fold_num}.{experiment}.bed.gz"
                ],
                [
                    str(cur_train_test_val.nonpeaks_testset),
                    f"nonpeaks.testset.fold_{fold_num}.{experiment}.bed.gz"
                ]
            ]
        }

    # check that all files exist
    validate_file_paths(upload_dict=upload_metadata)

    with outfile.open('w') as jsonFile:
        json.dump(upload_metadata, jsonFile, indent=4)


def collect_metadata_for_model_upload(
        experiment: str,
        assay: str,
        metadata_in: Path,
        bam_to_experiment_in: Path,
        model_tar_parent: Path,
        train_test_val_parent: Path,
        path_to_splits: Path,
        models_readme_in: Path,
        train_test_val_readme_in: Path,
        json_outfile: Path,
        metadata_sep: str = '\t'
) -> None:
    """Collect information for model upload json

    Args:
        experiment: experiment ENCID
        assay: DNase-seq or ATAC-seq
        metadata_in: path to metadata file with the following fields:
        *experiment DONE
        *bias_model_encid
        *peaks_bed - regions used to train model DONE
        *nonpeaks_fold[0-4] - nonpeak regions used to train model
        *signal_bw - signal bigwig
        *fold_[0-4]_model - {somePath}/{EXPID or not.}chombpnet_wo_bias.h5 or chrombpnet_no_bias.h5
        bam_to_experiment_in: bam_to_experiment_in: two-col file with expt ENCIDs in 1st col and BAM
        ENCFFs in second col. Maybe has a name like 'expt_bam_lookup.txt'
        model_tar_parent: parent directory for new model formats
        models_readme_in: path to models.README
        train_test_val_readme_in: path to train test val models.README
        metadata_sep: metadata file delimiter
        train_test_val_parent: parent directory for train/test/val peaks and non-peaks. Should have
        pattern like {parent_folder}/encid/train_test_val/; encid will be replaced with experiment
        path_to_splits: directory with train/test/val splits for each fold
        json_outfile: where to save json files
    """
    bam_to_experiment: Dict = {}
    with bam_to_experiment_in.open('r') as f:
        reader = DictReader(f, delimiter='\t')
        for line in reader:
            bam_to_experiment[line['ENCSRID']] = line['BAM_ENCFF'].lstrip(' ')

    metadata_df = pd.read_csv(metadata_in, delimiter=metadata_sep, index_col='experiment')

    bias_model_encid = metadata_df.loc[experiment, 'bias_model_encid']
    signal_bw = metadata_df.loc[experiment, 'signal_bw']
    peaks_bed = metadata_df.loc[experiment, 'peaks_bed']
    assert Path(signal_bw).is_file(), f"{signal_bw} does not exist!"

    # get splits files
    splits: List[str] = [str(path_to_splits / f"fold_{n}.json") for n in range(0, 5)]

    model_paths: List[ModelPaths] = []
    log_paths: List[LogPaths] = []
    train_test_val_paths: List[TrainTestValPaths] = []

    for fold in range(0, 5):
        col = f"fold_{fold}_model"

        # models first
        chrombpnet_no_bias_h5 = Path(metadata_df.loc[experiment, col])
        # some models have expt appended in front
        prefix = ''
        if experiment in chrombpnet_no_bias_h5.stem:
            prefix = f"{experiment}_"
        model_parent_dir = chrombpnet_no_bias_h5.parent
        model_tar_dir = model_tar_parent / f"{experiment}/fold_{fold}/new_model_format"
        model_paths.append(ModelPaths(
            chrombpnet_no_bias_h5=chrombpnet_no_bias_h5,
            chrombpnet_no_bias_tar=model_tar_dir / 'chrombpnet_nobias.tar',
            chrombpnet_h5=model_parent_dir / f'{prefix}chrombpnet.h5',
            chrombpnet_tar=model_tar_dir / 'chrombpnet.tar',
            bias_scaled_h5=model_parent_dir / f'{prefix}bias_model_scaled.h5',
            bias_scaled_tar=model_tar_dir / 'bias_model_scaled.tar')
        )

        # then logs
        if chrombpnet_no_bias_h5.stem == 'chrombpnet_wo_bias':
            logs_dir = chrombpnet_no_bias_h5.parent
        else:
            logs_dir = chrombpnet_no_bias_h5.parent.parent / "logs"
        log_paths.append(LogPaths(
            chrombpnet_args_json=logs_dir / f"{prefix}chrombpnet.args.json",
            chrombpnet_data_params_tsv=logs_dir / f"{prefix}chrombpnet_data_params.tsv",
            chrombpnet_model_params_tsv=logs_dir / f"{prefix}chrombpnet_model_params.tsv",
            chrombpnet_log=logs_dir / f"{prefix}chrombpnet.log",
            chrombpnet_log_batch=logs_dir / f"{prefix}chrombpnet.log.batch",
            h5_to_tar_log_chrombpnet_no_bias=model_tar_dir / "log_chrombpnet_nobias.txt",
            h5_to_tar_log_chrombpnet=model_tar_dir / "log_chrombpnet.txt",
            h5_to_tar_log_bias_scaled=model_tar_dir / "log_bias.txt"
        )
        )

        # then train/test/val regions
        train_test_val_cur_fold = train_test_val_parent / f"{experiment}/train_test_val/fold_{fold}"
        train_test_val_paths.append(TrainTestValPaths(
            negatives_with_summit_bed_gz=Path(metadata_df.loc[experiment, f"nonpeaks_fold{fold}"]),
            peaks_trainingset=train_test_val_cur_fold / 'peaks/regions_train.bed.gz',
            peaks_valset=train_test_val_cur_fold / 'peaks/regions_valid.bed.gz',
            peaks_testset=train_test_val_cur_fold / 'peaks/regions_test.bed.gz',
            nonpeaks_trainingset=train_test_val_cur_fold / 'nonpeaks/regions_train.bed.gz',
            nonpeaks_valset=train_test_val_cur_fold / 'nonpeaks/regions_valid.bed.gz',
            nonpeaks_testset=train_test_val_cur_fold / 'nonpeaks/regions_test.bed.gz',
        )
        )

    # finally: call function to create model json
    construct_model_upload_json(
        bias_model_encid=bias_model_encid,
        experiment=experiment,
        bam_to_experiment=bam_to_experiment,
        assay=assay,
        observed_signal_profile_bigwig=signal_bw,
        readme_filepath=str(models_readme_in),
        train_test_val_readme=str(train_test_val_readme_in),
        input_regions=peaks_bed,
        model_paths=model_paths,
        log_paths=log_paths,
        train_test_val_paths=train_test_val_paths,
        splits_json=splits,
        outfile=json_outfile
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate jsons to upload model files to ENCODE")
    parser.add_argument("--experiment", type=str, help="ENCSRID", required=True)
    parser.add_argument("--assay", type=str, help='DNase-seq or ATAC-seq')
    parser.add_argument("--bam_to_expt", type=Path, required=True, help="Two col file"
                                                                        "with expt in 1st, BAM ENCFF in 2nd")
    parser.add_argument("--json_outfile", type=Path, help="Where to save file")
    parser.add_argument("--json_outpath", type=Path,
                        help="Parent directory for where to save file")
    parser.add_argument("--metadata_in", type=Path, help='File with information about file metadata')
    parser.add_argument("--model_tar_parent", type=Path, help="Parent dir for chrombpnet model tar")
    parser.add_argument("--models_readme_in", type=str, help="Path to model.README")
    parser.add_argument("--train_test_val_readme_in", type=str, help="Path to train test val model.README")
    parser.add_argument("--metadata_sep", type=str, default='\t', help="Delimiter for metadata file")
    parser.add_argument("--project_parent_dir", type=Path)
    parser.add_argument("--train_test_val_parent", type=Path, help="Path to train_test_val files")
    parser.add_argument("--path_to_splits", type=Path)
    args = parser.parse_args()

    collect_metadata_for_model_upload(
        experiment=args.experiment,
        assay=args.assay,
        metadata_in=args.metadata_in,
        bam_to_experiment_in=args.bam_to_expt,
        model_tar_parent=args.model_tar_parent,
        train_test_val_parent=args.train_test_val_parent,
        path_to_splits=args.path_to_splits,
        models_readme_in=args.models_readme_in,
        train_test_val_readme_in=args.train_test_val_readme_in,
        json_outfile=args.json_outfile,
        metadata_sep=args.metadata_sep
    )