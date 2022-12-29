import argparse
from pathlib import Path

from src.pre_processing import DATA_PATH
from src.utils import get_certain_target_fields


def run():
    """
    input:
    targets
    output:
    pre-processed targets

    Keep document structure intact
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('targets', type=str, help='Input targets path as tsv file.')
    parser.add_argument('data', type=str, help='Name under which the documents should be stored.')
    parser.add_argument('-fields', type=str, nargs='+', default="all", help='Fields to keep in target file.')
    args = parser.parse_args()

    output_dir = DATA_PATH + args.data +'/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.fields != 'all':
        fields = ['id']
        fields.extend(args.fields)
        field_names = '_'.join(args.fields)
        output_path = output_dir + field_names + '_pp_targets.tsv'
        targets_df = get_certain_target_fields(args.targets, fields)
        targets_df.to_csv(output_path, index=False, header=True, sep='\t')


if __name__ == "__main__":
    run()


