import os

from pathlib import Path

import pandas as pd
import requests

from src.get_data import DATA_PATH


def run():

    data_name = "sv_ident_train"
    storage_path = DATA_PATH + 'training/' + data_name

    Path(storage_path).mkdir(parents=True, exist_ok=True)
    general_path = storage_path

    queries_path_train = general_path + "/queries_train.tsv"

    queries_path = general_path + "/queries.tsv"
    qrels_path = general_path + "/gold.tsv"

    queries_qrels_train = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/train/train.tsv")

    with open(queries_path_train, 'wb') as f:
        f.write(queries_qrels_train.content)

    df = pd.read_csv(queries_path_train, sep='\t')
    df = df.loc[df['is_variable'] != 0]

    queries_df = pd.DataFrame(columns=['uuid', 'text'])
    qrels_df = pd.DataFrame(columns=['uuid', '0', 'variable', '1'])

    research_data = []

    for index, row in df.iterrows():
        uuid = row['uuid']
        sentence = row['sentence']
        variables = row['variable'].split(";")
        if variables and variables != ["unk"]:
            current_research_data = row['research_data'].split(";")
            research_data.extend(current_research_data)
        for variable in variables:
            if variable != "unk":
                qrels_df_row = pd.DataFrame([[uuid, 0, variable, 1]], columns=['uuid', '0', 'variable', '1'])
                qrels_df = pd.concat([qrels_df, qrels_df_row])
                queries_df_row = pd.DataFrame([[uuid, sentence]], columns=['uuid', 'text'])
                queries_df = pd.concat([queries_df, queries_df_row])

    queries_df = queries_df.drop_duplicates()

    qrels_df.to_csv(qrels_path, sep= '\t', header = False, index=False)
    queries_df.to_csv(queries_path, sep= '\t', header = False, index=False)

    os.remove(general_path + "/queries_train.tsv")



if __name__ == "__main__":
    run()