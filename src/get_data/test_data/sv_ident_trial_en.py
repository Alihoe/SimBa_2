from pathlib import Path
import pandas as pd
import requests

from src.get_data import DATA_PATH


def get_variable_names(sequence):
    splitted = sequence.split("-")
    variable = splitted[0]
    relevance = splitted[1]
    if relevance == "Yes":
        return True, "v"+variable
    else:
        return False, variable

def run():

    data_name = "sv_ident_trial_en"
    general_path = DATA_PATH + data_name

    Path(general_path).mkdir(parents=True, exist_ok=True)

    queries_path_en = general_path + "/queries.tsv"
    qrels_path_en = general_path + "/gold.tsv"

    queries_qrels_en = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/test/en.tsv")

    with open(queries_path_en, 'wb') as f:
        f.write(queries_qrels_en.content)

    df_en = pd.read_csv(queries_path_en, sep='\t')
    df_en = df_en.loc[df_en['is_variable'] != 0]

    queries_df_en =  pd.DataFrame(columns=['uuid', 'text'])
    qrels_df_en = pd.DataFrame(columns=['uuid', '0', 'variable', '1'])

    for index, row in df_en.iterrows():
        uuid = row['uuid']
        text = row['text']
        variables = row['variable'].split(",")
        for variable in variables:
            has_variable, id = get_variable_names(variable)
            if has_variable:
                qrels_df = pd.DataFrame([[uuid, 0, id, 1]], columns=['uuid', '0', 'variable', '1'])
                qrels_df_en = pd.concat([qrels_df_en, qrels_df])
                queries_df = pd.DataFrame([[uuid, text]], columns=['uuid', 'text'])
                queries_df_en = pd.concat([queries_df_en, queries_df])

    queries_df_en = queries_df_en.drop_duplicates()

    qrels_df_en.to_csv(qrels_path_en, sep= '\t', header = False, index=False)
    queries_df_en.to_csv(queries_path_en, sep= '\t', header = False, index=False)

    targets_path_en = general_path + "/corpus"
    targets_en = requests.get("https://github.com/vadis-project/sv-ident/raw/main/data/trial/vocabulary/en.tsv")
    with open(targets_path_en, 'wb') as f:
        f.write(targets_en.content)


if __name__ == "__main__":
    run()