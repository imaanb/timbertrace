import pandas as pd
from tqdm import tqdm
#  Laod and clean data 
path = 'data/presave.csv'
df = pd.read_csv(path)
df.columns = [col.strip() for col in df.columns]

# indentify the genetic columns and make refence genome
gencolumns = [col for col in df.columns if col.startswith('P')]
gencolumns = gencolumns[3:]
print(list(gencolumns))

reference_genome = df.loc[0, gencolumns].to_dict()
for key in reference_genome.keys():
    reference_genome[key] = str(reference_genome[key])[1]


def transform_genetic_data(df, gencolumns, reference_genome):
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transforming genetic data"):
        for col in gencolumns:
            val = str(row[col])
            if len(val) >= 2:
                if val[1] == reference_genome[col] and val[3] == reference_genome[col]:
                    df.at[idx, col] = float(0)
                elif (val[1] != reference_genome[col] and val[3] == reference_genome[col]) or (val[1] == reference_genome[col] and val[3] != reference_genome[col]):
                    df.at[idx, col] = float(1)
                elif (val[1] != reference_genome[col] and val[3] != reference_genome[col]):
                    df.at[idx, col] = float(2)

    return df[gencolumns]



df[gencolumns] = transform_genetic_data(df, gencolumns, reference_genome)
df.to_csv('data/transformed_genetic_data.csv', index=False)