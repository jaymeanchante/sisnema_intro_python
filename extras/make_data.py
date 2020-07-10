import pandas as pd
from sklearn.datasets import load_iris

# load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame({"iris": iris.target})
y.loc[:, "iris"] = y["iris"].apply(lambda x: iris.target_names[x])
df = pd.concat([X, y], axis=1)
# resampling
df = df.sample(frac=1).sample(frac=1).reset_index(drop=True).reset_index().rename(columns={"index": "id"})
# splitting
test = df[:15].copy()
test_answers = test[["id", "iris"]]
del test["iris"]
valid = df[15:45].copy()
train = df[45:].copy()
# dump to database
conn = sqlite3.connect("iris.sqlite")
test.to_sql(name="test", con=conn, if_exists="replace", index=False)
test_answers.to_sql(name="test_answers", con=conn, if_exists="replace", index=False)
valid.to_sql(name="valid", con=conn, if_exists="replace", index=False)
train.to_sql(name="train", con=conn, if_exists="replace", index=False)
conn.close()