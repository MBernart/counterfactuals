# extra requirement: pip install ucimlrepo
from ucimlrepo import fetch_ucirepo 
from explainers_lib.datasets import Dataset

student_performance = fetch_ucirepo(id=320) 
  
X = student_performance.data.features 
y = student_performance.data.targets 
vars = student_performance.variables

cat = vars.loc[((vars['type'] == 'Categorical') | (vars['type'] == 'Binary')) & (vars['role'] == 'Feature'), 'name'].tolist()
num = vars.loc[(vars['type'] == 'Integer') & (vars['role'] == 'Feature'), 'name'].tolist()

print(X)

data = Dataset(X, y['G3'].tolist(), X.columns.tolist(),
               immutable_features=["sex", "age"],
               categorical_features=cat,
               continuous_features=num,
               allowable_ranges={
                   "Medu": (0, 4),
                   "Fedu": (0, 4),
                   "traveltime": (1, 4),
                   "studytime": (1, 4),
                   "failures": (1, 4),
                   "famrel": (1, 5),
                   "freetime": (1, 5),
                   "goout": (1, 5),
                   "Dalc": (1, 5),
                   "Walc": (1, 5),
                   "health": (1, 5),
                   "absences": (0, 93),
               })

print(data.data.shape)

print(data.inverse_transform(data.data))