import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ucimlrepo import fetch_ucirepo
from explainers_lib.datasets import Dataset
from explainers_lib.explainers.growing_spheres import GrowingSpheresExplainer
from explainers_lib.model import TorchModel
from explainers_lib.ensemble import print_cfs

student_performance = fetch_ucirepo(id=320)

X = student_performance.data.features
y = student_performance.data.targets

y_binary = (y["G3"] >= 10).astype(int)

vars = student_performance.variables

cat = vars.loc[
    ((vars["type"] == "Categorical") | (vars["type"] == "Binary"))
    & (vars["role"] == "Feature"),
    "name",
].tolist()
num = vars.loc[
    (vars["type"] == "Integer") & (vars["role"] == "Feature"), "name"
].tolist()

ds = Dataset(
    X,
    y_binary.tolist(),
    X.columns.tolist(),
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
    },
)

input_size = ds.data.shape[1]
output_size = len(set(y_binary))


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


net = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

X_tensor = torch.tensor(ds.data, dtype=torch.float32)
y_tensor = torch.tensor(ds.target, dtype=torch.long)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

torch_model = TorchModel(net)

explainer = GrowingSpheresExplainer(num_samples=1000)
explainer.fit(torch_model, ds)

instances_to_explain = ds[:5]
cfs = explainer.explain(torch_model, instances_to_explain)

print("Counterfactuals")
print_cfs(cfs, postprocess=ds.inverse_transform, feature_names=ds.features)

print("\nVerification")
for i, cf in enumerate(cfs):
    original_df = ds.inverse_transform(cf.original_data.reshape(1, -1))
    counterfactual_df = ds.inverse_transform(cf.data.reshape(1, -1))

    original = original_df.iloc[0]
    counterfactual = counterfactual_df.iloc[0]

    print(f"\nInstance {i}:")

    immutable_features = ds.immutable_features
    for feature in immutable_features:
        val_orig = original[feature]
        val_cf = counterfactual[feature]

        is_equal = False
        if isinstance(val_orig, float) or isinstance(val_cf, float):
            is_equal = np.allclose([val_orig], [val_cf])
        else:
            is_equal = val_orig == val_cf

        if not is_equal:
            print(
                f"  [FAIL] Immutable feature '{feature}' changed: {val_orig} -> {val_cf}"
            )
        else:
            print(f"  [OK] Immutable feature '{feature}' unchanged: {val_orig}")

    for feature, (min_val, max_val) in ds.allowable_ranges.items():
        if feature in ds.immutable_features:
            continue
        if feature in counterfactual.index:
            cf_val = counterfactual[feature]
            if not (min_val - 1e-6 <= cf_val <= max_val + 1e-6):
                print(
                    f"  [FAIL] Feature '{feature}' out of range ({min_val}, {max_val}): {cf_val}"
                )
            else:
                print(
                    f"  [OK] Feature '{feature}' within range ({min_val}, {max_val}): {cf_val}"
                )
