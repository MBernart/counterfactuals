import json
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from explainers_lib.counterfactual import Counterfactual
from explainers_lib.explainers import GrowingSpheresExplainer
from explainers_lib.ensemble import Ensemble
from explainers_lib.datasets import Dataset
from explainers_lib.model import Model
import numpy as np
import torch
from pydantic import BaseModel
from typing import List, Sequence
from torch import nn


class CounterfactualModel(BaseModel):
    """This is a helper class"""

    data: Sequence[float]
    original_class: int
    target_class: int


class InstanceRequest(BaseModel):
    data: Sequence[float]
    target_class: int


app = FastAPI()


@app.post("/growing-spheres/")
async def generate_growing_spheres_cf(
    instance_json: str = Form(...), model_file: UploadFile = File(...)
):
    instance_dict = json.loads(instance_json)
    instance = InstanceRequest(**instance_dict)

    model = torch.jit.load(model_file.file)

    gse = GrowingSpheresExplainer()
    gse.fit(model, None)
    instance_np = np.array(
        [
            instance.data,
        ]
    )
    original_class = instance.target_class

    cf = gse.explain(model, instance_np)

    target_class = model(torch.tensor(cf[0].data)).argmax().item()
    print(model(torch.tensor(cf[0].data)))

    return CounterfactualModel(
        data=cf[0].data.tolist(),
        original_class=original_class,
        target_class=target_class,
    )
