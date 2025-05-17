# Docker container with PoC explainer
:warning: Boys, I know it's dirty and uses HTTP.
## Run

Firstly, build the container. Simply run the command below in this directory: 

```bash
docker build -t cf_poc .  
```

and run it: 

```bash
docker run -p 8000:8000 cf_poc
```

## Usage

Check [swagger docs](http://127.0.0.1:8000/docs).

## Remarks

I have also included a notebook with my sample model for tests under `experiments/iris torch.ipynb`. I'd keep it on this branch only.

Ready model can be found here: `temp_model.pt`.
