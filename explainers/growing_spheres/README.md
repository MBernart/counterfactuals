# Docker container with PoC explainer
## Run

Firstly, build the container. Simply run the command below in the root of this repository: 

```bash
docker build -t cf_poc -f explainers/growing_spheres/Dockerfile .  
```

and run it: 

```bash
docker run -p 8000:8000 cf_poc
```

## Usage

Check [examples](../../examples/).