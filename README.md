# On the Importance of Step-wise Embeddings for Clinical Sequence Modeling

Repository for the Benchmark: „On the Importance of Step-wise Embeddings for Clinical Sequence Modeling“

![Pipeline Overview](https://github.com/ratschlab/clinical-embeddings/blob/main/files/figures/pipeline_overview.png?raw=true)

## Setup

We provide a `conda` environment file: `environment.yml`. To create the environment, run:

```bash
conda env create -f environment.yml
```

## Data and Preprocessing 

We base our pipeline on the work of Yèche et al. in Temporal Label Smoothing. For instructions on
how to obtain access to the MIMIC-III and HiRID datasets, as well as how to preprocess them, please
consult the repository here: [https://github.com/ratschlab/tls](https://github.com/ratschlab/tls).

The resulting `.h5` files in the `ml_stage` working directory of the preprocessing are the designated
inputs to our pipeline.

## Training

The core code of the models is located in `icu_benchmarks/models/encoders.py`, `icu_benchmarks/models/layers.py`, and  `icu_benchmarks/models/wrappers.py`.

We use [`.gin` configuration](https://github.com/google/gin-config) files to specify the hyperparameters of the models. The configuration files for the models used in the paper are located in `configs/`.

To run a training run, modify the chosen configuration file to point to the correct input files by setting `train_common.data_path` and `TASK` (HiRID: `Mortality_At24Hours, Dynamic_CircFailure_12Hours, Dynamic_RespFailure_12Hours`, MIMIC: `decomp_24Hours, ihm`) to the desired task to solve for (refer to [https://github.com/ratschlab/tls](https://github.com/ratschlab/tls)). Then, run:

```bash
 accelerate launch \
    --config_file ./configs/accelerate/accel_config_gpu1.yml \
    ./icu_benchmarks/run.py train \
    -c ./configs/examples/hirid/backbone/GRU.gin \
    -l logs/backbone_GRU
```

We use [HuggingFace Accelerate](https://huggingface.co/docs/accelerate/index) for launching (distributed) training runs. The configuration file `accel_config_gpu1.yml` is provided in `configs/accelerate/` and specifies the number of GPUs to use. The `train` command will create a new directory in `logs/` with the name specified by the `-l` argument. The training logs, performance results on validation and test, as well as checkpoints will be saved in this directory. `-c` specifies the path to the configuration file.

Please note that some of the example use very small model dimension for demonstration purposes. To reproduce the results from the paper, please refer to the search space configuration files (`.yaml`) in `configs/`.

### Hyperparameter Search

We use a [Slurm](https://slurm.schedmd.com) based compute cluster and provide a wrapper script (`./run_wrapper.py`) to launch hyperparameter sweeps. The script takes a configuration file (`.yaml`) as input and launches a hyperparameter sweep based on the specified search space. Individual run directories are created in the designated directory and the runs are submitted to the cluster. The configuration file should specify the following parameters:

```yaml
# Specify compute resources
compute:
  gpus: 1 # Number of GPUs (should align with the number of GPUs in the accelerate config file)
  gpu_type: rtx2080ti # Type of GPU
  cores: 4 # Number of CPU cores, specify the number of workers in the gin config file
  memory: 4 # Amount of memory in GB per core
  time: 4 # Time in hours

# Hyperparameters
# Specify hyperparameters to be tuned and their search space
# Specify them exactly as they are specified in the gin config file
params:

  Transformer.hidden: 42
  Transformer.depth: [1, 2, 3]
  Transformer.heads: [1, 2, 3]
  Transformer.dropout: [0.0, 0.1, 0.2]

  DLWrapper.reg: 'l1'
  DLWrapper.reg_weight: [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 20]


# Different Random Seeds for each configuration
seeds: [1111, 2222, 3333]

base_gin: './configs/examples/hirid/backbone/transformer.gin' # the gin config file to be used as a base
accelerate_config: './configs/accelerate/accel_config_gpu1.yml' # the accelerate config file to be used
task: 'decomp_24Hours' # the task to be run
```

To launch a hyperparameter sweep, run:

```bash
python run_wrapper.py \
    --config ./configs/examples/hirid/backbone/transformer_base.yaml \ # Base config file
    --shuffle \ # Shuffle the hyperparameter combinations, together with `num_runs` this creates a random search
     --num_runs 20 \ # Maximun number of different configurations to be run, if not set, all possible combinations are run
    --name example-sweep
```
