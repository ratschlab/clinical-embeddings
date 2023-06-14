# On the Importance of Step-wise Embeddings for Clinical Sequence Modeling

Repository for the paper: „On the Importance of Step-wise Embeddings for Clinical Sequence Modeling“

<object data="./files/figures/pipeline_overview.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="./files/figures/pipeline_overview.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="./files/figures/pipeline_overview.pdf">Download PDF</a>.</p>
    </embed>
</object>

## Setup

We provide a `conda` environment file: `environment.yml`. To create the environment, run:

```bash
conda env create -f environment.yml
```

## Data and Preprocessing 

We base our pipeline on the work of Yèche et al. in Temporal Label Smoothing. For instructions on
how to obtain access to the MIMIC-III and HiRID datasets, as well as how to preprocess them, please
consult the repository here: [https://github.com/ratschlab/tls](https://github.com/ratschlab/tls).

The resulting `.h5` files in the `ml_stage` output folders of the preprocessing are the designated
inputs to our pipeline.

## Training

The core code of the models is located in `icu_benchmarks/models/encoders.py`, `icu_benchmarks/models/layers.py`, and  `icu_benchmarks/models/wrappers.py`.

We use `.gin` configuration files to specify the hyperparameters of the models. The configuration files for the models used in the paper are located in `configs/`.

To run a training run, modify the chosen configuration file to point to the correct input files by setting `train_common.data_path` and `TASK` to the desired task to solve for (refer to [https://github.com/ratschlab/tls](https://github.com/ratschlab/tls)). Then, run:

```bash

```

Please note that some of the example use very small model dimension for demonstration purposes. To reproduce the results from the paper, please refer to the search space configuration files (`.yaml`) in `configs/`.
