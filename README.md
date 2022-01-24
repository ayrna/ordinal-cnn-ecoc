# Error-correcting output codes in the framework of deep ordinal classification

This is the companion code for the paper "Error-correcting output codes in the framework of deep ordinal classification". The code is prepared to automatically initialize and run all experiments as described in the paper.

The experimentation results from the original paper are available in file [`results.xlsx`](/results.xlsx).

Authors are:
* Javier Barbero-Gómez (@javierbg)
* Pedro Antonio Gutiérrez (@pagutierrez)
* César Hervás-Martínez (chervas@uco.es)


## Instructions

The following has been tested to run on an up-to-date Linux installation (Debian 10 buster).

### Preparing the environment

You should first install the latest version of `conda`. The small version known as `miniconda` is recommended.

For instructions on how to install `miniconda` check this [link](https://docs.conda.io/en/latest/miniconda.html).

Then, create a new environment named `"ordinal-cnn-ecoc"` populated with all the needed packages using the [`requirements.txt`](/requirements.txt) file:
```bash
conda create --name ordinal-cnn-ecoc --file requirements.txt
```

Once finished, activate the environment with the command:
```bash
conda activate ordinal-cnn-ecoc
```

### Initializing the project

First you should initialize all jobs to submit. Just run the [`init.py`](/init.py) script:
```bash
python init.py
```

A `"workspace"` folder should have been created containing all project data, as well as other project related files.

### Running the experiments

The experiments are designed to be ran in a GPU cluster using a workload manager such as [Slurm](https://slurm.schedmd.com/documentation.html), [PBS](https://linux.die.net/man/1/qsub-torque), [LSF](https://www.ibm.com/docs/en/spectrum-lsf/10.1.0) or [HTCondor](https://htcondor.org/). Refer to the [`signac` project documentation](https://docs.signac.io/projects/flow/en/latest/supported_environments.html#supported-environments) for instructions on how to configure your environment.

Once configured, all experiments can be submitted simply by using the command:
```bash
python project.py submit -o experiment
```

### Obtaining the results

When all jobs have been completed, the results can be extracted in tabular format using the [`extract_results.py`](/extract_results.py) script:
```bash
python extract_results.py results.xlsx
```