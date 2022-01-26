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

### Preparing the data

* [Diabetic Retinopathy dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/data):
  * Download files train.zip.001-005 and decompress
  * Download trainLabels.csv.zip and decompress
  * Run the partitioning script as:
  
```
python partition_retinopathy.py <path to trainLabels.csv> <path to train images folder> <path to folder to contain transformed images> <path to folder to contain all partitions>
```
* [Adience dataset](https://talhassner.github.io/home/projects/Adience/Adience-data.html):
  * Download files fold_0_data.txt-fold_4_data.txt and place in a common folder
  * Download aligned.tar.gz and decompress
  * Run the partitioning script as:

```
python partition_adience.py <path to folder containing folds> <path to images folder> <path to folder to contain transformed images> <path to folder to contain all partitions>
```

### Initializing the project

First you should initialize all jobs to submit. Just run the [`init.py`](/init.py) script:
```bash
python init.py <retinopathy_dataset_partitions_path> <adience_dataset_partitions_path>
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