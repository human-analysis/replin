# Incorporating Interventional Independence Improves Robustness against Interventional Distribution Shift

---

<p style="text-align: center;">Authors: <a href="https://gautamsreekumar.github.io">Gautam Sreekumar</a>, <a href="https://vishnuboddeti.net">Vishnu Naresh Boddeti</a><br>Michigan State University <br> Transactions on Machine Learning Research, 2025 <br> <a href="https://arxiv.org/abs/2507.05412">Paper</a> | <a href="https://openreview.net/forum?id=kXfcEyNIrf">OpenReview</a> | <a href="https://x.com/gautamsree_/status/1942794242751095295">X Thread</a></p>

---

## Paper Abstract

[image](./assets/replin-architecture.png)

**One-line summary**: Explicitly enforcing statistical independence between the representations of an intervened variable and its parent variables in a causal graph improves the robustness of those representations against similar interventions during inference.

We consider the problem of learning robust discriminative representations of causally related latent variables given the underlying directed causal graph and a training set comprising passively collected observational data and interventional data obtained through targeted interventions on some of these latent variables. We desire to learn representations that are robust against the resulting interventional distribution shifts. Existing approaches treat interventional data like observational data and ignore the independence relations that arise from these interventions, even when the underlying causal model is known. As a result, their representations lead to large disparities in predictive performance between observational and interventional data. This performance disparity worsens when interventional training samples are scarce. In this paper, (1) we first identify a strong correlation between this performance disparity and the representations' violation of statistical independence induced during interventions. (2) For linear models, we derive sufficient conditions on the proportion of interventional training data, for which enforcing statistical independence between representations of the intervened node and its non-descendants during interventions lowers the test-time error on interventional data. Combining these insights, (3) we propose RepLIn, a training algorithm that explicitly enforces this statistical independence between representations during interventions. We demonstrate the utility of RepLIn on a synthetic dataset, and on real image and text datasets on facial attribute classification and toxicity detection, respectively, with semi-synthetic causal structures. Our experiments show that RepLIn is scalable with the number of nodes in the causal graph and is suitable to improve robustness against interventional distribution shifts of both continuous and discrete latent variables compared to the ERM baselines.

---

## Repo Description

This repository contains the code for the paper "Incorporating Interventional Independence Improves Robustness against Interventional Distribution Shift," which was published in Transactions on Machine Learning Research (TMLR) in 2025.

The repository contains a basic implementation of our proposed method, RepLIn, and the ERM baseline for the Windmill dataset that was used for our primary experiments.

## Installing the conda environment

To run this repo, install the conda environment from the file ```replin.yml```.

```bash
conda env create -f replin.yml
```

## Running the experiments

The generation arguments for the Windmill dataset are provided in ```configs/windmill.ini```. The default training arguments for ERM and RepLIn are provided in ```configs/erm_windmill.ini``` and ```configs/replin_windmill.ini```, respectively. Using these files, we can train ERM and RepLIn models.

For training and evaluating ERM models:

```bash
python main.py --c configs/erm_windmill.ini \
	--beta [BETA] \
	--seed [SEED] \
	--logs_folder [WHERE_TO_STORE_RESULTS]
```

Similarly, to train and evaluate RepLIn models,

```bash
python main.py --c configs/replin_windmill.ini \
	--beta [BETA] \
	--seed [SEED] \
	--logs_folder [WHERE_TO_STORE_RESULTS]
```

Please see ```args.py``` for a detailed description of the available command line flags.

## BibTeX citation

```
@inproceedings{replin,
      title={{Incorporating Interventional Independence Improves Robustness against Interventional Distribution Shift}}, 
      author={Gautam Sreekumar and Vishnu Naresh Boddeti},
      year={2025},
      booktitle={{Transactions on Machine Learning Research}},
      url={https://openreview.net/forum?id=kXfcEyNIrf}, 
}
```

