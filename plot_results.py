import matplotlib.pyplot as plt
import json
from os.path import join as opj
from glob import glob
import numpy as np

path = "/home/gautamsree/research/hal-sreekum1/new_replin/logs_Windmill"
folders = glob(opj(path, "*Windmill*"))

erm_results = {}
replin_results = {}

for folder in folders:
    args_file = opj(folder, "args.json")
    results_file = opj(folder, "results.json")

    with open(args_file, "r") as f:
        args = json.load(f)
    with open(results_file, "r") as f:
        results = json.load(f)
    
    beta = args['beta']

    if args['model_name'] == "ERM" and args['seed'] < 200:
        continue
    if args['model_name'] == "RepLIn" and args['seed'] >= 140 and beta == 0.9:
        continue
    
    if args['model_name'] == "ERM":
        if beta in erm_results:
            erm_results[beta].append(results['val_perf_intB/A'])
        else:
            erm_results[beta] = [results['val_perf_intB/A']]
    elif args['model_name'] == "RepLIn":
        if beta in replin_results:
            replin_results[beta].append(results['val_perf_intB/A'])
        else:
            replin_results[beta] = [results['val_perf_intB/A']]

erm_mean = {k: 100*np.mean(v) for k, v in erm_results.items()}
replin_mean = {k: 100*np.mean(v) for k, v in replin_results.items()}
erm_std = {k: 100*np.std(v) for k, v in erm_results.items()}
replin_std = {k: 100*np.std(v) for k, v in replin_results.items()}

keys = sorted(erm_mean.keys())

# plt.bar([k-0.4 for k in range(len(keys))],
#         [erm_mean[k] for k in keys],
#         width=0.3,
#         yerr=[erm_std[k] for k in keys],
#         label="ERM",
#         capsize=5)
# plt.bar([k+0.4 for k in range(len(keys))],
#         [replin_mean[k] for k in keys],
#         width=0.3,
#         yerr=[replin_std[k] for k in keys],
#         label="RepLIn",
#         capsize=5)
# plt.xlabel("Beta")
# plt.ylabel("Test Accuracy on Intervened Data (%)")
# plt.xticks(range(len(keys)), keys)
# plt.ylim(0, 100)
# plt.legend()
# plt.title("Performance Comparison on Windmill Dataset")
# plt.show()

for k in keys:
    print(f"Beta: {k} | ERM: {erm_mean[k]:.2f} ± {erm_std[k]:.2f} | RepLIn: {replin_mean[k]:.2f} ± {replin_std[k]:.2f}")

