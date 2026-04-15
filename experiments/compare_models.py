import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from utils import plot_comparison_table
from experiments.evaluate import evaluate_model
from experiments.train import run_training


def run_full_comparison(datasets=None, config_path='configs/config.yaml'):
    if datasets is None:
        datasets = ['mnist', 'fashion_mnist']

    results = {}
    for ds in datasets:
        for mtype in ['gaussian', 'spherical']:
            key = f"{mtype}_{ds}"
            ckpt = f'results/models/{mtype}_{ds}_best.pth'
            if not os.path.exists(ckpt):
                print(f"Training {mtype} on {ds}...")
                _, train_metrics = run_training(mtype, ds, config_path)
            else:
                print(f"Checkpoint exists for {mtype} on {ds}, skipping training.")
                with open(f'results/logs/{mtype}_{ds}_metrics.json') as f:
                    train_metrics = json.load(f)

            eval_metrics = evaluate_model(mtype, ds, config_path)
            results[key] = {
                'Best Val Loss': train_metrics.get('best_val', 0.0),
                'Silhouette Score': eval_metrics['silhouette'],
                'SSIM': eval_metrics['ssim'],
            }

    plot_comparison_table(results, 'results/plots/comparison_table.png')

    with open('results/logs/comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== COMPARISON RESULTS ===")
    for k, v in results.items():
        print(f"{k}: {v}")

    return results


if __name__ == '__main__':
    run_full_comparison()
