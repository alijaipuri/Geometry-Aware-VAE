import subprocess
import sys
import os


def main():
    print("=" * 60)
    print("  Geometry-Aware VAE — Main Runner")
    print("=" * 60)
    mode = sys.argv[1] if len(sys.argv) > 1 else 'web'

    if mode == 'train':
        model = sys.argv[2] if len(sys.argv) > 2 else 'gaussian'
        dataset = sys.argv[3] if len(sys.argv) > 3 else 'mnist'
        subprocess.run([sys.executable, 'experiments/train.py',
                        '--model', model, '--dataset', dataset])
    elif mode == 'evaluate':
        model = sys.argv[2] if len(sys.argv) > 2 else 'gaussian'
        dataset = sys.argv[3] if len(sys.argv) > 3 else 'mnist'
        subprocess.run([sys.executable, 'experiments/evaluate.py',
                        '--model', model, '--dataset', dataset])
    elif mode == 'compare':
        subprocess.run([sys.executable, 'experiments/compare_models.py'])
    elif mode == 'web':
        subprocess.run([sys.executable, 'web/app.py'])
    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python main.py [train|evaluate|compare|web] [model] [dataset]")


if __name__ == '__main__':
    main()
