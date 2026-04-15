import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, jsonify, request, send_file
import torch
import numpy as np
import json
import base64
import io
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import threading
import time
from groq import Groq

from models import GaussianVAE, SphericalVAE
from utils import get_dataloader, load_checkpoint, compute_silhouette

app = Flask(__name__)

# =====================================================
# ⚠️  PASTE YOUR GROQ API KEY BELOW
GROQ_API_KEY = "your_groq_api_key_here"
# =====================================================

groq_client = Groq(api_key=GROQ_API_KEY)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOADED_MODELS = {}
TRAINING_STATUS = {}
TRAINING_LOGS = {}


def load_config():
    with open('configs/config.yaml') as f:
        return yaml.safe_load(f)


def get_or_load_model(model_type, dataset):
    key = f"{model_type}_{dataset}"
    if key in LOADED_MODELS:
        return LOADED_MODELS[key]

    config = load_config()
    cfg = config.get(model_type, config['gaussian'])
    img_size = 32 if dataset == 'cifar10' else 28

    _, in_channels = get_dataloader(dataset, 2, train=False)

    if model_type == 'gaussian':
        model = GaussianVAE(latent_dim=cfg['latent_dim'], in_channels=in_channels, img_size=img_size).to(DEVICE)
    else:
        model = SphericalVAE(latent_dim=cfg['latent_dim'], in_channels=in_channels, img_size=img_size).to(DEVICE)

    ckpt_path = f'results/models/{model_type}_{dataset}_best.pth'
    if os.path.exists(ckpt_path):
        load_checkpoint(model, None, ckpt_path, DEVICE)

    model.eval()
    LOADED_MODELS[key] = model
    return model


def tensor_to_base64(tensor):
    tensor = tensor.squeeze().cpu()
    if tensor.dim() == 2:
        arr = (tensor.numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr, mode='L')
    else:
        arr = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/datasets')
def api_datasets():
    return jsonify({'datasets': ['mnist', 'fashion_mnist', 'cifar10']})


@app.route('/api/status')
def api_status():
    results = {}
    for ds in ['mnist', 'fashion_mnist', 'cifar10']:
        for mt in ['gaussian', 'spherical']:
            key = f"{mt}_{ds}"
            ckpt = f'results/models/{mt}_{ds}_best.pth'
            log_path = f'results/logs/{mt}_{ds}_metrics.json'
            best_val = None
            if os.path.exists(log_path):
                with open(log_path) as f:
                    m = json.load(f)
                best_val = m.get('best_val')
            results[key] = {
                'trained': os.path.exists(ckpt),
                'best_val': best_val,
                'training': TRAINING_STATUS.get(key, False)
            }
    return jsonify(results)


@app.route('/api/train', methods=['POST'])
def api_train():
    data = request.json
    model_type = data.get('model_type', 'gaussian')
    dataset = data.get('dataset', 'mnist')
    key = f"{model_type}_{dataset}"

    if TRAINING_STATUS.get(key):
        return jsonify({'error': 'Already training'}), 400

    def train_thread():
        TRAINING_STATUS[key] = True
        TRAINING_LOGS[key] = []
        try:
            from experiments.train import run_training
            run_training(model_type, dataset)
            if key in LOADED_MODELS:
                del LOADED_MODELS[key]
        except Exception as e:
            TRAINING_LOGS[key].append(f"ERROR: {e}")
        finally:
            TRAINING_STATUS[key] = False

    t = threading.Thread(target=train_thread)
    t.daemon = True
    t.start()
    return jsonify({'message': f'Training {model_type} on {dataset} started'})


@app.route('/api/training_logs/<model_type>/<dataset>')
def api_training_logs(model_type, dataset):
    key = f"{model_type}_{dataset}"
    log_path = f'results/logs/{model_type}_{dataset}_metrics.json'
    logs = TRAINING_LOGS.get(key, [])
    metrics = {}
    if os.path.exists(log_path):
        with open(log_path) as f:
            metrics = json.load(f)
    return jsonify({'logs': logs, 'metrics': metrics,
                    'training': TRAINING_STATUS.get(key, False)})


@app.route('/api/latent_space', methods=['POST'])
def api_latent_space():
    data = request.json
    model_type = data.get('model_type', 'gaussian')
    dataset = data.get('dataset', 'mnist')

    try:
        model = get_or_load_model(model_type, dataset)
        loader, _ = get_dataloader(dataset, 128, train=False)

        all_z, all_labels = [], []
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                if i >= 15:
                    break
                x = x.to(DEVICE)
                if model_type == 'gaussian':
                    _, _, _, z = model(x)
                else:
                    _, _, _, z = model(x)
                all_z.append(z.cpu().numpy())
                all_labels.append(y.numpy())

        all_z = np.concatenate(all_z)
        all_labels = np.concatenate(all_labels)

        from sklearn.decomposition import PCA
        if all_z.shape[1] > 2:
            pca = PCA(n_components=2)
            plot_z = pca.fit_transform(all_z)
        else:
            plot_z = all_z

        sil = compute_silhouette(all_z, all_labels)

        points = [{'x': float(plot_z[i, 0]), 'y': float(plot_z[i, 1]),
                   'label': int(all_labels[i])} for i in range(len(plot_z))]

        return jsonify({'points': points, 'silhouette': sil,
                        'model_type': model_type, 'dataset': dataset})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/reconstruct', methods=['POST'])
def api_reconstruct():
    data = request.json
    model_type = data.get('model_type', 'gaussian')
    dataset = data.get('dataset', 'mnist')
    n = data.get('n', 8)

    try:
        model = get_or_load_model(model_type, dataset)
        loader, _ = get_dataloader(dataset, n, train=False)
        x, _ = next(iter(loader))
        x = x[:n].to(DEVICE)

        with torch.no_grad():
            if model_type == 'gaussian':
                recon, _, _, _ = model(x)
            else:
                recon, _, _, _ = model(x)

        originals = [tensor_to_base64(x[i]) for i in range(n)]
        reconstructions = [tensor_to_base64(recon[i]) for i in range(n)]

        return jsonify({'originals': originals, 'reconstructions': reconstructions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def api_generate():
    data = request.json
    model_type = data.get('model_type', 'gaussian')
    dataset = data.get('dataset', 'mnist')
    n = data.get('n', 16)

    try:
        model = get_or_load_model(model_type, dataset)
        with torch.no_grad():
            samples = model.sample(n, DEVICE)
        images = [tensor_to_base64(samples[i]) for i in range(n)]
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/interpolate', methods=['POST'])
def api_interpolate():
    data = request.json
    model_type = data.get('model_type', 'gaussian')
    dataset = data.get('dataset', 'mnist')
    steps = data.get('steps', 10)

    try:
        model = get_or_load_model(model_type, dataset)
        config = load_config()
        cfg = config.get(model_type, config['gaussian'])
        ldim = cfg['latent_dim']

        z_start = torch.randn(1, ldim).to(DEVICE)
        z_end = torch.randn(1, ldim).to(DEVICE)

        if model_type == 'spherical':
            import torch.nn.functional as F
            z_start = F.normalize(z_start, dim=1)
            z_end = F.normalize(z_end, dim=1)

        images = []
        for alpha in np.linspace(0, 1, steps):
            z = (1 - alpha) * z_start + alpha * z_end
            if model_type == 'spherical':
                import torch.nn.functional as F
                z = F.normalize(z, dim=1)
            with torch.no_grad():
                img = model.decode(z)
            images.append(tensor_to_base64(img[0]))

        return jsonify({'images': images, 'steps': steps})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/metrics', methods=['POST'])
def api_metrics():
    data = request.json
    model_type = data.get('model_type', 'gaussian')
    dataset = data.get('dataset', 'mnist')

    results = {}
    for mt in ['gaussian', 'spherical']:
        log_path = f'results/logs/{mt}_{dataset}_metrics.json'
        eval_path = f'results/logs/{mt}_{dataset}_eval.json'
        entry = {'model': mt, 'dataset': dataset}
        if os.path.exists(log_path):
            with open(log_path) as f:
                m = json.load(f)
            entry['best_val_loss'] = m.get('best_val', 'N/A')
            entry['train_losses'] = m.get('train_losses', [])
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                e = json.load(f)
            entry.update(e)
        results[mt] = entry

    return jsonify(results)


@app.route('/api/plot/<filename>')
def api_plot(filename):
    path = f'results/plots/{filename}'
    if os.path.exists(path):
        return send_file(path, mimetype='image/png')
    return jsonify({'error': 'Plot not found'}), 404


@app.route('/api/ai_analysis', methods=['POST'])
def api_ai_analysis():
    """AI-powered analysis using Groq llama-3.3-70b-versatile"""
    data = request.json
    metrics = data.get('metrics', {})
    context = data.get('context', '')

    prompt = f"""You are an expert deep learning researcher specializing in Variational Autoencoders and geometric deep learning.

Analyze the following experiment results comparing Gaussian VAE vs Spherical VAE:

Context: {context}

Metrics:
{json.dumps(metrics, indent=2)}

Provide:
1. Key insights about the geometry-aware latent space advantages
2. Interpretation of silhouette scores and reconstruction quality
3. Recommendations for improving the spherical VAE
4. Suggestions for future research directions
5. Brief explanation suitable for a student exam report

Be concise, technical, and insightful. Format your response in clear sections."""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.7,
        )
        analysis = response.choices[0].message.content
        return jsonify({'analysis': analysis})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai_chat', methods=['POST'])
def api_ai_chat():
    """Interactive AI chat about the VAE project"""
    data = request.json
    messages = data.get('messages', [])
    system_prompt = """You are an expert AI researcher specializing in Variational Autoencoders, 
geometric deep learning, and representation learning. Help the user understand their 
Geometry-Aware VAE project. Be technical but clear, and provide mathematical intuitions 
when helpful. You know about: standard Gaussian VAEs, von Mises-Fisher distribution, 
spherical latent spaces, hyperbolic spaces, MNIST, Fashion-MNIST, latent space visualization,
silhouette scores, SSIM, t-SNE, and all aspects of this project."""

    groq_messages = [{"role": "system", "content": system_prompt}]
    groq_messages.extend(messages)

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_messages,
            max_tokens=800,
            temperature=0.7,
        )
        reply = response.choices[0].message.content
        return jsonify({'reply': reply})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare_all', methods=['POST'])
def api_compare_all():
    data = request.json
    dataset = data.get('dataset', 'mnist')

    comparison = {}
    for mt in ['gaussian', 'spherical']:
        metrics_path = f'results/logs/{mt}_{dataset}_metrics.json'
        eval_path = f'results/logs/{mt}_{dataset}_eval.json'
        entry = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            entry['best_val_loss'] = round(m.get('best_val', 0), 4)
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                e = json.load(f)
            entry.update({k: round(v, 4) if isinstance(v, float) else v for k, v in e.items()})
        comparison[mt] = entry

    return jsonify({'comparison': comparison, 'dataset': dataset})


if __name__ == '__main__':
    os.makedirs('results/models', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/logs', exist_ok=True)
    app.run(host='0.0.0.0', port=5050, debug=False)
