import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from scipy.signal import stft
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import json
import os
from datetime import datetime

#Data preprocessing functions
def normalize_and_swap(data):
    """
    Normalize each channel of the input and swap axes to (N, T, C).
    """
    stds = np.std(data, axis=-1, keepdims=True)
    normed = data / stds
    return np.swapaxes(data, 1, 2), np.swapaxes(normed, 1, 2)

# Shared dataset directory path
DATASET_DIR = '/content/drive/MyDrive/LIGO/DatasetSplitting/Datasplitting01/pytorch/new_dataset'

def load_signals_by_snr(
    snr_thresh=0,
    data_type="combined",
    snr_key="snr",
    dataset_dir= DATASET_DIR,
    preprocess_fn=None,
    preprocess_kwargs=None
):
    """
    Loads and preprocesses signals from fixed test datasets, filtering by SNR.
    The file names are constructed using the data_type argument.
    Returns:
        - test_signals: dict[label] = preprocessed signal array for each label
        - combined_signals: concatenated array of all signals
        - snr_dict: dict[label] = SNR array for each label
    Args:
        snr_thresh: minimum SNR to include
        data_type: which type of data to load ('combined', 'normal', etc.)
        snr_key: which key in the npz file to use for SNR
        preprocess_fn: function to apply to the data (e.g., normalize_and_swap)
        preprocess_kwargs: dict of kwargs for preprocess_fn
    """
    import os
    test_datasets = {
        'WNB':      f'O3_WhiteNoiseBurst_{data_type}.npz',
        'KinkKink': f'O3_KinkKink_{data_type}.npz',
        'Kink':     f'O3_Kink_{data_type}.npz',
        'SG':       f'O3_SineGaussian_{data_type}.npz',
        'Cusp':     f'O3_Cusp_{data_type}.npz',
        'BBH':      f'O3_BBH_{data_type}.npz',
    }
    test_signals = {}
    snr_dict = {}
    for label, fname in test_datasets.items():
        full_path = os.path.join(dataset_dir, fname)
        data = np.load(full_path)
        snr_vals = data[snr_key]
        idx = np.where(snr_vals > snr_thresh)[0]
        x = data['data'][idx]
        # Always normalize and swap
        x, _ = normalize_and_swap(x)
        # Then apply any additional preprocessing
        if preprocess_fn is not None:
            if preprocess_kwargs is None:
                preprocess_kwargs = {}
            x = preprocess_fn(x, **preprocess_kwargs)
        test_signals[label] = x
        snr_dict[label] = snr_vals[idx]
    combined_signals = np.concatenate(list(test_signals.values()), axis=0)
    return test_signals, combined_signals, snr_dict


def load_bg_data(split_idx=56000, preprocess_fn=None, preprocess_kwargs=None, dataset_dir=DATASET_DIR):
    """
    Loads background data, applies normalize_and_swap, and returns BG_train and BG_test.
    Optionally applies preprocess_fn to both BG_train and BG_test.
    Uses the shared DATASET_DIR for the file path.
    Args:
        split_idx: number of samples to use for test set (default 56000)
        preprocess_fn: function to apply after normalize_and_swap
        preprocess_kwargs: dict of kwargs for preprocess_fn
    Returns:
        BG_train, BG_test (both normalized and swapped, and optionally further processed)
    """
    import os
    path = os.path.join(dataset_dir, 'O3_Background_dataset.npz')
    BG = np.load(path)
    BG_train = BG['data'][:-split_idx]
    BG_test = BG['data'][-split_idx:]
    BG_train, _ = normalize_and_swap(BG_train)
    BG_test, _ = normalize_and_swap(BG_test)
    if preprocess_fn is not None:
        if preprocess_kwargs is None:
            preprocess_kwargs = {}
        BG_train = preprocess_fn(BG_train, **preprocess_kwargs)
        BG_test = preprocess_fn(BG_test, **preprocess_kwargs)
    return BG_train, BG_test

def compute_correlation_channel(data):
    """
    Compute the correlation channel for each sample in a batch.
    Input: (N, 200, 2) or (N, 2, 200). Output: (N, 200, 1)
    """
    if data.shape[-1] == 2:
        a = data[..., 0]
        b = data[..., 1]
    elif data.shape[1] == 2:
        data = np.swapaxes(data, 1, 2)
        a = data[..., 0]
        b = data[..., 1]
    else:
        raise ValueError("Input data must have shape (N, 200, 2) or (N, 2, 200)")
    corr = np.array([np.correlate(a[i], b[i], mode='same') for i in range(data.shape[0])])
    return corr[..., np.newaxis]


def compute_stft_features_flexible(
    data, nperseg, noverlap, mode="time"
):
    """
    Compute STFT features for each channel, stacking real and imaginary parts.
    Flexible output for different convolutional approaches.

    Args:
        data: (N, T, C)
        nperseg, noverlap: STFT params
        mode: 
            "time"         -> (N, time_bins, freq_bins*2*C)         # 1D conv over time
            "freq"         -> (N, freq_bins, time_bins*2*C)         # 1D conv over frequency
            "flat"         -> (N, time_bins*freq_bins, 2*C)         # 1D conv over flat axis
            "2d"           -> (N, 2*C, freq_bins, time_bins)        # 2D conv (image style)
    Returns:
        Array in the selected format.
    """
    N, T_raw, C = data.shape
    stft_feats = []
    for i in range(N):
        feats = []
        for ch in range(C):
            f, t, Zxx = stft(data[i,:,ch], nperseg=nperseg, noverlap=noverlap)
            feats.append(np.real(Zxx))  # (freq_bins, time_bins)
            feats.append(np.imag(Zxx))
        feats = np.stack(feats, axis=0)  # (2*C, freq_bins, time_bins)
        if mode == "2d":
            stft_feats.append(feats)  # (2*C, freq_bins, time_bins)
        elif mode == "time":
            # (2*C, freq_bins, time_bins) -> (time_bins, freq_bins*2*C)
            feats = np.transpose(feats, (2, 1, 0))  # (time_bins, freq_bins, 2*C)
            feats = feats.reshape(feats.shape[0], -1)  # (time_bins, freq_bins*2*C)
            stft_feats.append(feats)
        elif mode == "freq":
            # (2*C, freq_bins, time_bins) -> (freq_bins, time_bins*2*C)
            feats = np.transpose(feats, (1, 2, 0))  # (freq_bins, time_bins, 2*C)
            feats = feats.reshape(feats.shape[0], -1)  # (freq_bins, time_bins*2*C)
            stft_feats.append(feats)
        elif mode == "flat":
            # (2*C, freq_bins, time_bins) -> (freq_bins*time_bins, 2*C)
            feats = feats.reshape(feats.shape[0], -1).T  # (freq_bins*time_bins, 2*C)
            stft_feats.append(feats)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    stft_feats = np.stack(stft_feats, axis=0)
    if mode == "2d":
        # (N, 2*C, freq_bins, time_bins)
        return stft_feats
    return stft_feats

#Autoencoder model
class FlexibleConv1DAutoencoder(nn.Module):
    """
    Flexible 1D convolutional autoencoder. Architecture is defined by config dict.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        encoder_layers = []
        in_channels = config['input_channels']
        for layer_cfg in config['encoder_layers']:
            padding = layer_cfg.get('padding', 0)
            encoder_layers.append(nn.Conv1d(
                in_channels,
                layer_cfg['out_channels'],
                kernel_size=layer_cfg['kernel_size'],
                stride=layer_cfg['stride'],
                padding=padding
            ))
            if layer_cfg['activation']:
                encoder_layers.append(config['activation_functions'][layer_cfg['activation']]())
            in_channels = layer_cfg['out_channels']
        # Bottleneck (optional, flexible)
        if 'bottleneck_layers' in config:
            for layer_cfg in config['bottleneck_layers']:
                padding = layer_cfg.get('padding', 0)
                encoder_layers.append(nn.Conv1d(
                    in_channels,
                    layer_cfg['out_channels'],
                    kernel_size=layer_cfg['kernel_size'],
                    stride=layer_cfg['stride'],
                    padding=padding
                ))
                if layer_cfg.get('activation'):
                    encoder_layers.append(config['activation_functions'][layer_cfg['activation']]())
                in_channels = layer_cfg['out_channels']
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = []
        for layer_cfg in config['decoder_layers']:
            padding = layer_cfg.get('padding', 0)
            output_padding = layer_cfg.get('output_padding', 0)
            decoder_layers.append(nn.ConvTranspose1d(
                in_channels,
                layer_cfg['out_channels'],
                kernel_size=layer_cfg['kernel_size'],
                stride=layer_cfg['stride'],
                padding=padding,
                output_padding=output_padding
            ))
            if layer_cfg['activation']:
                decoder_layers.append(config['activation_functions'][layer_cfg['activation']]())
            in_channels = layer_cfg['out_channels']
        self.decoder = nn.Sequential(*decoder_layers)
        # Add adaptive layer to ensure output matches input size
        self.target_length = config.get('target_length', 200)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(self.target_length)  # Output will match target_length
        # Add adaptive layer to ensure output matches input size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(200)  # Force output to be 200
    def forward(self, x):
        x = x.permute(0, 2, 1)
        z = self.encoder(x)
        out = self.decoder(z)
        # Apply adaptive pooling to ensure correct output size
        out = self.adaptive_pool(out)
        out = out.permute(0, 2, 1)
        return out
    
    def print_layer_shapes(self, input_shape=None):
        """
        Print the shape of each layer in the model for a given input shape.
        Useful for debugging and understanding the model architecture.
        If input_shape is None, try to infer a reasonable default shape from config.
        """
        if input_shape is None:
            # Try to infer input shape from config
            channels = self.config.get('input_channels', 2)
            # Try to infer sequence length from config or use 200 as default
            seq_len = self.config.get('input_length', 200)
            input_shape = (1, seq_len, channels)
            print(f"[print_layer_shapes] No input_shape provided. Using inferred shape: {input_shape}")
        else:
            print(f"Model Architecture Analysis for input shape: {input_shape}")
        print("=" * 60)
        
        # Create a dummy input
        x = torch.randn(input_shape)
        print(f"Input shape: {x.shape}")
        
        # Encoder analysis
        print("\nENCODER LAYERS:")
        print("-" * 30)
        x_enc = x.permute(0, 2, 1)  # (batch, channels, time)
        print(f"After permute: {x_enc.shape}")
        
        for i, layer in enumerate(self.encoder):
            x_enc = layer(x_enc)
            print(f"Layer {i+1} ({type(layer).__name__}): {x_enc.shape}")
        
        # Decoder analysis
        print("\nDECODER LAYERS:")
        print("-" * 30)
        x_dec = x_enc.clone()
        print(f"Decoder input: {x_dec.shape}")
        
        for i, layer in enumerate(self.decoder):
            x_dec = layer(x_dec)
            print(f"Layer {i+1} ({type(layer).__name__}): {x_dec.shape}")
        
        # Final output
        x_final = self.adaptive_pool(x_dec)
        x_final = x_final.permute(0, 2, 1)
        print(f"\nFinal output: {x_final.shape}")
        print("=" * 60)
        
#Training functions
def train_autoencoder(model, dataloader, optimizer, criterion, n_epochs, save_interval, device, scheduler=None, verbose=True):
    """
    Train the autoencoder, saving model checkpoints and losses.
    """
    model = model.to(device)
    models = {}
    train_losses = {}
    
    if verbose:
        print(f"Starting training for {n_epochs} epochs...")
        print(f"Model will be saved every {save_interval} epochs")
        print(f"Total batches per epoch: {len(dataloader)}")
        print("-" * 50)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        for i, (batch,) in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            total_loss += loss.item() * batch.size(0)
        
        avg_loss = total_loss / len(dataloader.dataset)
        train_losses[epoch+1] = avg_loss
        
        if verbose:
            print(f"Epoch {epoch+1}/{n_epochs} - Avg Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % save_interval == 0:
            model_copy = FlexibleConv1DAutoencoder(model.config).to(device)
            model_copy.load_state_dict(model.state_dict())
            models[epoch+1] = model_copy
            if verbose:
                print(f"  ✓ Saved model checkpoint at epoch {epoch+1}")
    
    if verbose:
        print("-" * 50)
        print(f"Training completed! Saved {len(models)} model checkpoints")
        print(f"Final loss: {avg_loss:.6f}")
    
    return models, train_losses


def get_clr_scheduler(optimizer, clr_params):
    """
    Returns a PyTorch CyclicLR scheduler from clr_params dict.
    """
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=clr_params['base_lr'],
        max_lr=clr_params['max_lr'],
        step_size_up=clr_params['step_size_up'],
        mode=clr_params.get('mode', 'triangular'),
        cycle_momentum=False
    )



#Evaluation functions
def evaluate_and_plot_best_combined(
    models, BG_test, test_signals, snr_dict, device, fpr_target=1/56000, n_snr_bins=10, verbose=True
):
    """
    1. Evaluate all models on the combined dataset (all signals vs BG).
    2. Find the best epoch (highest combined AUC).
    3. Plot AUC vs epochs for combined and individual signals.
    4. For the best model, plot TPR vs SNR for each signal type at fixed FPR.
    Uses the most general axis form for MSE calculation.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # Store all results
    all_errors = {}
    aucs_combined = {}
    aucs_individual = {label: {} for label in test_signals.keys()}
    
    # Evaluate all models
    if verbose:
        print(f"Evaluating {len(models)} model checkpoints...")
        print("-" * 50)
    
    for epoch, model in models.items():
        if verbose:
            print(f"Evaluating epoch {epoch}...")
        model.eval()
        with torch.no_grad():
            bg_tensor = torch.tensor(BG_test, dtype=torch.float32).to(device)
            bg_recon = model(bg_tensor).cpu().numpy()
            axes = tuple(range(1, BG_test.ndim))
            bg_error = np.mean((BG_test - bg_recon) ** 2, axis=axes)
            errors = {'BG': bg_error}
            
            # Calculate individual signal AUCs
            for label, data in test_signals.items():
                sig_tensor = torch.tensor(data, dtype=torch.float32).to(device)
                sig_recon = model(sig_tensor).cpu().numpy()
                axes = tuple(range(1, data.ndim))
                sig_error = np.mean((data - sig_recon) ** 2, axis=axes)
                errors[label] = sig_error
                
                # Calculate individual AUC for this signal
                y_true = np.concatenate([np.zeros_like(bg_error), np.ones_like(sig_error)])
                y_score = np.concatenate([bg_error, sig_error])
                auc_val = roc_auc_score(y_true, y_score)
                aucs_individual[label][epoch] = auc_val
            
            all_errors[epoch] = errors
            
            # Calculate combined AUC
            sig_error = np.concatenate([v for k, v in errors.items() if k != 'BG'])
            y_true = np.concatenate([np.zeros_like(bg_error), np.ones_like(sig_error)])
            y_score = np.concatenate([bg_error, sig_error])
            auc_val = roc_auc_score(y_true, y_score)
            aucs_combined[epoch] = auc_val
            
            if verbose:
                print(f"  Combined AUC: {auc_val:.4f}")
                for label in test_signals.keys():
                    print(f"    {label}: {aucs_individual[label][epoch]:.4f}")
    
    # Find best epoch
    best_epoch = max(aucs_combined, key=aucs_combined.get)
    
    if verbose:
        print("-" * 50)
        print(f"BEST MODEL RESULTS:")
        print(f"Best epoch: {best_epoch}")
        print(f"Best combined AUC: {aucs_combined[best_epoch]:.4f}")
        print("\nIndividual signal AUCs at best epoch:")
        for label in test_signals.keys():
            print(f"  {label}: {aucs_individual[label][best_epoch]:.4f}")
        print("-" * 50)
    else:
        print(f'Best epoch: {best_epoch} (Combined AUC: {aucs_combined[best_epoch]:.4f})')
    
    # Plot 1: AUC vs Epochs (combined and individual signals)
    plt.figure(figsize=(10, 6))
    epochs = sorted(aucs_combined.keys())
    combined_aucs = [aucs_combined[epoch] for epoch in epochs]
    plt.plot(epochs, combined_aucs, 'k-o', linewidth=2, markersize=6, label='Combined')
    for label in test_signals.keys():
        signal_aucs = [aucs_individual[label][epoch] for epoch in epochs]
        plt.plot(epochs, signal_aucs, '-o', markersize=4, label=label, alpha=0.8)
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7)
    plt.text(best_epoch, max(combined_aucs), f'Best: {best_epoch}', color='r', va='bottom', ha='right', fontsize=10)
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('AUC vs Epochs (Combined and Individual Signals)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot 2: ROC curve for the best model (all signals and individual signals)
    best_model = models[best_epoch]
    errors = all_errors[best_epoch]
    bg_error = errors['BG']
    plt.figure(figsize=(8, 6))
    # Combined ROC
    all_sig_errors = np.concatenate([v for k, v in errors.items() if k != 'BG'])
    y_true_combined = np.concatenate([np.zeros_like(bg_error), np.ones_like(all_sig_errors)])
    y_score_combined = np.concatenate([bg_error, all_sig_errors])
    fpr_combined, tpr_combined, _ = roc_curve(y_true_combined, y_score_combined)
    auc_combined = roc_auc_score(y_true_combined, y_score_combined)
    plt.plot(fpr_combined, tpr_combined, 'k-', linewidth=2, label=f'Combined (AUC={auc_combined:.3f})')
    # Individual signal ROCs
    for label, sig_error in errors.items():
        if label == 'BG':
            continue
        y_true = np.concatenate([np.zeros_like(bg_error), np.ones_like(sig_error)])
        y_score = np.concatenate([bg_error, sig_error])
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc_val = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, linewidth=1.5, alpha=0.8, label=f"{label} (AUC={auc_val:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves at Best Epoch ({best_epoch})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot 3: TPR at fixed FPR vs SNR threshold (not binned)
    snr_thresholds = np.linspace(4, 100, 10)  # Adjust as needed
    plt.figure(figsize=(10, 6))
    for label, sig_error in errors.items():
        if label == 'BG':
            continue
        snr_vals = snr_dict[label]
        tpr_list = []
        for snr_thresh in snr_thresholds:
            mask = (snr_vals >= snr_thresh)
            if np.sum(mask) == 0:
                tpr_list.append(np.nan)
                continue
            sig_selected_error = sig_error[mask]
            y_true = np.concatenate([np.zeros_like(bg_error), np.ones_like(sig_selected_error)])
            y_score = np.concatenate([bg_error, sig_selected_error])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            idx = np.argmin(np.abs(fpr - fpr_target))
            tpr_list.append(tpr[idx])
        plt.plot(snr_thresholds, tpr_list, marker='o', label=label)
    plt.xlabel('SNR Threshold')
    plt.ylabel(f'TPR at FPR={fpr_target:.6f}')
    plt.ylim(0, 1)
    plt.title('TPR at Fixed FPR vs SNR Threshold (All signals above threshold)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if verbose:
        print(f"Generated ROC plots and analysis for best model (epoch {best_epoch})")
    
    return best_epoch, aucs_combined[best_epoch]

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    elif isinstance(obj, type):
        return obj.__name__  # For classes like torch.nn.ReLU
    elif callable(obj):
        return str(obj)
    else:
        return obj
        
def save_all_open_figures_and_history(config, eval_result, model_name="experiment"):
    """
    Save all open matplotlib figures, config, and evaluation result to a model-specific history file.
    - Figures are saved in a directory named <model_name>_figures/
    - History is saved in <model_name>_history.jsonl (created if not exists)
    - Each record includes: timestamp, config, eval_result, figures
    Returns (figure_files, history_file)
    """
    import matplotlib.pyplot as plt
    fig_dir = f"{model_name}_figures"
    os.makedirs(fig_dir, exist_ok=True)
    figure_files = []
    for i, fig_num in enumerate(plt.get_fignums()):
        fig = plt.figure(fig_num)
        fig_filename = os.path.join(fig_dir, f"{model_name}_fig_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        fig.savefig(fig_filename)
        figure_files.append(fig_filename)
    print(f"[save_all_open_figures_and_history] Saved {len(figure_files)} figures to {fig_dir}/")

    # Prepare history file
    history_file = f"{model_name}_history.jsonl"
    if not os.path.exists(history_file):
        with open(history_file, 'w') as f:
            pass  # Just create the file if it doesn't exist
    serializable_config = make_json_serializable(config)
    record = {
            'timestamp': datetime.now().isoformat(),
            'config': serializable_config,  # <-- Use the processed config here!
            'eval_result': eval_result,
            'figures': figure_files
        }
    with open(history_file, 'a') as f:
        f.write(json.dumps(record) + '\n')
    print(f"[save_all_open_figures_and_history] Saved experiment record to {history_file}")
    return figure_files, history_file