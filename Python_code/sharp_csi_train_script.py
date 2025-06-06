"""
Copyright (C) 2022 Francesca Meneghello
contact: meneghello@dei.unipd.it
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""Example Colab notebook showing how to train SHARP on raw CSI data.

The raw CSI is assumed to be a ``numpy`` array with dimensions
``(trials, APs, antennas, subcarriers, time)``. ``time`` is 500 samples,
``subcarriers`` are 64 and there are five APs. Labels are one of ``circle``,
``left-right``, ``up-down`` and ``push-pull``.

This notebook implements the full SHARP preprocessing chain: phase
sanitization using the official routines and Doppler spectrum generation.
Random data are created for demonstration purposes, but the same workflow can
be used with real measurements.  Each AP/antenna combination is used as a
separate channel during training.
"""

# %%


# %% Imports
import numpy as np
import tensorflow as tf
from network_utility import csi_network_inc_res
import CSI_phase_sanitization_signal_reconstruction as signal_reconstruction



# %% Constants
GESTURES = ["circle", "left-right", "up-down", "push-pull"]



# %% Data generation
def generate_example_data(num_trials=100, num_ap=5, num_ant=3,
                          subcarriers=64, time_samples=500):
    """Generate random complex CSI data and labels for demonstration."""
    rng = np.random.default_rng(0)
    real = rng.standard_normal((num_trials, num_ap, num_ant,
                                subcarriers, time_samples))
    imag = rng.standard_normal((num_trials, num_ap, num_ant,
                                subcarriers, time_samples))
    data = real + 1j * imag
    labels = rng.integers(0, len(GESTURES), size=(num_trials,))
    return data.astype(np.complex64), labels



# %% Phase sanitization
def sanitize_csi(stream):
    """Sanitize CSI using the official reconstruction utilities."""
    subc_slice = slice(6, -5 if stream.shape[0] > 11 else stream.shape[0])
    csi = stream[subc_slice, :]
    return signal_reconstruction.sanitize_stream(csi)



# %% Doppler computation
def compute_doppler(sanitized_csi, num_symbols=51, step=1, n_fft=100, noise_db=-30):
    """Compute Doppler profiles from sanitized CSI (time, subc)."""
    profiles = []
    for start in range(0, sanitized_csi.shape[0] - num_symbols, step):
        cut = sanitized_csi[start:start + num_symbols, :]
        cut = np.nan_to_num(cut)
        window = np.expand_dims(np.hanning(num_symbols), axis=-1)
        cut_win = cut * window
        prof = np.fft.fftshift(np.fft.fft(cut_win, n=n_fft, axis=0), axes=0)
        power = np.abs(prof * np.conj(prof))
        power = np.sum(power, axis=1)
        profiles.append(power)
    profiles = np.asarray(profiles)
    if profiles.size == 0:
        profiles = np.zeros((1, n_fft))
    profiles = profiles / np.max(profiles, axis=1, keepdims=True)
    min_val = 10 ** (noise_db / 10)
    profiles[profiles < min_val] = min_val
    return profiles



# %% Full preprocessing
def preprocess_raw_csi(raw_csi, sample_length=340, n_fft=100):
    """Apply phase sanitization and compute Doppler profiles for all streams."""
    trials, aps, ants, _, _ = raw_csi.shape

    doppler_maps = []
    for tr in range(trials):
        channels = []
        for ap in range(aps):
            for ant in range(ants):
                stream = raw_csi[tr, ap, ant]
                sanitized = sanitize_csi(stream)
                doppler = compute_doppler(sanitized, n_fft=n_fft)
                if doppler.shape[0] < sample_length:
                    pad = np.zeros((sample_length - doppler.shape[0], n_fft))
                    doppler = np.concatenate([doppler, pad], axis=0)
                else:
                    doppler = doppler[:sample_length]
                channels.append(doppler.astype(np.float32))

        sample = np.stack(channels, axis=-1)
        doppler_maps.append(sample)

    return np.array(doppler_maps, dtype=np.float32)



# %% Dataset utilities
def build_dataset(data, labels, batch_size=8, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    if shuffle:
        dataset = dataset.shuffle(len(labels))
    dataset = dataset.batch(batch_size)
    return dataset



# %% Training
def main():
    raw_csi, labels = generate_example_data()
    doppler_data = preprocess_raw_csi(raw_csi)

    split = int(0.8 * doppler_data.shape[0])
    train_ds = build_dataset(doppler_data[:split], labels[:split], shuffle=True)
    test_ds = build_dataset(doppler_data[split:], labels[split:], shuffle=False)

    input_shape = doppler_data.shape[1:]
    num_classes = len(GESTURES)

    model = csi_network_inc_res(input_shape, num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    model.fit(train_ds, epochs=5, verbose=2)
    loss, acc = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {acc:.3f}")


# %% Execute
if __name__ == "__main__":
    main()
