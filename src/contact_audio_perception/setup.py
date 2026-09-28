from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'contact_audio_perception'


def _bundle_fold_id(meta_path):
    """Read fold_id straight from a bundle's model.npz."""
    import numpy as np

    with np.load(meta_path, allow_pickle=False) as data:
        return str(data['fold_id']) if 'fold_id' in data else 'all'


def model_bundle_data_files():
    """
    Map each model bundle DIRECTORY to the files it contains.

    data_files cannot install a directory: glob('models/*/') would hand setup a
    directory name and install nothing. Every bundle therefore gets its own
    entry listing its actual files, so model.npz and encoder.safetensors both
    land under share/<pkg>/models/<bundle>/.

    This is also the second, independent layer of fold isolation: the build
    FAILS if a cross-validation artifact has found its way into models/, rather
    than trusting that the writer refused to put it there.
    """
    entries = []
    for bundle in sorted(glob('models/*/')):
        name = os.path.basename(os.path.normpath(bundle))
        meta = os.path.join(bundle, 'model.npz')
        if os.path.isfile(meta):
            fold_id = _bundle_fold_id(meta)
            if fold_id != 'all':
                raise SystemExit(
                    f'models/{name} is a cross-validation artifact '
                    f'(fold_id={fold_id!r}) and must not be installed. Fold models '
                    f'are trained without one trial; move it to eval_runs/.')
        files = sorted(p for p in glob(os.path.join(bundle, '*')) if os.path.isfile(p))
        if files:
            entries.append((os.path.join('share', package_name, 'models', name), files))
    return entries


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*.npz')),
    ] + model_bundle_data_files(),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Paolo Andoni',
    maintainer_email='paolo@example.com',
    description='Contact-audio perception nodes (ROS 2 port of the audio manipulation pipeline).',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Nodes are registered here as they are implemented.
            'ast_encoder_node = contact_audio_perception.ast_encoder_node:main',
            'audio_capture_node = contact_audio_perception.audio_capture_node:main',
            'audio_daq_node = contact_audio_perception.audio_daq_node:main',
            'audio_window_node = contact_audio_perception.audio_window_node:main',
            'contact_classifier_node = contact_audio_perception.contact_classifier_node:main',
            'contact_ml_node = contact_audio_perception.contact_ml_node:main',
            'observation_node = contact_audio_perception.observation_node:main',
            'wav_player_node = contact_audio_perception.wav_player_node:main',
        ],
    },
)
