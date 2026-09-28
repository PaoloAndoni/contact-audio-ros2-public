"""
Encoders: mel filterbank features in, fixed-length embedding out.

Every encoder consumes the SAME published Spectrogram, which is what makes the
Block 3 comparison fair -- no encoder gets to preprocess its input differently.
"""
from contact_audio_perception.encoders.base import Encoder, ENCODER_TYPES

__all__ = ['build_encoder', 'build_encoder_for_model', 'Encoder', 'ENCODER_TYPES']


def build_encoder(encoder_type, **kwargs):
    """Construct an encoder by name; imports lazily so unused deps stay unloaded."""
    if encoder_type == 'ast':
        from contact_audio_perception.encoders.ast_fbank import AstFbankEncoder
        return AstFbankEncoder(**kwargs)
    if encoder_type == 'ast_random':
        from contact_audio_perception.encoders.ast_random import AstRandomEncoder
        return AstRandomEncoder(**kwargs)
    if encoder_type == 'cnn':
        from contact_audio_perception.encoders.cnn import SmallCnnEncoder
        return SmallCnnEncoder(**kwargs)
    if encoder_type == 'melstats':
        from contact_audio_perception.encoders.melstats import MelStatsEncoder
        return MelStatsEncoder(**kwargs)
    raise ValueError(
        f'unknown encoder_type {encoder_type!r}; known types: {sorted(ENCODER_TYPES)}')


def build_encoder_for_model(model):
    """
    Construct the exact encoder a bundle was built with.

    The bundle is the authority: its checkpoint id and pinned revision decide
    which weights load. Resolving the checkpoint name independently would follow
    whatever the upstream branch points at today, which is how a model quietly
    starts running against weights it was never fitted to.
    """
    config = dict(model.encoder_config)
    if model.encoder_type == 'ast':
        return build_encoder('ast', model_id=model.checkpoint_id,
                             revision=model.checkpoint_revision or None)
    if model.encoder_type == 'melstats':
        return build_encoder('melstats', n_mels=config.get('n_mels', 128),
                             stats=config.get('stats'))
    if model.encoder_type in ('cnn', 'ast_random'):
        # load_weights() verifies the SHA-256 before anything is deserialised.
        state = model.load_weights()
        if model.encoder_type == 'cnn':
            return build_encoder(
                'cnn', state_dict=state, n_classes=config['n_classes'],
                n_mels=config['n_mels'], channels=config['channels'],
                embedding_dim=config['embedding_dim'], seed=config['seed'],
                weights_sha256=model.weights_sha256)
        return build_encoder(
            'ast_random', state_dict=state, seed=config['seed'],
            model_id=config['architecture_of'],
            revision=config['architecture_revision'],
            weights_sha256=model.weights_sha256)
    raise ValueError(
        f'no encoder builder for {model.encoder_type!r}; known types: '
        f'{sorted(ENCODER_TYPES)}')
