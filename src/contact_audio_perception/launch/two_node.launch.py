from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch the two-node pipeline: audio_daq_node -> contact_ml_node."""
    source = LaunchConfiguration('source')
    wav_path = LaunchConfiguration('wav_path')
    model_path = LaunchConfiguration('model_path')
    publish_rate_hz = LaunchConfiguration('publish_rate_hz')

    pkg_share = FindPackageShare('contact_audio_perception')
    config = PathJoinSubstitution([pkg_share, 'config', 'audio_pipeline.yaml'])
    default_model = PathJoinSubstitution([pkg_share, 'models', 'contact_model_ast'])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'source', default_value='mic',
                description="'mic' for live capture, 'wav' to replay a file.",
            ),
            DeclareLaunchArgument(
                'wav_path', default_value='',
                description='16-bit PCM WAV to replay (required when source is wav).',
            ),
            DeclareLaunchArgument(
                'model_path', default_value=default_model,
                description='Tagged contact model (default: the bundled AST model).',
            ),
            DeclareLaunchArgument(
                'publish_rate_hz', default_value='15.0',
                description=(
                    'Spectrogram publish rate. The ML node consumes this topic but '
                    'classifies only as fast as the encoder allows (~0.9 Hz for AST '
                    'on CPU), dropping stale inputs.'
                ),
            ),
            Node(
                package='contact_audio_perception',
                executable='audio_daq_node',
                name='audio_daq_node',
                parameters=[config, {'source': source, 'wav_path': wav_path,
                                     'publish_rate_hz': publish_rate_hz}],
                output='screen',
            ),
            Node(
                package='contact_audio_perception',
                executable='contact_ml_node',
                name='contact_ml_node',
                parameters=[config, {'model_path': model_path}],
                output='screen',
            ),
        ]
    )
