from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch wav_player -> audio_window -> ast_encoder -> contact_classifier."""
    wav_path = LaunchConfiguration('wav_path')
    model_path = LaunchConfiguration('model_path')
    pkg_share = FindPackageShare('contact_audio_perception')
    config = PathJoinSubstitution([pkg_share, 'config', 'audio_pipeline.yaml'])
    default_model = PathJoinSubstitution([pkg_share, 'models', 'contact_centroids.npz'])
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'wav_path', description='Path to a 16-bit PCM WAV to replay.'
            ),
            DeclareLaunchArgument(
                'model_path',
                default_value=default_model,
                description='Centroids .npz for the classifier (default: bundled model).',
            ),
            Node(
                package='contact_audio_perception',
                executable='wav_player_node',
                name='wav_player_node',
                parameters=[config, {'wav_path': wav_path}],
                output='screen',
            ),
            Node(
                package='contact_audio_perception',
                executable='audio_window_node',
                name='audio_window_node',
                parameters=[config],
                output='screen',
            ),
            Node(
                package='contact_audio_perception',
                executable='ast_encoder_node',
                name='ast_encoder_node',
                parameters=[config],
                output='screen',
            ),
            Node(
                package='contact_audio_perception',
                executable='contact_classifier_node',
                name='contact_classifier_node',
                parameters=[config, {'model_path': model_path}],
                output='screen',
            ),
        ]
    )
