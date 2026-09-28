from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Launch the DAQ node alone: audio -> /audio/spectrogram (+ image for Foxglove)."""
    source = LaunchConfiguration('source')
    wav_path = LaunchConfiguration('wav_path')
    config = PathJoinSubstitution(
        [FindPackageShare('contact_audio_perception'), 'config', 'audio_pipeline.yaml']
    )
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
            Node(
                package='contact_audio_perception',
                executable='audio_daq_node',
                name='audio_daq_node',
                parameters=[config, {'source': source, 'wav_path': wav_path}],
                output='screen',
            ),
        ]
    )
