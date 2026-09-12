from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    package_dir = get_package_share_directory('ar_webots_fms_ros2')

    webots_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            package_dir, '/launch/webots.launch.py'
        ])
    )

    rmc2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            package_dir, '/launch/spawn_rmc2.launch.py'
        ])
    )

    return LaunchDescription([
        webots_launch,

        rmc2_launch
    ])