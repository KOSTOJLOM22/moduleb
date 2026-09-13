# Модуль Б — навигация РМК-2

Автономный проезд от старта до цели и обратно с обходом препятствия лидаром.

## Запуск

```bash
# терминал 1 — симулятор
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch ar_webots_fms_ros2 module2.launch.py

# терминал 2 — Rviz2 (опционально)
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch ar_webots_fms_ros2 rviz_rmc2.launch.py

# терминал 3 — миссия
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
python3 main.py --ros-args -p start_id:=0 -p target_id:=22
```

Движение — **Enter** в терминале 3 после вывода маршрута. Второй раз — после появления обратного маршрута.

## Аварийная остановка

```bash
ros2 topic pub /chvt/emergency_stop std_msgs/Bool "{data: true}"  --once
ros2 topic pub /chvt/emergency_stop std_msgs/Bool "{data: false}" --once
```
