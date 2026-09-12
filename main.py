#!/usr/bin/env python3
"""Модуль Б. Автономная навигация РМК-2 по сетке ArUco-маркеров."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
os.environ.setdefault("ROS_DOMAIN_ID", "0")

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    Buffer,
)
from tf2_ros.transform_listener import TransformListener

from navigation_core import (
    MarkerGrid,
    OccupancyTracker,
    clamp,
    normalize_angle,
    points_to_markers,
    summarize_scan,
    yaw_from_odom,
    yaw_from_quaternion,
)

DEFAULTS = {
    "start_id": 0,
    "target_id": 22,
    # Поле. По умолчанию - поле финала ЧВТ по документации: 5x5, нумерация
    # по столбцам ("col_major") - правый столбец снизу вверх 0..4, следующий
    # влево 5..9, левый 20..24. Для тренировки в симуляторе (6x6, построчная
    # нумерация "row_major") переопредели явно:
    #   -p grid_rows:=6 -p grid_cols:=6 -p grid_order:=row_major \
    #   -p aruco_topic:=/RMC2/aruco_id -p scan_topic:=/RMC2/scan
    # Размер и нумерация задаются раздельно, менять надо оба сразу.
    "grid_rows": 5,
    "grid_cols": 5,
    "grid_order": "col_major",
    "marker_spacing": 1.0,

    # Топики РМК-2. По умолчанию - поле финала по документации:
    # /RMC2/camera_bottom/aruco_id и отдельный передний лидар
    # /RMC2/scan_front (на поле /RMC2/scan - это сумма переднего и заднего,
    # в ней видно корпус и заднюю полусферу). В симуляторе - /RMC2/aruco_id
    # и объединённый /RMC2/scan, см. переопределение выше.
    "aruco_topic": "/RMC2/camera_bottom/aruco_id",
    "scan_topic": "/RMC2/scan_front",
    "map_angle_offset": 0.009,

    "max_linear": 0.35,
    "max_angular": 0.30,
    "yaw_tolerance": 0.02,
    "yaw_settle_time": 0.20,
    "odom_reach_tolerance": 0.06,
    "odom_timeout": 2.0,
    "search_enabled": True,
    "search_speed": 0.05,
    "search_yaw_rate": 0.18,
    "search_timeout": 12.0,
    "center_tolerance": 0.02,
    "final_tolerance": 0.015,
    "final_align": True,
    "final_yaw_align": False,
    "align_at_target": True,
    "final_yaw_tolerance": 0.012,
    "final_align_timeout": 15.0,
    "center_speed": 0.09,
    "center_timeout": 5.0,
    "aruco_fresh_time": 0.8,
    "theta_gain": 0.4,
    "aruco_frame_prefix": "aruco_",

    "stop_distance": 0.35,
    "slow_distance": 0.80,
    "front_sector_deg": 50.0,
    "robot_radius": 0.28,
    "scan_max_range": 3.0,
    "obstacle_radius": 0.35,
    "attribution_range": 1.6,
    "hits_to_block": 3,
    "min_points_per_marker": 2,
    "block_forget_time": 6.0,
    "protect_border": False,
    "base_frame": "RMC2/base_link",
    "scan_yaw_offset_deg": 999.0,
    "replan_retry_period": 3.0,
    "blocked_markers": "",
    "cross_track_gain": 1.5,
    "heading_gain": 1.2,
    "leg_overshoot": 1.6,

    "wait_for_go": True,
    "go_topic": "/chvt/go",

    "auto_spawn_obstacle": False,
    "spawn_x": -3.0,
    "spawn_y": 3.0,
    "spawn_name": "obstacle",
}

CONTROL_PERIOD = 0.1


class Mission(Node):
    def __init__(self):
        super().__init__("rmc2_mission")

        for name, value in DEFAULTS.items():
            self.declare_parameter(name, value)

        def p(name):
            return self.get_parameter(name).value

        self.start_id = int(p("start_id"))
        self.target_id = int(p("target_id"))

        self.grid = MarkerGrid(rows=int(p("grid_rows")),
                               cols=int(p("grid_cols")),
                               spacing=float(p("marker_spacing")),
                               angle_offset=float(p("map_angle_offset")),
                               order=str(p("grid_order")))

        self.max_linear = float(p("max_linear"))
        self.max_angular = float(p("max_angular"))
        self.yaw_tolerance = float(p("yaw_tolerance"))
        self.yaw_settle_time = float(p("yaw_settle_time"))
        self.reach_tolerance = float(p("odom_reach_tolerance"))
        self.odom_timeout = float(p("odom_timeout"))
        self.search_enabled = bool(p("search_enabled"))
        self.search_speed = float(p("search_speed"))
        self.search_yaw_rate = float(p("search_yaw_rate"))
        self.search_timeout = float(p("search_timeout"))
        self.center_tolerance = float(p("center_tolerance"))
        self.final_tolerance = float(p("final_tolerance"))
        self.final_align = bool(p("final_align"))
        self.final_yaw_align = bool(p("final_yaw_align"))
        self.align_at_target = bool(p("align_at_target"))
        self.final_yaw_tolerance = float(p("final_yaw_tolerance"))
        self.final_align_timeout = float(p("final_align_timeout"))
        self.center_speed = float(p("center_speed"))
        self.center_timeout = float(p("center_timeout"))
        self.aruco_fresh_time = float(p("aruco_fresh_time"))
        self.theta_gain = float(p("theta_gain"))
        self.aruco_prefix = str(p("aruco_frame_prefix"))

        self.stop_distance = float(p("stop_distance"))
        self.slow_distance = float(p("slow_distance"))
        self.front_half_angle = math.radians(float(p("front_sector_deg")) / 2.0)
        self.robot_radius = float(p("robot_radius"))
        self.scan_max_range = float(p("scan_max_range"))
        self.obstacle_radius = float(p("obstacle_radius"))
        self.attribution_range = float(p("attribution_range"))
        self.protect_border = bool(p("protect_border"))
        self.base_frame = str(p("base_frame"))
        manual = float(p("scan_yaw_offset_deg"))
        self.manual_scan_yaw = None if manual > 900.0 else math.radians(manual)
        self.replan_retry_period = float(p("replan_retry_period"))
        self.cross_track_gain = float(p("cross_track_gain"))
        self.heading_gain = float(p("heading_gain"))
        self.leg_overshoot = float(p("leg_overshoot"))
        self.static_blocked = set()
        raw = str(p("blocked_markers")).replace(";", ",")
        for chunk in raw.split(","):
            chunk = chunk.strip()
            if chunk:
                self.static_blocked.add(int(chunk))

        self.occupancy = OccupancyTracker(hits_to_block=int(p("hits_to_block")),
                                          min_points=int(p("min_points_per_marker")),
                                          forget_time=float(p("block_forget_time")))

        self.wait_for_go = bool(p("wait_for_go"))
        self.go_topic = str(p("go_topic"))

        self.auto_spawn = bool(p("auto_spawn_obstacle"))
        self.spawn_x = float(p("spawn_x"))
        self.spawn_y = float(p("spawn_y"))
        self.spawn_name = str(p("spawn_name"))

        if not (self.grid.valid(self.start_id) and self.grid.valid(self.target_id)):
            raise ValueError(f"start_id/target_id вне сетки 0..{self.grid.size - 1}")

        self.pose = None
        self.yaw = None
        self.current_aruco = None
        self.aruco_stamp = 0.0
        self.front_dist = float("inf")
        self.scan_seen = False
        self.scan_offset = None
        self.scan_offset_warned = False
        self.retry_deadline = 0.0
        self.pending_goal = None
        self.pending_label = ""
        self.blocked_phase = None
        self.emergency = False
        self.emergency_logged = False

        self.route = []
        self.route_index = 0
        self.phase = "WAIT_START"
        self.phase_deadline = 0.0
        self.mission_start_time = None
        self.leg_start_time = None
        self.armed_phase = None
        self.armed_label = ""
        self.go_requested = False
        self.leg_times = []
        self.leg_names = []
        self.moving = False

        self.last_marker = None
        self.last_marker_odom = None
        self.centering_marker = None
        self.centering_time = 0.0
        self.turn_settled_at = None
        self.leg_origin = None
        self.leg_started_at = 0.0
        self.aligned = False
        self.pending_conflict = set()
        self.aruco_msgs = 0
        self.marker_tf_ok = False
        self.marker_tf_stale = 0
        self.sim_time = 0.0
        self.diag_deadline = 0.0
        self.odom_wait_marker = None
        self.odom_wait_time = 0.0
        self.obstacle_halt_since = None
        self.search_marker = None
        self.search_started = 0.0
        self.start_field_yaw = None
        self.align_started = 0.0
        self.align_step = "center"
        self.align_next_phase = "FINISHED"
        self.align_marker = None

        self.last_linear = 0.0
        self.last_angular = 0.0
        self.last_status_log = 0.0
        self.last_throttle_log = 0.0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, "/RMC2/cmd_vel", 10)
        self.create_subscription(String, p("aruco_topic"), self.aruco_cb, 10)
        self.create_subscription(Odometry, "/RMC2/odometry", self.odom_cb, 10)
        self.create_subscription(
            LaserScan, p("scan_topic"), self.scan_cb, qos_profile_sensor_data
        )
        self.create_subscription(Bool, "/chvt/emergency_stop", self.estop_cb, 10)
        self.create_subscription(Bool, self.go_topic, self.go_cb, 10)
        self.start_keyboard_listener()

        self.log_path = self.make_log_file()
        self.banner()
        self.timer = self.create_timer(CONTROL_PERIOD, self.tick)

    def make_log_file(self):
        log_dir = Path(__file__).resolve().parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"mission_{datetime.now():%Y%m%d_%H%M%S}.log"

    def log(self, text, level="INFO"):
        line = f"[{datetime.now():%H:%M:%S.%f}"[:-3] + f"] [{level}] {text}"
        if level in ("WARN", "EMERGENCY"):
            self.get_logger().warning(line)
        else:
            self.get_logger().info(line)
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            self.get_logger().error(f"Не удалось записать лог: {exc}")

    def log_throttled(self, text, level="INFO", period=1.0):
        now = self.now()
        if now - self.last_throttle_log > period:
            self.last_throttle_log = now
            self.log(text, level)

    def banner(self):
        self.log(f"Модуль Б. Старт {self.start_id}, цель {self.target_id}, "
                 f"сетка {self.grid.rows}x{self.grid.cols}, шаг {self.grid.spacing} м")
        if self.static_blocked:
            self.log(f"Занятыми объявлены: {sorted(self.static_blocked)}")
        self.log(f"Лог: {self.log_path}")
        self.log("Жду стартовый маркер")

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def aruco_cb(self, msg):
        match = re.search(r"-?\d+", msg.data)
        if match:
            self.current_aruco = int(match.group(0))
            self.aruco_stamp = self.now()
            self.aruco_msgs += 1

    def fresh_aruco(self):
        """ID маркера, если он виден ПРЯМО СЕЙЧАС, иначе None."""
        if self.current_aruco is None:
            return None
        if self.now() - self.aruco_stamp > self.aruco_fresh_time:
            return None
        return self.current_aruco

    def odom_cb(self, msg):
        self.pose = msg.pose.pose.position
        self.yaw = yaw_from_odom(msg)
        self.sim_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def estop_cb(self, msg):
        self.emergency = bool(msg.data)
        if self.emergency and not self.emergency_logged:
            self.log("АВАРИЙНАЯ ОСТАНОВКА по /chvt/emergency_stop", "EMERGENCY")
            self.emergency_logged = True
        elif not self.emergency and self.emergency_logged:
            self.log("Аварийная остановка снята, работа продолжается", "EMERGENCY")
            self.emergency_logged = False

    def request_go(self, source):
        if self.phase != "ARMED":
            self.log(f"Команда получена ({source}), но сейчас не жду её", "WARN")
            return False
        self.go_requested = True
        return True

    def go_cb(self, msg):
        if bool(msg.data):
            self.request_go("топик")

    def start_keyboard_listener(self):
        """Enter в терминале запуска = команда «ехать»."""
        try:
            if not sys.stdin.isatty():
                return
        except (AttributeError, ValueError):
            return

        def reader():
            while sys.stdin.readline():
                self.request_go("Enter")

        threading.Thread(target=reader, daemon=True).start()

    def route_array(self):
        return "[" + ", ".join(str(m) for m in self.route) + "]"

    def arm(self, drive_phase, label):
        """Маршрут выведен, ждём команду эксперта. Время паузы не в зачёт."""
        self.armed_phase = drive_phase
        self.armed_label = label
        self.go_requested = False
        self.stop()
        if not self.wait_for_go:
            self.start_leg()
            return
        self.phase = "ARMED"
        self.log(f"Жду команду на движение {label}: Enter в этом терминале "
                 f"или ros2 topic pub --once {self.go_topic} "
                 f"std_msgs/Bool \"{{data: true}}\"")

    def handle_armed(self):
        self.stop()
        if self.go_requested:
            self.go_requested = False
            self.start_leg()

    def start_leg(self):
        self.phase = self.armed_phase
        self.leg_start_time = self.now()
        if self.mission_start_time is None:
            self.mission_start_time = self.leg_start_time
        self.moving = True
        self.log(f"movement_start {self.armed_label} {self.route_array()}")

    def stop_leg(self, name):
        if not self.moving:
            return
        self.moving = False
        elapsed = self.now() - self.leg_start_time
        self.leg_times.append(elapsed)
        self.leg_names.append(name)
        self.log(f"movement_stop {name}, {elapsed:.1f} с")

    def resolve_scan_offset(self, scan_frame):
        """
        Трансформ base_link -> фрейм скана. У РМК-2 laser_merged повёрнут
        относительно base_link, и без этой поправки все точки лидара
        оказываются развёрнутыми, а робот тормозит перед пустотой.
        """
        if self.scan_offset is not None:
            return self.scan_offset
        if self.manual_scan_yaw is not None:
            self.scan_offset = (0.0, 0.0, self.manual_scan_yaw)
            self.log(f"Поворот фрейма скана задан вручную: "
                     f"{math.degrees(self.manual_scan_yaw):+.1f}°", "WARN")
            return self.scan_offset
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, scan_frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            if not self.scan_offset_warned:
                self.scan_offset_warned = True
                self.log(f"Жду TF {self.base_frame} -> {scan_frame}, "
                         f"пока препятствия не анализируются", "WARN")
            return None
        t = tf.transform.translation
        yaw = yaw_from_quaternion(tf.transform.rotation)
        self.scan_offset = (t.x, t.y, yaw)
        self.log(f"TF {self.base_frame} -> {scan_frame}: "
                 f"смещение ({t.x:+.3f}, {t.y:+.3f}) м, поворот {math.degrees(yaw):+.1f}°")
        if abs(yaw) > math.radians(5.0):
            self.log(f"Фрейм скана повёрнут на {math.degrees(yaw):+.1f}° — "
                     f"поправка учтена, точки лидара приводятся к base_link")
        return self.scan_offset

    def scan_cb(self, msg):
        if not self.scan_seen:
            self.scan_seen = True
            self.log(f"Лидар подключён: фрейм {msg.header.frame_id}, {len(msg.ranges)} лучей, "
                     f"сектор {math.degrees(msg.angle_min):.0f}..{math.degrees(msg.angle_max):.0f}°")
        offset = self.resolve_scan_offset(msg.header.frame_id)
        if offset is None:
            self.front_dist = float("inf")
            return
        self.front_dist, points = summarize_scan(
            msg, self.robot_radius, self.scan_max_range, self.front_half_angle,
            offset=offset)
        if not self.grid.anchored or self.pose is None:
            return
        hits = points_to_markers(points, self.grid,
                                 self.pose.x, self.pose.y, self.yaw,
                                 self.obstacle_radius,
                                 max_point_range=self.attribution_range)
        newly_blocked, newly_free = self.occupancy.update(
            hits, self.now(), protected=self.protected_markers())
        for mid in newly_blocked:
            self.log(f"ПРЕПЯТСТВИЕ: маркер {mid} занят по данным лидара "
                     f"({hits.get(mid, 0)} точек в радиусе {self.obstacle_radius} м)", "WARN")
        for mid in newly_free:
            self.log(f"Маркер {mid} снова свободен")

    def aruco_offset(self, marker_id):
        """Положение маркера относительно base_link по данным нижней камеры."""
        frame = f"{self.aruco_prefix}{marker_id}"
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, frame, rclpy.time.Time())
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        if not self.marker_tf_ok:
            self.marker_tf_ok = True
            self.log(f"TF маркеров доступен ({frame}), доводка идёт по замеру камеры")
        stamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9
        if stamp > 0.0 and self.sim_time > 0.0:
            age = self.sim_time - stamp
            if age > self.aruco_fresh_time or age < -self.aruco_fresh_time:
                self.marker_tf_stale += 1
                return None
        return tf.transform.translation.x, tf.transform.translation.y

    def offset_to_odom(self, offset):
        """Смещение в base_link -> точка в одометрии."""
        ox = self.pose.x + offset[0] * math.cos(self.yaw) - offset[1] * math.sin(self.yaw)
        oy = self.pose.y + offset[0] * math.sin(self.yaw) + offset[1] * math.cos(self.yaw)
        return ox, oy

    def protected_markers(self):
        """Маркеры, которые нельзя объявлять занятыми."""
        protected = {self.start_id}
        if self.last_marker is not None:
            protected.add(self.last_marker)
        current = self.fresh_aruco()
        if current is not None:
            protected.add(current)
        if self.protect_border:
            protected |= {m for m in range(self.grid.size) if self.grid.is_border(m)}
        return protected

    def diagnostics(self):
        """Один раз через 20 с после старта проверяем, что источники живы."""
        if self.diag_deadline <= 0.0 or self.now() < self.diag_deadline:
            return
        self.diag_deadline = 0.0
        if self.marker_tf_stale > 20 and self.marker_tf_ok:
            self.log(f"ДИАГНОСТИКА: TF маркера находится, но отбракован по "
                     f"возрасту {self.marker_tf_stale} раз. Проверь, что "
                     f"одометрия штампуется временем симулятора", "WARN")
        if self.aruco_msgs < 5:
            self.log(f"ДИАГНОСТИКА: за 20 с пришло всего {self.aruco_msgs} "
                     f"сообщений /RMC2/aruco_id. Робот едет вслепую, по счислению. "
                     f"Проверь: ros2 topic hz /RMC2/aruco_id", "WARN")
        else:
            self.log(f"ДИАГНОСТИКА: ArUco-кадров {self.aruco_msgs}, "
                     f"TF маркеров {'есть' if self.marker_tf_ok else 'НЕТ'}")
        if not self.marker_tf_ok:
            self.log(f"ДИАГНОСТИКА: TF {self.base_frame} -> "
                     f"{self.aruco_prefix}<id> не найден. Доводка идёт по ID, "
                     f"точность ниже. Проверь: ros2 run tf2_ros tf2_echo "
                     f"{self.base_frame} {self.aruco_prefix}0", "WARN")

    def tick(self):
        self.diagnostics()
        if self.emergency:
            self.hard_stop()
            return
        if self.pose is None or self.yaw is None:
            self.stop()
            self.log_throttled("Нет одометрии /RMC2/odometry, стою", "WARN", 5.0)
            return

        if self.phase == "WAIT_START":
            self.handle_wait_start()
        elif self.phase == "ARMED":
            self.handle_armed()
        elif self.phase in ("GO_TARGET", "GO_START"):
            self.handle_navigation()
        elif self.phase == "AT_TARGET":
            self.handle_at_target()
        elif self.phase == "WAIT_CLEAR":
            self.handle_wait_clear()
        elif self.phase == "ALIGN":
            self.handle_align()
        elif self.phase == "FINISHED":
            self.stop()

    def handle_wait_start(self):
        self.stop()
        current = self.fresh_aruco()
        if current is None:
            self.log_throttled("Стартовый маркер не виден нижней камерой, жду...", "INFO", 3.0)
            return
        if current != self.start_id:
            self.log_throttled(f"Виден маркер {current}, ожидается {self.start_id}. Жду...",
                               "WARN", 3.0)
            return

        self.grid.anchor(self.start_id, self.pose.x, self.pose.y, self.yaw)
        offset = self.aruco_offset(self.start_id)
        if offset is not None:
            measured = self.offset_to_odom(offset)
            self.grid.reanchor(self.start_id, measured[0], measured[1])
            self.last_marker_odom = measured
        else:
            self.last_marker_odom = (self.pose.x, self.pose.y)
            self.log("TF маркера недоступен, привязка по позе робота", "WARN")
        self.last_marker = self.start_id
        self.start_field_yaw = self.grid.field_yaw(self.yaw)
        self.log(f"Стартовый маркер {self.start_id} подтверждён")
        self.diag_deadline = self.now() + 20.0

        if not self.plan(self.start_id, self.target_id, "к цели"):
            self.enter_wait_clear("GO_TARGET")
            return
        self.arm("GO_TARGET", "к цели")

    def current_grid_direction(self):
        """Направление сетки, в котором РМК стоит прямо сейчас."""
        if self.yaw is None or not self.grid.anchored:
            return None
        return self.grid.direction_from_field_yaw(self.grid.field_yaw(self.yaw))

    def plan(self, start, goal, label):
        """
        Кратчайший маршрут в обход занятых ячеек: минимум ячеек, при равенстве
        минимум поворотов, с учётом текущего курса робота.
        """
        self.pending_goal = goal
        self.pending_label = label
        exclude = (set(self.occupancy.blocked) | self.static_blocked) - {start, goal}
        heading = self.current_grid_direction()
        route = self.grid.route(start, goal, exclude=exclude, start_direction=heading)
        if route is None:
            self.log(f"Нет пути {label}: {start} -> {goal} в обход "
                     f"{sorted(exclude)}, жду", "EMERGENCY")
            self.stop()
            return False
        self.route = route
        self.route_index = 1 if len(route) > 1 else 0
        turns = self.grid.turns_in_route(route, heading)
        occupied = f", занято {sorted(exclude)}" if exclude else ""
        self.log(f"МАРШРУТ {label}: {self.route_array()} "
                 f"(ячеек {len(route) - 1}, поворотов {turns}){occupied}")
        return True

    def replan_if_blocked(self):
        """Перестроение маршрута."""
        if not self.route or self.route_index >= len(self.route):
            return True
        goal = self.route[-1]
        blocked = (set(self.occupancy.blocked) | self.static_blocked) - {goal}
        if self.last_marker is not None:
            blocked.discard(self.last_marker)

        ahead = set(self.route[self.route_index:])
        conflict = ahead & blocked
        if not conflict:
            self.pending_conflict = set()
            return True

        next_marker = self.route[self.route_index]
        at_marker = not self.aligned and self.leg_origin is None
        if next_marker not in conflict and not at_marker:
            if conflict != self.pending_conflict:
                self.pending_conflict = set(conflict)
                self.log(f"Дальше по маршруту заняты маркеры {sorted(conflict)}. "
                         f"Сначала доеду до маркера {next_marker}, потом перестрою")
            return True

        label = "к цели" if self.phase == "GO_TARGET" else "на старт"
        self.log(f"Маршрут перекрыт маркерами {sorted(conflict)}, перестраиваю", "WARN")
        self.stop()
        self.pending_conflict = set()
        start = self.last_marker if self.last_marker is not None else self.route[0]
        if not self.plan(start, goal, label + " (перестроенный)"):
            self.enter_wait_clear(self.phase)
            return False
        if self.route[self.route_index if self.route_index < len(self.route) else -1] \
                in blocked:
            self.log("Новый маршрут всё равно упирается в занятый маркер, жду", "WARN")
            self.enter_wait_clear(self.phase)
            return False
        self.leg_origin = None
        self.aligned = False
        self.route_index = 1 if len(self.route) > 1 else 0
        return True

    def leg_frame(self, origin, target):
        """Геометрия текущего отрезка в системе поля."""
        ax, ay = self.grid.coords(origin)
        bx, by = self.grid.coords(target)
        heading = math.atan2(by - ay, bx - ax)
        leg_len = math.hypot(bx - ax, by - ay)
        px, py = self.grid.to_grid(self.pose.x, self.pose.y)
        dx, dy = px - ax, py - ay
        along = dx * math.cos(heading) + dy * math.sin(heading)
        cross = -dx * math.sin(heading) + dy * math.cos(heading)
        return heading, along, cross, leg_len

    def handle_navigation(self):
        if self.route_index >= len(self.route):
            self.arrive()
            return
        if not self.replan_if_blocked():
            return
        if self.route_index >= len(self.route):
            self.arrive()
            return

        target = self.route[self.route_index]
        origin = (self.route[self.route_index - 1] if self.route_index > 0
                  else self.last_marker)
        if origin is None or origin == target:
            origin = self.last_marker if self.last_marker is not None else target
        if self.leg_origin != origin:
            self.leg_origin = origin
            self.leg_started_at = self.now()
            self.aligned = False

        seen = self.fresh_aruco()
        if seen == target:
            self.center_on_marker(target, self.aruco_offset(target))
            return
        if seen is not None and seen != origin and self.grid.valid(seen):
            self.relocalize(seen)
            return

        if origin == target:
            self.reach_marker(target, "уже на маркере", measured=None)
            return

        if self.search_marker == target:
            self.search_for_marker(target)
            return

        heading, along, cross, leg_len = self.leg_frame(origin, target)
        remaining = leg_len - along
        desired_yaw = normalize_angle(heading + self.grid.theta)
        yaw_error = normalize_angle(desired_yaw - self.yaw)

        if remaining <= self.reach_tolerance:
            now = self.now()
            if self.odom_wait_marker != target:
                self.odom_wait_marker = target
                self.odom_wait_time = now
                self.log(f"Маркер {target} рядом по счислению (остаток "
                         f"{remaining:+.2f} м, снос {cross:+.2f} м), "
                         f"жду ArUco до {self.odom_timeout} с")
                self.stop()
            elif now - self.odom_wait_time >= self.odom_timeout:
                if self.search_enabled:
                    self.search_for_marker(target)
                else:
                    self.reach_marker(target, "по счислению, ArUco не подтвердил",
                                      measured=None)
            else:
                self.stop()
            return
        self.odom_wait_marker = None
        self.search_marker = None

        if along > leg_len * self.leg_overshoot:
            self.lost_position(origin, target, along, cross)
            return

        if self.obstacle_guard(target):
            return
        speed_scale = self.speed_scale()

        if not self.aligned or abs(yaw_error) > self.yaw_tolerance * 3:
            if abs(yaw_error) > self.yaw_tolerance:
                self.turn_settled_at = None
                angular = clamp(1.4 * yaw_error, -self.max_angular, self.max_angular)
                if abs(angular) < 0.06:
                    angular = math.copysign(0.06, angular)
                self.publish(0.0, angular)
                self.status_log(target, remaining, cross)
                return
            if self.turn_settled_at is None:
                self.turn_settled_at = self.now()
                self.log(f"Курс на маркер {target} взят: "
                         f"{math.degrees(normalize_angle(heading)):+.0f}° в поле")
            if self.now() - self.turn_settled_at < self.yaw_settle_time:
                self.stop()
                return
            self.aligned = True

        linear = clamp(0.4 * remaining, 0.05, self.max_linear) * speed_scale
        angular = clamp(self.heading_gain * yaw_error - self.cross_track_gain * cross,
                        -self.max_angular / 2, self.max_angular / 2)
        self.publish(linear, angular)
        self.status_log(target, remaining, cross)

    def search_for_marker(self, target):
        """Приехали по счислению, а маркера под камерой нет."""
        if self.search_marker != target:
            self.search_marker = target
            self.search_started = self.now()
            self.log(f"Маркер {target} не найден камерой, начинаю поиск "
                     f"(до {self.search_timeout} с)", "WARN")

        elapsed = self.now() - self.search_started
        if elapsed > self.search_timeout:
            self.log(f"Поиск маркера {target} не дал результата, засчитываю "
                     f"по счислению. Снос не скорректирован", "WARN")
            self.search_marker = None
            self.reach_marker(target, "по счислению, поиск не помог", measured=None)
            return

        v, w = self.search_speed, self.search_yaw_rate
        phase = elapsed % 6.0
        if phase < 1.0:
            self.publish(-v, 0.0)
        elif phase < 2.5:
            self.publish(v, 0.0)
        elif phase < 3.5:
            self.publish(0.0, w)
        elif phase < 5.0:
            self.publish(0.0, -w)
        else:
            self.publish(v * 0.5, w * 0.5)

    def relocalize(self, marker):
        """
        Камера показала маркер, которого мы не ждали. Значит счисление увело
        робота не туда. Привязываемся к тому, что реально видим, и строим
        маршрут заново отсюда — вместо того чтобы ехать вслепую дальше.
        """
        self.log(f"Вижу маркер {marker}, а ожидался другой. Перепривязка и "
                 f"перестроение маршрута от {marker}", "WARN")
        self.stop()
        offset = self.aruco_offset(marker)
        measured = (self.offset_to_odom(offset) if offset is not None
                    else (self.pose.x, self.pose.y))
        self.grid.reanchor(marker, measured[0], measured[1])
        self.last_marker = marker
        self.last_marker_odom = measured
        self.leg_origin = None
        self.aligned = False
        goal = self.route[-1] if self.route else self.target_id
        label = "к цели" if self.phase == "GO_TARGET" else "на старт"
        if self.plan(marker, goal, label + " (после перепривязки)"):
            self.route_index = 1 if len(self.route) > 1 else 0
        else:
            self.enter_wait_clear(self.phase)

    def lost_position(self, origin, target, along, cross):
        """
        Проехали мимо маркера и не видим ни одного. Останавливаемся и ищем.
        """
        self.stop()
        self.log(f"Потеря позиции: отрезок {origin} -> {target} пройден на "
                 f"{along:.2f} м при сносе {cross:+.2f} м, маркер не найден. "
                 f"Стою и жду распознавания", "EMERGENCY")
        self.log("Проверь, что публикуется /RMC2/aruco_id "
                 "(ros2 topic hz /RMC2/aruco_id)", "WARN")
        self.leg_started_at = self.now()
        self.enter_wait_clear(self.phase)

    def center_on_marker(self, target, offset):
        """
        Доводка на маркер. Если доступен TF маркера — ведём по замеру смещения
        (камера стоит на оси робота, поэтому нулевое смещение = робот точно над
        маркером). Если TF нет, подтверждаем маркер по ID и привязываемся по
        позе робота: точность хуже, но снос всё равно обнуляется.
        """
        if self.centering_marker != target:
            self.centering_marker = target
            self.centering_time = self.now()
            where = (f"смещение ({offset[0]:+.3f}, {offset[1]:+.3f}) м"
                     if offset is not None else "TF маркера нет, беру по ID")
            self.log(f"Маркер {target} в поле зрения, {where}")

        if offset is None:
            self.stop()
            if self.now() - self.centering_time >= 0.4:
                self.reach_marker(target, "по ArUco (по ID, без TF)",
                                  measured=(self.pose.x, self.pose.y))
            return

        dist = math.hypot(offset[0], offset[1])
        if dist <= self.center_tolerance:
            self.reach_marker(target, f"по ArUco, смещение {dist:.3f} м",
                              measured=self.offset_to_odom(offset))
            return
        if self.now() - self.centering_time > self.center_timeout:
            self.log(f"Доводка на маркер {target} заняла больше "
                     f"{self.center_timeout} с, принимаю смещение {dist:.3f} м", "WARN")
            self.reach_marker(target, f"по ArUco, смещение {dist:.3f} м (таймаут)",
                              measured=self.offset_to_odom(offset))
            return

        angle = math.atan2(offset[1], offset[0])
        if abs(angle) < math.pi / 2:
            linear = clamp(0.6 * dist, 0.02, self.center_speed)
            angular = clamp(1.2 * angle, -self.max_angular / 2, self.max_angular / 2)
        else:
            back = normalize_angle(angle - math.pi)
            linear = -clamp(0.6 * dist, 0.02, self.center_speed)
            angular = clamp(1.2 * back, -self.max_angular / 2, self.max_angular / 2)
        self.publish(linear, angular)

    def obstacle_guard(self, target):
        """
        Экстренный стоп по лидару. True — движение на этом такте запрещено.
        """
        if self.front_dist >= self.stop_distance:
            if self.obstacle_halt_since is not None:
                self.log(f"Курс свободен ({self.front_dist:.2f} м), продолжаю")
                self.obstacle_halt_since = None
            return False
        if self.obstacle_halt_since is None:
            self.obstacle_halt_since = self.now()
            self.log(f"ЭКСТРЕННЫЙ СТОП: препятствие в {self.front_dist:.2f} м по курсу "
                     f"(порог {self.stop_distance} м, сектор "
                     f"±{math.degrees(self.front_half_angle):.0f}°)", "EMERGENCY")
        self.stop()
        self.log_throttled(f"Стою перед препятствием в {self.front_dist:.2f} м, "
                           f"занятые маркеры: {self.occupancy.ids() or 'нет'}",
                           "WARN", 3.0)
        return True

    def speed_scale(self):
        if self.front_dist >= self.slow_distance:
            return 1.0
        span = max(self.slow_distance - self.stop_distance, 1e-3)
        self.log_throttled(f"Замедление: препятствие в {self.front_dist:.2f} м",
                           "WARN", 2.0)
        return clamp((self.front_dist - self.stop_distance) / span, 0.25, 1.0)

    def reach_marker(self, marker, how, measured=None):
        """
        Маркер пройден. Если есть замер камерой — переобвязываем систему
        координат по нему и уточняем угол поля по пройденному отрезку. Именно
        это не даёт одометрии копить перекос от маркера к маркеру.
        """
        self.log(f"Маркер {marker} достигнут ({how}). Осталось "
                 f"{max(len(self.route) - self.route_index - 1, 0)} маркеров.")
        if measured is not None:
            if self.last_marker is not None and self.last_marker_odom is not None:
                correction = self.grid.refine_theta(
                    self.last_marker, self.last_marker_odom, marker, measured,
                    gain=self.theta_gain)
                if correction is not None and abs(correction) > math.radians(0.2):
                    self.log(f"Угол поля уточнён на {math.degrees(correction):+.2f}°, "
                             f"итого {math.degrees(self.grid.theta):+.2f}°")
            self.grid.reanchor(marker, measured[0], measured[1])
            self.last_marker_odom = measured
        else:
            self.last_marker_odom = None
        self.last_marker = marker
        self.centering_marker = None
        self.odom_wait_marker = None
        self.search_marker = None
        self.turn_settled_at = None
        self.aligned = False
        self.leg_origin = marker
        self.leg_started_at = self.now()
        self.pending_conflict = set()
        self.route_index += 1
        self.stop()

    def status_log(self, target, remaining, cross=0.0):
        now = self.now()
        if now - self.last_status_log < 2.0:
            return
        self.last_status_log = now
        front = "inf" if math.isinf(self.front_dist) else f"{self.front_dist:.2f}"
        blocked = sorted(set(self.occupancy.ids()) | self.static_blocked)
        self.log(f"[статус] маркер {target}: осталось {remaining:.2f} м, "
                 f"снос {cross:+.3f} м, впереди {front} м, "
                 f"угол поля {math.degrees(self.grid.theta):+.1f}°, "
                 f"ArUco-кадров {self.aruco_msgs}, занято: {blocked or 'нет'}")

    def arrive(self):
        """
        Конечный маркер под камерой. movement_stop пишется после доводки:
        доводка — это ещё движение.
        """
        if self.phase == "GO_TARGET":
            self.log(f"Целевой маркер {self.target_id} под камерой")
            self.stop()
            if self.align_at_target:
                self.enter_align(self.target_id, "AT_TARGET")
            else:
                self.phase = "AT_TARGET"
                self.finish_phase()
        else:
            self.log(f"Стартовый маркер {self.start_id} под камерой")
            self.stop()
            self.enter_align(self.start_id, "FINISHED")

    def enter_align(self, marker, next_phase):
        """
        Финальная доводка: точно на центр маркера и в исходную ориентацию.
        """
        if not self.final_align or self.start_field_yaw is None:
            self.phase = next_phase
            self.finish_phase()
            return
        self.align_marker = marker
        self.align_next_phase = next_phase
        self.align_started = self.now()
        self.align_step = "center"
        self.phase = "ALIGN"
        self.stop()

    def handle_align(self):
        """
        Раньше миссия заканчивалась там, где робот остановился, — с остаточным
        разворотом от последнего отрезка и парой сантиметров недоезда. Теперь
        на конечном маркере он сначала подтягивается на центр (тут можно
        подруливать), а потом доворачивается в исходную ориентацию. Порядок
        важен: разворот на месте происходит вокруг оси робота, а камера стоит
        на этой же оси, поэтому доворот уже не сбивает наведение.
        """
        elapsed = self.now() - self.align_started
        if elapsed > self.final_align_timeout:
            self.log(f"Финальная доводка прервана по таймауту "
                     f"{self.final_align_timeout} с", "WARN")
            self.finish_align()
            return

        desired_yaw = self.grid.odom_yaw(self.start_field_yaw)
        yaw_error = normalize_angle(desired_yaw - self.yaw)

        if self.align_step == "center":
            offset = self.aruco_offset(self.align_marker)
            if offset is None:
                self.log("Маркер из-под камеры не виден, точная центровка "
                         "пропущена, доворачиваю", "WARN")
                self.align_step = "turn"
                self.stop()
                return
            dist = math.hypot(offset[0], offset[1])
            if dist <= self.final_tolerance:
                self.align_step = "turn"
                self.stop()
                return
            angle = math.atan2(offset[1], offset[0])
            if abs(angle) < math.pi / 2:
                linear = clamp(0.5 * dist, 0.02, self.center_speed)
                angular = clamp(1.0 * angle, -self.max_angular / 3,
                                self.max_angular / 3)
            else:
                back = normalize_angle(angle - math.pi)
                linear = -clamp(0.5 * dist, 0.02, self.center_speed)
                angular = clamp(1.0 * back, -self.max_angular / 3,
                                self.max_angular / 3)
            self.publish(linear, angular)
            return

        if not self.final_yaw_align:
            self.stop()
            self.finish_align(self.aruco_offset(self.align_marker))
            return
        if abs(yaw_error) > self.final_yaw_tolerance:
            angular = clamp(1.2 * yaw_error, -self.max_angular / 2,
                            self.max_angular / 2)
            if abs(angular) < 0.05:
                angular = math.copysign(0.05, angular)
            self.publish(0.0, angular)
            return
        self.stop()
        self.finish_align(self.aruco_offset(self.align_marker))

    def finish_align(self, offset=None):
        if offset is not None:
            dist = math.hypot(offset[0], offset[1])
            self.grid.reanchor(self.align_marker, *self.offset_to_odom(offset))
            self.log(f"Маркер {self.align_marker}: {dist * 100:.1f} см от центра")
        self.stop()
        self.phase = self.align_next_phase
        self.finish_phase()

    def finish_phase(self):
        """Что делать после доводки: пауза на цели или конец миссии."""
        if self.phase == "AT_TARGET":
            self.stop_leg(f"старт {self.start_id} -> цель {self.target_id}")
            self.phase_deadline = self.now() + 2.0
            if self.auto_spawn:
                self.spawn_obstacle()
        elif self.phase == "FINISHED":
            self.stop_leg(f"цель {self.target_id} -> старт {self.start_id}")
            legs = ", ".join(f"{s:.1f}" for s in self.leg_times)
            self.log(f"МИССИЯ ЗАВЕРШЕНА. Этапы: {legs} с, "
                     f"суммарно {sum(self.leg_times):.1f} с")

    def enter_wait_clear(self, resume_phase):
        """Пути нет. Не сдаёмся: стоим и периодически пробуем снова."""
        self.blocked_phase = resume_phase
        self.retry_deadline = self.now() + self.replan_retry_period
        self.phase = "WAIT_CLEAR"
        self.stop()

    def handle_wait_clear(self):
        self.stop()
        if self.now() < self.retry_deadline:
            return
        self.retry_deadline = self.now() + self.replan_retry_period
        self.log_throttled(f"Пути нет, занятые: {self.occupancy.ids()}. "
                           f"Повторная попытка планирования", "WARN", 2.0)
        if self.plan(self.last_marker, self.pending_goal, self.pending_label):
            self.leg_origin = None
            self.aligned = False
            self.route_index = 1 if self.fresh_aruco() == self.last_marker else 0
            self.log("Путь освободился, маршрут перестроен")
            if self.moving:
                self.phase = self.blocked_phase
            else:
                self.arm(self.blocked_phase, self.pending_label)

    def handle_at_target(self):
        self.stop()
        if self.now() < self.phase_deadline:
            return
        self.last_marker = self.target_id
        offset = self.aruco_offset(self.target_id)
        self.last_marker_odom = (self.offset_to_odom(offset) if offset is not None
                                 else (self.pose.x, self.pose.y))
        if not self.plan(self.target_id, self.start_id, "на старт"):
            self.enter_wait_clear("GO_START")
            return
        self.arm("GO_START", "на старт")

    def spawn_obstacle(self):
        cmd = ["ros2", "launch", "ar_webots_fms_ros2", "spawn_object.launch.py",
               f"x:={self.spawn_x}", f"y:={self.spawn_y}", "angle_z:=0.0",
               f"object_name:={self.spawn_name}"]
        self.log(f"САМОТЕСТ: спавню препятствие в ({self.spawn_x}, {self.spawn_y})")
        try:
            subprocess.Popen(cmd)
        except OSError as exc:
            self.log(f"Не удалось заспавнить препятствие: {exc}", "WARN")

    def publish(self, linear, angular):
        linear = self.last_linear + clamp(linear - self.last_linear, -0.08, 0.08)
        angular = self.last_angular + clamp(angular - self.last_angular, -0.10, 0.10)
        self.last_linear, self.last_angular = linear, angular
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish(0.0, 0.0)

    def hard_stop(self):
        """Аварийный стоп: нули без сглаживания."""
        self.last_linear = 0.0
        self.last_angular = 0.0
        if not rclpy.ok():
            return
        try:
            self.cmd_pub.publish(Twist())
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = Mission()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node:
            node.hard_stop()
            try:
                node.log("Миссия прервана пользователем (Ctrl+C).", "WARN")
            except Exception:
                pass
    finally:
        if node:
            node.hard_stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
