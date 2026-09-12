#!/usr/bin/env python3
"""Модуль Б. Диагностика обнаружения препятствий. Робот НЕ ДВИГАЕТСЯ."""

from __future__ import annotations

import math
import re

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, ConnectivityException, ExtrapolationException, LookupException
from tf2_ros.transform_listener import TransformListener

from navigation_core import (
    MarkerGrid,
    OccupancyTracker,
    points_to_markers,
    summarize_scan,
    yaw_from_odom,
    yaw_from_quaternion,
)

DEFAULTS = {
    "start_id": 0,
    "target_id": 22,
    # По умолчанию - поле финала (5x5, нумерация по столбцам), как в
    # main.py. Для диагностики в симуляторе переопредели:
    #   -p grid_rows:=6 -p grid_cols:=6 -p grid_order:=row_major
    "grid_rows": 5,
    "grid_cols": 5,
    "grid_order": "col_major",
    "marker_spacing": 1.0,
    "map_angle_offset": 0.009,
    "robot_radius": 0.28,
    "scan_max_range": 3.0,
    "obstacle_radius": 0.35,
    "attribution_range": 1.6,
    "front_sector_deg": 50.0,
    "hits_to_block": 3,
    "min_points_per_marker": 2,
    "block_forget_time": 6.0,
    "refresh_period": 1.0,
    "base_frame": "RMC2/base_link",
    "scan_yaw_offset_deg": 999.0,
}


class ScanCheck(Node):
    def __init__(self):
        super().__init__("rmc2_scan_check")
        for name, value in DEFAULTS.items():
            self.declare_parameter(name, value)

        def p(name):
            return self.get_parameter(name).value

        self.start_id = int(p("start_id"))
        self.target_id = int(p("target_id"))
        self.grid = MarkerGrid(int(p("grid_rows")), int(p("grid_cols")),
                               float(p("marker_spacing")), float(p("map_angle_offset")),
                               order=str(p("grid_order")))
        self.robot_radius = float(p("robot_radius"))
        self.scan_max_range = float(p("scan_max_range"))
        self.obstacle_radius = float(p("obstacle_radius"))
        self.attribution_range = float(p("attribution_range"))
        self.front_half_angle = math.radians(float(p("front_sector_deg")) / 2.0)
        self.occupancy = OccupancyTracker(int(p("hits_to_block")),
                                          int(p("min_points_per_marker")),
                                          float(p("block_forget_time")))

        self.pose = None
        self.yaw = 0.0
        self.aruco = None
        self.aruco_stamp = 0.0
        self.front = float("inf")
        self.points_total = 0
        self.hits = {}
        self.base_frame = str(p("base_frame"))
        manual = float(p("scan_yaw_offset_deg"))
        self.manual_scan_yaw = None if manual > 900.0 else math.radians(manual)
        self.scan_offset = None
        self.scan_frame = "?"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(String, "/RMC2/aruco_id", self.aruco_cb, 10)
        self.create_subscription(Odometry, "/RMC2/odometry", self.odom_cb, 10)
        self.create_subscription(LaserScan, "/RMC2/scan", self.scan_cb, qos_profile_sensor_data)
        self.create_timer(float(p("refresh_period")), self.render)

        print("Диагностика лидара. Робот не двигается. Ctrl+C для выхода.")
        print(f"Ждём стартовый маркер {self.start_id} под нижней камерой...")

    def now(self):
        return self.get_clock().now().nanoseconds / 1e9

    def aruco_cb(self, msg):
        match = re.search(r"-?\d+", msg.data)
        if match:
            self.aruco = int(match.group(0))
            self.aruco_stamp = self.now()
            if not self.grid.anchored and self.aruco == self.start_id and self.pose is not None:
                self.grid.anchor(self.start_id, self.pose.x, self.pose.y, self.yaw)
                print(f"Привязка выполнена по маркеру {self.start_id}.")

    def fresh_aruco(self):
        if self.aruco is None or self.now() - self.aruco_stamp > 0.8:
            return None
        return self.aruco

    def odom_cb(self, msg):
        self.pose = msg.pose.pose.position
        self.yaw = yaw_from_odom(msg)

    def scan_cb(self, msg):
        self.front, points = summarize_scan(msg, self.robot_radius,
                                            self.scan_max_range, self.front_half_angle)
        self.points_total = len(points)
        if not self.grid.anchored or self.pose is None:
            return
        self.hits = points_to_markers(points, self.grid, self.pose.x, self.pose.y,
                                      self.yaw, self.obstacle_radius,
                                      max_point_range=self.attribution_range)
        protected = {self.start_id}
        current = self.fresh_aruco()
        if current is not None:
            protected.add(current)
        self.occupancy.update(self.hits, self.now(), protected=protected)

    def render(self):
        if self.pose is None:
            print("Нет одометрии /RMC2/odometry")
            return
        if not self.grid.anchored:
            yaw_txt = ("нет TF" if self.scan_offset is None
                       else f"{math.degrees(self.scan_offset[2]):+.1f}°")
            print(f"Ждём маркер {self.start_id} (сейчас виден: {self.fresh_aruco()}), "
                  f"точек лидара {self.points_total}, поворот фрейма скана {yaw_txt}")
            return

        gx, gy = self.grid.to_grid(self.pose.x, self.pose.y)
        robot_marker, robot_dist = self.grid.nearest(gx, gy)
        blocked = set(self.occupancy.blocked)

        print("\n" + "=" * 58)
        front = "inf" if math.isinf(self.front) else f"{self.front:.2f} м"
        print(f"робот: поле x={gx:+.2f} y={gy:+.2f} | ближайший маркер {robot_marker} "
              f"({robot_dist:.2f} м) | ArUco: {self.fresh_aruco()}")
        yaw_txt = ("нет TF" if self.scan_offset is None
                   else f"{math.degrees(self.scan_offset[2]):+.1f}°")
        print(f"впереди: {front} | точек лидара в работе: {self.points_total} | "
              f"фрейм скана {self.scan_frame}, поворот к base_link {yaw_txt}")
        print(f"занятые маркеры: {self.occupancy.ids() or 'нет'}")
        if self.hits:
            top = sorted(self.hits.items(), key=lambda kv: -kv[1])[:6]
            print("точек по маркерам: " + ", ".join(f"{m}:{c}" for m, c in top))

        print("     " + " ".join(f"c{c}" for c in range(self.grid.cols)))
        for row in range(self.grid.rows):
            cells = []
            for col in range(self.grid.cols):
                mid = row * self.grid.cols + col
                if mid == robot_marker:
                    cells.append(" R")
                elif mid in blocked:
                    cells.append(" X")
                elif mid == self.target_id:
                    cells.append(" T")
                elif mid == self.start_id:
                    cells.append(" S")
                else:
                    cells.append(" .")
            ids = f"r{row} "
            print(f"{ids:<5}" + " ".join(cells))
        print("  R робот, X занято лидаром, S старт, T цель")

        start = robot_marker if robot_marker is not None else self.start_id
        route = self.grid.route(start, self.target_id,
                                exclude=blocked - {start, self.target_id})
        if route is None:
            print(f"МАРШРУТ {start} -> {self.target_id}: ПУТИ НЕТ в обход {sorted(blocked)}")
        else:
            print(f"МАРШРУТ {start} -> {self.target_id}: " + " -> ".join(str(m) for m in route))


def main(args=None):
    rclpy.init(args=args)
    node = ScanCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
