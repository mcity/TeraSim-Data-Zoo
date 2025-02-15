from pathlib import Path
from typing import Union

from shapely import LineString

from sumolib.net import *
from sumolib.net.connection import Connection
from sumolib.net.edge import Edge
from sumolib.net.lane import Lane
from sumolib.net.node import Node

from waymo_open_dataset.protos import map_pb2, scenario_pb2

from scenparse.utils import *
from scenparse.utils.geometry import interpolate
from scenparse.utils.generic import ms_to_mph


class SUMO2Waymo:
    """
    # SUMO -> Waymo

    A SUMO to Waymo map converter.

    ### Usage

    1. Instantiate a class object ``converter = SUMO2Waymo(sumo_file)``.
    """

    def __init__(self, sumo_file) -> None:

        self.sumonet: Net = readNet(sumo_file, withInternal=True)

        # SUMO stuffs
        self.normal_edges: dict[str, Edge] = {}
        self.internal_edges: dict[str, Edge] = {}

        # Waymo stuffs
        self.feature_counter: int = 1
        self.lane_centers: dict = {}  # feature id, LaneCenter
        self.road_lines: dict = {}  # feature id, RoadLine
        self.road_edges: dict = {}  # feature id, RoadEdge

        self.map_lane2featureid: dict = {}  # Lane -> feature_id

    def parse(self, starter_edges: list[str] = None, have_road_edges=False, have_road_lines=False):
        """
        Parses the edges and performs several steps to process them.

        Args:
            starter_edges (list[str], optional): List of starter edges. Defaults to None.
        """
        print("Parsing SUMO data to Waymo...")

        if starter_edges:
            edge_set: set[Edge] = self._find_neighboring_edges(starter_edges)
        else:
            edge_set = set(self.sumonet.getEdges())

        # I. parse edges
        print(f"Parsing edges...")
        for edge in edge_set:
            if not edge.isSpecial():
                self.normal_edges[edge.getID()] = edge
            else:
                self.internal_edges[edge.getID()] = edge
        print(f"there are {len(self.normal_edges)} normal edges, {len(self.internal_edges)} internal edges")

        # II. normal edges
        print("creating lanecenters for normal edges...")
        for _, edge in self.normal_edges.items():
            # 1. create lanecenter for each lane
            lanes: list[Lane] = edge.getLanes()
            for lane in lanes:
                self._create_lanecenter(lane)

            # 2. add neighbor information between them
            for i in range(0, len(lanes) - 1):
                self._set_neighborship(lanes[i], lanes[i + 1])

            # 3. create road edge for this edge
            if have_road_edges:
                self._create_roadedge(lanes[0], side="right")
                self._create_roadedge(lanes[-1], side="left")

            # 4. create road line
            if have_road_lines:
                for i in range(0, len(lanes) - 1):
                    self.create_roadline(lanes[i], side="left")

        print("creating lanecenters for internal edges...")
        # III. internal edges
        for _, edge in self.internal_edges.items():
            # 1. create lanecenter for each lane
            lanes: list[Lane] = edge.getLanes()
            for lane in lanes:
                self._create_lanecenter(lane)

            # 2. add neighbor information between them
            for i in range(0, len(lanes) - 1):
                self._set_neighborship(lanes[i], lanes[i + 1])

            # entry lanes and exit lanes information
            all_conns: list[Connection] = list(edge.getOutgoing().values()) + list(
                edge.getIncoming().values()
            )
            for conns in all_conns:
                for con in conns:
                    assert type(con) == Connection
                    self._set_connectionship(con.getFromLane(), con.getToLane())

    def _create_lanecenter(self, lane: Lane):
        """
        given a sumo lane, create a LaneCenter() and store it in self.lane_centers
        """
        lanecenter = map_pb2.LaneCenter()
        polyline = interpolate(lane.getShape3D(), lane.getLength())
        assert len(polyline) >= 2
        lanecenter.polyline.extend([map_pb2.MapPoint(x=pt[0], y=pt[1]) for pt in polyline])
        lanecenter.speed_limit_mph = ms_to_mph(lane.getSpeed())
        lanecenter.type = 2

        self.map_lane2featureid[lane] = self.feature_counter
        self.lane_centers[self.feature_counter] = lanecenter
        self.feature_counter += 1

        print(f"lanecenter {self.feature_counter-1} created")

    def _create_roadline(self, lane: Lane, side: str):
        """
        given a sumo lane, create a RoadLine() on its left or right, and store it in self.road_lines
        return the created road_line

        side = left | right
        """

        assert lane in self.map_lane2featureid
        lanecenter = self.lane_centers[self.map_lane2featureid[lane]]
        polyline = lanecenter.polyline[1:-1]
        if len(polyline) <= 1:
            return
        polyline = LineString([(pt.x, pt.y, pt.z) for pt in polyline])

        offset = lane.getWidth() / 2 if side == "left" else -lane.getWidth() / 2
        boundary: LineString = polyline.offset_curve(offset, join_style=1)
        boundary: list[tuple] = list(boundary.coords)

        roadline = map_pb2.RoadLine()
        roadline.type = 1  # default to be DASH WHITE LINE
        roadline.polyline.extend([map_pb2.MapPoint(x=pt[0], y=pt[1]) for pt in boundary[::-1]])

        self.road_lines[self.feature_counter] = roadline
        self.feature_counter += 1

    def _create_roadedge(self, lane: Lane, side: str):
        """
        given a sumo lane, create a RoadEdge() on its left or right, and store it in self.road_edges
        return the created road_line

        side = left | right
        """

        assert lane in self.map_lane2featureid
        lanecenter = self.lane_centers[self.map_lane2featureid[lane]]
        polyline = lanecenter.polyline[1:-1]
        if len(polyline) <= 1:
            return
        polyline = LineString([(pt.x, pt.y, pt.z) for pt in polyline])

        offset = lane.getWidth() / 2 if side == "left" else -lane.getWidth() / 2
        try:
            boundaries = polyline.offset_curve(offset, join_style=1)
        except:
            return
        if type(boundaries) == LineString:
            boundaries = [boundaries]
        else:
            print(f"{self.map_lane2featureid[lane]}, {side}")
            boundaries = list(boundaries.geoms)

        for bd in boundaries:
            boundary: list[tuple] = list(bd.coords)

            roadedge = map_pb2.RoadEdge()
            roadedge.type = 2 if side == "left" else 1
            # right boundary: type == ROAD_EDGE_BOUNDARY | left boundary: type == ROAD_EDGE_MEDIAN
            roadedge.polyline.extend([map_pb2.MapPoint(x=pt[0], y=pt[1]) for pt in boundary[::-1]])

            self.road_edges[self.feature_counter] = roadedge
            self.feature_counter += 1

    def _set_neighborship(self, left_lane: Lane, right_lane: Lane):
        """
        add neighborship information on the corresponding feature of left_lane and right_lane
        left_lane is one of the left neighbors of right_lane, and vice versa

        create_lanecenter() should have been called on self_lane and neighbor_lane
        """

        neighbor1 = map_pb2.LaneNeighbor()
        neighbor1.feature_id = self.map_lane2featureid[right_lane]
        neighbor1.self_start_index = 0
        neighbor1.neighbor_start_index = 0
        neighbor1.self_end_index = len(left_lane.getShape3D()) - 1
        neighbor1.neighbor_end_index = len(right_lane.getShape3D()) - 1

        neighbor2 = map_pb2.LaneNeighbor()
        neighbor2.feature_id = self.map_lane2featureid[left_lane]
        neighbor2.self_start_index = 0
        neighbor2.neighbor_start_index = 0
        neighbor2.self_end_index = len(right_lane.getShape3D()) - 1
        neighbor2.neighbor_end_index = len(left_lane.getShape3D()) - 1

        left_lanecenter = self.lane_centers[self.map_lane2featureid[left_lane]]
        right_lanecenter = self.lane_centers[self.map_lane2featureid[right_lane]]
        left_lanecenter.right_neighbors.append(neighbor1)
        right_lanecenter.left_neighbors.append(neighbor2)

    def _set_connectionship(self, from_lane: Lane, to_lane: Lane):
        """
        add connectionship information on the corresponding features of from_lane and to_lane,
        to_lane is one of from_lane's exit lanes, from_lane is one of to_lane's entry lanes

        create_lanecenter() should have been called on from_lane and to_lane, otherwise do nothing
        """

        if from_lane in self.map_lane2featureid and to_lane in self.map_lane2featureid:
            from_feature_id = self.map_lane2featureid[from_lane]
            to_feature_id = self.map_lane2featureid[to_lane]
            self.lane_centers[to_feature_id].entry_lanes.append(from_feature_id)
            self.lane_centers[from_feature_id].exit_lanes.append(to_feature_id)

    def _find_neighboring_edges(self, starter_edge_ids: list[str], radius: float = 50):
        edge_set = set()

        for edge_id in starter_edge_ids:
            edge: Edge = self.sumonet.getEdge(edge_id)
            lane_sample: Lane = edge.getLane(edge.getLaneNumber() // 2)
            lane_shape: list[tuple] = lane_sample.getShape3D()

            edge_set1 = self.sumonet.getNeighboringEdges(
                lane_shape[0][0], lane_shape[0][1], r=radius, includeJunctions=True
            )
            edge_set2 = self.sumonet.getNeighboringEdges(
                lane_shape[-1][0], lane_shape[-1][1], r=radius, includeJunctions=True
            )
            edge_set.update([item[0] for item in edge_set1])
            edge_set.update([item[0] for item in edge_set2])

        return edge_set

    def save_scenario(self, scenario_id: str = "test", output_dir: str = None):
        """
        save the parsed data into a scenario file
        """

        print("saving Waymo scenario file...")
        scenario = scenario_pb2.Scenario()
        scenario.scenario_id = scenario_id
        for id, lanecenter in self.lane_centers.items():
            feature = map_pb2.MapFeature()
            feature.id = id
            feature.lane.CopyFrom(lanecenter)
            scenario.map_features.append(feature)

        for id, roadedge in self.road_edges.items():
            feature = map_pb2.MapFeature()
            feature.id = id
            feature.road_edge.CopyFrom(roadedge)
            scenario.map_features.append(feature)

        for id, roadline in self.road_lines.items():
            feature = map_pb2.MapFeature()
            feature.id = id
            feature.road_line.CopyFrom(roadline)
            scenario.map_features.append(feature)

        serialized_scenario = scenario.SerializeToString()
        if output_dir is not None:
            output_dir = Path(output_dir)
        else:
            output_dir = Path(f"map/{scenario_id}")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{scenario_id}_waymo"
        with open(output_path, "wb") as f:
            f.write(serialized_scenario)
