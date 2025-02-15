from sumolib.net import *
from sumolib.net.connection import Connection
from sumolib.net.edge import Edge
from sumolib.net.lane import Lane
from sumolib.net.node import Node
from scenparse.utils import read_data, Pt, point_side_of_polyline
import numpy as np

import xml.dom.minidom
import xml.etree.ElementTree as ET
from typing import Union, Optional
from pathlib import Path
import os

class PedInfoEmbedder:

    def __init__(self, scenario, net_file_path):

        self.scenario = scenario
        self.sumonet: Net = readNet(net_file_path, withInternal=True, withPedestrianConnections=True)
        self.route_root = ET.Element("routes")

        for track in scenario.tracks:
            if track.object_type == 2 and track.states[10].valid:
                self.embed_ped_info(track)

    def embed_ped_info(self, track) -> None:

        location, speed = self.get_loc_and_speed(track)
        nearest_lane, pos, latoffset = self.get_assigned_lane(location)
        if not nearest_lane:
            return

        start_edge = nearest_lane.getEdge()
        route = self.find_route(start_edge)
        if len(route) <= 1 and start_edge.isSpecial():
            print(f"attention: next noraml edge not found for curr_edge {start_edge.getID()}")
            return

        edge_list = [edge.getID() for edge in route]
        person_elem = ET.SubElement(self.route_root, "person")
        person_elem.set("id", str(track.id))
        person_elem.set("depart", "0.00")
        person_elem.set("departPos", f"{pos:.2f}")
        walk_elem = ET.SubElement(person_elem, "walk")
        walk_elem.set("edges", str(" ".join(edge_list)))
        walk_elem.set("departPosLat", f"{latoffset:.2f}")  # find its departPosLat
        walk_elem.set("speed", f"{speed:.2f}")

    def get_loc_and_speed(self, track) -> tuple[Pt, float]:

        last_state = track.states[10]
        location = Pt(last_state.center_x, last_state.center_y, last_state.center_z)
        speed = np.mean(
            [
                np.linalg.norm([state.velocity_x, state.velocity_y])
                for state in track.states[:11]
                if state.valid
            ]
        )
        speed = min(speed, 2)
        return location, speed

    def get_assigned_lane(self, location: Pt) -> Optional[tuple[Lane, float, float]]:

        neighboring_lanes = self.sumonet.getNeighboringLanes(
            location.x, location.y, r=4, includeJunctions=True, allowFallback=True
        )
        neighboring_lanes = [item for item in neighboring_lanes if item[0].allows("pedestrian")]
        if not neighboring_lanes:
            return None, None, None

        nearest_lane, min_dist = min(neighboring_lanes, key=lambda item: item[1])
        
        min_pos, min_dis = nearest_lane.getClosestLanePosAndDist(location.to_list(), perpendicular=False)
        if min_dis > nearest_lane.getWidth()/2 + 0.5:
            return None, None, None
        min_dis = min(min_dis, nearest_lane.getWidth()/2)
        min_pos = min(min_pos, nearest_lane.getLength())
        
        print(nearest_lane, min_dis)

        
        side = point_side_of_polyline(location, [Pt(*pt) for pt in nearest_lane.getShape()])
        min_dist = min(min_dist, nearest_lane.getWidth()/2 - 0.1)
        if side == "Right":
            min_dis = -min_dis

        print(min_dis)
        return nearest_lane, min_pos, min_dis

    def find_route(self, start_edge: Edge) -> list[Edge]:
        reachable_edges: set[Edge] = self.sumonet.getReachable(
            start_edge, vclass="pedestrian", useIncoming=False
        )
        reachable_edges = [
            edge for edge in reachable_edges if edge.getID() != start_edge.getID() and not edge.isSpecial()
        ]
        paths_and_costs: dict[Edge, tuple[list[Edge], float]] = {
            edge: self.sumonet.getOptimalPath(
                fromEdge=start_edge, toEdge=edge, vClass="pedestrian", withInternal=False
            )
            for edge in reachable_edges
        }
        paths_and_costs = {
            the_edge: (path, cost)
            for the_edge, (path, cost) in paths_and_costs.items()
            if path is not None and not the_edge.isSpecial()
        }
        if paths_and_costs:
            _, (optimal_path, highest_cost) = max(paths_and_costs.items(), key=lambda item: item[1][1])
        else:
            optimal_path = [start_edge]

        return optimal_path

    def save_route_file(self, base_dir: Union[str, None]) -> None:
        route_xml = xml.dom.minidom.parseString(ET.tostring(self.route_root)).toprettyxml()
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, f"{self.scenario.scenario_id}.rou.xml"), "w") as f:
            f.write(route_xml)

# 5d6f6adb8f60a8e 140117c81b4ef9fb a2fd6176c7695739 ee5e5fe91cd6c6e7 e9653159c09f5c72

if __name__ == "__main__":

    SCENARIO_ID = "a2fd6176c7695739"

    scenario_list = read_data(
        "/media/led/WD_2TB/WOMD/validation/validation.tfrecord-00001-of-00150", 
    )

    for scenario in scenario_list:
        sumo_file = f"/home/led/Documents/maps/{scenario.scenario_id}/{scenario.scenario_id}.net.xml"
        if os.path.exists(sumo_file):
            pie = PedInfoEmbedder(scenario, sumo_file)
            pie.save_route_file(base_dir=f"/home/led/Documents/maps/{scenario.scenario_id}")
        else:
            assert False

