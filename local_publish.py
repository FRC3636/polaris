import argparse
import os
from os.path import basename
import logging
import time

import ntcore
import robotpy_apriltag

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("ip", type=str, help="IP address to connect to")
    args = parser.parse_args()

    # Initialize NT4 client
    inst = ntcore.NetworkTableInstance.getDefault()

    identity = "polaris-debug"
    inst.startClient4(identity)

    inst.setServer(args.ip)

    # publish two values
    table = inst.getTable("polaris/config")
    robotpy_apriltag.AprilTagFieldLayout.loadField(robotpy_apriltag.AprilTagField.k2025ReefscapeWelded).serialize("apriltagmap.json")
    # Read apriltagmap.json and publish to NT
    with open("apriltagmap.json", "r") as f:
        tag_layout_json = f.read()
    camera_id = table.getStringTopic("camera_id").publish()
    camera_resolution_width = table.getIntegerTopic("camera_resolution_width").publish()
    camera_resolution_height = table.getIntegerTopic("camera_resolution_height").publish()
    tag_layout = table.getStringTopic("tag_layout").publish()

    while True:
        camera_id.set("/dev/video2")
        camera_resolution_width.set(640)
        camera_resolution_height.set(480)
        tag_layout.set(tag_layout_json)
        time.sleep(0.5)
    