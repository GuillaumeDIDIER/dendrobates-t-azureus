#!/usr/bin/python3
"""
Generate .cores.csv file from the output of `lstopo --of xml`
"""
import sys
import subprocess
from xml.etree import ElementTree


def get_elements(root, attribute):
    return [i for i in root if "type" in i.attrib and i.attrib["type"] == attribute]

def get_element(root, attribute):
    elems = get_elements(root, attribute)

    if len(elems) > 1:
        print(f"More than one '{attribute}' found")
    if len(elems) == 0:
        print(f"No '{attribute}' found !")
        sys.exit(1)

    return elems[0]



if len(sys.argv) <= 1:
    print("No file provided, parsing data for this machine", file=sys.stderr)
    xml_data = subprocess.check_output(['lstopo', '--of', 'xml']).decode("utf8")
    root = ElementTree.fromstring(xml_data)
else:
    tree = ElementTree.parse(sys.argv[1])
    root = tree.getroot()

machine = get_element(root, "Machine")

core_count = 0
sockets = []
for pack in get_elements(machine, "Package"):
    socket = [] # Each L3Cache corresponds to 1 socket
    l3 = get_element(pack, "L3Cache")
    for l2 in get_elements(l3, "L2Cache"):
        core = [] # Each L2Cache corresponds to 1 core
        l1 = get_element(l2, "L1Cache")
        l1i = get_element(l1, "L1iCache")
        core_obj = get_element(l1i, "Core")
        
        for PU in get_elements(core_obj, "PU"):
            core.append(int(PU.attrib["os_index"]))
            core_count += 1

        socket.append((int(l2.attrib["os_index"]), core))
    sockets.append(socket)

# socket, core, hyper-thread
out_data = [
    (-1, -1, -1) for _ in range(core_count)
]

for si, sock in enumerate(sockets):
    for ci, (c_os, core) in enumerate(sock):
        for ti, pu_os in enumerate(core):
            out_data[pu_os] = (si, c_os, ti)

print("socket,core,hthread")
for i in out_data:
    print(f"{i[0]},{i[1]},{i[2]}")
