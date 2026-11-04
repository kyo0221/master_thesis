import xml.etree.ElementTree as ET, rclpy, sys
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
osm = sys.argv[1]
route = [10000000000, 10000000001, 10000000002, 10000000006, 10000000003, 10000000004]
root = ET.parse(osm).getroot()
nodes = {}
for n in root.findall('node'):
    tags = {t.get('k'): float(t.get('v')) for t in n.findall('tag')}
    nodes[int(n.get('id'))] = (tags['local_x'], tags['local_y'])
ways = [[int(nd.get('ref')) for nd in w.findall('nd')] for w in root.findall('way')]
def mk(i, typ, ns, r, g, b, a=1.0):
    m = Marker(); m.header.frame_id = 'map'; m.ns = ns; m.id = i; m.type = typ; m.action = Marker.ADD
    m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a; m.pose.orientation.w = 1.0
    return m
arr = MarkerArray()
for i, way in enumerate(ways):
    m = mk(i, Marker.LINE_STRIP, 'edges', 0.75, 0.75, 0.75); m.scale.x = 0.5
    m.points = [Point(x=nodes[r][0], y=nodes[r][1], z=0.0) for r in way]; arr.markers.append(m)
for i, nid in enumerate(nodes):
    if nid == 0: continue
    x, y = nodes[nid]
    m = mk(i, Marker.SPHERE, 'nodes', 1.0, 0.85, 0.1); m.scale.x = m.scale.y = m.scale.z = 2.4
    m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 0.2; arr.markers.append(m)
    t = mk(i, Marker.TEXT_VIEW_FACING, 'ids', 1.0, 1.0, 1.0); t.scale.z = 3.2; t.text = "node " + str(nid)[-1]
    t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y + 3.4, 0.5; arr.markers.append(t)
# 青コースの経路選択に用いる順序（route）
for i in range(len(route) - 1):
    a, b = nodes[route[i]], nodes[route[i+1]]
    m = mk(i, Marker.ARROW, 'route', 0.2, 0.45, 1.0, 0.9); m.scale.x, m.scale.y, m.scale.z = 0.5, 1.4, 1.4
    m.points = [Point(x=a[0], y=a[1], z=0.4), Point(x=b[0], y=b[1], z=0.4)]; arr.markers.append(m)
o = mk(0, Marker.CYLINDER, 'origin', 0.9, 0.2, 0.2); o.scale.x = o.scale.y = 1.6; o.scale.z = 0.1
o.pose.position.x, o.pose.position.y = nodes[0]; arr.markers.append(o)
rclpy.init(); node = Node('topomap_marker_publisher')
pub = node.create_publisher(MarkerArray, '/topomap_markers', 1)
def tick():
    for m in arr.markers: m.header.stamp = node.get_clock().now().to_msg()
    pub.publish(arr)
node.create_timer(0.5, tick); rclpy.spin(node)
