#include "path_smoother/path_smoother_node.hpp"
#include "local_planner/local_planner_server.hpp"
#include "speed_path_planner/speed_path_planner_node.hpp"
#include "trajectory_follower/controller_server.hpp"
#include "traffic_signal_stop/traffic_signal_stop_node.hpp"
#include <rclcpp/rclcpp.hpp>
#include <cstdlib>
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor exec;
    rclcpp::NodeOptions o; o.allow_undeclared_parameters(true); o.automatically_declare_parameters_from_overrides(true);
    auto a = std::make_shared<path_smoother::PathSmootherNode>(o);
    auto b = std::make_shared<local_planner::LocalPlannerServer>(o);
    auto c = std::make_shared<speed_path_planner::SpeedPathPlannerNode>(o);
    auto d = std::make_shared<trajectory_follower::ControllerServer>(o);
    exec.add_node(a); exec.add_node(b); exec.add_node(c); exec.add_node(d);
    std::shared_ptr<traffic_signal_stop::TrafficSignalStopNode> e;
    if (std::getenv("SIL_SIGNAL_NODE")) { e = std::make_shared<traffic_signal_stop::TrafficSignalStopNode>(o); exec.add_node(e); }
    exec.spin(); rclcpp::shutdown(); return 0;
}
