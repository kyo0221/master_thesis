"""条件ごとの cfg を cfg_mpc_v70.yaml から生成する"""
import yaml, sys, copy
BASE = yaml.safe_load(open('/tmp/claude-1000/-home-kyo-formula-ws-src/2ce42fe3-f430-480e-b923-ae2606bdf87f/scratchpad/silmatrix/cfg_mpc_v70.yaml'))
def make(path, vel=7.0, plugin='trajectory_follower::PidMpcPlugin', merge_fixed=None, branch_vel=4.0, stop_decel=1.5, stop_decel_max=4.0, acc=3.0, signal=True, extend_to=None, fit_max=None):
    c = copy.deepcopy(BASE)
    c['/**']['ros__parameters']['linear_max']['vel'] = float(vel)
    c['/**']['ros__parameters']['linear_max']['acc'] = float(acc)
    c['controller_server_node']['ros__parameters']['controller_plugin'] = plugin
    f = c['local_planner_server_node']['ros__parameters']['frenet']
    if merge_fixed is not None:
        f['merge_length_min_m'] = float(merge_fixed); f['merge_length_max_m'] = float(merge_fixed)
    sp = c['speed_path_planner_node']['ros__parameters']
    sp['branch_linear_max'] = {'vel': float(branch_vel)}
    sp['stop_decel'] = float(stop_decel); sp['stop_decel_max'] = float(stop_decel_max)
    ps = c['path_smoother_node']['ros__parameters']
    if extend_to is not None: ps['extend_to_x_m'] = float(extend_to)
    if fit_max is not None: ps['fit_max_x_m'] = float(fit_max)
    c['traffic_signal_stop_node'] = {'ros__parameters': {'stop_distance_to_panel_pylon_m': 3.3, 'use_sim_time': True}}
    yaml.safe_dump(c, open(path, 'w'), default_flow_style=False, sort_keys=True)
