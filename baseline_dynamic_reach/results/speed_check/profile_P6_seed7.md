# Where the time goes: P6, seed 7, 30,720 transitions, in-process environments

476 transitions/s under the profiler (it slows Python code, so shares matter more than the speed). Own time by package:

| Part | Share of time |
|---|---:|
| simulator (highway-env) | 44.5% |
| other (Python built-ins, C extensions) | 24.9% |
| numpy | 18.2% |
| observation encoder | 3.3% |
| torch (networks, updates) | 3.1% |
| SB3 (rollout bookkeeping, buffers) | 2.0% |
| gymnasium wrappers | 1.9% |
| dynamic-goal environment | 1.7% |
| reach environment and logging | 0.4% |
| TensorBoard | 0.0% |
| CSV logging | 0.0% |

Top functions by cumulative time:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.000    0.000   64.528   64.528 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\ppo\ppo.py:302(learn)
       40    0.143    0.004   63.083    1.577 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\on_policy_algorithm.py:162(collect_rollouts)
        1    0.001    0.001   62.960   62.960 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\on_policy_algorithm.py:300(learn)
    10240    0.016    0.000   57.188    0.006 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\vec_env\base_vec_env.py:214(step)
    10240    0.134    0.000   57.171    0.006 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\vec_env\dummy_vec_env.py:56(step_wait)
    30720    0.043    0.000   53.155    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\monitor.py:87(step)
    30720    0.120    0.000   53.101    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\baseline_dynamic_reach\environment.py:176(step)
    30720    0.352    0.000   52.714    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\baseline_dynamic_waypoint\environment.py:239(step)
    30720    0.022    0.000   49.174    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\gymnasium\wrappers\common.py:420(step)
    30720    0.014    0.000   49.152    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\gymnasium\core.py:338(step)
    30720    0.020    0.000   49.138    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\gymnasium\wrappers\common.py:287(step)
    30720    0.135    0.000   49.118    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\envs\common\abstract.py:259(step)
    30720    0.075    0.000   39.667    0.001 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\envs\common\abstract.py:287(_simulate)
    30720    0.080    0.000   25.483    0.001 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\road.py:464(act)
   245760    0.891    0.000   24.385    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\behavior.py:93(act)
    30720    0.345    0.000   12.117    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\road.py:469(step)
  5422392    9.008    0.000    9.694    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\lane.py:209(local_coordinates)
   298571    1.855    0.000    9.190    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\road.py:483(neighbour_vehicles)
  2567872    1.521    0.000    9.143    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\numpy\_core\fromnumeric.py:2245(clip)
  2872052    1.571    0.000    8.496    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\numpy\_core\fromnumeric.py:48(_wrapfunc)
    31786    1.174    0.000    8.182    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\baseline_DQN\fast_observation.py:35(observe)
   276480    0.984    0.000    6.767    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\kinematics.py:130(step)
   307200    1.753    0.000    6.653    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\controller.py:145(steering_control)
   382269    1.504    0.000    6.332    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\behavior.py:150(acceleration)
   245760    0.144    0.000    5.885    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\behavior.py:139(step)
   281277    0.207    0.000    5.670    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\kinematics.py:170(on_state_update)
   286607    0.872    0.000    5.539    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\road.py:55(get_closest_lane_index)
  1105920    0.298    0.000    4.861    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\objects.py:92(handle_collisions)
249320/12760    0.154    0.000    4.664    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\torch\nn\modules\module.py:1783(_wrapped_call_impl)
249320/12760    0.253    0.000    4.640    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\torch\nn\modules\module.py:1791(_call_impl)
  1105920    1.282    0.000    4.563    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\objects.py:122(_is_colliding)
    10240    0.063    0.000    4.403    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\policies.py:636(forward)
   215040    0.150    0.000    3.672    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\behavior.py:219(change_lane_policy)
  1290679    1.322    0.000    3.575    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\numpy\_core\fromnumeric.py:38(_wrapit)
   859821    0.998    0.000    3.422    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\lane.py:132(distance_with_heading)
```
