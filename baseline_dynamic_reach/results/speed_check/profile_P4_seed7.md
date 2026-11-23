# Where the time goes: P4, seed 7, 30,720 transitions, in-process environments

447 transitions/s under the profiler (it slows Python code, so shares matter more than the speed). Own time by package:

| Part | Share of time |
|---|---:|
| simulator (highway-env) | 44.4% |
| other (Python built-ins, C extensions) | 24.6% |
| numpy | 18.3% |
| observation encoder | 3.6% |
| torch (networks, updates) | 2.9% |
| gymnasium wrappers | 2.3% |
| SB3 (rollout bookkeeping, buffers) | 1.9% |
| dynamic-goal environment | 1.7% |
| reach environment and logging | 0.3% |
| TensorBoard | 0.0% |
| CSV logging | 0.0% |

Top functions by cumulative time:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      2/1    0.000    0.000   68.686   68.686 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\ppo\ppo.py:302(learn)
       40    0.146    0.004   67.179    1.679 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\on_policy_algorithm.py:162(collect_rollouts)
        1    0.001    0.001   66.940   66.940 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\on_policy_algorithm.py:300(learn)
    10240    0.016    0.000   61.246    0.006 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\vec_env\base_vec_env.py:214(step)
    10240    0.135    0.000   61.228    0.006 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\vec_env\dummy_vec_env.py:56(step_wait)
    30720    0.047    0.000   52.426    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\monitor.py:87(step)
    30720    0.072    0.000   52.368    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\baseline_dynamic_reach\environment.py:176(step)
    30720    0.367    0.000   52.189    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\baseline_dynamic_waypoint\environment.py:239(step)
    30720    0.022    0.000   48.642    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\gymnasium\wrappers\common.py:420(step)
    30720    0.014    0.000   48.620    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\gymnasium\core.py:338(step)
    30720    0.019    0.000   48.606    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\gymnasium\wrappers\common.py:287(step)
    30720    0.136    0.000   48.587    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\envs\common\abstract.py:259(step)
    30720    0.077    0.000   39.140    0.001 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\envs\common\abstract.py:287(_simulate)
    30720    0.077    0.000   24.637    0.001 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\road.py:464(act)
   245760    0.875    0.000   23.542    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\behavior.py:93(act)
    30720    0.340    0.000   12.487    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\road.py:469(step)
  5657119    9.426    0.000   10.172    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\lane.py:209(local_coordinates)
    37482    1.363    0.000    9.509    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\baseline_DQN\fast_observation.py:35(observe)
  2592468    1.569    0.000    9.191    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\numpy\_core\fromnumeric.py:2245(clip)
   287960    1.756    0.000    8.865    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\road.py:483(neighbour_vehicles)
  2945777    1.600    0.000    8.627    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\numpy\_core\fromnumeric.py:48(_wrapfunc)
   276480    0.975    0.000    6.745    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\kinematics.py:130(step)
   307200    1.758    0.000    6.639    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\controller.py:145(steering_control)
   336996    1.040    0.000    6.567    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\road\road.py:55(get_closest_lane_index)
   305376    0.215    0.000    6.140    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\kinematics.py:170(on_state_update)
   354676    1.425    0.000    5.934    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\behavior.py:150(acceleration)
   245760    0.144    0.000    5.867    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\behavior.py:139(step)
      628    0.002    0.000    5.854    0.009 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\monitor.py:66(reset)
      628    0.017    0.000    5.853    0.009 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\baseline_dynamic_reach\environment.py:115(reset)
     3162    0.029    0.000    5.617    0.002 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\baseline_waypoint\environment.py:74(reset)
  1105920    0.303    0.000    5.259    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\objects.py:92(handle_collisions)
  1105920    1.315    0.000    4.955    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\highway_env\vehicle\objects.py:122(_is_colliding)
249485/12793    0.145    0.000    4.688    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\torch\nn\modules\module.py:1783(_wrapped_call_impl)
249485/12793    0.252    0.000    4.668    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\torch\nn\modules\module.py:1791(_call_impl)
    10240    0.061    0.000    4.434    0.000 E:\Multi_Agent_Auto_Drive\third_party\HighwayEnv-Automaton\.venv\Lib\site-packages\stable_baselines3\common\policies.py:636(forward)
```
