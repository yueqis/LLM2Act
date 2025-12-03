import glob
import sys
import os
from sys import platform

# Needs to be fixed!
package_dir = os.path.dirname(os.path.abspath(__file__))
simulation_path = os.path.join(package_dir, 'simulation')
if simulation_path not in sys.path:
    sys.path.insert(0, simulation_path)

from unity_simulator.comm_unity import UnityCommunication
from unity_simulator import utils_viz