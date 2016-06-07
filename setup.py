
# 1. Copy tf_rl folder to lib
import os

os.system('git submodule foreach git pull origin master')

if os.path.islink(os.path.abspath('./lib/tf_rl')):
	os.remove(os.path.abspath('./lib/tf_rl'));

os.symlink(os.path.abspath('./submodules/tensorflow_rl/tf_rl'), './lib/tf_rl') 
import lib.tf_rl # Test if library correctly linked
