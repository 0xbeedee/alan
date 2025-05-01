# alan

An exploration of intrinsic motivation in a lifelong learning setting. 
More specifically, we attempt to build an agent that utilises multiple Reinforcement Learning techniques to act competently and autonomously in challenging environments.

For our own experiments, we use [NetHack](https://github.com/heiner/nle), but the machinery developed is expected to be useful in arbitrary environments.


## Running the code

The code uses `python 3.11.12`. To run it, do the following:

1. (Optional) Set up a virtual environment;
2. Install all the requirements (i.e., `pip install -r requirements.txt`);

    2. `nle` is missing from the file because it requires `gymnasium==0.29.1`, while `tianshou` requires `gymnasium<0.29.0`. Forcing `tianshou` to use `0.29.1` seems to be fine, though, so running `pip install nle` after installing all the requirements and ignoring the warning should work.

3. Install the project in editable mode (i.e., `pip install -e .`);
4. Run the `scripts/run_experiment.py` script with whichever arguments interest you!

    4. Note that some permutations of arguments might not work. This is either on purpose (if the arguments do not make sense together), or because the permutation is not yet supported.
