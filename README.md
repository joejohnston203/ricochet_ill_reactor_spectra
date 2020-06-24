Requirements:
```
python 2
matplotlib
numpy
scipy
pyROOT
```

Python 2 and pyROOT must be installed separately. The rest can be set up with a virtual environment:

```
$ virtualenv --python=/usr/bin/python2.7 --no-site-packages venv
. venv/bin/activate
pip install -r requirements.txt
```

To make plots cd to the top directory, then do:

```
$ . update_pythonpath.sh
$ cd results/ill_spectrum
$ python make_ill_plots.py
```

This will also generate text files with the neutrino spectrum
from the reactor and the recoil spectrum in Zn and Ge CEvNS
detectors
