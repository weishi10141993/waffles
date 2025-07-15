## PDVD light noise hunt within waffles framework

Under Alma9, set up the DAQ environment:
```
cd PDVDLightNoiseHunt
source /cvmfs/dunedaq.opensciencegrid.org/setup_dunedaq.sh
setup_dbt latest
dbt-create -l
dbt-create fddaq-v5.2.1-a9
```

Installing WAFFLES inside the DAQ environment:
```
cd fddaq-v5.2.1-a9
source env.sh # Activate your DAQ environment
# Ensure your SSH keys are properly set up, then:
cd ..
git clone git@github.com:DUNE/waffles.git
cd waffles
which python3 # Should show the .venv Python
python3 -m pip install -r requirements.txt .
```

Setting up Rucio to fetch raw data (alma9, lxplus at CERN, gvpn cluster at Fermilab):
```
cd fddaq-v5.2.1-a9
# from daq folder
source env.sh
cd waffles/scripts
source setup_rucio_a9.sh # need FNAL service account and authentication
python3 fetch_rucio_replicas.py --runs 28676 --max-files 5
```

To run the ``00_HDF5toROOT.py`` it is necessary to build ROOT on the dbt environment, due to versions incompatibility.
To build ROOT you need to source the dbt ``env.sh`` first:

```bash
source env.sh

git clone --branch latest-stable --depth=1 https://github.com/root-project/root.git root_src

mkdir root_build root_install && cd root_build

cmake -DCMAKE_INSTALL_PREFIX=../root_install -Ddataframe=OFF ../root_src

cmake --build . -- install -j1
```

If some error appears during the installation related with Roofit, just disable it by adding the command ``-Droofit=OFF`` on the cmake.

After that, everytime you login, you need to source ROOT, or you can edit ``env.sh`` and add this command:

``source ../root_install/bin/thisroot.sh``
