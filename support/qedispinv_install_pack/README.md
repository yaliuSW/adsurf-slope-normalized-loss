# QEDispInv installation helper

This directory installs the QEDispInv Python modules used only to generate or
independently evaluate dispersion curves for final model selection.
QEDispInv is not called by the root-free inversion loss.

The installer checks out the upstream QEDispInv repository at the pinned
commit

```text
ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa
```

and applies the bundled Python/pybind11 interface patch. No precompiled shared
library is distributed.

## Linux installation

Create the main environment first, then install the build dependencies:

```bash
conda create -n adsurf_epvr -c conda-forge python=3.12
conda activate adsurf_epvr
python -m pip install -r ../../requirements.txt
conda install -n adsurf_epvr -c conda-forge \
  cmake make pybind11 c-compiler cxx-compiler fortran-compiler \
  gcc_linux-64 gxx_linux-64 gfortran_linux-64
```

Run the installer from this directory:

```bash
chmod +x install_qedispinv_to_conda.sh
./install_qedispinv_to_conda.sh --env adsurf_epvr
```

Verify the installation:

```bash
conda run -n adsurf_epvr python -c \
  "import qedispinv, qedispinv_forward, qedispinv_inversion; print('ok')"
```

The saved observation and prediction files in this repository allow the
saved example inversions and summary checks to be inspected without rebuilding
QEDispInv.

## License

QEDispInv and the bundled patch are distributed under the GNU Lesser General
Public License v3.0. See `LICENSE.LGPL-3.0`. The surrounding reproduction
scripts remain under the repository's MIT license.
