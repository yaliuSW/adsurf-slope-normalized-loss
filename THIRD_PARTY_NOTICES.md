# Third-party notices

## ADsurf

`vendor/ADsurf/` contains the upstream numerical implementation used by both
objectives. Upstream: https://github.com/liufeng2317/ADsurf, recorded revision
`7f7d289740f9f433b58079ab9c808adb1c0cae76`.
Its MIT license is retained in `vendor/ADsurf_LICENSE`.

## QEDispInv

`support/qedispinv_install_pack/` pins
https://github.com/pan3rock/QEDispInv at
`ed1b5dd7b449a2b3a27bb8e6581278790f5df8aa` and applies the included Python-interface
patch. The LGPL-3.0 license is retained with the installer. No compiled binary
is distributed. QEDispInv is used for forward curve evaluation, not gradients.

## Numerical observations

The Mirandola observations and borehole reference were digitized from Figures
15 and 17 of Yang et al. (2025), https://doi.org/10.6038/cjg2024R0556.
No publisher PDF or original published image is included.
