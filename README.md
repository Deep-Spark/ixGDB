# README for ixGDB release

## INTRODUCTION

ixGDB is Iluvatar CUDA source-level debugger for Linux OS, based on NVIDIA [CUDA-GDB](https://github.com/NVIDIA/cuda-gdb) 10.2.

ixGDB provides the following capabilities:
- Provides a seamless debugging environment that allows simultaneous debugging of both GPU and CPU code within the same application.
- Supports debugging C/C++ applications and all CUDA applications, which might use CUDA driver APIs or CUDA runtime APIs.
- Supports setting breakpoints.

## BUILD INSTRUCTIONS (example only, adjust as needed)

First, make sure that libtermcap and other required dependent packages are
installed (try "sudo yum install ncurses-devel"). The "configure" command will
report an error if some packages are missing.

Please note that the libexpat development headers must be present if ixGDB is to be used for cross-platform debugging.

Issue the following commands to build ixGDB:

```
./configure --program-prefix=cuda- \
    --enable-cuda \
    --enable-targets="x86_64-apple-darwin,x86_64-unknown-linux-gnu,\
    arm-elf-linux-gnu,m68k-unknown-linux-gnu" \
    CFLAGS='-I/usr/local/cuda/include' \
    LDFLAGS='-lpthread'
make
```

## USING ixGDB

All standard GDB commands could be used for both CPU and GPU code debugging. In addition to that, ixGDB provides CUDA-specific command families like "info cuda ..." to query GPU states, "cuda .." to control debugger focus on GPU and "[get|set] cuda .." to alter/query CUDA debugger configuration. If you want to know more about how to use ixGDB, please go to Iluvatar CoreX support [official site](https://support.iluvatar.com/#/DocumentCentre?id=1&nameCenter=1&productId=) and use "ixgdb" as the keyword to find document "SDK Tools User Guide", which includes detailed usage of ixGDB.

## COMMUNICATION

[Gitee Issues](https://gitee.com/deep-spark/ixgdb/issues): bug reports, feature requests, install issues, usage issues, etc.

## LICENSE

Licensee's use of the GDB third party component is subject to the terms and conditions of GNU GPL v3:
```
This product includes copyrighted third-party software licensed under the terms of the GNU General Public License v3 ("GPL v3"). All third-party software packages are copyright by their respective authors.
```
Consistent with these licensing requirements, the software listed below is provided under the terms of the specified open source software licenses.
```
Component    License
ixGDB        GPL v3
```
