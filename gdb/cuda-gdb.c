/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2021 NVIDIA Corporation
 * Written by CUDA-GDB team at NVIDIA <cudatools@nvidia.com>
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, see <http://www.gnu.org/licenses/>.
 */

/* Copyright (C) 2023 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
   Modified from the original CUDA-GDB file referenced above by the ixGDB
   team at Iluvatar. */

#include "defs.h"
#include "cuda.h"
#include "cuda-gdb.h"

void
cuda_print_message_nvidia_version (struct ui_file *stream)
{
  fprintf_unfiltered (stream,
                      "NVIDIA (R) CUDA Debugger\n"
                      "%d.%d release\n"
                      "Portions Copyright (C) 2007-2021 NVIDIA Corporation\n",
		      CUDA_VERSION / 1000, (CUDA_VERSION % 1000) / 10);
}

#if 0
void
cuda_print_message_iluvatar_version (struct ui_file *stream){
  int iMain = 2;
  int iSub = 3;
  int iSubSub = 0;
  fprintf_unfiltered (stream,
                      "ixgdb %d.%d.%d release\n"
                      "Portions Copyright © 2021-2023 Iluvatar CoreX.\n",
		                  iMain, iSub, iSubSub);

}
#else
void
cuda_print_message_iluvatar_version (struct ui_file *stream){
  fprintf_unfiltered (stream,
                      "ixgdb 3.0.0 release\n"
                      "Portions Copyright © 2021-2023 Iluvatar CoreX.\n");

}
#endif
