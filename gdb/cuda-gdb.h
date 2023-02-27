/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2007-2020 NVIDIA Corporation
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

#ifndef _CUDA_GDB_H
#define _CUDA_GDB_H 1

#include "ui-file.h"

#define DEFAULT_PROMPT   "(ixgdb) "

void cuda_print_message_nvidia_version (struct ui_file *stream);
void cuda_print_message_iluvatar_version (struct ui_file *stream);

#endif

