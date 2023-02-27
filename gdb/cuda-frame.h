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

#ifndef _CUDA_FRAME_H
#define _CUDA_FRAME_H 1

#include "defs.h"
#include "cuda-defs.h"
#include "frame.h"
#include "frame-unwind.h"
#include "frame-base.h"


extern const struct frame_unwind cuda_frame_unwind;
extern const struct frame_base   cuda_frame_base;

const struct frame_unwind * cuda_frame_sniffer (struct frame_info *next_frame);
const struct frame_base *   cuda_frame_base_sniffer (struct frame_info *next_frame);

bool cuda_frame_p (struct frame_info *next_frame);
bool cuda_frame_outermost_p (struct frame_info *next_frame);

CORE_ADDR cuda_unwind_pc (struct gdbarch *gdbarch, struct frame_info *next_frame);

#endif

