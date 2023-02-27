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

#include "defs.h"
#include "frame.h"
#include "common/common-defs.h"
#include "ui-out.h"

#include "cuda-api.h"
#include "cuda-asm.h"
#include "cuda-context.h"
#include "cuda-iterator.h"
#include "cuda-modules.h"
#include "cuda-options.h"
#include "cuda-state.h"
#include "cuda-tdep.h"

/* counter for the CUDA kernel ids */
static uint64_t next_kernel_id = 0;

uint64_t
cuda_latest_launched_kernel_id (void)
{
  return next_kernel_id - 1;
}

/* forward declaration */
static void
kernels_add_parent_kernel (uint32_t dev_id, uint64_t grid_id, uint64_t *parent_grid_id);

/******************************************************************************
 *
 *                                   Kernel
 *
 *****************************************************************************/

struct kernel_st {
  bool              grid_status_p;
  uint64_t          id;              /* unique kernel id per GDB session */
  uint32_t          dev_id;          /* device where the kernel was launched */
  uint64_t          grid_id;         /* unique kernel id per device */
  CUDBGGridStatus   grid_status;     /* current grid status of the kernel */
  kernel_t          parent;          /* the kernel that launched this grid */
  kernel_t          children;        /* list of children */
  kernel_t          siblings;        /* next sibling when traversing the list of children */
  char             *name;            /* name of the kernel if available */
  char             *args;            /* kernel arguments in string format */
  uint64_t          virt_code_base;  /* virtual address of the kernel entry point */
  module_t          module;          /* CUmodule handle of the kernel */
  bool              launched;        /* Has the kernel been seen on the hw? */
  CuDim3            grid_dim;        /* The grid dimensions of the kernel. */
  CuDim3            block_dim;       /* The block dimensions of the kernel. */
  char              dimensions[128]; /* A string repr. of the kernel dimensions. */
  CUDBGKernelType   type;            /* The kernel type: system or application. */
  CUDBGKernelOrigin origin;          /* The kernel origin: CPU or GPU */
  disasm_cache_t    disasm_cache;    /* the cached disassembled instructions */
  kernel_t          next;            /* next kernel on the same device */
  unsigned int      depth;           /* kernel nest level (0 - host launched kernel) */
};

static void
kernel_add_child (kernel_t parent, kernel_t child)
{
  gdb_assert (child);

  if (!parent)
    return;

  child->siblings = parent->children;
  parent->children = child;
}

static void
kernel_remove_child (kernel_t parent, kernel_t child)
{
  kernel_t cur, prev;

  gdb_assert (child);

  if (!parent)
    return;

  if (parent->children == child)
    {
      parent->children = child->siblings;
      return;
    }

  for (prev = parent->children, cur = parent->children->siblings;
       cur != NULL;
       prev = cur, cur = cur->siblings)
    if (cur == child)
      {
        prev->siblings = cur->siblings;
        break;
      }
}

static bool
should_print_kernel_event (kernel_t kernel)
{
  unsigned int depth_or_disabled = cuda_options_show_kernel_events_depth ();

  if (depth_or_disabled && kernel->depth > depth_or_disabled - 1)
    return false;

  return (kernel->type == CUDBG_KNL_TYPE_SYSTEM && cuda_options_show_kernel_events_system ()) ||
         (kernel->type == CUDBG_KNL_TYPE_APPLICATION && cuda_options_show_kernel_events_application ());
}

static kernel_t
kernel_new (uint32_t dev_id, uint64_t grid_id, uint64_t virt_code_base,
            const char *name, module_t module, CuDim3 grid_dim, CuDim3 block_dim,
            CUDBGKernelType type, uint64_t parent_grid_id,
            CUDBGKernelOrigin origin)
{
  kernel_t   kernel;
  uint32_t   name_len;
  char      *name_copy;
  kernel_t   parent_kernel;

  parent_kernel = kernels_find_kernel_by_grid_id (dev_id, parent_grid_id);
  if (!parent_kernel && origin == CUDBG_KNL_ORIGIN_GPU)
    {
      kernels_add_parent_kernel (dev_id, grid_id, &parent_grid_id);
      parent_kernel = kernels_find_kernel_by_grid_id (dev_id, parent_grid_id);
    }

  if (name)
    {
      name_len  = strlen (name);
      name_copy = (char *) xmalloc (name_len + 1);
      memcpy (name_copy, name, name_len + 1);
    }
  else
    name_copy = NULL;

  kernel = (kernel_t) xmalloc (sizeof *kernel);

  kernel->grid_status_p            = false;

  kernel->id                       = next_kernel_id++;
  kernel->dev_id                   = dev_id;
  kernel->grid_id                  = grid_id;
  kernel->parent                   = parent_kernel;
  kernel->children                 = NULL;
  kernel->siblings                 = NULL;
  kernel->virt_code_base           = virt_code_base;
  kernel->name                     = name_copy;
  kernel->args                     = NULL;
  kernel->module                   = module;
  kernel->grid_dim                 = grid_dim;
  kernel->block_dim                = block_dim;
  kernel->type                     = type;
  kernel->origin                   = origin;
  kernel->disasm_cache             = disasm_cache_create ();
  kernel->next                     = NULL;
  kernel->depth                    = !parent_kernel ? 0 : parent_kernel->depth + 1;

  snprintf (kernel->dimensions, sizeof (kernel->dimensions), "<<<(%d,%d,%d),(%d,%d,%d)>>>",
            grid_dim.x, grid_dim.y, grid_dim.z, block_dim.x, block_dim.y, block_dim.z);

  kernel->launched = false;

  kernel_add_child (parent_kernel, kernel);

  if (should_print_kernel_event(kernel))
    printf_unfiltered (_("[Launch of CUDA Kernel %llu (%s%s) on Device %u, level %u]\n"),
                       (unsigned long long)kernel->id, kernel->name, kernel->dimensions,
                       kernel->dev_id, kernel->depth);

  return kernel;
}

static void
kernel_delete (kernel_t kernel)
{
  gdb_assert (kernel);

  kernel_remove_child (kernel->parent, kernel);

  if (should_print_kernel_event(kernel))
    printf_unfiltered (_("[Termination of CUDA Kernel %llu (%s%s) on Device %u, level %u]\n"),
                       (unsigned long long)kernel->id, kernel->name, kernel->dimensions,
                       kernel->dev_id, kernel->depth);

  disasm_cache_destroy (kernel->disasm_cache);
  xfree (kernel->name);
  xfree (kernel->args);
  xfree (kernel);
}

void
kernel_invalidate (kernel_t kernel)
{
  cuda_trace ("kernel %llu: invalidate", (unsigned long long)kernel->id);

  kernel->grid_status_p = false;
}

uint64_t
kernel_get_id (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->id;
}

const char *
kernel_get_name (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->name ? kernel->name : "??";
}

static void
kernel_populate_args (kernel_t kernel)
{
  cuda_coords_t *coords, requested, candidates[CK_MAX];
  struct frame_info *prev_frame, *frame;
  cuda_focus_t focus;

  cuda_focus_init (&focus);

  string_file stream;

  /* Find an active lane for the kernel */
  requested = CUDA_WILDCARD_COORDS;
  requested.kernelId = kernel_get_id (kernel);
  cuda_coords_find_valid (requested, candidates, CUDA_SELECT_VALID);
  coords = &candidates[CK_EXACT_LOGICAL];
  if (!coords->valid || !cuda_coords_equal (&requested, coords)) {
    return;
  }

  /* Save environment */
  cuda_focus_save (&focus);
  current_uiout->redirect (&stream);

  TRY
    {
      /* Switch focus to that lane/kernel, temporarily */
      switch_to_cuda_thread (coords);

      /* Find the outermost frame */
      frame = get_current_frame ();
      while ((prev_frame = get_prev_frame (frame)))
        frame = prev_frame;

      /* Print the arguments */
      print_args_frame (frame);
      kernel->args = xstrdup (stream.string ().c_str ());
    }
  CATCH (except, RETURN_MASK_ERROR)
    {
      kernel->args = NULL;
    }
  END_CATCH

  /* Restore environment */
  current_uiout->redirect (NULL);
  cuda_focus_restore (&focus);
}

const char *
kernel_get_args (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->args;
}

uint64_t
kernel_get_grid_id (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->grid_id;
}

kernel_t
kernel_get_parent (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->parent;
}

kernel_t
kernel_get_children (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->children;
}

kernel_t
kernel_get_sibling (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->siblings;
}

uint64_t
kernel_get_virt_code_base (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->virt_code_base;
}

context_t
kernel_get_context (kernel_t kernel)
{
  gdb_assert (kernel);
  return module_get_context (kernel->module);
}

module_t
kernel_get_module (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->module;
}

uint32_t
kernel_get_dev_id (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->dev_id;
}

CuDim3
kernel_get_grid_dim (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->grid_dim;
}

CuDim3
kernel_get_block_dim (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->block_dim;
}

const char*
kernel_get_dimensions (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->dimensions;
}

CUDBGKernelType
kernel_get_type (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->type;
}

CUDBGGridStatus
kernel_get_status (kernel_t kernel)
{
  gdb_assert (kernel);

  if (!kernel->grid_status_p)
    {
      cuda_api_get_grid_status (kernel->dev_id, kernel->grid_id, &kernel->grid_status);
      kernel->grid_status_p = CACHED;
    }

  return kernel->grid_status;
}

CUDBGKernelOrigin
kernel_get_origin (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->origin;
}

uint32_t
kernel_get_depth (kernel_t kernel)
{
  kernel_t k;
  uint32_t depth = -1;

  gdb_assert (kernel);

  for (k = kernel; k; k = kernel_get_parent (k))
    ++depth;

  return depth;
}

uint32_t
kernel_get_num_children (kernel_t kernel)
{
  kernel_t k;
  uint32_t num_children = 0;

  gdb_assert (kernel);

  for (k = kernel_get_children (kernel); k; k = kernel_get_sibling (k))
    ++num_children;

  return num_children;
}

bool
kernel_has_launched (kernel_t kernel)
{
  gdb_assert (kernel);
  return kernel->launched;
}

bool
kernel_is_present (kernel_t kernel)
{
  CUDBGGridStatus status;
  bool present;

  gdb_assert (kernel);

  status = kernel_get_status (kernel);
  present = (status == CUDBG_GRID_STATUS_ACTIVE ||
             status == CUDBG_GRID_STATUS_SLEEPING);

  return present;
}

uint32_t
kernel_compute_sms_mask (kernel_t kernel)
{
  cuda_coords_t current;
  cuda_coords_t filter;
  cuda_iterator itr;
  uint32_t      sms_mask;

  gdb_assert (kernel);

  filter        = CUDA_WILDCARD_COORDS;
  filter.dev    = kernel->dev_id;
  filter.gridId = kernel->grid_id;
  itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_WARPS, &filter, CUDA_SELECT_VALID);

  sms_mask = 0U;
  for (cuda_iterator_start (itr); !cuda_iterator_end (itr); cuda_iterator_next (itr))
    {
      current = cuda_iterator_get_current (itr);
      sms_mask |= 1U << current.sm;
    }

  return sms_mask;
}

const char*
kernel_disassemble (kernel_t kernel, uint64_t pc, uint32_t *inst_size)
{
  gdb_assert (kernel);
  gdb_assert (inst_size);

  return disasm_cache_find_instruction (kernel->disasm_cache, pc, inst_size);
}

void
kernel_flush_disasm_cache (kernel_t kernel)
{
  gdb_assert (kernel);

  disasm_cache_flush (kernel->disasm_cache);
}

void
kernel_print (kernel_t kernel)
{
  gdb_assert (kernel);

  fprintf (stderr, "    Kernel %llu:\n", (unsigned long long)kernel->id);
  fprintf (stderr, "        name        : %s\n", kernel->name);
  fprintf (stderr, "        device id   : %u\n", kernel->dev_id);
  fprintf (stderr, "        grid id     : %lld\n", (long long)kernel->grid_id);
  fprintf (stderr, "        module id   : 0x%llx\n", (unsigned long long)module_get_id (kernel->module));
  fprintf (stderr, "        entry point : 0x%llx\n", (unsigned long long)kernel->virt_code_base);
  fprintf (stderr, "        dimensions  : %s\n", kernel->dimensions);
  fprintf (stderr, "        launched    : %s\n", kernel->launched ? "yes" : "no");
  fprintf (stderr, "        present     : %s\n", kernel_is_present (kernel)? "yes" : "no");
  fprintf (stderr, "        next        : 0x%llx\n", (unsigned long long)(uintptr_t)kernel->next);
  fflush (stderr);
}


/******************************************************************************
 *
 *                                   Kernels
 *
 *****************************************************************************/

/* head of the system list of kernels */
static kernel_t kernels = NULL;

void
kernels_print (void)
{
  kernel_t kernel;

  for (kernel = kernels; kernel; kernel = kernels_get_next_kernel (kernel))
    kernel_print (kernel);
}

void
kernels_start_kernel (uint32_t dev_id, uint64_t grid_id,
                      uint64_t virt_code_base, uint64_t context_id,
                      uint64_t module_id, CuDim3 grid_dim,
                      CuDim3 block_dim, CUDBGKernelType type,
                      uint64_t parent_grid_id, CUDBGKernelOrigin origin)
{
  context_t context;
  modules_t modules;
  module_t  module;
  kernel_t  kernel;
  const char     *kernel_name = NULL;

  context = device_find_context_by_id (dev_id, context_id);
  modules = context_get_modules (context);
  module = modules_find_module_by_id (modules, module_id);

  if (context)
    set_current_context (context);

  kernel_name = cuda_find_function_name_from_pc (virt_code_base, true);

  kernel = kernel_new (dev_id, grid_id, virt_code_base, kernel_name, module,
                       grid_dim, block_dim, type, parent_grid_id, origin);


  kernel->next = kernels;
  kernels = kernel;
}

static void
kernels_add_parent_kernel (uint32_t dev_id, uint64_t grid_id, uint64_t *parent_grid_id)
{
  CUDBGGridInfo grid_info;
  CUDBGGridInfo parent_grid_info;
  CUDBGGridStatus grid_status;

  cuda_api_get_grid_status (dev_id, grid_id, &grid_status);
  if (grid_status == CUDBG_GRID_STATUS_INVALID) return;

  cuda_api_get_grid_info (dev_id, grid_id, &grid_info);

  cuda_api_get_grid_status (dev_id, grid_info.parentGridId, &grid_status);
  if (grid_status == CUDBG_GRID_STATUS_INVALID) return;

  cuda_api_get_grid_info (dev_id, grid_info.parentGridId, &parent_grid_info);
  *parent_grid_id = parent_grid_info.gridId64;
  kernels_start_kernel (parent_grid_info.dev, parent_grid_info.gridId64,
                        parent_grid_info.functionEntry,
                        parent_grid_info.context, parent_grid_info.module,
                        parent_grid_info.gridDim, parent_grid_info.blockDim,
                        parent_grid_info.type, parent_grid_info.parentGridId,
                        parent_grid_info.origin);
}

void
kernels_terminate_kernel (kernel_t kernel)
{
  kernel_t  prev, ker;

  if (!kernel)
    return;

  // must keep kernel object until all the children have terminated
  if (kernel->children)
    return;

  for (ker = kernels, prev = NULL;
       ker && ker != kernel;
       prev = ker, ker = kernels_get_next_kernel (ker))
    ;
  gdb_assert (ker);

  if (prev)
    prev->next = kernels_get_next_kernel (kernel);
  else
    kernels = kernels_get_next_kernel (kernel);

  kernel_delete (kernel);
}

void
kernels_terminate_module (module_t module)
{
  kernel_t kernel, next_kernel;

  gdb_assert (module);

  kernel = kernels_get_first_kernel ();
  while (kernel)
    {
      next_kernel = kernels_get_next_kernel (kernel);
      if (kernel_get_module (kernel) == module)
        kernels_terminate_kernel (kernel);
      kernel = next_kernel;
    }
}

kernel_t
kernels_get_first_kernel (void)
{
  return kernels;
}

kernel_t
kernels_get_next_kernel (kernel_t kernel)
{
  if (!kernel)
    return NULL;

  return kernel->next;
}

kernel_t
kernels_find_kernel_by_grid_id (uint32_t dev_id, uint64_t grid_id)
{
  kernel_t kernel;

  for (kernel = kernels; kernel; kernel = kernels_get_next_kernel (kernel))
    if (kernel->dev_id == dev_id && kernel->grid_id == grid_id)
      return kernel;

  return NULL;
}

kernel_t
kernels_find_kernel_by_kernel_id (uint64_t kernel_id)
{
  kernel_t kernel;

  for (kernel = kernels; kernel; kernel = kernels_get_next_kernel (kernel))
    if (kernel->id == kernel_id)
      return kernel;

  return NULL;
}

void
kernels_update_args (void)
{
  kernel_t kernel;

  for (kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    if (!kernel->args && kernel_is_present (kernel))
      kernel_populate_args (kernel);
}

void
kernels_update_terminated (void)
{
  kernel_t      kernel;
  kernel_t      next_kernel;

  /* rediscover the kernels currently running on the hardware */
  kernel = kernels_get_first_kernel ();
  while (kernel)
    {
      next_kernel = kernels_get_next_kernel (kernel);

      if (kernel_is_present (kernel))
        kernel->launched = true;

      /* terminate the kernels that we had seen running at some point
         but are not here on the hardware anymore. If there is any child kernel
         still present, keep the data available. */
      if (kernel->launched && !kernel_is_present (kernel))
        kernels_terminate_kernel (kernel);

      kernel = next_kernel;
    }
}
