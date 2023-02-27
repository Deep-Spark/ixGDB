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

#include "defs.h"
#include "breakpoint.h"
#include "remote.h"
#include "common/common-defs.h"

#include "cuda-context.h"
#include "cuda-defs.h"
#include "cuda-iterator.h"
#include "cuda-state.h"
#include "cuda-utils.h"
#include "cuda-packet-manager.h"
#include "cuda-options.h"
#include "cuda-elf-image.h"

#ifdef __ANDROID__
#undef CUDBG_MAX_DEVICES
#define CUDBG_MAX_DEVICES 4
#endif /*__ANDROID__*/

typedef struct {
  bool thread_idx_p;
  bool pc_p;
  bool exception_p;
  bool virtual_pc_p;
  bool timestamp_p;
  CuDim3           thread_idx;
  uint64_t         pc;
  CUDBGException_t exception;
  uint64_t         virtual_pc;
  cuda_clock_t     timestamp;
} lane_state_t;

typedef struct {
  bool valid_p;
  bool broken_p;
  bool block_idx_p;
  bool kernel_p;
  bool grid_id_p;
  bool valid_lanes_mask_p;
  bool active_lanes_mask_p;
  bool timestamp_p;
  bool error_pc_p;
  bool     valid;
  bool     broken;
  bool     error_pc_available;
  CuDim3   block_idx;
  kernel_t kernel;
  uint64_t grid_id;
  uint64_t error_pc;
  uint64_t valid_lanes_mask;
  uint64_t active_lanes_mask;
  cuda_clock_t     timestamp;
  lane_state_t ln[CUDBG_MAX_LANES];
} warp_state_t;

typedef struct {
  bool valid_warps_mask_p;
  bool broken_warps_mask_p;
  cuda_api_warpmask valid_warps_mask;
  cuda_api_warpmask broken_warps_mask;
  warp_state_t wp[CUDBG_MAX_WARPS];
} sm_state_t;

typedef struct {
  bool valid_p;
  bool num_sms_p;
  bool num_warps_p;
  bool num_lanes_p;
  bool num_registers_p;
  bool num_predicates_p;
  bool num_uregisters_p;
  bool num_upredicates_p;
  bool pci_bus_info_p;
  bool dev_type_p;
  bool sm_type_p;
  bool inst_size_p;
  bool dev_name_p;
  bool sm_exception_mask_valid_p;
  bool valid;             // at least one active lane
  /* the above fields are invalidated on resume */
  bool suspended;         // true if the device is suspended
  char dev_type[256];
  char dev_name[256];
  char sm_type[16];
  uint32_t inst_size;
  uint32_t num_sms;
  uint32_t num_warps;
  uint32_t num_lanes;
  uint32_t num_registers;
  uint32_t num_predicates;
  uint32_t num_uregisters;
  uint32_t num_upredicates;
  uint32_t pci_dev_id;
  uint32_t pci_bus_id;
  uint64_t sm_exception_mask[(CUDBG_MAX_SMS + 63) / 64];    // Mask needs to be large enough to hold all the SMs, rounded up
  sm_state_t sm[CUDBG_MAX_SMS];
  contexts_t contexts;    // state for contexts associated with this device
} device_state_t;

typedef struct {
  bool num_devices_p;
  uint32_t num_devices;
  device_state_t *dev[CUDBG_MAX_DEVICES];
  uint32_t suspended_devices_mask;
} cuda_system_t;


/* GPU register cache */
#define CUDBG_CACHED_REGISTERS_COUNT 256
#define CUDBG_CACHED_PREDICATES_COUNT 8

typedef struct {
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t ln;
  uint32_t registers[CUDBG_CACHED_REGISTERS_COUNT];
  uint32_t register_valid_mask[CUDBG_CACHED_REGISTERS_COUNT>>5];
  uint32_t predicates[CUDBG_CACHED_PREDICATES_COUNT];
  bool     predicates_valid_p;
  uint32_t cc_register;
  bool     cc_register_valid_p;
} cuda_reg_cache_element_t;
DEF_VEC_O(cuda_reg_cache_element_t);
static VEC(cuda_reg_cache_element_t) *cuda_register_cache = NULL;

/* GPU uniform register cache */
#define CUDBG_CACHED_UREGISTERS_COUNT 64
#define CUDBG_CACHED_UPREDICATES_COUNT 8

typedef struct {
  uint32_t dev;
  uint32_t sm;
  uint32_t wp;
  uint32_t registers[CUDBG_CACHED_UREGISTERS_COUNT];
  uint32_t register_valid_mask[CUDBG_CACHED_UREGISTERS_COUNT>>5];
  uint32_t predicates[CUDBG_CACHED_UPREDICATES_COUNT];
  bool     predicates_valid_p;
} cuda_ureg_cache_element_t;
DEF_VEC_O(cuda_ureg_cache_element_t);
static VEC(cuda_ureg_cache_element_t) *cuda_uregister_cache = NULL;

const bool CACHED = true; // set to false to disable caching
typedef enum { RECURSIVE, NON_RECURSIVE } recursion_t;

static void device_initialize             (uint32_t dev_id);
static void device_cleanup_contexts       (uint32_t dev_id);
static void device_flush_disasm_cache     (uint32_t dev_id);
static void device_update_exception_state (uint32_t dev_id);
static void sm_invalidate                 (uint32_t dev_id, uint32_t sm_id, recursion_t);
static void sm_set_exception_none         (uint32_t dev_id, uint32_t sm_id);
static void warp_invalidate               (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id);
static void lane_invalidate               (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);
static void lane_set_exception_none       (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id);


/******************************************************************************
 *
 *                                  System
 *
 ******************************************************************************/

static cuda_system_t cuda_system_info;

static void cuda_system_cleanup (void)
{
  uint32_t dev_id;

  cuda_system_info.num_devices_p = 0;
  cuda_system_info.num_devices = 0;
  cuda_system_info.suspended_devices_mask = 0;
  for (dev_id = 0; dev_id < CUDBG_MAX_DEVICES; ++dev_id)
    if (cuda_system_info.dev[dev_id])
      memset (cuda_system_info.dev[dev_id], 0, sizeof(device_state_t));
}

void
cuda_system_initialize (void)
{
  uint32_t dev_id;

  cuda_trace ("system: initialize");
  gdb_assert (cuda_initialized);

  cuda_system_cleanup ();

  for (dev_id = 0; dev_id < cuda_system_get_num_devices (); ++dev_id)
     device_initialize (dev_id);

  cuda_options_force_set_launch_notification_update ();
}

void
cuda_system_finalize (void)
{

  cuda_trace ("system: finalize");
  gdb_assert (cuda_initialized);

  cuda_system_cleanup ();
}

uint32_t
cuda_system_get_num_devices (void)
{
  if (!cuda_initialized)
    return 0;

  if (cuda_system_info.num_devices_p)
    return cuda_system_info.num_devices;

  cuda_api_get_num_devices (&cuda_system_info.num_devices);
  gdb_assert (cuda_system_info.num_devices <= CUDBG_MAX_DEVICES);
  cuda_system_info.num_devices_p = CACHED;

  return cuda_system_info.num_devices;
}

uint32_t
cuda_system_get_num_present_kernels (void)
{
  kernel_t kernel;
  uint32_t num_present_kernel = 0;

  if (!cuda_initialized)
    return 0;

  for (kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    if (kernel_is_present (kernel))
      ++num_present_kernel;

  return num_present_kernel;
}

void
cuda_system_resolve_breakpoints (int bp_number_from)
{
  elf_image_t elf_image;

  cuda_trace ("system: resolve breakpoints\n");

  CUDA_ALL_LOADED_ELF_IMAGES (elf_image)
    cuda_resolve_breakpoints (bp_number_from, elf_image);
}

void
cuda_system_cleanup_breakpoints (void)
{
  elf_image_t elf_image;

  cuda_trace ("system: clean up breakpoints");

  CUDA_ALL_LOADED_ELF_IMAGES (elf_image)
    cuda_unresolve_breakpoints (elf_image);
}

void
cuda_system_cleanup_contexts (void)
{
  uint32_t dev_id;

  cuda_trace ("system: clean up contexts");

  for (dev_id = 0; dev_id < cuda_system_get_num_devices (); ++dev_id)
    device_cleanup_contexts (dev_id);
}

void
cuda_system_flush_disasm_cache (void)
{
  uint32_t dev_id;

  cuda_trace ("system: flush disassembly cache");

  for (dev_id = 0; dev_id < cuda_system_get_num_devices (); ++dev_id)
    device_flush_disasm_cache (dev_id);
}

bool
cuda_system_is_broken (cuda_clock_t clock)
{
  cuda_iterator itr;
  cuda_coords_t c, filter = CUDA_WILDCARD_COORDS;
  bool broken = false;

  itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_WARPS, &filter,
                               CUDA_SELECT_VALID);

  for (cuda_iterator_start (itr);
       !cuda_iterator_end (itr);
       cuda_iterator_next (itr))
    {
      /* if we hit a breakpoint at an earlier time, we do not report it again. */
      c = cuda_iterator_get_current (itr);
      if (warp_get_timestamp (c.dev, c.sm, c.wp) < clock)
        continue;

      if (!warp_is_broken (c.dev, c.sm, c.wp))
        continue;

      broken = true;
      break;
    }

  cuda_iterator_destroy (itr);

  return broken;
}

uint32_t
cuda_system_get_suspended_devices_mask (void)
{
  return cuda_system_info.suspended_devices_mask;
}

context_t
cuda_system_find_context_by_addr (CORE_ADDR addr)
{
  uint32_t  dev_id;
  context_t context;

  for (dev_id = 0; dev_id < cuda_system_get_num_devices (); ++dev_id)
    {
      context = device_find_context_by_addr (dev_id, addr);
      if (context)
        return context;
    }

  return NULL;
}

/******************************************************************************
 *
 *                                  Device
 *
 ******************************************************************************/

static inline device_state_t *
device_get (uint32_t dev_id)
{
  device_state_t *dev;

  gdb_assert (dev_id < cuda_system_get_num_devices ());

  dev = cuda_system_info.dev[dev_id];
  if (!dev)
    {
      dev = (device_state_t *) xmalloc (sizeof *dev);
      memset (dev, 0, sizeof *dev);
      cuda_system_info.dev[dev_id] = dev;
    }

  gdb_assert (dev);

  return dev;
}

static void
device_initialize (uint32_t dev_id)
{
  device_state_t *dev;

  cuda_trace ("device %u: initialize", dev_id);

  dev = device_get (dev_id);
  dev->contexts = contexts_new ();
}

static void
device_invalidate_kernels (uint32_t dev_id)
{
  kernel_t        kernel;

  cuda_trace ("device %u: invalidate kernels", dev_id);
  gdb_assert (dev_id < cuda_system_get_num_devices ());

  for (kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    kernel_invalidate (kernel);
}

void
device_invalidate (uint32_t dev_id)
{
  device_state_t *dev;
  uint32_t sm_id;

  cuda_trace ("device %u: invalidate", dev_id);
  dev = device_get (dev_id);

  for (sm_id = 0; sm_id < device_get_num_sms (dev_id); ++sm_id)
    sm_invalidate (dev_id, sm_id, RECURSIVE);

  device_invalidate_kernels(dev_id);

  dev->valid_p   = false;
}

static void
device_flush_disasm_cache (uint32_t dev_id)
{
  kernel_t        kernel;

  cuda_trace ("device %u: flush disassembly cache", dev_id);
  gdb_assert (dev_id < cuda_system_get_num_devices ());

  for (kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    kernel_flush_disasm_cache (kernel);
}

static void
device_cleanup_contexts (uint32_t dev_id)
{
  contexts_t      contexts;

  cuda_trace ("device %u: clean up contexts", dev_id);

  contexts = device_get_contexts (dev_id);

  contexts_delete (contexts);

  device_get(dev_id)->contexts = NULL;
}

const char*
device_get_device_type (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->dev_type_p)
    return dev->dev_type;

  cuda_api_get_device_type (dev_id, dev->dev_type, sizeof dev->dev_type);
  dev->dev_type_p = CACHED;
  return dev->dev_type;
}

const char*
device_get_sm_type (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->sm_type_p)
    return dev->sm_type;

  cuda_api_get_sm_type (dev_id, dev->sm_type, sizeof dev->sm_type);
  dev->sm_type_p = CACHED;
  return dev->sm_type;
}

const char*
device_get_device_name (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->dev_name_p)
    return dev->dev_name;

  cuda_api_get_device_name (dev_id, dev->dev_name, sizeof dev->dev_name);
  dev->dev_name_p = CACHED;
  return dev->dev_name;
}

/* This assumes that the GPU architecture has a uniform instruction size,
 * which is true on all GPU architectures except FERMI. Since cuda-gdb no
 * longer supports FERMI as of 9.0 toolkit, this assumption is valid.
 */
uint32_t
device_get_inst_size (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  return dev->inst_size_p ? dev->inst_size : 0;
}

void
device_set_inst_size (uint32_t dev_id, uint32_t inst_size)
{
  device_state_t *dev = device_get (dev_id);

  dev->inst_size = inst_size;
  dev->inst_size_p = true;
}

uint32_t
device_get_pci_bus_id (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->pci_bus_info_p)
    return dev->pci_bus_id;

  cuda_api_get_device_pci_bus_info (dev_id, &dev->pci_bus_id, &dev->pci_dev_id);
  gdb_assert (dev->num_sms <= CUDBG_MAX_SMS);
  dev->pci_bus_info_p = CACHED;

  return dev->pci_bus_id;
}

uint32_t
device_get_pci_dev_id (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->pci_bus_info_p)
    return dev->pci_dev_id;

  cuda_api_get_device_pci_bus_info (dev_id, &dev->pci_bus_id, &dev->pci_dev_id);
  gdb_assert (dev->num_sms <= CUDBG_MAX_SMS);
  dev->pci_bus_info_p = CACHED;

  return dev->pci_dev_id;
}

uint32_t
device_get_num_sms (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->num_sms_p)
    return dev->num_sms;

  cuda_api_get_num_sms (dev_id, &dev->num_sms);
  gdb_assert (dev->num_sms <= CUDBG_MAX_SMS);
  dev->num_sms_p = CACHED;

  return dev->num_sms;
}

uint32_t
device_get_num_warps (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->num_warps_p)
    return dev->num_warps;

  cuda_api_get_num_warps (dev_id, &dev->num_warps);
  gdb_assert (dev->num_warps <= CUDBG_MAX_WARPS);
  dev->num_warps_p = CACHED;

  return dev->num_warps;
}

uint32_t
device_get_num_lanes (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->num_lanes_p)
    return dev->num_lanes;

  cuda_api_get_num_lanes (dev_id, &dev->num_lanes);
  gdb_assert (dev->num_lanes <= CUDBG_MAX_LANES);
  dev->num_lanes_p = CACHED;

  return dev->num_lanes;
}

uint32_t
device_get_num_registers (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->num_registers_p)
    return dev->num_registers;

  cuda_api_get_num_registers (dev_id, &dev->num_registers);
  dev->num_registers_p = CACHED;

  return dev->num_registers;
}

uint32_t
device_get_num_predicates (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->num_predicates_p)
    return dev->num_predicates;

  cuda_api_get_num_predicates (dev_id, &dev->num_predicates);
  dev->num_predicates_p = CACHED;
  gdb_assert (dev->num_predicates <= CUDBG_CACHED_PREDICATES_COUNT);

  return dev->num_predicates;
}

uint32_t
device_get_num_uregisters (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->num_uregisters_p)
    return dev->num_uregisters;

  cuda_api_get_num_uregisters (dev_id, &dev->num_uregisters);
  dev->num_uregisters_p = CACHED;

  return dev->num_uregisters;
}

uint32_t
device_get_num_upredicates (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  if (dev->num_upredicates_p)
    return dev->num_upredicates;

  cuda_api_get_num_upredicates (dev_id, &dev->num_upredicates);
  dev->num_upredicates_p = CACHED;
  gdb_assert (dev->num_upredicates <= CUDBG_CACHED_UPREDICATES_COUNT);

  return dev->num_upredicates;
}

uint32_t
device_get_num_kernels (uint32_t dev_id)
{
  kernel_t kernel;
  uint32_t num_kernels = 0;

  gdb_assert (dev_id < cuda_system_get_num_devices ());

  for (kernel = kernels_get_first_kernel (); kernel; kernel = kernels_get_next_kernel (kernel))
    if (kernel_get_dev_id (kernel) == dev_id)
      ++num_kernels;

  return num_kernels;
}

bool
device_is_any_context_present (uint32_t dev_id)
{
  contexts_t contexts;

  gdb_assert (dev_id < cuda_system_get_num_devices ());

  contexts = device_get_contexts (dev_id);

  return contexts_is_any_context_present (contexts);
}

bool
device_is_active_context (uint32_t dev_id, context_t context)
{
  contexts_t contexts;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  contexts = device_get_contexts (dev_id);

  return contexts_is_active_context (contexts, context);
}

bool
device_is_valid (uint32_t dev_id)
{
  device_state_t *dev;
  uint32_t sm, wp;

  if (!cuda_initialized)
    return false;

  dev = device_get (dev_id);

  if (dev->valid_p)
    return dev->valid;

  dev->valid = false;

  if (!device_is_any_context_present (dev_id))
    return dev->valid;

  for (sm = 0; sm < device_get_num_sms (dev_id) && !dev->valid; ++sm)
    for (wp = 0; wp < device_get_num_warps (dev_id) && !dev->valid; ++wp)
      if (warp_is_valid (dev_id, sm, wp))
          dev->valid = true;

  dev->valid_p = CACHED;
  return dev->valid;
}

bool
device_has_exception (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  device_update_exception_state (dev_id);

  return dev->sm_exception_mask != 0;
}

void
device_get_active_sms_mask (uint32_t dev_id, uint32_t *mask)
{
  uint32_t        sm;
  uint32_t        wp;

  gdb_assert (mask);
  memset(mask, 0, ((CUDBG_MAX_SMS + 31) / 32) * sizeof(*mask));

  for (sm = 0; sm < device_get_num_sms (dev_id); ++sm)
    for (wp = 0; wp < device_get_num_warps (dev_id); ++wp)
      if (warp_is_valid (dev_id, sm, wp))
        {
          mask[sm / 32] |= 1UL << (sm % 32);
          break;
        }
}

contexts_t
device_get_contexts (uint32_t dev_id)
{
  device_state_t *dev = device_get (dev_id);

  gdb_assert (dev->contexts);

  return dev->contexts;
}

context_t
device_find_context_by_id (uint32_t dev_id, uint64_t context_id)
{
  contexts_t      contexts = device_get_contexts (dev_id);

  return contexts_find_context_by_id (contexts, context_id);
}

context_t
device_find_context_by_addr (uint32_t dev_id, CORE_ADDR addr)
{
  contexts_t      contexts = device_get_contexts (dev_id);

  return contexts_find_context_by_address (contexts, addr);
}

void
device_print (uint32_t dev_id)
{
  contexts_t      contexts;

  cuda_trace ("device %u:", dev_id);

  contexts = device_get_contexts (dev_id);

  contexts_print (contexts);
}

void
device_resume (uint32_t dev_id)
{
  device_state_t *dev;

  cuda_trace ("device %u: resume", dev_id);

  device_invalidate (dev_id);

  dev = device_get (dev_id);

  if (!dev->suspended)
    return;

  cuda_api_resume_device (dev_id);

  dev->suspended = false;

  cuda_system_info.suspended_devices_mask &= ~(1 << dev_id);
}

static void
device_create_kernel(uint32_t dev_id, uint64_t grid_id)
{
  CUDBGGridInfo gridInfo = {0};

  cuda_api_get_grid_info(dev_id, grid_id, &gridInfo);
  kernels_start_kernel(dev_id, grid_id,
                       gridInfo.functionEntry,
                       gridInfo.context,
                       gridInfo.module,
                       gridInfo.gridDim,
                       gridInfo.blockDim,
                       gridInfo.type,
                       gridInfo.parentGridId,
                       gridInfo.origin);
}

void
device_suspend (uint32_t dev_id)
{
  device_state_t *dev;

  cuda_trace ("device %u: suspend", dev_id);

  dev = device_get (dev_id);

  cuda_api_suspend_device (dev_id);

  dev->suspended = true;

  cuda_system_info.suspended_devices_mask |= (1 << dev_id);
}

static void
device_update_exception_state (uint32_t dev_id)
{
  device_state_t *dev;
  uint32_t sm_id;
  uint32_t nsms;

  dev = device_get (dev_id);

  if (dev->sm_exception_mask_valid_p)
    return;

  memset(&dev->sm_exception_mask, 0, sizeof(dev->sm_exception_mask));
  nsms = device_get_num_sms (dev_id);

  if (device_is_any_context_present (dev_id))
    cuda_api_read_device_exception_state (dev_id, dev->sm_exception_mask, (nsms+63) / 64);

  for (sm_id = 0; sm_id < nsms; ++sm_id)
    if (!((dev->sm_exception_mask[sm_id / 64] >> (sm_id % 64)) & 1))
      sm_set_exception_none (dev_id, sm_id);

  dev->sm_exception_mask_valid_p = true;
}

void
cuda_system_set_device_spec (uint32_t dev_id, uint32_t num_sms,
                             uint32_t num_warps, uint32_t num_lanes,
                             uint32_t num_registers, char *dev_type,
                             char *sm_type)
{
  device_state_t *dev = device_get (dev_id);

  gdb_assert (cuda_remote);
  gdb_assert (num_sms <= CUDBG_MAX_SMS);
  gdb_assert (num_warps <= CUDBG_MAX_WARPS);
  gdb_assert (num_lanes <= CUDBG_MAX_LANES);

  dev->num_sms         = num_sms;
  dev->num_warps       = num_warps;
  dev->num_lanes       = num_lanes;
  dev->num_registers   = num_registers;
  strcpy (dev->dev_type, dev_type);
  strcpy (dev->sm_type, sm_type);

  dev->num_sms_p        = CACHED;
  dev->num_warps_p      = CACHED;
  dev->num_lanes_p      = CACHED;
  dev->num_registers_p  = CACHED;
  dev->dev_type_p       = CACHED;
  dev->dev_name_p       = CACHED;
  dev->sm_type_p        = CACHED;
  dev->num_predicates_p = false;
}


/******************************************************************************
 *
 *                                    SM
 *
 ******************************************************************************/

static inline sm_state_t *
sm_get (uint32_t dev_id, uint32_t sm_id)
{
  gdb_assert (sm_id < device_get_num_sms (dev_id));

  return &device_get(dev_id)->sm[sm_id];
}

static void
sm_invalidate (uint32_t dev_id, uint32_t sm_id, recursion_t recursion)
{
  device_state_t *dev = device_get (dev_id);
  sm_state_t *sm = sm_get (dev_id, sm_id);
  uint32_t wp_id;

  if (recursion == RECURSIVE)
    for (wp_id = 0; wp_id < device_get_num_warps (dev_id); ++wp_id)
      warp_invalidate (dev_id, sm_id, wp_id);

  dev->sm_exception_mask_valid_p = false;

  sm->valid_warps_mask_p  = false;
  sm->broken_warps_mask_p = false;
}

bool
sm_is_valid (uint32_t dev_id, uint32_t sm_id)
{
  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));

  return sm_get_valid_warps_mask (dev_id, sm_id);
}

bool
sm_has_exception (uint32_t dev_id, uint32_t sm_id)
{
  device_state_t *dev = device_get (dev_id);

  gdb_assert (sm_id < device_get_num_sms (dev_id));

  device_update_exception_state (dev_id);

  return (dev->sm_exception_mask[sm_id / 64] >> (sm_id % 64)) & 1ULL;
}

cuda_api_warpmask*
sm_get_valid_warps_mask (uint32_t dev_id, uint32_t sm_id)
{
  sm_state_t *sm = sm_get (dev_id, sm_id);

  if (!sm->valid_warps_mask_p) {
      cuda_api_read_valid_warps (dev_id, sm_id, &sm->valid_warps_mask);
      sm->valid_warps_mask_p = CACHED;
  }

  return &sm->valid_warps_mask;
}

cuda_api_warpmask*
sm_get_broken_warps_mask (uint32_t dev_id, uint32_t sm_id)
{
  sm_state_t *sm = sm_get (dev_id, sm_id);

  if (!sm->broken_warps_mask_p) {
      cuda_api_read_broken_warps (dev_id, sm_id, &sm->broken_warps_mask);
      sm->broken_warps_mask_p = CACHED;
  }

  return &sm->broken_warps_mask;
}

static void
sm_set_exception_none (uint32_t dev_id, uint32_t sm_id)
{
  uint32_t wp_id;
  uint32_t ln_id;

  for (wp_id = 0; wp_id < device_get_num_warps (dev_id); ++wp_id)
    for (ln_id = 0; ln_id < device_get_num_lanes (dev_id); ++ln_id)
      lane_set_exception_none (dev_id, sm_id, wp_id, ln_id);
}

/******************************************************************************
 *
 *                                   Warps
 *
 ******************************************************************************/

/* Warps register cache */
static cuda_ureg_cache_element_t *
cuda_ureg_cache_find_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  int idx;
  cuda_ureg_cache_element_t *elem;
  cuda_ureg_cache_element_t new_elem;

  for (idx = 0;
       VEC_iterate (cuda_ureg_cache_element_t, cuda_uregister_cache, idx, elem);
       ++idx)
    if (elem->dev == dev_id && elem->sm == sm_id && elem->wp == wp_id)
      return elem;

  memset (&new_elem, 0, sizeof(new_elem));
  new_elem.dev = dev_id;
  new_elem.sm = sm_id;
  new_elem.wp = wp_id;

  return VEC_safe_push (cuda_ureg_cache_element_t, cuda_uregister_cache, &new_elem);
}

static void
cuda_ureg_cache_remove_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  int idx;
  cuda_ureg_cache_element_t *elem;

  for (idx = 0;
       VEC_iterate (cuda_ureg_cache_element_t, cuda_uregister_cache, idx, elem);
       ++idx)
    {
       if (elem->dev != dev_id || elem->sm != sm_id || elem->wp != wp_id)
         continue;
       VEC_unordered_remove(cuda_ureg_cache_element_t, cuda_uregister_cache, idx);
       break;
    }
}

uint32_t
warp_get_uregister (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno)
{
  cuda_ureg_cache_element_t *elem;

  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);
  if ( (elem->register_valid_mask[regno>>5]&(1UL<<(regno&31))) != 0)
    return elem->registers[regno];

  cuda_api_read_uregister_range (dev_id, sm_id, wp_id,
				 0, CUDBG_CACHED_UREGISTERS_COUNT, elem->registers);
  elem->register_valid_mask[0] = 0xffffffff;
  elem->register_valid_mask[1] = 0xffffffff;

  return elem->registers[regno];
}

void
warp_set_uregister (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t regno, uint32_t value)
{
  cuda_ureg_cache_element_t *elem;

  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  cuda_api_write_uregister (dev_id, sm_id, wp_id, regno, value);

  /* If register can not be cached - return */
  if (regno > CUDBG_CACHED_UREGISTERS_COUNT)
      return;

  elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);
  elem->registers[regno] = value;
  elem->register_valid_mask[regno>>5] |= 1UL << (regno & 31);
}

bool
warp_get_upredicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t predicate)
{
  cuda_ureg_cache_element_t *elem;

  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  /* NVIDIA 470 driver with "CUDA 11.3 Update 1" release fixes UP7 predicate
     which should have always been reported as true. This is a temporary change
     and will be removed for next major CUDA release. */
  if (predicate == 7)
    return true;

  elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);

  if (elem->predicates_valid_p)
    return elem->predicates[predicate] != 0;

  cuda_api_read_upredicates (dev_id, sm_id, wp_id,
			     device_get_num_upredicates (dev_id),
			     elem->predicates);
  elem->predicates_valid_p = CACHED;

  return elem->predicates[predicate] != 0;
}

void
warp_set_upredicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t predicate, bool value)
{
  cuda_ureg_cache_element_t *elem;

  gdb_assert (predicate < device_get_num_upredicates (dev_id));
  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));
  elem = cuda_ureg_cache_find_element (dev_id, sm_id, wp_id);

  if (!elem->predicates_valid_p)
    {
      cuda_api_read_upredicates (dev_id, sm_id, wp_id,
                                device_get_num_upredicates (dev_id),
                                elem->predicates);
      elem->predicates_valid_p = CACHED;
    }

  elem->predicates[predicate] = value;

  cuda_api_write_upredicates (dev_id, sm_id, wp_id,
			      device_get_num_upredicates (dev_id),
			      elem->predicates);
}

static inline warp_state_t *
warp_get (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  return &sm_get(dev_id, sm_id)->wp[wp_id];
}

static void
warp_invalidate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  sm_state_t   *sm = sm_get (dev_id, sm_id);
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);
  uint32_t      ln_id;

  cuda_ureg_cache_remove_element (dev_id, sm_id, wp_id);

  for (ln_id = 0; ln_id < device_get_num_lanes (dev_id); ++ln_id)
    lane_invalidate (dev_id, sm_id, wp_id, ln_id);

  // XXX decouple the masks from the SM state data structure to avoid this
  // little hack.
  /* If a warp is invalidated, we have to invalidate the warp masks in the
     corresponding SM. */
  sm->valid_warps_mask_p  = false;
  sm->broken_warps_mask_p = false;

  wp->valid_p             = false;
  wp->broken_p            = false;
  wp->block_idx_p         = false;
  wp->kernel_p            = false;
  wp->grid_id_p           = false;
  wp->valid_lanes_mask_p  = false;
  wp->active_lanes_mask_p = false;
  wp->timestamp_p         = false;
}

bool
warps_resume_until (uint32_t dev_id, uint32_t sm_id, cuda_api_warpmask* mask, uint64_t pc)
{
  uint32_t i;

  /* No point in resuming warps, if one them is already there */
  for (i = 0; i < device_get_num_warps (dev_id); ++i)
    if (cuda_api_get_bit(mask, i))
      if (pc == warp_get_active_virtual_pc (dev_id, sm_id, i))
        return false;

  /* If resume warps is not possible - abort */
  if (!cuda_api_resume_warps_until_pc (dev_id, sm_id, mask, pc))
    return false;

  if (cuda_options_software_preemption ())
    {
      device_invalidate (dev_id);
      return true;
    }
  /* invalidate the cache for the warps that have been single-stepped. */
  for (i = 0; i < device_get_num_warps (dev_id); ++i)
    if (cuda_api_get_bit(mask, i))
          warp_invalidate (dev_id, sm_id, i);

  /* must invalidate the SM since that's where the warp valid mask lives */
  sm_invalidate (dev_id, sm_id, NON_RECURSIVE);

  return true;
}

bool
warp_single_step (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
                  uint32_t nsteps, cuda_api_warpmask *single_stepped_warp_mask)
{
  uint32_t i;
  bool rc;
  cuda_api_warpmask tmp;

  cuda_trace ("device %u sm %u warp %u nsteps %u: single-step", dev_id, sm_id, wp_id, nsteps);

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  cuda_api_clear_mask(single_stepped_warp_mask);
  cuda_api_clear_mask(&tmp);
  cuda_api_set_bit(&tmp, wp_id, 1);
  cuda_api_not_mask(&tmp, &tmp);    // Select all but the single-stepped warp in the mask

  rc = cuda_api_single_step_warp (dev_id, sm_id, wp_id, nsteps, single_stepped_warp_mask);
  if (!rc)
    return rc;

  if (cuda_options_software_preemption ())
    {
      device_invalidate (dev_id);
      return true;
    }

  cuda_api_and_mask(&tmp, &tmp, single_stepped_warp_mask);

  if (cuda_api_has_bit(&tmp))
    {
      warning ("Warp(s) other than the current warp had to be single-stepped:%" WARP_MASK_FORMAT,
          cuda_api_mask_string(single_stepped_warp_mask));
      device_invalidate (dev_id);
    }

  /* invalidate the cache for the warps that have been single-stepped. */
  for (i = 0; i < device_get_num_warps (dev_id); ++i)
    if (cuda_api_get_bit(single_stepped_warp_mask, i))
      warp_invalidate (dev_id, sm_id, i);

  /* must invalidate the SM since that's where the warp valid mask lives */
  sm_invalidate (dev_id, sm_id, NON_RECURSIVE);

  return true;
}

bool
warp_is_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  gdb_assert (wp_id < device_get_num_warps (dev_id));
  return cuda_api_get_bit(sm_get_valid_warps_mask (dev_id, sm_id), wp_id) != 0;
}

bool
warp_is_broken (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  gdb_assert (wp_id < device_get_num_warps (dev_id));
  return cuda_api_get_bit(sm_get_broken_warps_mask (dev_id, sm_id), wp_id) != 0;
}

bool
warp_has_error_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);
  bool error_pc_available = false;
  uint64_t error_pc = 0ULL;

  //if (wp->error_pc_p)
  //  return wp->error_pc_available;

  cuda_api_read_error_pc (dev_id, sm_id, wp_id, &error_pc, &error_pc_available);

  wp->error_pc = error_pc;
  wp->error_pc_available = error_pc_available;
  wp->error_pc_p = CACHED;

  return wp->error_pc_available;
}

static void
update_warp_cached_info (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  lane_state_t *ln;
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);
  CUDBGWarpState state;
  uint32_t ln_id;

  cuda_api_read_warp_state (dev_id, sm_id, wp_id, &state);

  wp->error_pc = state.errorPC;
  wp->error_pc_available = state.errorPCValid;
  wp->error_pc_p = CACHED;

  wp->block_idx = state.blockIdx;
  wp->block_idx_p = CACHED;

  wp->grid_id = state.gridId;
  wp->grid_id_p = CACHED;

  wp->active_lanes_mask   = state.activeLanes;
  wp->active_lanes_mask_p = CACHED;

  wp->valid_lanes_mask   = state.validLanes;
  wp->valid_lanes_mask_p = CACHED;

  for (ln_id = 0; ln_id < device_get_num_lanes (dev_id); ln_id++) {
    ln = &wp->ln[ln_id];
    if ( !(state.validLanes & (1U<<ln_id)) ) continue;
    ln->thread_idx = state.lane[ln_id].threadIdx;
    ln->virtual_pc = state.lane[ln_id].virtualPC;
    ln->exception = state.lane[ln_id].exception;
    ln->exception_p = ln->thread_idx_p = ln->virtual_pc_p = CACHED;

    if (!ln->timestamp_p)
      {
        ln->timestamp_p = true;
        ln->timestamp = cuda_clock ();
      }
  }
  if (!wp->timestamp_p)
    {
      wp->timestamp_p = true;
      wp->timestamp = cuda_clock ();
    }
}

uint64_t
warp_get_grid_id (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);

  if (cuda_remote && !(wp->grid_id_p)
      && sm_is_valid (dev_id, sm_id))
    cuda_remote_update_grid_id_in_sm (get_current_remote_target (), dev_id, sm_id);

  if (wp->grid_id_p)
    return wp->grid_id;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return wp->grid_id;
}

kernel_t
warp_get_kernel (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);
  uint64_t      grid_id;
  kernel_t      kernel;

  if (wp->kernel_p)
    return wp->kernel;

  grid_id = warp_get_grid_id (dev_id, sm_id, wp_id);
  kernel  = kernels_find_kernel_by_grid_id (dev_id, grid_id);

  if (!kernel)
    {
      device_create_kernel (dev_id, grid_id);
      kernel = kernels_find_kernel_by_grid_id (dev_id, grid_id);
    }

  wp->kernel   = kernel;
  wp->kernel_p = CACHED;

  return wp->kernel;
}

CuDim3
warp_get_block_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);

  if (cuda_remote && !(wp->block_idx_p)
      && sm_is_valid (dev_id, sm_id))
    cuda_remote_update_block_idx_in_sm (get_current_remote_target (), dev_id, sm_id);

  if (wp->block_idx_p)
    return wp->block_idx;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return wp->block_idx;
}

uint64_t
warp_get_valid_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);

  if (wp->valid_lanes_mask_p)
    return wp->valid_lanes_mask;

  if (warp_is_valid (dev_id, sm_id, wp_id))
    {
      update_warp_cached_info (dev_id, sm_id, wp_id);
      return wp->valid_lanes_mask;
    }

  wp->valid_lanes_mask   = 0;
  wp->valid_lanes_mask_p = CACHED;

  if (!wp->timestamp_p)
    {
      wp->timestamp_p = true;
      wp->timestamp = cuda_clock ();
    }

  return 0;
}

uint64_t
warp_get_active_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);

  if (wp->active_lanes_mask_p)
    return wp->active_lanes_mask;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return wp->active_lanes_mask;
}

uint64_t
warp_get_divergent_lanes_mask (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  uint64_t valid_lanes_mask;
  uint64_t active_lanes_mask;
  uint64_t divergent_lanes_mask;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  valid_lanes_mask     = warp_get_valid_lanes_mask  (dev_id, sm_id, wp_id);
  active_lanes_mask    = warp_get_active_lanes_mask (dev_id, sm_id, wp_id);
  divergent_lanes_mask = valid_lanes_mask & ~active_lanes_mask;

  return divergent_lanes_mask;
}

uint32_t
warp_get_lowest_active_lane (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  uint64_t active_lanes_mask;
  uint32_t ln_id;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  active_lanes_mask = warp_get_active_lanes_mask (dev_id, sm_id, wp_id);

  for (ln_id = 0; ln_id < device_get_num_lanes (dev_id); ++ln_id)
    if ((active_lanes_mask >> ln_id) & 1)
      break;

  return ln_id;
}

uint64_t
warp_get_active_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  uint32_t ln_id;
  uint64_t pc;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  ln_id = warp_get_lowest_active_lane (dev_id, sm_id, wp_id);
  pc = lane_get_pc (dev_id, sm_id, wp_id, ln_id);

  return pc;
}

uint64_t
warp_get_active_virtual_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  uint32_t ln_id;
  uint64_t pc;

  gdb_assert (dev_id < cuda_system_get_num_devices ());
  gdb_assert (sm_id < device_get_num_sms (dev_id));
  gdb_assert (wp_id < device_get_num_warps (dev_id));

  ln_id = warp_get_lowest_active_lane (dev_id, sm_id, wp_id);
  pc = lane_get_virtual_pc (dev_id, sm_id, wp_id, ln_id);

  return pc;
}

uint64_t
warp_get_error_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);
  bool error_pc_available = false;
  uint64_t error_pc = 0ULL;

  /*if (wp->error_pc_p)
    {
      gdb_assert (wp->error_pc_available);
      return wp->error_pc;
    }
*/
  cuda_api_read_error_pc (dev_id, sm_id, wp_id, &error_pc, &error_pc_available);

  wp->error_pc = error_pc;
  wp->error_pc_available = error_pc_available;
  wp->error_pc_p = CACHED;

  gdb_assert (wp->error_pc_available);
  return wp->error_pc;
}

cuda_clock_t
warp_get_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);

  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  gdb_assert (wp->timestamp_p);

  return wp->timestamp;
}

void
warp_set_grid_id (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint64_t grid_id)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);

  gdb_assert (cuda_remote);

  wp->grid_id = grid_id;
  wp->grid_id_p = true;
}

void
warp_set_block_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, CuDim3 *block_idx)
{
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);

  gdb_assert (cuda_remote);
  gdb_assert (warp_is_valid (dev_id, sm_id, wp_id));

  wp->block_idx = *block_idx;
  wp->block_idx_p = true;
}

/* Lanes register cache */
static cuda_reg_cache_element_t *
cuda_reg_cache_find_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  int idx;
  cuda_reg_cache_element_t *elem;
  cuda_reg_cache_element_t new_elem;

  for (idx = 0;
       VEC_iterate (cuda_reg_cache_element_t, cuda_register_cache, idx, elem);
       ++idx)
    if (elem->dev == dev_id && elem->sm == sm_id &&
        elem->wp == wp_id && elem->ln == ln_id)
      return elem;

  memset (&new_elem, 0, sizeof(new_elem));
  new_elem.dev = dev_id;
  new_elem.sm = sm_id;
  new_elem.wp = wp_id;
  new_elem.ln = ln_id;

  return VEC_safe_push (cuda_reg_cache_element_t, cuda_register_cache, &new_elem);
}

static void
cuda_reg_cache_remove_element (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  int idx;
  cuda_reg_cache_element_t *elem;

  for (idx = 0;
       VEC_iterate (cuda_reg_cache_element_t, cuda_register_cache, idx, elem);
       ++idx)
    {
       if (elem->dev != dev_id || elem->sm != sm_id ||
           elem->wp != wp_id || elem->ln != ln_id)
         continue;
         VEC_unordered_remove(cuda_reg_cache_element_t, cuda_register_cache, idx);
         break;
    }
}

/******************************************************************************
 *
 *                                   Lanes
 *
 ******************************************************************************/

static inline lane_state_t *
lane_get (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  gdb_assert (ln_id < device_get_num_lanes (dev_id));

  return &warp_get(dev_id, sm_id, wp_id)->ln[ln_id];
}

static void
lane_invalidate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  lane_state_t *ln = lane_get (dev_id, sm_id, wp_id, ln_id);


  ln->pc_p         = false;
  ln->virtual_pc_p = false;
  ln->thread_idx_p = false;
  ln->exception_p  = false;
  ln->timestamp_p  = false;

  cuda_reg_cache_remove_element (dev_id, sm_id, wp_id, ln_id);
}

bool
lane_is_valid (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  uint64_t valid_lanes_mask;
  bool     valid;
  lane_state_t *ln = lane_get (dev_id, sm_id, wp_id, ln_id);

  valid_lanes_mask = warp_get_valid_lanes_mask (dev_id, sm_id, wp_id);
  valid = (valid_lanes_mask >> ln_id) & 1;

  if (!ln->timestamp_p)
    {
      ln->timestamp_p = true;
      ln->timestamp = cuda_clock ();
    }

  return valid;
}

bool
lane_is_active (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  uint64_t active_lanes_mask;
  bool     active;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  active_lanes_mask = warp_get_active_lanes_mask (dev_id, sm_id, wp_id);
  active = (active_lanes_mask >> ln_id) & 1;

  return active;
}

bool
lane_is_divergent (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  uint64_t divergent_lanes_mask;
  bool     divergent;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  divergent_lanes_mask = warp_get_divergent_lanes_mask (dev_id, sm_id, wp_id);
  divergent = (divergent_lanes_mask >> ln_id) & 1;

  return divergent;
}

CuDim3
lane_get_thread_idx (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  lane_state_t *ln = lane_get(dev_id, sm_id, wp_id, ln_id);

  /* In a remote session, we fetch the threadIdx of all valid thread in the warp using
   * one rsp packet to reduce the amount of communication. */
  if (cuda_remote && !(ln->thread_idx_p)
      && warp_is_valid (dev_id, sm_id, wp_id))
    cuda_remote_update_thread_idx_in_warp (get_current_remote_target (), dev_id, sm_id, wp_id);

  if (ln->thread_idx_p)
    return ln->thread_idx;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return ln->thread_idx;
}

uint64_t
lane_get_virtual_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  lane_state_t *ln = lane_get (dev_id, sm_id, wp_id, ln_id);

  if (ln->virtual_pc_p)
    return ln->virtual_pc;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return ln->virtual_pc;
}

uint64_t
lane_get_pc (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  lane_state_t *ln = lane_get (dev_id, sm_id, wp_id, ln_id);
  warp_state_t *wp = warp_get (dev_id, sm_id, wp_id);
  uint64_t      pc;
  uint32_t      other_ln_id;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  if (ln->pc_p)
    return ln->pc;

  cuda_api_read_pc (dev_id, sm_id, wp_id, ln_id, &pc);

  ln->pc_p = CACHED;
  ln->pc   = pc;

  /* Optimization: all the active lanes share the same virtual PC */
  if (lane_is_active (dev_id, sm_id, wp_id, ln_id))
    for (other_ln_id = 0; other_ln_id < device_get_num_lanes (dev_id); ++other_ln_id)
      if (lane_is_valid (dev_id, sm_id, wp_id, other_ln_id) &&
          lane_is_active (dev_id, sm_id, wp_id, other_ln_id))
        {
          wp->ln[other_ln_id].pc_p = CACHED;
          wp->ln[other_ln_id].pc   = pc;
        }

  return ln->pc;
}

CUDBGException_t
lane_get_exception (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  lane_state_t    *ln = lane_get (dev_id, sm_id, wp_id, ln_id);

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  if (ln->exception_p)
    return ln->exception;

  update_warp_cached_info (dev_id, sm_id, wp_id);

  return ln->exception;
}

uint32_t
lane_get_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
                   uint32_t regno)
{
  uint32_t value;
  cuda_reg_cache_element_t *elem;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  /* If register can not be cached - read it directly */
  if (regno > CUDBG_CACHED_REGISTERS_COUNT)
    {
      cuda_api_read_register (dev_id, sm_id, wp_id, ln_id, regno, &value);
      return value;
    }

  elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);
  if ( (elem->register_valid_mask[regno>>5]&(1UL<<(regno&31))) != 0)
    return elem->registers[regno];

  if (regno < CUDBG_CACHED_REGISTERS_COUNT)
  {
    cuda_api_read_register_range (dev_id, sm_id, wp_id, ln_id, regno&~31, 32, &elem->registers[regno&~31]);
    elem->register_valid_mask[regno>>5]|=0xffffffff;
  } else {
    cuda_api_read_register (dev_id, sm_id, wp_id, ln_id, regno, &elem->registers[regno]);
    elem->register_valid_mask[regno>>5]|=1UL<<(regno&31);
  }

  return elem->registers[regno];
}

void
lane_set_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
                   uint32_t regno, uint32_t value)
{
  cuda_reg_cache_element_t *elem;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  cuda_api_write_register (dev_id, sm_id, wp_id, ln_id, regno, value);
  /* If register can not be cached - read it directly */
  if (regno > CUDBG_CACHED_REGISTERS_COUNT)
      return;

  elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);
  elem->registers[regno] = value;
  elem->register_valid_mask[regno>>5]|=1UL<<(regno&31);
}

bool
lane_get_predicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
                    uint32_t predicate)
{
  cuda_reg_cache_element_t *elem;

  /* NVIDIA 470 driver with "CUDA 11.3 Update 1" release fixes P7 predicate
     which should have always been reported as true. With this change CUDA
     Debugger API from newer driver will report 8 device predicates instead of
     7. Following assert excludes P7 from the check to allow cuda-gdb to
     continue to work with older driver. This is a temporary change and will be
     removed for next major CUDA release. */
  gdb_assert (!((predicate >= device_get_num_predicates (dev_id)) && (predicate != 7)));
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  /* P7 is always true */
  if (predicate == 7)
    return true;

  elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  if (elem->predicates_valid_p)
    return elem->predicates[predicate] != 0;

  cuda_api_read_predicates (dev_id, sm_id, wp_id, ln_id,
                            device_get_num_predicates (dev_id),
                            elem->predicates);
  elem->predicates_valid_p = CACHED;

  return elem->predicates[predicate] != 0;
}

void
lane_set_predicate (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
                    uint32_t predicate, bool value)
{
  cuda_reg_cache_element_t *elem;

  /* NVIDIA 470 driver with "CUDA 11.3 Update 1" release fixes P7 predicate
     which should have always been reported as true. With this change CUDA
     Debugger API from newer driver will report 8 device predicates instead of
     7. Following assert excludes P7 from the check to allow cuda-gdb to
     continue to work with older driver. This is a temporary change and will be
     removed for next major CUDA release. */
  gdb_assert (!((predicate >= device_get_num_predicates (dev_id)) && (predicate != 7)));
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  /* Do nothing as P7 is always true */
  if (predicate == 7)
    return;

  elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  if (!elem->predicates_valid_p)
    {
      cuda_api_read_predicates (dev_id, sm_id, wp_id, ln_id,
                                device_get_num_predicates (dev_id),
                                elem->predicates);
      elem->predicates_valid_p = CACHED;
    }

  elem->predicates[predicate] = value;

  cuda_api_write_predicates (dev_id, sm_id, wp_id, ln_id,
                             device_get_num_predicates (dev_id),
                             elem->predicates);
}

uint32_t
lane_get_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  cuda_reg_cache_element_t *elem;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));
  elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  if (elem->cc_register_valid_p)
    return elem->cc_register;

  cuda_api_read_cc_register (dev_id, sm_id, wp_id, ln_id,
                            &elem->cc_register);
  elem->cc_register_valid_p = CACHED;

  return elem->cc_register;
}

void
lane_set_cc_register (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id,
                      uint32_t value)
{
  cuda_reg_cache_element_t *elem;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));
  elem = cuda_reg_cache_find_element (dev_id, sm_id, wp_id, ln_id);

  elem->cc_register = value;
  elem->cc_register_valid_p = CACHED;

  cuda_api_write_cc_register (dev_id, sm_id, wp_id, ln_id, elem->cc_register);
}

int32_t
lane_get_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  int32_t call_depth;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  cuda_api_read_call_depth (dev_id, sm_id, wp_id, ln_id, &call_depth);

  return call_depth;
}

int32_t
lane_get_syscall_call_depth (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id, uint32_t ln_id)
{
  int32_t syscall_call_depth;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  cuda_api_read_syscall_call_depth (dev_id, sm_id, wp_id, ln_id, &syscall_call_depth);

  return syscall_call_depth;
}

uint64_t
lane_get_virtual_return_address (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
                                 uint32_t ln_id, int32_t level)
{
  uint64_t virtual_return_address;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  cuda_api_read_virtual_return_address (dev_id, sm_id, wp_id, ln_id, level,
                                             &virtual_return_address);

  return virtual_return_address;
}

cuda_clock_t
lane_get_timestamp (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,uint32_t ln_id)
{
  lane_state_t *ln = lane_get (dev_id, sm_id, wp_id, ln_id);;

  gdb_assert (ln->timestamp_p);

  return ln->timestamp;
}

uint64_t
lane_get_memcheck_error_address (uint32_t dev_id, uint32_t sm_id,
                                 uint32_t wp_id, uint32_t ln_id)
{
  CUDBGException_t exception;
  uint64_t address = 0;
  ptxStorageKind segment = ptxUNSPECIFIEDStorage;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  exception = lane_get_exception (dev_id, sm_id, wp_id, ln_id);

  if (exception == CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS)
    cuda_api_memcheck_read_error_address (dev_id, sm_id, wp_id, ln_id,
                                          &address, &segment);
  return address;
}

ptxStorageKind
lane_get_memcheck_error_address_segment (uint32_t dev_id, uint32_t sm_id,
                                         uint32_t wp_id, uint32_t ln_id)
{
  CUDBGException_t exception;
  uint64_t address = 0;
  ptxStorageKind segment = ptxUNSPECIFIEDStorage;

  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  exception = lane_get_exception (dev_id, sm_id, wp_id, ln_id);

  if (exception == CUDBG_EXCEPTION_LANE_ILLEGAL_ADDRESS)
    cuda_api_memcheck_read_error_address (dev_id, sm_id, wp_id, ln_id,
                                          &address, &segment);
  return segment;
}

void
lane_set_thread_idx (uint32_t dev_id, uint32_t sm_id,
                     uint32_t wp_id, uint32_t ln_id, CuDim3 *thread_idx)
{
  lane_state_t *ln = lane_get (dev_id, sm_id, wp_id, ln_id);

  gdb_assert (cuda_remote);
  gdb_assert (lane_is_valid (dev_id, sm_id, wp_id, ln_id));

  ln->thread_idx = *thread_idx;
  ln->thread_idx_p = true;
}

static void
lane_set_exception_none (uint32_t dev_id, uint32_t sm_id, uint32_t wp_id,
                         uint32_t ln_id)
{
  lane_state_t *ln = lane_get (dev_id, sm_id, wp_id, ln_id);

  ln->exception = CUDBG_EXCEPTION_NONE;
  ln->exception_p = true;
}
