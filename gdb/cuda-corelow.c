/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2015-2020 NVIDIA Corporation
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

#include "cuda-corelow.h"

#include "inferior.h"
#include "target.h"
#include "gdbthread.h"
#include "regcache.h"
#include "completer.h"
#include "readline/readline.h"
#include "common/common-defs.h"

#include "cuda-api.h"
#include "cuda-tdep.h"
#include "cuda-events.h"
#include "cuda-state.h"
#include "cuda-exceptions.h"
#include "cuda-context.h"
#include "cuda-iterator.h"
#include "cuda-linux-nat.h"

#include "../libcudacore/libcudacore.h"

class cuda_core_target_ops : public target_ops
{
public:
  strata stratum () const override
  {
    return process_stratum;
  }

  /* Return a reference to this target's unique target_info
     object.  */
  virtual const target_info &info () const override;

  virtual void close () override;

  virtual void detach (inferior *inf, int from_tty) override;

  virtual bool has_memory () override;
  virtual bool has_stack () override;
  virtual bool has_registers () override;

  virtual bool thread_alive (ptid_t ptid) override;
  virtual struct address_space *thread_address_space (ptid_t) override;
  virtual struct gdbarch *thread_architecture (ptid_t ptid) override;

  virtual const char *pid_to_str (ptid_t) override;
  virtual void fetch_registers (struct regcache *, int) override;
};

void _initialize_cuda_corelow (void);

static cuda_core_target_ops cudacore_target;

static CudaCore *cuda_core = NULL;

bool
cuda_core_target_ops::has_memory ()
{
  return true;
}

bool
cuda_core_target_ops::has_stack ()
{
  return true;
}

bool
cuda_core_target_ops::has_registers ()
{
  return true;
}

bool
cuda_core_target_ops::thread_alive (ptid_t ptid)
{
  return 1;
}

struct address_space *
cuda_core_target_ops::thread_address_space (ptid_t ptid)
{
  struct inferior *inf = find_inferior_ptid (ptid);

  gdb_assert (inf);

  return inf->aspace;
}

struct gdbarch *
cuda_core_target_ops::thread_architecture (ptid_t ptid)
{
  struct inferior *inf = find_inferior_ptid (ptid);

  gdb_assert (inf);

  return inf->gdbarch;
}

const char *
cuda_core_target_ops::pid_to_str (ptid_t ptid)
{
  static char buf[64];

  xsnprintf (buf, sizeof buf, "Thread %ld", ptid.tid ());
  return buf;
}

void
cuda_core_target_ops::fetch_registers (struct regcache *regcache, int regno)
{
  cuda_core_fetch_registers (regcache, regno);
}

void
cuda_core_fetch_registers (struct regcache *regcache, int regno)
{
  cuda_coords_t c;
  unsigned reg_no, reg_value, num_regs;
  uint64_t pc;
  struct gdbarch *gdbarch = cuda_get_gdbarch();
  uint32_t pc_regnum = gdbarch ? gdbarch_pc_regnum (gdbarch): 256;

  if (cuda_coords_get_current (&c))
    return;

  num_regs = device_get_num_registers (c.dev);
  for (reg_no = 0; reg_no < num_regs; ++reg_no)
    {
      reg_value = lane_get_register (c.dev, c.sm, c.wp, c.ln, reg_no);
      regcache->raw_supply (reg_no, &reg_value);
    }

  /* Save PC as well */
  pc = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
  regcache->raw_supply (pc_regnum, &pc);

  if (gdbarch)
    {
      int i;

      num_regs = device_get_num_uregisters (c.dev);
      for (reg_no = 0; reg_no < num_regs; ++reg_no)
	{
	  reg_t reg = CUDA_REG_CLASS_AND_REGNO (REG_CLASS_UREG_FULL, reg_no);
	  uint32_t regnum = cuda_reg_to_regnum (gdbarch, reg);

	  reg_value = warp_get_uregister (c.dev, c.sm, c.wp, reg_no);
	  regcache->raw_supply (regnum, &reg_value);
	}

      num_regs = device_get_num_upredicates (c.dev);
      for (reg_no = 0; reg_no < num_regs; ++reg_no)
	{
	  reg_t reg = CUDA_REG_CLASS_AND_REGNO (REG_CLASS_UREG_PRED, reg_no);
	  uint32_t regnum = cuda_reg_to_regnum (gdbarch, reg);

	  reg_value = warp_get_upredicate (c.dev, c.sm, c.wp, reg_no);
	  regcache->raw_supply (regnum, &reg_value);
	}

      /* Mark all registers not found in the core as unavailable.  */
      for (i = 0; i < gdbarch_num_regs (gdbarch); i++)
	if (regcache->get_register_status (i) == REG_UNKNOWN)
	  regcache->raw_supply (i, NULL);
    }
}

#define CUDA_CORE_PID 966617

void
cuda_core_register_tid (uint32_t tid)
{
  if (inferior_ptid != null_ptid)
    return;

  ptid_t ptid (CUDA_CORE_PID, tid, tid);
  add_thread (ptid);
  inferior_ptid = ptid;
}

void
cuda_core_load_api (const char *filename)
{
  CUDBGAPI api;

  printf_unfiltered (_("Opening GPU coredump: %s\n"), filename);

  cuda_core = cuCoreOpenByName (filename);
  if (cuda_core == NULL)
    error ("Failed to read core file: %s", cuCoreErrorMsg());
  api = cuCoreGetApi (cuda_core);
  if (api == NULL)
    error ("Failed to get debugger APIs: %s", cuCoreErrorMsg());

  cuda_api_set_api (api);

  /* Initialize the APIs */
  cuda_initialize ();
  if (!cuda_initialized)
    error ("Failed to initialize CUDA Core debugger API!");

  /* Set debuggers architecture to CUDA */
  set_target_gdbarch (cuda_get_gdbarch ());
}

void
cuda_core_free (void)
{
  if (cuda_core == NULL)
    return;

  cuda_cleanup ();
  cuda_gdb_session_destroy ();
  cuCoreFree(cuda_core);
  cuda_core = NULL;
}

void
cuda_core_initialize_events_exceptions (void)
{
  CUDBGEvent event;

  /* Flush registers cache */
  registers_changed ();

  /* Create session directory */
  if (cuda_gdb_session_create ())
    error ("Failed to create session directory");

  /* Drain the event queue */
  while (true) {
    cuda_api_get_next_sync_event (&event);

    if (event.kind == CUDBG_EVENT_INVALID)
      break;

    if (event.kind == CUDBG_EVENT_CTX_CREATE)
      cuda_core_register_tid (event.cases.contextCreate.tid);

    cuda_process_event (&event);
  }

  /* Figure out, where exception happened */
  if (cuda_exception_hit_p (cuda_exception))
    {
      uint64_t kernelId;
      cuda_coords_t c = cuda_exception_get_coords (cuda_exception);

      cuda_coords_set_current (&c);

      /* Set the current coordinates context to current */
      if (!cuda_coords_get_current_logical (&kernelId, NULL, NULL, NULL))
        {
          kernel_t kernel = kernels_find_kernel_by_kernel_id (kernelId);
          context_t ctx = kernel ? kernel_get_context (kernel) : get_current_context ();
          if (ctx != NULL)
             set_current_context (ctx);
        }

      cuda_exception_print_message (cuda_exception);
    }

  /* Fetch latest information about coredump grids */
  kernels_update_args ();
}

static void
cuda_find_first_valid_lane (void)
{
  cuda_iterator itr;
  cuda_coords_t c;
  itr = cuda_iterator_create (CUDA_ITERATOR_TYPE_THREADS, NULL,
                              (cuda_select_t) (CUDA_SELECT_VALID | CUDA_SELECT_SNGL));
  cuda_iterator_start (itr);
  c  = cuda_iterator_get_current (itr);
  cuda_iterator_destroy (itr);
  if (!c.valid)
    {
      cuda_coords_update_current (false, false);
      return;
    }
  cuda_coords_set_current (&c);
}

void
cuda_core_open (const char *filename, int from_tty)
{
  struct inferior *inf;
  struct cleanup *old_chain;
  char *expanded_filename;

  target_preopen (from_tty);

  if (filename == NULL)
    error (_("No core file specified."));

  expanded_filename = tilde_expand (filename);
  old_chain = make_cleanup (xfree, expanded_filename);

  cuda_core_load_api (filename);

  TRY
    {
      /* Push the target */
      push_target (&cudacore_target);

      /* Flush existing thread information */
      init_thread_list ();

      /* Switch focus to null ptid */
      inferior_ptid = null_ptid;

      /* Add fake PID*/
      inf = current_inferior();
      if (inf->pid == 0)
        {
          inferior_appeared (inf, CUDA_CORE_PID);
          inf->fake_pid_p = true;
        }

      post_create_inferior (&cudacore_target, from_tty);

      cuda_core_initialize_events_exceptions ();

      /* If no exception found try to set focus to first valid thread */
      if (!cuda_focus_is_device())
        {
          warning ("No exception was found on the device");
          cuda_find_first_valid_lane ();
        }

      if (!cuda_focus_is_device())
        throw_error (GENERIC_ERROR, "No focus could be set on device");

      cuda_print_message_focus (false);

      /* Fetch all registers from core file.  */
      target_fetch_registers (get_current_regcache (), -1);

      /* Now, set up the frame cache, and print the top of stack.  */
      reinit_frame_cache ();
      print_stack_frame (get_selected_frame (NULL), 1, SRC_AND_LOC, 1);
    }
  CATCH (e, RETURN_MASK_ALL)
    {
      if (e.reason < 0)
	{
	  pop_all_targets_at_and_above (process_stratum);

	  inferior_ptid = null_ptid;
	  discard_all_inferiors ();

	  registers_changed ();
	  reinit_frame_cache ();
	  cuda_cleanup ();

	  error (_("Could not open CUDA core file: %s"), e.message);
	}
    }
  END_CATCH

  do_cleanups (old_chain);
}

void
cuda_core_target_ops::close ()
{
  inferior_ptid = null_ptid;
  discard_all_inferiors ();

  cuda_core_free ();
}

void
cuda_core_target_ops::detach (inferior *inf, int from_tty)
{
  unpush_target (this);
  reinit_frame_cache ();
  if (from_tty)
    printf_filtered (_("No core file now.\n"));
}

static target_info cuda_core_target_info = {
  "cudacore",
  "CUDA core dump file",
  "Use CUDA core file as a target. Specify the filename to the core file."
};

const target_info &
cuda_core_target_ops::info () const
{
  return cuda_core_target_info;
}

void
_initialize_cuda_corelow (void)
{
  add_target (cuda_core_target_info, cuda_core_open, filename_completer);
}
