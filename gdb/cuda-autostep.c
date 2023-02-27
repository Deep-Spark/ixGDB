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

#include "defs.h"
#include "inferior.h"
#include "gdbthread.h"
#include "arch-utils.h"
#include "regcache.h"

#include <block.h>

#include "cuda-autostep.h"
#include "cuda-state.h"
#include "cuda-iterator.h"
#include "cuda-frame.h"
#include "cuda-tdep.h"
#include "cuda-options.h"

DEF_VEC_I (CORE_ADDR);

/* When inside an autostep range, we go into single-step mode.
   When true, it means we are actively handling an autostep region.  Otherwise
   it means we are not handling any autostepping region.  */
static bool autostep_pending = false;

/* Global autostep state.  */
static struct autostep_state astep_state;

/* Getters and setters */

/* Fetch the state of autostepping.
   TRUE if autostep is running.
   FALSE otherwise.  */

bool
cuda_get_autostep_pending (void)
{
  return autostep_pending;
}

/* Set the state of autostepping to PENDING.
   TRUE means a autostep region is currently being handled.
   FALSE means we are not currently handling an autostep region.  */

void
cuda_set_autostep_pending (bool pending)
{
  autostep_pending = pending;
}

/* Report an exception in host code based on BEFORE_PC.  */

static void
autostep_report_exception_host (uint64_t before_pc)
{
  /* We know the exception must have been at the previous pc */
  struct gdbarch *gdbarch = get_current_arch ();
  struct type *type_uint32   = builtin_type (gdbarch)->builtin_uint32;
  struct type *type_data_ptr = builtin_type (gdbarch)->builtin_data_ptr;

  struct symtab_and_line before_sal = find_pc_line (before_pc, 0);

  printf_filtered (_("Autostep precisely caught exception at %s:%d (0x%llx)\n"),
    before_sal.symtab->filename, before_sal.line, (unsigned long long)before_pc);

  set_internalvar (lookup_internalvar ("autostep_exception_pc"),
    value_from_longest (type_data_ptr, (LONGEST) before_pc));
  set_internalvar (lookup_internalvar ("autostep_exception_line"),
    value_from_longest (type_uint32, (LONGEST) before_sal.line));
}

/* Report an exception in device code based on the number of steps NSTEPS, the
   previous lane information BEFORE_LN, the previous PC BEFORE_PC and the
   current PC AFTER_PC.  */

static void
autostep_report_exception_device (int nsteps, int before_ln, uint64_t before_pc, uint64_t after_pc)
{
  struct gdbarch *gdbarch    = get_current_arch ();
  struct type *type_uint32   = builtin_type (gdbarch)->builtin_uint32;
  struct type *type_data_ptr = builtin_type (gdbarch)->builtin_data_ptr;
  uint64_t exception_pc;
  struct symtab_and_line exception_sal;
  cuda_coords_t c;
  bool divergent;
  char exception_pc_line_info[200];
  struct symtab_and_line before_sal;

  gdb_assert (nsteps >= 1);

  /* If the thread before stepping is also active, the exception didn't occur
     in a divergent thread */
  cuda_coords_get_current (&c);
  divergent = !lane_is_active (c.dev, c.sm, c.wp, before_ln);

  /* Calculate exception PC - if more than 1 one instruction was executed
   * it means there was no control flow instructions so after_pc should
   * be used as a reference (before_pc is far behind after_pc).
   * If just one instruction was executed use the last stepped pc as a reference,
   * because the executed instruction could be a control flow instruction. */
  if (nsteps > 1)
    cuda_api_get_adjusted_code_address (c.dev, after_pc, &exception_pc, CUDBG_ADJ_PREVIOUS_ADDRESS);
  else
    exception_pc = cuda_sstep_get_last_pc ();

  cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                     "Autostep: nsteps %d divergent %d after_pc 0x%llx exception_pc 0x%llx last_pc 0x%llx",
                     nsteps, divergent, (long long)after_pc,
                     (long long)exception_pc,
                     (long long)cuda_sstep_get_last_pc ());

  /* We could have stepped a control flow instruction like NOP.S or SYNC, which
   * means another set of lanes ran while we tried to step the lowest active
   * lane.  Switch focus to one of the lanes that we last stepped as the closest
   * approximation of the failure. */
  if (!cuda_sstep_lane_stepped (before_ln))
    cuda_coords_set_current_physical (c.dev, c.sm, c.wp, cuda_sstep_get_lowest_lane_stepped ());

  cuda_print_message_focus (false);

  exception_sal = find_pc_line (exception_pc, 0);

  if (exception_sal.symtab && exception_sal.line)
    snprintf (exception_pc_line_info, sizeof (exception_pc_line_info),
              "%s:%d (0x%llx)",
              exception_sal.symtab->filename, exception_sal.line,
              (unsigned long long)exception_pc);
  else
    snprintf (exception_pc_line_info, sizeof (exception_pc_line_info),
              "0x%llx",
              (unsigned long long)exception_pc);

  if (divergent)
    printf_filtered (_("Autostep caught exception at instruction before 0x%llx\n"
                       "This is probably %s\n"),
                     (unsigned long long)after_pc, exception_pc_line_info);
  else
    printf_filtered (_("Autostep precisely caught exception at %s\n"),
                     exception_pc_line_info);

  set_internalvar (lookup_internalvar ("autostep_exception_pc"),
                   value_from_longest (type_data_ptr, (LONGEST) exception_pc));
  set_internalvar (lookup_internalvar ("autostep_exception_line"),
                   value_from_longest (type_uint32, (LONGEST) exception_sal.line));
}

static uint64_t
find_end_pc (uint64_t pc)
{
  const struct block *bl;
  struct bound_minimal_symbol msymbol;

  bl = block_for_pc ((CORE_ADDR)pc);
  if (bl)
       return BLOCK_END (bl);

  msymbol = lookup_minimal_symbol_by_pc ((CORE_ADDR)pc);
  if (msymbol.minsym)
       return MSYMBOL_VALUE_RAW_ADDRESS (msymbol.minsym) + MSYMBOL_SIZE (msymbol.minsym);

  return (uint64_t)-1LL;
}

static int
count_instructions (uint64_t pc, uint64_t end_pc)
{
  const char *inst;
  uint32_t inst_size;
  kernel_t kernel = cuda_current_kernel ();
  int count = 0;

  for (; pc < end_pc; pc += inst_size)
    {
      inst = kernel_disassemble (kernel, pc, &inst_size);
      if (!inst)
        break; /* Abort the loop if pc is outside of the routine boundary */
      if (inst[0] == 0)
        continue; /* Ignore empty instructions */
      ++count;
    }

  return count;
}

static int
count_lines (uint64_t pc, uint64_t end_pc, uint32_t inst_size)
{
  struct symtab_and_line cur_sal, next_sal;
  int nlines = 0;

  for (cur_sal = find_pc_line(pc, 0);
       pc <= end_pc;
       pc += inst_size, cur_sal = next_sal)
    {
      next_sal = find_pc_line(pc, 0);
      /* Check if line numbers differ.
       * If no line information exists treat each instruction as one line. */
      if (!(cur_sal.symtab && cur_sal.line) ||
          !(next_sal.symtab && next_sal.line) ||
          cur_sal.line != next_sal.line)
        ++nlines;
    }

  return nlines;
}

/* Check if we are dealing with a device that is fermi or better.
   Return true if we are, false otherwise.  */

static bool
fermi_or_better (CORE_ADDR astep_pc, uint32_t device)
{
  const char *sm_type;
  struct breakpoint *astep = cuda_find_autostep_by_addr (astep_pc);

  /* Sanity check.  */
  gdb_assert (cuda_focus_is_device ());
  gdb_assert (astep);

  /* Check that the device is Fermi or better */
  /* Must check here in case user re-enabled it */
  sm_type = device_get_sm_type (device);
  if (strncmp (sm_type, "sm_1", 4) == 0)
    {
      warning ("Disabling autostep %d on device %d because autostep "
	       "requires compute capability 2.0 or higher.", astep->number,
	       device);
      astep->enable_state = bp_disabled;
      return false;
    }
  return true;
}

/* Reset the autostep state to reflect the autostep region that starts
   at PC.  */

static void
initialize_autostep_state (CORE_ADDR astep_pc)
{
  struct breakpoint *astep = cuda_find_autostep_by_addr (astep_pc);

  /* Sanity check.  */
  gdb_assert (astep);

  /* Initialize autostep data based on astep_pc.  */
  memset ((void *)&astep_state, 0, sizeof (struct autostep_state));

  astep_state.insn_stepping =
    (astep->cuda_autostep_length_type == cuda_autostep_insts);

  if (astep_state.insn_stepping)
    astep_state.insns_to_step = astep->cuda_autostep_length;
  else
    astep_state.lines_to_step = astep->cuda_autostep_length;

  astep_state.remaining = astep->cuda_autostep_length;
  astep_state.start_pc = astep_pc;
  astep_state.start_sal = find_pc_line (astep_pc, 0);
  astep_state.device = cuda_focus_is_device ();
  astep_state.cur_sal = astep_state.start_sal;
  astep_state.cur_pc = astep_pc;
}

/* Return true if coords is a valid astep warp, false otherwise.  */

static bool
astep_warp_valid_p (cuda_coords_t coords)
{
  cuda_coords_t c = coords;
  struct breakpoint *astep;

  if (cuda_coords_complete_physical (&c))
    return false;

  c = coords;

  if (!warp_is_valid (c.dev, c.sm, c.wp))
    return false;

  astep = cuda_find_autostep_by_addr (astep_state.start_pc);

  if (astep->enable_state != bp_enabled)
    return false;

  if (!fermi_or_better (astep_state.start_pc, coords.dev))
    return false;

  if (warp_get_active_virtual_pc (c.dev, c.sm, c.wp) != (uint64_t) astep_state.start_pc)
    return false;

  return true;
}

/* Select the next valid warp based on the currently active iterator.
   Return true if found or false otherwise.

   When a new valid warp is found, this function resets the autostep
   state so we have correct information about what to step and for how
   long, since we could've switched back to handling a different autostep
   region.  */

static bool
select_next_valid_warp (void)
{
  cuda_iterator iter = astep_state.data.device.iter;
  cuda_coords_t c = astep_state.data.device.cur_coords;
  cuda_coords_t nextc = c;

  cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                     "Autostep: handling next warp! Previous was: tId=(%d,%d,%d) bId=(%d,%d,%d)",
                     c.threadIdx.x, c.threadIdx.y, c.threadIdx.y,
                     c.blockIdx.x, c.blockIdx.y, c.blockIdx.z);

  /* Skip to next warp (by using possibly outdated physical coordinates,
     but sorted correctly by logical coordinates) */
  while (cuda_focus_is_device () && !cuda_iterator_end (iter) &&
	 c.dev == nextc.dev && c.sm == nextc.sm && c.wp == nextc.wp)
    {
      cuda_iterator_next (iter);
      nextc = cuda_iterator_get_current (iter);

      if (astep_warp_valid_p (nextc))
	{
	  int ln;
	  CORE_ADDR warp_pc;

	  c = nextc;
	  ln = warp_get_lowest_active_lane (c.dev, c.sm, c.wp);
	  cuda_coords_set_current_physical (c.dev, c.sm, c.wp, ln);
	  warp_pc = warp_get_active_virtual_pc (c.dev, c.sm, c.wp);

	  cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
			     "Autostep: next warp: tId=(%d,%d,%d) bId=(%d,%d,%d)",
			     c.threadIdx.x, c.threadIdx.y, c.threadIdx.y,
			     c.blockIdx.x, c.blockIdx.y, c.blockIdx.z);

	  /* Now that we have found a valid warp, reset the autostep state
	     to reflect the region this warp will autostep through.  */
	  initialize_autostep_state (warp_pc);
	  astep_state.data.device.cur_coords = nextc;
	  astep_state.data.device.cur_ln = ln;
	  astep_state.data.device.iter = iter;
	  return true;
	}
    }

  return false;
}

/* Setup the next round of single-steps.  It is assumed a valid warp is
   already selected.  */

static int
set_next_device_iteration (void)
{
  cuda_coords_t c = astep_state.data.device.cur_coords;
  uint64_t cur_pc, end_pc;
  int nsteps;
  struct symtab_and_line cur_sal;
  bool single_inst = astep_state.insn_stepping;
  uint32_t inst_size;
  int remaining = astep_state.remaining;

  /* A valid warp is already in place and we are starting to step this warp
     from the start.  */
  cur_pc = (CORE_ADDR) warp_get_active_virtual_pc (c.dev, c.sm, c.wp);
  cur_sal = find_pc_line(cur_pc, 0);
  end_pc = -1;

  /* Limit end_pc; at first assume there are no control flow instructions */
  if (!cuda_options_single_stepping_optimizations_enabled ())
    {
      end_pc = cur_pc;
    }
  else if (!(cur_sal.symtab && cur_sal.line) || single_inst)
    {
      /* Get instruction size */
      kernel_disassemble (cuda_current_kernel (), cur_pc, &inst_size);
      end_pc = cur_pc + remaining * inst_size;
    }
  else
    {
      struct linetable_entry *best_item = NULL; /* ignored */

      /* Search for all PCs which correspond to (current + remaining) line.
       * Try to pick up lowest possible address after current pc
       * corresponding to limiting line. */

      std::vector<CORE_ADDR> line_pcs = find_pcs_for_symtab_line (cur_sal.symtab,
								  cur_sal.line + remaining,
								  &best_item);

      for (CORE_ADDR &line_pc : line_pcs)
	{
	  if (cur_pc < line_pc && line_pc < end_pc)
	    end_pc = line_pc;
	}
    }

  /* If pc is in the top frame - do not allow autostepping outside of kernel boundaries */
  if (cuda_frame_outermost_p (get_next_frame (get_current_frame ())))
    {
      uint64_t kernel_end_pc;
      kernel_end_pc = find_end_pc (cur_pc);
      if (cur_pc >= kernel_end_pc)
	{
	  /* STOP AUTOSTEPPING!!!!!! */
	  return 1;
	}
      if (kernel_end_pc < end_pc)
	end_pc = kernel_end_pc;
    }

  /* Calculate how many steps should be taken */
  if (cuda_options_single_stepping_optimizations_enabled ())
    end_pc = cuda_find_next_control_flow_instruction (cur_pc, end_pc, false, &inst_size);
  if (end_pc == cur_pc)
    nsteps = 1; /* Currently at control flow instruction */
  else
    nsteps = count_instructions (cur_pc, end_pc);
  nsteps = std::max (nsteps, 1);

  cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                     "Autostep: issuing single step %d steps (from %llx to %llx).",
                     nsteps, cur_pc, end_pc);

  /* Does stepi, but cuda_sstep_execute takes cuda_sstep_nsteps into
   * account to execute 'single step nsteps times' */
  cuda_sstep_set_nsteps(nsteps);

  /* The device is ready to single-step now.  */
  astep_state.cur_pc = cur_pc;
  astep_state.end_pc = end_pc;
  astep_state.cur_sal = cur_sal;
  astep_state.remaining = remaining;
  astep_state.data.device.inst_size = inst_size;
  astep_state.data.device.nsteps = nsteps;

  return 0;
}

/* Initialize device-specific data for autostep.  This takes care of
   initializing the warp iterator and setting other information.  */

static int
cuda_initialize_device_autostep (CORE_ADDR pc)
{

  /* Iterate through all warps in current grid that are at a breakpoint */
  astep_state.data.device.filter = CUDA_WILDCARD_COORDS;
  astep_state.data.device.filter.valid = true;
  astep_state.data.device.filter.gridId = CUDA_CURRENT;

  cuda_coords_evaluate_current (&astep_state.data.device.filter, false);

  /* Create the iterator.  */
  astep_state.data.device.iter = cuda_iterator_create (CUDA_ITERATOR_TYPE_THREADS,
						       &astep_state.data.device.filter,
						       (cuda_select_t)(CUDA_SELECT_BKPT | CUDA_SELECT_VALID));

  /* Start the iteration process through all warps.  */
  cuda_iterator_start (astep_state.data.device.iter);

  astep_state.data.device.cur_coords = cuda_iterator_get_current (astep_state.data.device.iter);

  if (!astep_warp_valid_p (astep_state.data.device.cur_coords))
    if (!select_next_valid_warp ())
      return 1;

  /* Now that we've got the basic things out of the way, set the first iteration next.  */
  set_next_device_iteration ();

  return 0;
}

/* Initialize host-specific data for autostep.  This is currently not
   used, but it can be implemented in the future if needed.  */

static int
cuda_initialize_host_autostep (CORE_ADDR pc)
{
  /* Nothing special to do here.  Just return.  */
  return 0;
}

/* See cuda-autostep.h */

int
cuda_initialize_autostep (CORE_ADDR pc)
{
  /* Set basic autostep information.  */
  initialize_autostep_state (pc);

  /* Set host/device-specific autostep information.  */
  if (astep_state.device)
    {
      if (!fermi_or_better (pc, cuda_current_device ()))
	return 1;

      if (cuda_initialize_device_autostep (pc) != 0)
	return 1;
    }
  else
    cuda_initialize_host_autostep (pc);

  cuda_set_autostep_pending (true);

  return 0;
}

/* Given an address ASTEP_PC and a number of remaining step/instructions to
   conclude an autostep region, check if we have an overlapping autostep
   region and warn the user.  */

static void
check_overlapping_astep (CORE_ADDR astep_pc, int remaining)
{
  struct breakpoint *overlap = cuda_find_autostep_by_addr (astep_pc);

  if (overlap && remaining > 0)
    warning (_("Overlapping autostep %d ignored"), overlap->number);
}

/* Update the host autostep state.  */

static int
update_host_autostep_state (CORE_ADDR pc)
{
  if (astep_state.insn_stepping)
    {
      /* We are instruction-stepping.  Move to the next instruction.  */
      if (pc != astep_state.cur_pc)
	{
	  astep_state.insns_stepped++;
	  astep_state.remaining--;
	  astep_state.cur_pc = pc;
	}
    }
  else
    {
      struct symtab_and_line sal;

      sal = find_pc_line (pc, 0);

      if (sal.line != 0 && sal.line != astep_state.cur_sal.line)
	{
	  astep_state.lines_stepped++;
	  astep_state.remaining--;
	  astep_state.cur_sal = sal;
	  astep_state.cur_pc = pc;
	}
    }

  /* Check if we have an overlapping autostep region that we should ignore.  */
  check_overlapping_astep (pc, astep_state.remaining);

  if (astep_state.remaining <= 0)
    cuda_cleanup_autostep_state ();

  return 0;
}

/* Update the device autostep state.

   This function should only be called while autostepping is active.  It takes
   care of updating the autostep state with current information about the
   device.

   If a warp has executed to completion or if we are done with the autostep
   range for a particular warp, select the next valid warp and set things
   up so we can continue handling autostepping.

   If we happen to find an adjacent autostep region, switch to handling that
   region without changing focus.  */

static int
update_device_autostep_state (CORE_ADDR pc)
{
  cuda_coords_t c = astep_state.data.device.cur_coords;
  uint64_t after_pc, before_pc, end_pc;
  struct symtab_and_line after_sal, before_sal;
  int nsteps, lines, remaining;
  int single_inst;
  int before_ln;
  uint32_t inst_size;
  struct breakpoint *overlap;

  /* Sanity check.  */
  gdb_assert (cuda_get_autostep_pending ());

  /* The device was stepped.  Check what has changed and update data
     accordingly.  */

  /* Check if logical coordinates are still valid and update physical
     coordinates. If logical coordinates are not valid, warp ran to
     completion. */

  /* Make sure we can continue stepping this warp */
  if (cuda_coords_complete_physical (&c)
      || !cuda_focus_is_device ()
      || !warp_is_valid (c.dev, c.sm, c.wp))
    {
      /* It looks like this warp ran to completion or became invalid.  Switch
	 to the next valid one.  */
	cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
			   "Autostep: warp complete! Previous was: tId=(%d,%d,%d) bId=(%d,%d,%d)",
			   c.threadIdx.x, c.threadIdx.y, c.threadIdx.y,
			   c.blockIdx.x, c.blockIdx.y, c.blockIdx.z);

      if (!select_next_valid_warp ())
	return 1;

      /* Set things up to we can autostep the warp that is currently
	 selected.  */
      set_next_device_iteration ();
      return 0;
    }

  /* The warp did not run to completion.  Continue handling this warp.  */

  c = astep_state.data.device.cur_coords;
  before_pc = astep_state.cur_pc;
  end_pc = astep_state.end_pc;
  nsteps = astep_state.data.device.nsteps;
  lines = astep_state.data.device.lines;
  remaining = astep_state.remaining;
  single_inst = astep_state.insn_stepping;
  before_sal = astep_state.cur_sal;
  before_ln = astep_state.data.device.cur_ln;
  inst_size = astep_state.data.device.inst_size;

  /* If the lane is not active, keep stepping it until it is.  Keep all the
     data unchanged until the lane becomes active.  */
  if (!lane_is_active (c.dev, c.sm, c.wp, c.ln))
    return 0;

  /* Fetch the updated PC for the active warp.  Also fetch its line number
     information.  */
  after_pc = warp_get_active_virtual_pc (c.dev, c.sm, c.wp);
  after_sal = find_pc_line(after_pc, 0);

  cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                     "Autostep: issued single step %d steps (from %llx to %llx).",
                     nsteps, before_pc, end_pc);

  /* Find out how many lines/nsteps were actually stepped */
  if (nsteps > 1)
    {
      /* We were supposed to instruction-step multiple instructions at
	 once.  */
      nsteps = count_instructions (before_pc, after_pc);
      lines = count_lines (before_pc, after_pc, inst_size);
    }
  else /* Control flow instruction */
    {
      /* We were sitting at a control flow instruction, so we needed to
	 instruction-step only once to see where that instruction
	 would take us.  */
      gdb_assert(nsteps == 1);

      /* Calculate lines - if no line information exists treat it as one
	 instruction */
      lines = !(before_sal.symtab && before_sal.line) ||
              !(after_sal.symtab && after_sal.line) ||
              before_sal.line != after_sal.line;
    }

  /* Update the number of remaining instructions/lines we must step through.  */
  remaining -= single_inst ? nsteps : lines;

  cuda_trace_domain (CUDA_TRACE_BREAKPOINT,
                     "Autostep: in fact single stepped %d steps / %d lines (%d %s left). "
                     "PC after is %llx (%d).",
                     nsteps, lines, remaining, single_inst ? "instructions" : "lines",
                     (unsigned long long)after_pc, after_sal.line);

  /* We are done updating things.  Check if we are done with the autostep
     range.  */
  overlap = cuda_find_autostep_by_addr (after_pc);

  /* Check if we have an overlapping autostep region that we should ignore.  */
  check_overlapping_astep (after_pc, remaining);

  if (remaining <= 0)
    {
      /* We are done with the autostep range.  Check if there is an adjacent
	 autostep region we should handle.  */
      if (overlap)
	{
	  /* There is an adjacent autostep region.  Proceed to handle it without
	     switching focus to a different warp.  We will get back to the other
	     warps later on.  */
	  cuda_iterator iter = astep_state.data.device.iter;
	  initialize_autostep_state (after_pc);
	  astep_state.data.device.iter = iter;
	  astep_state.data.device.cur_coords = c;
	}
      else
	{
	  /* We are done with this autostep region and there are no adjacent
	     autostep regions to handle.  Just switch to the next valid warp
	     and set it up so we can autostep it through the autostep
	     region.  */

	  /* Select the next valid warp.  */
	  if (!select_next_valid_warp ())
	    return 1;
	}
    }
  else
    {
      /* Update the remaining data for the existing autostep and warp.  */
      astep_state.remaining = remaining;
    }

  /* Set the next iteration for the currently-selected warp.  */
  set_next_device_iteration ();

  return 0;
}

/* Return true if we should continue autostepping, false otherwise.  */

static bool
should_autostep_p (struct autostep_state *as)
{
  struct breakpoint *astep;

  gdb_assert (as);

  astep = cuda_find_autostep_by_addr (as->start_pc);

  if (astep == NULL)
    return false;

  if (astep->enable_state != bp_enabled)
    return false;

  return true;
}

/* See cuda-autostep.h */

int
cuda_update_autostep_state (CORE_ADDR pc)
{
  int status;

  /* Check if we should continue autostepping before we go about updating the
     state.  The user may have disabled an autostep we were currently handling
     or even deleted it.  */
  if (!should_autostep_p (&astep_state))
    return 1;

  if (cuda_focus_is_device ())
    status = update_device_autostep_state (pc);
  else
    status = update_host_autostep_state (pc);

  if (status != 0)
    cuda_cleanup_autostep_state ();

  return status;
}

/* See cuda-autostep.h */

int
cuda_cleanup_autostep_state (void)
{
  if (cuda_focus_is_device ())
    cuda_iterator_destroy (astep_state.data.device.iter);

  cuda_set_autostep_pending (false);
  return 0;
}

/* See cuda-autostep.h */

void
cuda_autostep_print_exception (void)
{
  struct thread_info *tp = NULL;

  if (target_has_execution && !(inferior_ptid.pid () == 0))
    tp = inferior_thread ();

  if (tp && signal_pass_state (tp->suspend.stop_signal))
    {
      /* This is an exception */
      if (cuda_focus_is_device ())
	autostep_report_exception_device (astep_state.data.device.nsteps,
					  astep_state.data.device.cur_ln,
					  astep_state.cur_pc,
					  tp->suspend.stop_pc);
      else
	autostep_report_exception_host (astep_state.cur_pc);

      /* We are done with autostepping.  */
      cuda_cleanup_autostep_state ();
    }
}
