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

/*Warning: this isn't intended as a standalone compile module! */

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#ifndef __ANDROID__
#include <sys/signal.h>
#else
#include <signal.h>
#endif
#include <defs.h>
#include <time.h>
#include <objfiles.h>
#include "block.h"
#include "gdbthread.h"
#include "language.h"
#include "demangle.h"
#include "regcache.h"
#include "arch-utils.h"
#include "buildsym.h"
#include "dictionary.h"
#include "cuda-commands.h"
#include "cuda-events.h"
#include "cuda-exceptions.h"
#include "cuda-notifications.h"
#include "cuda-options.h"
#include "cuda-tdep.h"
#include "cuda-parser.h"
#include "cuda-state.h"
#include "cuda-utils.h"
#include "cuda-packet-manager.h"
#include "cuda-convvars.h"
#include "inferior.h"
#include "valprint.h"
#include "command.h"
#include "gdbcmd.h"
#include "observable.h"
#if defined(__linux__) && defined(GDB_NM_FILE)
#include "linux-nat.h"
#endif
#include "inf-child.h"
#include "cuda-linux-nat.h"
#include "top.h"
#include "event-top.h"

template <class BaseTarget>
cuda_nat_linux<BaseTarget>::cuda_nat_linux ()
{
  char shortname[128];
  char longname[128];
  char doc[256];

  /* Build the meta-data strings without using malloc */
  strncpy (shortname, BaseTarget::info ().shortname, sizeof (shortname) - 1);
  strncat (shortname, " + cuda", sizeof (shortname) - 1 - strlen (shortname));
  strncpy (longname, BaseTarget::info ().longname, sizeof (longname) - 1);
  strncat (longname, " + CUDA support", sizeof (longname) - 1 - strlen (longname));
  strncpy (doc, BaseTarget::info ().doc, sizeof (doc) - 1);
  strncat (doc, " with CUDA support", sizeof (doc) - 1 - strlen (doc));
  
  m_info.shortname = xstrdup (shortname);
  m_info.longname = xstrdup (N_(longname));
  m_info.doc = xstrdup (N_(doc));
}

template <class BaseTarget>
void
cuda_nat_linux<BaseTarget>::open (const char *args, int from_tty)
{
  BaseTarget::open (args, from_tty);
}

/* The whole Linux siginfo structure is presented to the user, but, internally,
   only the si_signo matters. We do not save the siginfo object. Instead we
   save only the signo. Therefore any read/write to any other field of the
   siginfo object will have no effect or will return 0. */
template <class BaseTarget>
enum target_xfer_status
cuda_nat_linux<BaseTarget>::xfer_siginfo (enum target_object object,
					  const char *annex, gdb_byte *readbuf,
					  const gdb_byte *writebuf, ULONGEST offset, LONGEST len,
					  ULONGEST *xfered_len)
{
  /* the size of siginfo is not consistent between ptrace and other parts of
     GDB. On 32-bit Linux machines, the layout might be 64 bits. It does not
     matter for CUDA because only signo is used and the rest is set to zero. We
     just allocate 8 extra bytes and bypass the issue. On 64-bit Mac, the
     difference is 24 bytes. Therefore take the max of the 2 values. */
  gdb_byte buf[sizeof (siginfo_t) + 24];
  siginfo_t *siginfo = (siginfo_t *) buf;

  gdb_assert (object == TARGET_OBJECT_SIGNAL_INFO);
  gdb_assert (readbuf || writebuf);

  if (!cuda_focus_is_device ())
    return TARGET_XFER_E_IO;

  if (offset >= sizeof (buf))
    return TARGET_XFER_E_IO;

  if (offset + len > sizeof (buf))
    len = sizeof (buf) - offset;

  memset (buf, 0 , sizeof buf);

  if (readbuf)
    {
      siginfo->si_signo = cuda_get_signo ();
      memcpy (readbuf, siginfo + offset, len);
    }
  else
    {
      memcpy (siginfo + offset, writebuf, len);
      cuda_set_signo (siginfo->si_signo);
    }

  *xfered_len = len;
  return TARGET_XFER_OK;
}

template <class BaseTarget>
enum target_xfer_status
cuda_nat_linux<BaseTarget>::xfer_partial (enum target_object object, const char *annex,
					  gdb_byte *readbuf, const gdb_byte *writebuf,
					  ULONGEST offset, ULONGEST len, ULONGEST *xfered_len)
{
  enum target_xfer_status status = TARGET_XFER_E_IO;
  uint32_t dev, sm, wp, ln;
  *xfered_len = 0;

  /* Either readbuf or writebuf must be a valid pointer */
  gdb_assert (readbuf != NULL || writebuf != NULL);

  /* If focus is not set on device, call the host routines directly */
  if (!cuda_focus_is_device ())
    {
      status = BaseTarget::xfer_partial (object, annex, readbuf,
					 writebuf, offset, len,
					 xfered_len);
#ifdef __arm__
      /*
       * FIXME - Temporary workaround for mmap()/ptrace() issue.
       * If xfer partial targets object other than memory and error is hit,
       * return right away to let cuda-gdb return the right error.
       */
       if (*xfered_len <= 0 && object != TARGET_OBJECT_MEMORY)
         return status;

       /*
       * If the host memory xfer operation fails (i.e. *xfered_len is 0),
       * fallthrough to see if the CUDA Debug API can access
       * the specified address.
       * This can happen with ordinary mmap'd allocations.
       */
      if (*xfered_len > 0)
	return status;
#else
      return status;
#endif
    }

  switch (object)
  {
    /* If focus is on GPU, still try to read the address using host routines,
       if it fails, see if this address is in pinned system memoryi, i.e. to
       system memory that was allocated by the inferior through the CUDA API */
    case TARGET_OBJECT_MEMORY:

      status = BaseTarget::xfer_partial (object, annex, readbuf,
					 writebuf, offset, len,
					 xfered_len);
      if (*xfered_len)
        return TARGET_XFER_OK;

      if (readbuf ? cuda_api_read_pinned_memory  (offset, readbuf, len) :
		    cuda_api_write_pinned_memory (offset, writebuf, len))
	{
	  *xfered_len = len;
	  return TARGET_XFER_OK;
	}

      /* If all else failed, try to read memory from the device.  */
      if (cuda_coords_get_current_physical (&dev, &sm, &wp, &ln))
        return TARGET_XFER_E_IO;

      if (readbuf)
        cuda_api_read_local_memory (dev, sm, wp, ln, offset, readbuf, len);
      else
        cuda_api_write_local_memory (dev, sm, wp, ln, offset, writebuf, len);

      *xfered_len = len;
      return TARGET_XFER_OK;

    /* The stack lives in local memory for ABI compilations. */
    case TARGET_OBJECT_STACK_MEMORY:

      if (cuda_coords_get_current_physical (&dev, &sm, &wp, &ln))
        return TARGET_XFER_E_IO;
      if (readbuf)
        cuda_api_read_local_memory (dev, sm, wp, ln, offset, readbuf, len);
      else
        cuda_api_write_local_memory (dev, sm, wp, ln, offset, writebuf, len);

      *xfered_len = len;
      return status;

    /* When stopping on the device, build a simple siginfo object */
    case TARGET_OBJECT_SIGNAL_INFO:

      return xfer_siginfo (object, annex, readbuf, writebuf,
			   offset, len, xfered_len);
  }

  /* Fallback to host routines for other types of memory objects */
  return BaseTarget::xfer_partial (object, annex, readbuf,
				   writebuf, offset, len,
				   xfered_len);
}

template <class BaseTarget>
void
cuda_nat_linux<BaseTarget>::kill (void)
{
   /* XXX potential race condition here. we kill the application, and will later kill
      the device when finalizing the API. Should split the process into smaller steps:
      kill the device, then kill the app, then kill the dispatcher thread, then free
      resources in gdb (cuda_cleanup). OR do one initialize/finalize of the API per
      gdb run. */
  cuda_cleanup ();
  if (cuda_exception_is_valid (cuda_exception))
    cuda_exception_reset (cuda_exception);

  BaseTarget::kill ();
}

/* We don't want to issue a full mourn if we've encountered a cuda exception,
   because the host application has not actually reported that it has
   terminated yet. */
template <class BaseTarget>
void
cuda_nat_linux<BaseTarget>::mourn_inferior ()
{
  /* Mark breakpoints uninserted in case something tries to delete a
     breakpoint while we delete the inferior's threads (which would
     fail, since the inferior is long gone).  */
  mark_breakpoints_out ();

  cuda_cleanup ();
  if (cuda_exception_is_valid (cuda_exception))
    cuda_exception_reset (cuda_exception);

  BaseTarget::mourn_inferior ();
}

/* This discusses how CUDA device exceptions are handled.  This
   includes hardware exceptions that are detected and propagated
   through the Debug API.

   Similarly, we optionally integrate CUDA Memcheck with CUDA GDB
   (set cuda memcheck on), to get memory access checking in CUDA code.

   We adopt the host semantics such that a device exception will
   terminate the host application as well. This is the simplest option
   for now. For this purpose we use the boolean cuda_exception_is_valid
   (cuda_exception) to track the propagation of this event.

   This is a 3-step process:

   1. cuda_wait ()

     Device exception detected.  We indicate this process has "stopped"
     (i.e. is not yet terminated) with a signal.  We suspend the
     device, and allow a user to inspect their program for the reason
     why they hit the fault.  cuda_exception_is_valid (cuda_exception) is set
     to true at this point.

   2. cuda_resume ()

     The user has already inspected any and all state in the
     application, and decided to resume.  On host systems, you cannot
     resume your app beyond a terminal signal (the app dies).  So,
     since in our case the app doesn't die, we need to enforce this if
     we desire the same behavior.  This is done by seeeing if
     cuda_exception_is_valid (cuda_exception) is set to true.

   3. cuda_wait ()

     If cuda_exception_is_valid (cuda_exception) is set, then we know we've
     killed the app due to an exception.  We need to indicate the process has
     been "signalled" (i.e. app has terminated) with a signal.  At this point,
     cuda_exception_is_valid (cuda_exception) is set back to false.  Process
     mourning ensues and the world is a better place.
*/

/*CUDA_RESUME:

  For the meaning and interaction of ptid and sstep, read gnu-nat.c,
  line 1924.

  The actions of cuda_resume are based on 3 inputs: sstep, host_sstep
  and cuda_focus_is_device(). The actions are summarized in this
  table. 'sstep/resume dev' means single-stepping/resuming the device
  in focus if any, respectively.  'resume other dev' means resume any
  active device that is not in focus.

      device   sstep sstep | sstep   resume   resume    resume    sstep
      focus           host |  dev     dev    other dev   host      host
      ------------------------------------------------------------------
        0        0     0   |   0       1         1        1(b)      0
        0        0     1   |   0       0         0        1(c)      0
        0        1     0   |   -       -         -        -         -
        0        1     1   |   0       0         0        0         1
        1        0     0   |   0       1         1        1         0
        1        0     1   |   0       1         1        1         0
        1        1     0   |   1       0         0        0(a)      0
        1        1     1   |   1       0         0        0(a)      0

     (a) because we fake single-stepping to GDB by not calling the
     wait() routine, there is no need to resume the host. We used to
     resume the host so that the host could capture any SIGTRAP signal
     sent during single-stepping.

     (b) currently, there is no way to resume a single device, without
     resuming the rest of the world. That would lead to a deadlock.

     (c) In case host is resumed to simulate a single stepping,
     device should remain suspended.
*/
template <class BaseTarget>
void
cuda_nat_linux<BaseTarget>::resume (ptid_t ptid, int sstep, int host_sstep, enum gdb_signal ts)
{
  uint32_t dev;

  cuda_sstep_reset (sstep);

  // Is focus on host?
  if (!cuda_focus_is_device())
    {
      // If not sstep - resume devices
      if (!host_sstep)
        for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
            device_resume (dev);

      // resume the host
      BaseTarget::resume (ptid, sstep, ts);
      return;
    }

   // sstep the device
  if (sstep)
    {
      if (cuda_sstep_execute (inferior_ptid)) {
        device_invalidate (cuda_current_device ()); //iluvatar: device_invalidate after sstep
        return;
      }
      /* If single stepping failed, plant a temporary breakpoint
         at the previous frame and resume the device */
      if (cuda_options_software_preemption ())
        {
          /* Physical coordinates might change even if API call has failed
           * if software preemption is enabled */
          int rc;
          uint64_t kernel_id, grid_id;
          CuDim3 block_idx, thread_idx;

          rc = cuda_coords_get_current_logical (&kernel_id, &grid_id, &block_idx, &thread_idx);
          if (rc)
            error (_("Failed to get current logical coordinates on GPU!"));
          /* Invalidate current coordinates as well as device cache */
          device_invalidate (cuda_current_device ());
          cuda_coords_invalidate_current ();
          rc = cuda_coords_set_current_logical (kernel_id, grid_id, block_idx, thread_idx);
          if (rc)
            error (_("Failed to find physical coordinates matching logical ones!"));
        }
      cuda_sstep_reset (false);
      insert_step_resume_breakpoint_at_caller (get_current_frame ());
      cuda_insert_breakpoints ();
    }

  // resume the device
  device_resume (cuda_current_device ());

  // resume other devices
  if (!cuda_notification_pending ())
    for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
      if (dev != cuda_current_device ())
        device_resume (dev);

  // resume the host
  BaseTarget::resume (ptid, 0, ts);
}

template <class BaseTarget>
void
cuda_nat_linux<BaseTarget>::resume (ptid_t ptid, int sstep, enum gdb_signal ts)
{
  bool cuda_event_found = false;
  int host_want_sstep = cuda_host_want_singlestep;
  CUDBGEvent event;

  /* We're about to resume - so flush the device-disassembly disasm cache.
     ELF disassembly is kept as the ELF file contents do not change.
     See cuda-asm.c */
  cuda_system_flush_disasm_cache ();

  cuda_trace ("cuda_resume: sstep=%d", sstep);
  cuda_host_want_singlestep = 0;

  if (!cuda_options_device_resume_on_cpu_dynamic_function_call () &&
      inferior_thread ()->control.in_infcall)
    {
      BaseTarget::resume (ptid, 0, ts);
      return;
    }

  /* In cuda-gdb we have two types of device exceptions :
     Recoverable : CUDA_EXCEPTION_WARP_ASSERT
     Nonrecoverable : All others (e.g. CUDA_EXCEPTION_LANE_ILLEGAL_ADDRESS)

     The main difference is that a recoverable exception ensures that device
     state is consistent. Therefore, the user can request that the device
     continue execution. Currently, CUDA_EXCEPTION_WARP_ASSERT is the only
     recoverable exception.

     When a device side exception is hit, it sets cuda_exception in cuda_wait.
     In the case of a nonrecoverable exception, the cuda_resume call
     kills the host application and return early. The subsequent cuda_wait
     call cleans up the exception state.
     In the case of a recoverable exception, cuda-gdb must reset the exception
     state here and can then continue executing.
     In the case of CUDA_EXCEPTION_WARP_ASSERT, the handling of the
     exception (i.e. printing the assert message) is done as part of the
     cuda_wait call.
  */
  if (cuda_exception_is_valid (cuda_exception) &&
      !cuda_exception_is_recoverable (cuda_exception))
    {
      cuda_cleanup ();
      cuda_exception_reset (cuda_exception);
      BaseTarget::resume (ptid, 0, GDB_SIGNAL_KILL);
      cuda_trace ("cuda_resume: exception found");
      return;
    }

  if (cuda_exception_is_valid (cuda_exception) &&
      cuda_exception_is_recoverable (cuda_exception))
    {
      cuda_exception_reset (cuda_exception);
      cuda_trace ("cuda_resume: recoverable exception found\n");
    }

  /* We have now handled all the CUDA notifications. We are ready to
     handle the next batch when the world resumes. Pending CUDA
     timeout events will be ignored until next time. */
  cuda_notification_mark_consumed ();
  cuda_sigtrap_restore_settings ();

  /* Check if a notification was received while a previous event was being
     serviced. If yes, check the event queue for a pending event, and service
     the event if one is found. */
  if (cuda_notification_aliased_event ())
    {
      cuda_notification_reset_aliased_event ();
      cuda_api_get_next_sync_event (&event);
      cuda_event_found = event.kind != CUDBG_EVENT_INVALID;

      if (cuda_event_found)
        {
          cuda_process_events (&event, CUDA_EVENT_SYNC);
          sendAck = true;
        }
    }

  /* Acknowledge the CUDA debugger API (for synchronous events) */
  if (sendAck)
    {
      cuda_api_acknowledge_sync_events ();
      sendAck = false;
    }

  resume (ptid, sstep, host_want_sstep, ts);

  cuda_clock_increment ();
  cuda_trace ("cuda_resume: done");
}

/*CUDA_WAIT:

  The wait function freezes the world, update the cached information
  about the CUDA devices, and cook the wait_status.

  If we hit a SIGTRAP because of a debugger API event, qualify the
  signal as spurious, so that GDB ignores it.

  If we are single-stepping the device, we never resume the host. But
  GDB needs to believe a SIGTRAP has been received. We fake the
  target_wait_status accordingly. If we are stepping instruction
  (stepi, not step), GDB cannot guarantee that there is an actual next
  instruction (unlike stepping a source line). If the kernel dies, we
  have to recognize the case.

  Device exceptions (including memcheck violations) are presented to GDB as
  unique signals (defined in signal.[c,h]).  Everything else is presented
  as a SIGTRAP, spurious in the case of a debugger API event.
 */
template <class BaseTarget>
ptid_t
cuda_nat_linux<BaseTarget>::wait (ptid_t ptid,
				  struct target_waitstatus *ws,
				  int target_options)
{
  ptid_t r;
  uint32_t dev, dev_id;
  uint64_t grid_id;
  kernel_t kernel;
  bool cuda_event_found = false;
  CUDBGEvent event, asyncEvent;
  struct thread_info *tp;
  cuda_coords_t c;

  cuda_trace ("cuda_wait");

  if (!cuda_options_device_resume_on_cpu_dynamic_function_call () &&
      inferior_thread ()->control.in_infcall)
    return BaseTarget::wait (ptid, ws, target_options);

  if (cuda_exception_is_valid (cuda_exception))
    {
      ws->kind = TARGET_WAITKIND_SIGNALLED;
      ws->value.sig = (enum gdb_signal) cuda_exception_get_value (cuda_exception);
      cuda_exception_reset (cuda_exception);
      cuda_trace ("cuda_wait: exception found");
      return inferior_ptid;
    }
  else if (cuda_sstep_is_active ())
    {
      /* Cook the ptid and wait_status if single-stepping a CUDA device. */
      cuda_trace ("cuda_wait: single-stepping");
      r = cuda_sstep_ptid ();

      /* When stepping the device, the host process remains suspended.
         So, if the user issued a Ctrl-C, we wouldn't detect it since
         we never actually check its wait status.  We must explicitly
         check for a pending SIGINT here.
         if quit_flag is set then C-c was pressed in gdb session
         but signal was yet not forwarded to debugged process */
      if (cuda_check_pending_sigint (r.pid ()) || check_quit_flag() )
        {
          ws->kind = TARGET_WAITKIND_STOPPED;
          ws->value.sig = GDB_SIGNAL_INT;
          cuda_set_signo (GDB_SIGNAL_INT);
        }
      else
        {
          ws->kind = TARGET_WAITKIND_STOPPED;
          ws->value.sig = GDB_SIGNAL_TRAP;
          cuda_set_signo (GDB_SIGNAL_TRAP);

          /* If we single stepped the last warp on the device, then the
             launch has completed.  However, we do not see the event for
             kernel termination until we resume the application.  We must
             explicitly handle this here by indicating the kernel has
             terminated and switching to the remaining host thread. */
          if (cuda_sstep_kernel_has_terminated ())
            {
              /* Only destroy the kernel that has been stepped to its exit */
              dev_id  = cuda_sstep_dev_id ();
              grid_id = cuda_sstep_grid_id ();
              kernel = kernels_find_kernel_by_grid_id (dev_id, grid_id);
              kernels_terminate_kernel (kernel);

              /* Invalidate current coordinates and device state */
              cuda_coords_invalidate_current ();
              device_invalidate (dev_id);

              /* Consume any asynchronous events, if necessary.  We need to do
                 this explicitly here, since we're taking the quick path out of
                 this routine (and bypassing the normal check for API events). */
              cuda_api_get_next_async_event (&asyncEvent);
              if (asyncEvent.kind != CUDBG_EVENT_INVALID)
                cuda_process_events (&asyncEvent, CUDA_EVENT_ASYNC);

              /* Update device state/kernels */
              kernels_update_terminated ();
              cuda_update_convenience_variables ();

              switch_to_thread (r);
              tp = inferior_thread ();
              tp->control.step_range_end = 1;
              return r;
            }
        }
    }
  else
    {
      cuda_coords_invalidate_current ();
      cuda_trace ("cuda_wait: host_wait");
      r = BaseTarget::wait (ptid, ws, target_options);
      cuda_trace ("cuda_wait: host_wait done");

      /* GDB reads events asynchronously without blocking. The target may have
	 taken too long to reply and GDB did not get any events back.  Check if
	 this is the case and just return.  */
      if (ws->kind == TARGET_WAITKIND_IGNORE
	  || ws->kind == TARGET_WAITKIND_NO_RESUMED)
	return r;

      if (ws->kind == TARGET_WAITKIND_STOPPED &&
	  ws->value.sig == GDB_SIGNAL_0)
	{
	  /* GDB is trying to stop this thread.  Let it do it.  */
	  return r;
	}
    }

  /* Immediately detect if the inferior is exiting.
     In these situations, do not investigate the device. */
  if (ws->kind == TARGET_WAITKIND_EXITED)
    {
      cuda_trace ("cuda_wait: target is exiting, avoiding device inspection");
      return r;
    }

  /* Return if cuda has not been initialized yet */
  if (!cuda_initialize_target ())
    return r;

  /* Suspend all the CUDA devices. */
  cuda_trace ("cuda_wait: suspend devices");
  for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
    device_suspend (dev);

  /* Check for asynchronous events.  These events do not require
     acknowledgement to the debug API, and may arrive at any time
     without an explicit notification. */
  cuda_api_get_next_async_event (&asyncEvent);
  if (asyncEvent.kind != CUDBG_EVENT_INVALID)
    cuda_process_events (&asyncEvent, CUDA_EVENT_ASYNC);

  /* Analyze notifications.  Only check for new events if we've
     we've received a notification, or if we're single stepping
     the device (since if we're stepping we wouldn't receive an
     explicit notification). */
  tp = inferior_thread ();
  cuda_notification_analyze (r, ws, tp->control.trap_expected);
  if (cuda_notification_received ())
    {
      /* Check if there is any CUDA event to be processed */
      cuda_api_get_next_sync_event (&event);
      cuda_event_found = event.kind != CUDBG_EVENT_INVALID;
    }

  /* Handle all the CUDA events immediately.  In particular, for
     GPU events that may happen without prior notification (GPU
     grid launches, for example), API events will be packed
     alongside of them, so we need to process the API event first. */
  if (cuda_event_found)
    {
      cuda_process_events (&event, CUDA_EVENT_SYNC);
      sendAck = true;
    }

  /* Update the info about the kernels */
  kernels_update_terminated ();

  /* Decide which thread/kernel to switch focus to. */
  if (cuda_exception_hit_p (cuda_exception))
    {
      cuda_trace ("cuda_wait: stopped because of an exception");
      c = cuda_exception_get_coords (cuda_exception);
      cuda_coords_set_current (&c);
      cuda_exception_print_message (cuda_exception);
      ws->kind = TARGET_WAITKIND_STOPPED;
      ws->value.sig = (enum gdb_signal) cuda_exception_get_value (cuda_exception);
      cuda_set_signo (cuda_exception_get_value (cuda_exception));
    }
  else if (cuda_sstep_is_active ())
    {
      cuda_trace ("cuda_wait: stopped because we are single-stepping");
      cuda_coords_update_current (false, false);
    }
  else if (cuda_breakpoint_hit_p (cuda_clock ()))
    {
      cuda_trace ("cuda_wait: stopped because of a breakpoint");
      /* Alias received signal to SIGTRAP when hitting a trap */
      cuda_set_signo (GDB_SIGNAL_TRAP);
      ws->value.sig = GDB_SIGNAL_TRAP;
      cuda_coords_update_current (true, false);
    }
  else if (cuda_system_is_broken (cuda_clock ()))
    {
      cuda_trace ("cuda_wait: stopped because there are broken warps (induced trap?)");
      /* Alias received signal to SIGTRAP when hitting a breakpoint */
      cuda_set_signo (GDB_SIGNAL_TRAP);
      ws->value.sig = GDB_SIGNAL_TRAP;
      cuda_coords_update_current (false, false);
    }
  else if (cuda_api_get_attach_state () == CUDA_ATTACH_STATE_APP_READY)
    {
      /* Finished attaching to the CUDA app.
         Preferably switch focus to a device if possible */
      struct inferior *inf = find_inferior_pid (r.pid ());
      cuda_trace ("cuda_wait: stopped because we attached to the CUDA app");
      cuda_api_set_attach_state (CUDA_ATTACH_STATE_COMPLETE);
      inf->control.stop_soon = STOP_QUIETLY;
      cuda_coords_update_current (false, false);
    }
  else if (cuda_api_get_attach_state () == CUDA_ATTACH_STATE_DETACH_COMPLETE)
    {
      /* Finished detaching from the CUDA app. */
      struct inferior *inf = find_inferior_pid (r.pid ());
      cuda_trace ("cuda_wait: stopped because we detached from the CUDA app");
      inf->control.stop_soon = STOP_QUIETLY;
    }
  else if (ws->value.sig == GDB_SIGNAL_INT)
    {
      /* CTRL-C was hit. Preferably switch focus to a device if possible */
      cuda_trace ("cuda_wait: stopped because a SIGINT was received.");
      cuda_set_signo (GDB_SIGNAL_INT);
      cuda_coords_update_current (false, false);
    }
  else if (check_quit_flag ())
    {
      /* cuda-gdb received sigint, probably Nsight tries to stop the app. */
      cuda_trace ("cuda_wait: stopped because SIGINT was received by debugger.");
      ws->kind = TARGET_WAITKIND_STOPPED;
      ws->value.sig = GDB_SIGNAL_INT;
      cuda_set_signo (GDB_SIGNAL_INT);
      cuda_coords_update_current (false, false);
    }
  else if (cuda_event_found)
    {
      cuda_trace ("cuda_wait: stopped because of a CUDA event");
      cuda_sigtrap_set_silent ();
      cuda_coords_update_current (false, false);
    }
  else if (cuda_notification_received ())
    {
      /* No reason found when actual reason was consumed in a previous iteration (timeout,...) */
      cuda_trace ("cuda_wait: stopped for no visible CUDA reason.");
      cuda_set_signo (GDB_SIGNAL_TRAP); /* Dummy signal. We stopped after all. */
      cuda_coords_invalidate_current ();
    }
  else
    {
      cuda_trace ("cuda_wait: stopped for a non-CUDA reason.");
      cuda_set_signo (GDB_SIGNAL_TRAP);
      cuda_coords_invalidate_current ();
    }

  cuda_adjust_host_pc (r);

  /* CUDA - managed memory */
  if (ws->kind == TARGET_WAITKIND_STOPPED &&
      (ws->value.sig == GDB_SIGNAL_BUS || ws->value.sig == GDB_SIGNAL_SEGV))
    {
      uint64_t addr = 0;
      struct gdbarch *arch = get_current_arch();
      int arch_ptr_size = gdbarch_ptr_bit (arch) / 8;
      LONGEST len = arch_ptr_size;
      LONGEST offset = arch_ptr_size == 8 ? 0x10 : 0x0c;
      LONGEST read = 0;
      gdb_byte *buf = (gdb_byte *)&addr;
      int inf_exec = is_executing (inferior_ptid);

      /* Mark inferior_ptid as not executing while reading object signal info*/
      set_executing (inferior_ptid, 0);
      read = target_read (this, TARGET_OBJECT_SIGNAL_INFO, NULL, buf, offset, len);
      set_executing (inferior_ptid, inf_exec);

      /* Check the results */
      if (read == len && cuda_managed_address_p (addr))
        {
          ws->value.sig = GDB_SIGNAL_CUDA_INVALID_MANAGED_MEMORY_ACCESS;
          cuda_set_signo (ws->value.sig);
        }
    }
  cuda_managed_memory_clean_regions();

  /* Switch focus and update related data */
  cuda_update_convenience_variables ();
  if (cuda_focus_is_device ())
    /* Must be last, once focus and elf images have been updated */
    switch_to_cuda_thread (NULL);

  cuda_trace ("cuda_wait: done");
  return r;
}

template <class BaseTarget>
void
cuda_nat_linux<BaseTarget>::fetch_registers (struct regcache *regcache,
					     int regno)
{
  uint64_t val = 0;
  cuda_coords_t c;
  struct gdbarch *gdbarch = get_regcache_arch (regcache);
  uint32_t pc_regnum = gdbarch_pc_regnum (gdbarch);
  enum register_status status;

  /* delegate to the host routines when not on the device */
  if (!cuda_focus_is_device ())
    {
      BaseTarget::fetch_registers (regcache, regno);
      return;
    }

  cuda_coords_get_current (&c);

  /* if all the registers are wanted, then we need the host registers and the
     device PC */
  if (regno == -1)
    {
      BaseTarget::fetch_registers (regcache, regno);
      val = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
      regcache->raw_supply (pc_regnum, &val);
      return;
    }

  /* get the PC */
  if (regno == pc_regnum )
    {
      val = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
      regcache->raw_supply (pc_regnum, &val);
      return;
    }

  if (cuda_regular_register_p (gdbarch, regno))
    {
      /* raw register */
      val = lane_get_register (c.dev, c.sm, c.wp, c.ln, regno);
      regcache->raw_supply (regno, &val);
      return;
    }

  status = cuda_pseudo_register_read (gdbarch, regcache, regno, (gdb_byte *)&val);
  gdb_assert (status == REG_VALID);
  
  regcache->raw_supply (regno, &val);
}

template <class BaseTarget>
void
cuda_nat_linux<BaseTarget>::store_registers (struct regcache *regcache,
					     int regno)
{
  uint64_t val = 0;
  struct gdbarch *gdbarch = get_regcache_arch (regcache);
  int num_regs = gdbarch_num_regs (gdbarch);

  gdb_assert (regno >= 0 && regno < num_regs);

  if (!cuda_focus_is_device ())
    {
      BaseTarget::store_registers (regcache, regno);
      return;
    }

  regcache->raw_collect (regno, &val);
  cuda_register_write (gdbarch, regcache, regno, (gdb_byte *) &val);
}

template <class BaseTarget>
int
cuda_nat_linux<BaseTarget>::insert_breakpoint (struct gdbarch *gdbarch,
					       struct bp_target_info *bp_tgt)
{
  uint32_t dev;
  bool inserted;

  gdb_assert (bp_tgt->owner != NULL ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_arm ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_aarch64);

  if (!bp_tgt->owner || !bp_tgt->owner->cuda_breakpoint)
    return BaseTarget::insert_breakpoint (gdbarch, bp_tgt);

  /* Insert the breakpoint on whatever device accepts it (valid address). */
  inserted = false;
  for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
    {
      inserted |= cuda_api_set_breakpoint (dev, bp_tgt->reqstd_address);
    }

  /* Make sure we save the address where the actual breakpoint was placed.  */
  if (inserted)
    bp_tgt->placed_address = bp_tgt->reqstd_address;

  return !inserted;
}

template <class BaseTarget>
int
cuda_nat_linux<BaseTarget>::remove_breakpoint (struct gdbarch *gdbarch,
					       struct bp_target_info *bp_tgt,
					       enum remove_bp_reason reason)
{
  uint32_t dev;
  bool removed;

  gdb_assert (bp_tgt->owner != NULL ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_arm ||
              gdbarch_bfd_arch_info (gdbarch)->arch == bfd_arch_aarch64);

  if (!bp_tgt->owner || !bp_tgt->owner->cuda_breakpoint)
    return BaseTarget::remove_breakpoint (gdbarch, bp_tgt, reason);

  /* Removed the breakpoint on whatever device accepts it (valid address). */
  removed = false;
  for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
    {
      /* We need to remove breakpoints even if no kernels remain on the device */
      removed |= cuda_api_unset_breakpoint (dev, bp_tgt->placed_address);
    }
  return !removed;
}

template <class BaseTarget>
struct gdbarch *
cuda_nat_linux<BaseTarget>::thread_architecture (ptid_t ptid)
{
  if (cuda_focus_is_device ())
    return cuda_get_gdbarch ();
  else
    return target_gdbarch ();
}

template <class BaseTarget>
void
cuda_nat_linux<BaseTarget>::detach (inferior *inf, int from_tty)
{
  /* If the Debug API is not initialized,
   * treat the inferior as a host-only process */
  if (cuda_api_get_state () == CUDA_API_STATE_INITIALIZED)
    cuda_do_detach (false);

  /* Do not try to detach from an already dead process */
  if (inferior_ptid.pid () == 0)
    return;

  /* Call the host detach routine. */
  BaseTarget::detach (inf, from_tty);
}
