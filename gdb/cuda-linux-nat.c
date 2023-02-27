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
#include "buildsym-legacy.h"
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
#include "inf-loop.h"
#include "cuda-linux-nat.h"
#include "top.h"
#include "event-top.h"

#define CUDA_NUM_CUDART_FRAME_ENTRIES 3

#if !defined(__APPLE__)
extern struct lwp_info *find_lwp_pid (ptid_t ptid);
#endif

bool sendAck = false;
bool cuda_debugging_enabled = false;

struct cuda_cudart_symbols_st cuda_cudart_symbols;
static struct cuda_signal_info_st cuda_sigtrap_info;

/* Provide a prototype to silence -Wmissing-prototypes.  */
extern initialize_file_ftype _initialize_cuda_nat;

target_info cuda_nat_linux_target_info = {
  NULL,
  NULL,
  NULL
};

#if defined(__APPLE__)
static void smuggle_dyld_settings (struct gdb_environ *e);
#endif

#if defined(__linux__) && defined(GDB_NM_FILE)
static void
cuda_clear_pending_sigint (pid_t pid)
{
  int status = 0, options = 0;
  ptrace (PTRACE_CONT, pid, 0, 0); /* Resume the host to consume the pending SIGINT */
  waitpid (pid, &status, options); /* Ensure we return for the right reason */
  gdb_assert (WIFSTOPPED (status) && WSTOPSIG (status) == SIGINT);
}
#endif

int
cuda_check_pending_sigint (pid_t pid)
{
#if defined(__linux__) && defined(GDB_NM_FILE)
  sigset_t pending, blocked, ignored;

  linux_proc_pending_signals (pid, &pending, &blocked, &ignored);
  if (sigismember (&pending, SIGINT))
    {
      cuda_clear_pending_sigint (pid);
      return 1;
    }
#endif

   /* No pending SIGINT */
   return 0;
}

void
cuda_signal_set_silent (int sig, struct cuda_signal_info_st *save)
{
  enum gdb_signal gdb_sig = gdb_signal_from_host (sig);

  gdb_assert (save);
  gdb_assert (gdb_sig != GDB_SIGNAL_UNKNOWN);
  gdb_assert (cuda_options_stop_signal() != gdb_sig);

  save->stop = signal_stop_state (gdb_sig);
  save->print = signal_print_state (gdb_sig);
  save->saved = true;

  signal_stop_update (gdb_sig, 0);
  signal_print_update (gdb_sig, 0);
}

void
cuda_signal_restore_settings (int sig, struct cuda_signal_info_st *save)
{
  enum gdb_signal gdb_sig = gdb_signal_from_host (sig);

  gdb_assert(save);
  gdb_assert(gdb_sig != GDB_SIGNAL_UNKNOWN);
  gdb_assert(cuda_options_stop_signal() != gdb_sig);

  if (save->saved)
    {
      signal_stop_update (gdb_sig, save->stop);
      signal_print_update (gdb_sig, save->print);
      save->saved = false;
    }
}

void
cuda_sigtrap_set_silent (void)
{
  cuda_signal_set_silent(SIGTRAP, &cuda_sigtrap_info);
}

void
cuda_sigtrap_restore_settings (void)
{
  cuda_signal_restore_settings(SIGTRAP, &cuda_sigtrap_info);
}

/* If a host event is hit while there are valid threads
   on the GPU, the focus ends up being switched to the 
   GPU, leaving the host PC not rewound.

   This function determines if the host is at a breakpoint,
   and if so it manually rewinds the host PC so that the
   breakpoint can be hit again after a resume.
   r here is the return value of host_wait().
*/
void 
cuda_adjust_host_pc (ptid_t r)
{
  bool pc_rewound = false;
  struct regcache *regcache;
  CORE_ADDR pc;
  cuda_coords_t coords;

  if (!cuda_focus_is_device ())
    return;

  /* Rewind host PC and consume pending SIGTRAP
     Sometimes, one thread can hit both a host and a device
     breakpoint at the same time, in which case host SIGTRAP
     is triggered while SIGTRAP from back end is blocked (pending).
     When resuming, host PC is not rewound because focus is on the
     device.

     Before switching to CUDA thread, we check if that's the case.
     If so, manually rewind the host PC and consume the pending SIGTRAP.
     This allows the host breakpoint to be hit again after resuming. */

  /* r is guaranteed to be the return of host_wait in this case */
  cuda_coords_get_current (&coords);

  /* Temporarily invalidate the current coords so that the focus
     is set on the host. */
  cuda_coords_invalidate_current ();

  regcache = get_thread_arch_regcache (r, target_gdbarch());
  pc = regcache_read_pc (regcache) - gdbarch_decr_pc_after_break (target_gdbarch());
  if (breakpoint_inserted_here_p (regcache->aspace (), pc))
    {
        /* Rewind the PC */
      regcache_write_pc (regcache, pc);
      pc_rewound = true;
    }

  /* Restore coords */
  cuda_coords_set_current (&coords);

  /* Remove the pending notification if we rewound the pc */
  if (pc_rewound)
      cuda_notification_consume_pending ();
}

void
cuda_set_environment (void)
{
#if defined(__APPLE__)
  /* CUDA - cuda-gdb wrapper */
  smuggle_dyld_settings (&current_inferior ()->environment);
#endif
}

static bool
cuda_get_cudbg_api (void)
{
  CUDBGAPI api = NULL;
  CUDBGResult res;


  res = cudbgGetAPI (CUDBG_API_VERSION_MAJOR,
                     CUDBG_API_VERSION_MINOR,
                     CUDBG_API_VERSION_REVISION,

                     &api);
  if (res == CUDBG_SUCCESS)
    cuda_api_set_api (api);

  cuda_api_handle_get_api_error (res);

  return (res != CUDBG_SUCCESS);
}

void
_initialize_cuda_nat (void)
{
  /* Check the required CUDA debugger files are present */
  if (cuda_get_cudbg_api ())
    {
      warning ("CUDA support disabled: could not obtain the CUDA debugger API\n");
      cuda_debugging_enabled = false;
      return;
    }

  /* Initialize the CUDA modules */
  cuda_commands_initialize ();
  cuda_options_initialize ();
  cuda_notification_initialize ();

  /* Initialize the cleanup routines */
  make_final_cleanup (cuda_final_cleanup, NULL);

  cuda_debugging_enabled = true;
}

void
cuda_nat_attach (void)
{
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *cmd = NULL;
  const char *cudbgApiAttach = "(void) cudbgApiAttach()";
  CORE_ADDR debugFlagAddr;
  CORE_ADDR resumeAppOnAttachFlagAddr;
  unsigned char resumeAppOnAttach;
  unsigned int timeOut = 5000; // ms
  unsigned int timeElapsed = 0;
  unsigned dev = 0;
  bool need_retry = 0;
  unsigned retry_count = 0;
  unsigned retry_delay = 100; // ms
  unsigned app_init_timeout = 5000; // ms
  const unsigned int sleepTime = 1; // ms
  uint64_t internal_error_code;
  struct cleanup *cleanup = NULL;

  /* Return early if CUDA driver isn't available. Attaching to the host
     process has already been completed at this point. */
  cuda_api_set_attach_state (CUDA_ATTACH_STATE_IN_PROGRESS);
  if (!cuda_initialize_target ())
    {
      cuda_api_set_attach_state (CUDA_ATTACH_STATE_NOT_STARTED);
      return;
    }

   /* If the CUDA driver has been loaded but software preemption has been turned
      on, stop the attach process. */
   if (cuda_options_software_preemption ())
    {
       cuda_api_set_attach_state (CUDA_ATTACH_STATE_NOT_STARTED);
       error (_("Attaching to a running CUDA process with software preemption "
                "enabled in the debugger is not supported."));
    }


  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_("Failed to initiate attach."));

  cuda_sigtrap_set_silent ();

  do
  {
    /* Try to init debugger's backend */
    cleanup = cuda_gdb_bypass_signals ();
    cmd_func (cmd, cudbgApiAttach, 0);
    do_cleanups (cleanup);

    internal_error_code = cuda_get_last_driver_internal_error_code();

    /* CUDBG_ERROR_ATTACH_NOT_POSSIBLE can be returned in two scenarios:
     * 1. Attach is really not possible
     * 2. Critical section's mutex is taken, attaching would cause a deadlock */
    need_retry = (unsigned int)internal_error_code == CUDBG_ERROR_ATTACH_NOT_POSSIBLE;

    if (need_retry)
    {
      /* Resume the target */
      prepare_execution_command (current_top_target (), true);
      continue_1 (true);

      usleep(retry_delay * 1000);

      /* Trigger the future wait() */
      kill (current_inferior ()->pid, SIGTRAP);

      /* Get control back */
      wait_for_inferior ();
      set_running (minus_one_ptid, 0);

      retry_count++;
    }
  } while (need_retry && (retry_count * retry_delay < app_init_timeout));

  /* We are re-using the ATTACH_NOT_POSSIBLE error code for delayed attach,
   * therefore a timeout will allow us to determine if the error code is
   * genuinely not possible. */
  if (need_retry)
    error (_("Attaching not possible. "
             "Please verify that software preemption is disabled "
             "and that nvidia-cuda-mps-server is not running."));

  cuda_sigtrap_restore_settings ();

  if ((unsigned int)internal_error_code == CUDBG_ERROR_SOME_DEVICES_WATCHDOGGED)
     error (_("Attaching to process running on watchdogged GPU is not possible.\n"
              "Please repeat the attempt in console mode or "
              "restart the process with CUDA_VISIBLE_DEVICES environment variable set."));
  if (internal_error_code)
    error (_("Attach failed due to the internal driver error 0x%llx\n"),
            (unsigned long long) internal_error_code);

  debugFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));
  resumeAppOnAttachFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_RESUME_FOR_ATTACH_DETACH));

  /* If this is not available, the CUDA driver doesn't support attaching.  */
  if (resumeAppOnAttachFlagAddr == 0 || debugFlagAddr == 0)
    error (_("This CUDA driver does not support attaching to a running CUDA process."));

  /* Wait till the backend has started up and is ready to service API calls */
  while (cuda_api_initialize () != CUDBG_SUCCESS)
    {
      internal_error_code = cuda_get_last_driver_internal_error_code();

      if ((unsigned int)internal_error_code == CUDBG_ERROR_ATTACH_NOT_POSSIBLE)
         error (_("Attaching not possible. "
                  "Please verify that software preemption is disabled "
                  "and that nvidia-cuda-mps-server is not running."));
      if (internal_error_code)
        error (_("Attach failed due to the internal driver error 0x%llx\n"),
                (unsigned long long) internal_error_code);

      if (timeElapsed < timeOut)
        usleep(sleepTime * 1000);
      else
        error (_("Timed out waiting for the CUDA API to initialize."));

      timeElapsed += sleepTime;
    }

  /* Check if the inferior needs to be resumed */
  target_read_memory (resumeAppOnAttachFlagAddr, &resumeAppOnAttach, 1);

  if (resumeAppOnAttach)
    {
      int  cnt;
      cleanup = cuda_gdb_bypass_signals ();
      /* Resume the inferior to collect more data. CUDA_ATTACH_STATE_COMPLETE and
         CUDBG_IPC_FLAG_NAME will be set once this completes. */
      for (cnt=0;
	   cnt < 1000
           && cuda_api_get_attach_state () == CUDA_ATTACH_STATE_IN_PROGRESS;
           cnt++)
	  {
	      prepare_execution_command (current_top_target (), true);
	      continue_1 (false);
	      wait_for_inferior ();
	      normal_stop ();
	      /* infrun's async_event_handler is in the "ready" state after running `continue_1`
		 above. Since we've waited for inferior above, we now run the completions and
		 reset the "ready" state by calling the below function. Doing this will lead
		 to CUDA's wait function not being called after `cuda_nat_attach` completes. */
	      inferior_event_handler (INF_EXEC_COMPLETE, NULL);
	  }

      /* No threads are running at this point.  */
      set_running (minus_one_ptid, 0);

      do_cleanups (cleanup);
      if (cuda_api_get_attach_state () != CUDA_ATTACH_STATE_APP_READY &&
          cuda_api_get_attach_state () != CUDA_ATTACH_STATE_COMPLETE)
          error ("Unexpected CUDA attach state, further debugging session might be unreliable");
    }
  else
    {
      /* Enable debugger callbacks from the CUDA driver */
      cuda_write_bool (debugFlagAddr, true);

      /* No data to collect, attach complete. */
      cuda_api_set_attach_state (CUDA_ATTACH_STATE_COMPLETE);
    }

  /* Initialize CUDA and suspend the devices */
  cuda_initialize ();
  for (dev = 0; dev < cuda_system_get_num_devices (); ++dev)
    device_suspend (dev);

  /* The inferior just got signaled, we're not expecting any other stop */
  current_inferior ()->control.stop_soon = NO_STOP_QUIETLY;
}

void cuda_do_detach(bool remote)
{
  struct cmd_list_element *alias = NULL;
  struct cmd_list_element *prefix_cmd = NULL;
  struct cmd_list_element *cmd = NULL;
  const char *cudbgApiDetach = "(void) cudbgApiDetach()";
  CORE_ADDR debugFlagAddr;
  CORE_ADDR rpcFlagAddr;
  CORE_ADDR resumeAppOnDetachFlagAddr;
  unsigned char resumeAppOnDetach;
  struct cleanup *cleanup = NULL;

  debugFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));

  /* Bail out if the CUDA driver isn't available */
  if (!debugFlagAddr)
      return;

  /* Update the suspended devices mask in the inferior */
  cuda_inferior_update_suspended_devices_mask ();

  cuda_api_set_attach_state (CUDA_ATTACH_STATE_DETACHING);

  /* Make sure the focus is set on the host */
  switch_to_thread (inferior_ptid);

  if (!lookup_cmd_composition ("call", &alias, &prefix_cmd, &cmd))
    error (_("Failed to initiate detach."));

  /* Figure out if we need to clean up driver state before detaching */
  resumeAppOnDetachFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_RESUME_FOR_ATTACH_DETACH));

  if (!resumeAppOnDetachFlagAddr)
    error (_("Failed to detach cleanly from the inferior."));

  /* Make dynamic call for cleanup. */
  cleanup = cuda_gdb_bypass_signals ();
  cmd_func (cmd, cudbgApiDetach, 0);
  do_cleanups (cleanup);

  /* Read the updated value of the flag */
  target_read_memory (resumeAppOnDetachFlagAddr, &resumeAppOnDetach, 1);

  /* If this flag is set, the debugger backend needs to be notified to cleanup on detach */
  if (resumeAppOnDetach)
    cuda_api_request_cleanup_on_detach (resumeAppOnDetach);

  /* Make sure the debugger is reinitialized from scratch on reattaching
     to the inferior */
  rpcFlagAddr = cuda_get_symbol_address (_STRING_(CUDBG_DEBUGGER_INITIALIZED));

  if (!rpcFlagAddr)
    error (_("Failed to detach cleanly from the inferior."));

  cuda_write_bool (rpcFlagAddr, false);

  /* If a cleanup is needed, resume the app to allow the cleanup to complete.
     The debugger backend will send a cleanup event to stop the app when the
     cleanup finishes. */
  if (resumeAppOnDetach)
    {
      int cnt;

      /* Clear all breakpoints */
      delete_command (NULL, 0);
      cuda_system_cleanup_breakpoints ();
      cuda_options_disable_break_on_launch ();

      /* Now resume the app and wait for CUDA_ATTACH_STATE_DETACH_COMPLETE event. */
      for (cnt=0;
	   cnt < 100
           && cuda_api_get_attach_state () != CUDA_ATTACH_STATE_DETACH_COMPLETE;
           cnt++)
	  {
	      prepare_execution_command (current_top_target (), true);
	      continue_1 (false);
	      wait_for_inferior ();
	      normal_stop ();
	  }

      /* No threads are running at this point.  */
      set_running (minus_one_ptid, 0);
    }
  else
    cuda_api_set_attach_state (CUDA_ATTACH_STATE_DETACH_COMPLETE);

  if (cuda_api_get_attach_state () != CUDA_ATTACH_STATE_DETACH_COMPLETE)
    warning (_("Unexpected CUDA API attach state."));

  cuda_write_bool (debugFlagAddr, false);

  cuda_cleanup ();
}

void
cuda_focus_init (cuda_focus_t *focus)
{
  gdb_assert (focus);

  focus->valid = false;
}

void
cuda_focus_save (cuda_focus_t *focus)
{
  gdb_assert (focus);
  gdb_assert (!focus->valid);

  focus->ptid = inferior_ptid;
  focus->coords = CUDA_INVALID_COORDS;
  cuda_coords_get_current (&focus->coords);

  focus->valid = true;
}

void
cuda_focus_restore (cuda_focus_t *focus)
{
  gdb_assert (focus);
  gdb_assert (focus->valid);

  if (focus->coords.valid)
    switch_to_cuda_thread  (&focus->coords);
  else
    switch_to_thread (focus->ptid);

  focus->valid = false;
}

void
switch_to_cuda_thread (cuda_coords_t *coords)
{
  uint64_t pc;
  cuda_coords_t c;

  gdb_assert (coords || cuda_focus_is_device ());

  if (coords)
    cuda_coords_set_current (coords);
  cuda_coords_get_current (&c);

  cuda_update_cudart_symbols ();
  reinit_frame_cache ();
  registers_changed ();

  if (c.valid)
    pc = lane_get_virtual_pc (c.dev, c.sm, c.wp, c.ln);
  else
    pc = (CORE_ADDR) ~0;

#if !defined(__APPLE__) && !defined(__QNXTARGET__)
  struct lwp_info *lp = find_lwp_pid (inferior_ptid);
  if (lp)
    lp->stop_pc = pc;
#endif

  thread_info *thr = find_thread_ptid (inferior_ptid);
  if (thr)
    thr->suspend.stop_pc = pc;
}


struct objfile *cuda_create_builtins_objfile (void);

void
cuda_update_cudart_symbols (void)
{
  /* If not done yet, create a CUDA runtime symbols file */
  if (!cuda_cudart_symbols.objfile)
    {
      cuda_cudart_symbols.objfile = cuda_create_builtins_objfile ();
    }

}

void
cuda_cleanup_cudart_symbols (void)
{
  /* Free the objfile if allocated */
  if (cuda_cudart_symbols.objfile)
    {
      delete cuda_cudart_symbols.objfile;
      cuda_cudart_symbols.objfile = NULL;
    }
}

#if defined(__APPLE__)
/* CUDA - cuda-gdb wrapper */
/* Smuggle the DYLD_* environement variables like GDB 6.3.5 used to do. Because
   we cuda-gdb must be part of the procmod group, those variables are not
   passed, for security reasons. Instead we pass them in a GDB_DYLD_*
   equivalent and restore them when launching the inferior. This code was taken
   from GDB 6.3.5, the Apple editon. */ 

struct dyld_smuggle_pairs {
  const char *real_name;
  const char *smuggled_name;
};

static void
smuggle_dyld_settings (struct gdb_environ *e)
{
  int i;
  struct dyld_smuggle_pairs env_names[] = { 
       {"DYLD_FRAMEWORK_PATH",          "GDB_DYLD_FRAMEWORK_PATH"},
       {"DYLD_FALLBACK_FRAMEWORK_PATH", "GDB_DYLD_FALLBACK_FRAMEWORK_PATH"},
       {"DYLD_LIBRARY_PATH",            "GDB_DYLD_LIBRARY_PATH"},
       {"DYLD_FALLBACK_LIBRARY_PATH",   "GDB_DYLD_FALLBACK_LIBRARY_PATH"},
       {"DYLD_ROOT_PATH",               "GDB_DYLD_ROOT_PATH"},
       {"DYLD_PATHS_ROOT",              "GDB_DYLD_PATHS_ROOT"},
       {"DYLD_IMAGE_SUFFIX",            "GDB_DYLD_IMAGE_SUFFIX"},
       {"DYLD_INSERT_LIBRARIES",        "GDB_DYLD_INSERT_LIBRARIES"},
       { NULL,                          NULL } };

  for (i = 0; env_names[i].real_name != NULL; i++)
    {
      const char *real_val     = e->get (env_names[i].real_name);
      const char *smuggled_val = e->get (env_names[i].smuggled_name);

      if (real_val == NULL && smuggled_val == NULL)
        continue;

      if (smuggled_val == NULL)
        continue;

      /* Is the value of the DYLD_* env var truncated to ""? */
      if (real_val != NULL && real_val[0] != '\0')
        continue;

      /* real_val has a value and it looks legitimate - don't overwrite it
         with the smuggled version.  */
      if (real_val != NULL)
        continue;

      e->set (env_names[i].real_name, smuggled_val);
    }
}
#endif

/*
 * CUDA builtins construction routines
 */

/* cuda_alloc_dim3_type helper routine: initializes one of the structure fields
 * with a given name, offset and type */
static void
cuda_init_field (struct field *fp, const char *name, const int offs, struct type *type)
{
  fp->name = name;
  fp->type = type;
  SET_FIELD_BITPOS(*fp, offs*8);
  FIELD_BITSIZE(*fp) = type->length*8;
}


/* Allocates dim3 type as structure of 3 packed unsigned int: x, y and z */
static struct type *
cuda_alloc_dim3_type (struct objfile *objfile)
{
  struct gdbarch *gdbarch = get_objfile_arch (objfile); 
  struct type *uint32_type = builtin_type (gdbarch)->builtin_unsigned_int;
  struct type *dim3 = NULL;

  dim3 = alloc_type (objfile);

  TYPE_NAME(dim3) = "dim3";
  TYPE_LENGTH(dim3) = 12;
  TYPE_CODE(dim3) = TYPE_CODE_STRUCT;

  TYPE_NFIELDS(dim3) = 3;
  TYPE_FIELDS(dim3) = (struct field *)TYPE_ALLOC(dim3, sizeof (struct field) * 3);

  cuda_init_field (&TYPE_FIELD(dim3, 0), "x", 0, uint32_type);
  cuda_init_field (&TYPE_FIELD(dim3, 1), "y", 4, uint32_type);
  cuda_init_field (&TYPE_FIELD(dim3, 2), "z", 8, uint32_type);

  return dim3;
}


/* Create symbol of a given type inside the objfile */
static struct symbol *
cuda_create_symbol (struct objfile *objfile, struct blockvector *bv, const char *name, CORE_ADDR addr, struct type *type)
{
  struct symbol *sym = NULL;

  /* Allocate a new symbol in OBJFILE's obstack.  */
  sym = allocate_symbol (objfile);

  SYMBOL_SET_LANGUAGE (sym, language_c, &objfile->per_bfd->storage_obstack);
  SYMBOL_SET_NAMES (sym, name, strlen (name), 0, objfile);
  SYMBOL_TYPE (sym) = type;
  SYMBOL_DOMAIN (sym) = VAR_DOMAIN;
  SYMBOL_ACLASS_INDEX (sym) = LOC_STATIC;
  SYMBOL_VALUE_ADDRESS (sym) = addr;

  /* Register symbol as global symbol with symtab */
  symbol_set_symtab (sym, COMPUNIT_FILETABS (objfile->compunit_symtabs));
  add_symbol_to_list (sym, get_global_symbols ());
  mdict_add_symbol (BLOCK_MULTIDICT (BLOCKVECTOR_BLOCK (bv, GLOBAL_BLOCK)), sym);

  return sym;
}

/* Symtab initialization helper routine:
 * Allocates blockvector as well as global and static blocks inside symtab 
 */
struct blockvector *
cuda_alloc_blockvector (struct symtab *symtab, int nblocks)
{
  struct obstack *obstack = &(SYMTAB_OBJFILE (symtab)->objfile_obstack);

  /* At least enough room for the global and static blocks */
  gdb_assert (nblocks >= 2);

  /* allocate and zero the blockvector */
  uint32_t len = sizeof (struct blockvector) + (nblocks - 1) * sizeof (struct block *);
  struct blockvector *bv = (struct blockvector *) obstack_alloc (obstack, len);
  memset ((void *)bv, 0, len);

  BLOCKVECTOR_MAP (bv) = 0;
  BLOCKVECTOR_NBLOCKS (bv) = nblocks;

  /* Allocate the GLOBAL block */
  struct block *global_block = allocate_global_block (obstack);
  BLOCK_MULTIDICT (global_block) = mdict_create_hashed_expandable (symtab->language);
  BLOCKVECTOR_BLOCK (bv, GLOBAL_BLOCK) = global_block;

  /* Only allowed for the GLOBAL block */
  set_block_compunit_symtab (global_block, SYMTAB_COMPUNIT (symtab));

  /* Allocate the STATIC block*/
  struct block *static_block = allocate_block (obstack);
  BLOCK_MULTIDICT (static_block) = mdict_create_hashed_expandable (symtab->language);
  BLOCKVECTOR_BLOCK (bv, STATIC_BLOCK) = static_block;

  /* superblock of the static block is the global block - see block.h */
  BLOCK_SUPERBLOCK (static_block) = global_block;

  return bv;
}

/* Allocate virtual objfile and construct the following symbols inside it:
 * threadIdx of type dim3 located at CUDBG_THREADIDX_OFFSET
 * blockIdx of type dim3 located at CUDBG_BLOCKIDX_OFFSET
 * blockDim of type dim3 located at CUDBG_BLOCKDIM_OFFSET
 * threadDim of type dim3 located at CUDBG_THREADDIM_OFFSET
 * warpSize of type int located at CUDBG_WARPSIZE_OFFSET
 */
struct objfile *
cuda_create_builtins_objfile (void)
{
  struct objfile *objfile = NULL;
  struct type *int32_type = NULL;
  struct type *dim3_type = NULL;
  struct symtab *symtab = NULL;

  /* Set the cleanup chain so we get things properly set after we're done
     assembling the symbol table.  */
  scoped_free_pendings free_pending;

  /* This is not a real objfile.  Mark it as so by passing
     OBJF_NOT_FILENAME.  */
  objfile = new struct objfile (NULL, NULL, OBJF_SHARED | OBJF_NOT_FILENAME);
  objfile->per_bfd->gdbarch = cuda_get_gdbarch ();

  /* Get/allocate types */
  int32_type = builtin_type (get_objfile_arch (objfile))->builtin_int32;
  dim3_type = cuda_alloc_dim3_type (objfile);

  /* Now that the objfile structure has been allocated, we need to allocate all
     the required data structures for symbols.  */
  objfile->compunit_symtabs = start_symtab(objfile, "<cuda-builtins>", NULL, 0, language_c);

  symtab = allocate_symtab (objfile->compunit_symtabs, "<cuda-builtins>");

  symtab->language = language_c;

  struct blockvector *bv = cuda_alloc_blockvector (symtab, 2);
  SYMTAB_BLOCKVECTOR (symtab) = bv;

  cuda_create_symbol (objfile, bv, "threadIdx", CUDBG_THREADIDX_OFFSET, dim3_type);
  cuda_create_symbol (objfile, bv, "blockIdx", CUDBG_BLOCKIDX_OFFSET, dim3_type);
  cuda_create_symbol (objfile, bv, "blockDim", CUDBG_BLOCKDIM_OFFSET, dim3_type);
  cuda_create_symbol (objfile, bv, "gridDim", CUDBG_GRIDDIM_OFFSET, dim3_type);
  cuda_create_symbol (objfile, bv, "warpSize", CUDBG_WARPSIZE_OFFSET, int32_type);

  return objfile;
}

void
_initialize_cuda (void)
{
}
