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

#include <stdbool.h>
#include "defs.h"
#include "gdbthread.h"
#include "inferior.h"
#include "remote.h"
#ifdef __QNXTARGET__
# include "remote-nto.h"
# define getpkt qnx_getpkt
# define putpkt qnx_putpkt
# define putpkt_binary qnx_putpkt_binary
#else
#endif
#include "remote-cuda.h"
#include "cuda-packet-manager.h"
#include "cuda-context.h"
#include "cuda-events.h"
#include "cuda-state.h"
#include "cuda-textures.h"
#include "cuda-utils.h"
#include "cuda-options.h"

#ifdef __QNXTARGET__
/* Maximum supported data size in QNX protocol is DS_DATA_MAX_SIZE (1024).
   cuda-gdbserver can be modified to handle 16384 instead, but in order to
   use bigger packets for CUDA, we would need first ensure that they can be
   packed/unpacked by the pdebug putpkt/getpkt functions.

   Until then, use pdebug max allowed size.
   Each DS_DATA_MAX_SIZE can be escaped (*2), 2 frame chars (+2) plus a checksum
   that can be escaped (+2). */
# define PBUFSIZE (DS_DATA_MAX_SIZE * 2 + 4)
#else
# define PBUFSIZE 16384
#endif

static gdb::char_vector pktbuf(PBUFSIZE);

static char *
append_string (const char *src, char *dest, bool sep)
{
  char *p;

  if (dest + strlen (src) - pktbuf.data () >= pktbuf.size ())
    error (_("Exceed the size of cuda packet.\n"));

  sprintf (dest, "%s", src);
  p = strchr (dest, '\0');

  if (sep)
    {
      *p = ';';
      *(++p) = '\0';
    }
  return p;
}

static char *
append_bin (const gdb_byte *src, char *dest, int size, bool sep)
{
  char *p;

  if (dest + size * 2 - pktbuf.data () >= pktbuf.size ())
    error (_("Exceed the size of cuda packet.\n"));

  bin2hex (src, dest, size);
  p = strchr (dest, '\0');

  if (sep)
    {
      *p = ';';
      *(++p) = '\0';
    }
  return p;
}

static char *
extract_string (char *src)
{
  return strtok (src, ";");
}

static char *
extract_bin (char *src, gdb_byte *dest, int size)
{
  char *p;

  p = extract_string (src);
  if (!p)
    error (_("The data in the cuda packet is not complete.\n")); 
  hex2bin (p, dest, size);
  return p;
}

bool
cuda_remote_notification_pending (remote_target *ops)
{
  char *p;
  bool ret_val;
  cuda_packet_type_t packet_type = NOTIFICATION_PENDING;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) &ret_val, sizeof (ret_val));
  return ret_val;
}

bool
cuda_remote_notification_received (remote_target *ops)
{
  char *p;
  bool ret_val;
  cuda_packet_type_t packet_type = NOTIFICATION_RECEIVED;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) &ret_val, sizeof (ret_val));
  return ret_val;
}

bool
cuda_remote_notification_aliased_event (remote_target *ops)
{
  char *p;
  bool ret_val;
  cuda_packet_type_t packet_type = NOTIFICATION_ALIASED_EVENT;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) &ret_val, sizeof (ret_val));
  return ret_val;
}

void
#ifdef __QNXTARGET__
cuda_remote_notification_analyze (remote_target *ops, ptid_t ptid, struct target_waitstatus *ws)
#else /* __QNXTARGET__ */
cuda_remote_notification_analyze (remote_target *ops, ptid_t ptid)
#endif /* __QNXTARGET__ */
{
  char *p;
  cuda_packet_type_t packet_type = NOTIFICATION_ANALYZE;
  struct thread_info *tp;

  /* Upon connecting to gdbserver, we may not have stablished an inferior_ptid,
     so it is still null_ptid.  In that case, use the event ptid that should be
     the thread that triggered this code path.  */
  if (inferior_ptid == null_ptid)
    tp = find_thread_ptid (ptid);
  else
    tp = inferior_thread ();

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
#ifdef __QNXTARGET__
  p = append_bin ((gdb_byte *) &ptid, p, sizeof (ptid), true);
  p = append_bin ((gdb_byte *) ws, p, sizeof (*ws), true);
#endif
  p = append_bin ((gdb_byte *) &(tp->control.trap_expected), p, sizeof (tp->control.trap_expected), false);
  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);
}

void
cuda_remote_notification_mark_consumed (remote_target *ops)
{
  char *p;
  cuda_packet_type_t packet_type = NOTIFICATION_MARK_CONSUMED;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);
}

void
cuda_remote_notification_consume_pending (remote_target *ops)
{
  char *p;
  cuda_packet_type_t packet_type = NOTIFICATION_CONSUME_PENDING;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);
}

void
cuda_remote_update_grid_id_in_sm (remote_target *ops, uint32_t dev, uint32_t sm)
{
#ifdef __QNXTARGET__
  uint32_t wp;
  cuda_api_warpmask* valid_warps_mask_c;
  cuda_api_warpmask valid_warps_mask_s;
  uint32_t num_warps;
  uint64_t grid_id;

  /* On QNX the packet size is limited and the response for this packet
     doesn't fit. Therefore gather all the data from the client side
     without server-side batching, using individual vCUDA packets via
     the CUDA API which are constant in size in this case. */

  valid_warps_mask_c = sm_get_valid_warps_mask (dev, sm);
  num_warps = device_get_num_warps (dev);

  cuda_api_read_valid_warps (dev, sm, &valid_warps_mask_s);
  gdb_assert (cuda_api_eq_mask(&valid_warps_mask_s, valid_warps_mask_c));

  for (wp = 0; wp < num_warps; wp++)
    {
      if (warp_is_valid (dev, sm, wp))
        {
          cuda_api_read_grid_id (dev, sm, wp, &grid_id);
          warp_set_grid_id (dev, sm, wp, grid_id);
        }
    }
#else
  CUDBGResult res;
  char *p;
  uint32_t wp;
  cuda_api_warpmask* valid_warps_mask_c;
  cuda_api_warpmask valid_warps_mask_s;
  uint32_t num_warps;
  uint64_t grid_id;
  cuda_packet_type_t packet_type = UPDATE_GRID_ID_IN_SM;

  valid_warps_mask_c = sm_get_valid_warps_mask (dev, sm);
  num_warps = device_get_num_warps (dev);
  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &dev, p, sizeof (dev), true);
  p = append_bin ((gdb_byte *) &sm,  p, sizeof (sm), true);
  p = append_bin ((gdb_byte *) &num_warps, p, sizeof (num_warps), false);

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) &valid_warps_mask_s, sizeof (valid_warps_mask_s));
  gdb_assert (cuda_api_eq_mask(&valid_warps_mask_s, valid_warps_mask_c));
  for (wp = 0; wp < num_warps; wp++)
    {
      if (warp_is_valid (dev, sm, wp))
        {
          extract_bin (NULL, (gdb_byte *) &grid_id, sizeof (grid_id));
          warp_set_grid_id (dev, sm, wp, grid_id);
        }
    }
  extract_bin (NULL, (gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read the grid index (error=%u).\n"), res);
#endif
}

void
cuda_remote_update_block_idx_in_sm (remote_target *ops, uint32_t dev, uint32_t sm)
{
#ifdef __QNXTARGET__
  uint32_t wp;
  cuda_api_warpmask* valid_warps_mask_c;
  cuda_api_warpmask valid_warps_mask_s;
  uint32_t num_warps;
  CuDim3 block_idx;

  /* On QNX a response to this packet won't fit in the packet size,
     use multiple vCUDA packets instead.
     See explanation in cuda_remote_update_grid_id_in_sm() for details. */

  valid_warps_mask_c = sm_get_valid_warps_mask (dev, sm);
  num_warps = device_get_num_warps (dev);

  cuda_api_read_valid_warps (dev, sm, &valid_warps_mask_s);
  gdb_assert (cuda_api_eq_mask(&valid_warps_mask_s, valid_warps_mask_c));

  for (wp = 0; wp < num_warps; wp++)
    {
      if (warp_is_valid (dev, sm, wp))
        {
          cuda_api_read_block_idx(dev, sm, wp, &block_idx);
          warp_set_block_idx (dev, sm, wp, &block_idx);
        }
    }
#else
  CUDBGResult res;
  char *p;
  uint32_t wp;
  cuda_api_warpmask* valid_warps_mask_c;
  cuda_api_warpmask valid_warps_mask_s;
  uint32_t num_warps;
  CuDim3 block_idx;
  cuda_packet_type_t packet_type = UPDATE_BLOCK_IDX_IN_SM;

  valid_warps_mask_c = sm_get_valid_warps_mask (dev, sm);
  num_warps = device_get_num_warps (dev);
  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &dev, p, sizeof (dev), true);
  p = append_bin ((gdb_byte *) &sm,  p, sizeof (sm), true);
  p = append_bin ((gdb_byte *) &num_warps, p, sizeof (num_warps), false);

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) &valid_warps_mask_s, sizeof (valid_warps_mask_s));
  gdb_assert (cuda_api_eq_mask(&valid_warps_mask_s, valid_warps_mask_c));
  for (wp = 0; wp < num_warps; wp++)
    {
       if (warp_is_valid (dev, sm, wp))
         {
           extract_bin (NULL, (gdb_byte *) &block_idx, sizeof (block_idx));
           warp_set_block_idx (dev, sm, wp, &block_idx);
         }
    }
  extract_bin (NULL, (gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read the block index (error=%u).\n"), res);
#endif
}

void
cuda_remote_update_thread_idx_in_warp (remote_target *ops, uint32_t dev, uint32_t sm, uint32_t wp)
{
  CUDBGResult res;
  char *p;
  uint32_t ln;
  uint64_t valid_lanes_mask_c;
  uint64_t valid_lanes_mask_s;
  uint32_t num_lanes;
  CuDim3 thread_idx;
  cuda_packet_type_t packet_type = UPDATE_THREAD_IDX_IN_WARP;

  valid_lanes_mask_c = warp_get_valid_lanes_mask (dev, sm, wp);
  num_lanes = device_get_num_lanes (dev);
  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &dev, p, sizeof (dev), true);
  p = append_bin ((gdb_byte *) &sm,  p, sizeof (sm), true);
  p = append_bin ((gdb_byte *) &wp,  p, sizeof (wp), true);
  p = append_bin ((gdb_byte *) &num_lanes, p, sizeof (num_lanes), false);

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) &valid_lanes_mask_s, sizeof (valid_lanes_mask_s));
  gdb_assert (valid_lanes_mask_s == valid_lanes_mask_c);
  for (ln = 0; ln < num_lanes; ln++)
    {
       if (lane_is_valid (dev, sm, wp, ln))
         {
           extract_bin (NULL, (gdb_byte *) &thread_idx, sizeof (thread_idx));
           lane_set_thread_idx (dev, sm, wp, ln, &thread_idx);
         }
    }
  extract_bin (NULL, (gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read the thread index (error=%u).\n"), res);
}

#ifdef __QNXTARGET__
void
cuda_remote_set_symbols (remote_target *ops, bool *symbols_are_set)
{
  char *p;
  cuda_packet_type_t packet_type = SET_SYMBOLS;
  CORE_ADDR address;
  const unsigned char symbols_count = 13;
  enum target_xfer_status status;
  ULONGEST len;

  *symbols_are_set = false;

  /* Remote side will also check for zeros, here we test only one symbol
     to avoid unnecessary back and forth with it.
     Sent symbols must be kept in sync with those in cuda_symbol_list[] */
  address = cuda_get_symbol_address (_STRING_(CUDBG_IPC_FLAG_NAME));
  if (address == 0)
    {
      return;
    }

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &symbols_count, p, sizeof (symbols_count), true);
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_RPC_ENABLED));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_APICLIENT_PID));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_APICLIENT_REVISION));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_SESSION_ID));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_ATTACH_HANDLER_AVAILABLE));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_DEBUGGER_INITIALIZED));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_REPORTED_DRIVER_API_ERROR_CODE));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_REPORTED_DRIVER_INTERNAL_ERROR_CODE));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_DETACH_SUSPENDED_DEVICES_MASK));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_ENABLE_INTEGRATED_MEMCHECK));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_ENABLE_LAUNCH_BLOCKING));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), true);
  address = cuda_get_symbol_address (_STRING_(CUDBG_ENABLE_PREEMPTION_DEBUGGING));
  p = append_bin ((gdb_byte *) &address, p, sizeof (address), false);

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) symbols_are_set, sizeof (*symbols_are_set));
}
#endif /* __QNXTARGET__ */

void
cuda_remote_initialize (remote_target *ops,
			CUDBGResult *get_debugger_api_res, CUDBGResult *set_callback_api_res,
                        CUDBGResult *initialize_api_res, bool *cuda_initialized,
                        bool *cuda_debugging_enabled, bool *driver_is_compatible)
{
  char *p;
  cuda_packet_type_t packet_type = INITIALIZE_TARGET;
  bool preemption          = cuda_options_software_preemption ();
  bool memcheck            = cuda_options_memcheck ();
  bool launch_blocking     = cuda_options_launch_blocking ();

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type,     p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &preemption,      p, sizeof (preemption), true);
  p = append_bin ((gdb_byte *) &memcheck,        p, sizeof (memcheck), true);
  p = append_bin ((gdb_byte *) &launch_blocking, p, sizeof (launch_blocking), false);

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) get_debugger_api_res, sizeof (*get_debugger_api_res));
  extract_bin (NULL, (gdb_byte *) set_callback_api_res, sizeof (*set_callback_api_res));
  extract_bin (NULL, (gdb_byte *) initialize_api_res, sizeof (*initialize_api_res));
  extract_bin (NULL, (gdb_byte *) cuda_initialized, sizeof (*cuda_initialized));
  extract_bin (NULL, (gdb_byte *) cuda_debugging_enabled, sizeof (*cuda_debugging_enabled));
  extract_bin (NULL, (gdb_byte *) driver_is_compatible, sizeof (*driver_is_compatible));
}

void
cuda_remote_query_device_spec (remote_target *ops,
			       uint32_t dev_id, uint32_t *num_sms, uint32_t *num_warps,
                               uint32_t *num_lanes, uint32_t *num_registers,
                               char **dev_type, char **sm_type)
{
  char *p;
  CUDBGResult res;
  cuda_packet_type_t packet_type = QUERY_DEVICE_SPEC;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &dev_id, p, sizeof (uint32_t), false);

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) &res, sizeof (res));
  if (res != CUDBG_SUCCESS)
    error (_("Error: Failed to read device specification (error=%u).\n"), res);  
  extract_bin (NULL, (gdb_byte *) num_sms, sizeof (num_sms));
  extract_bin (NULL, (gdb_byte *) num_warps, sizeof (num_warps));
  extract_bin (NULL, (gdb_byte *) num_lanes, sizeof (num_lanes));
  extract_bin (NULL, (gdb_byte *) num_registers, sizeof (num_registers));
  *dev_type = extract_string (NULL);
  *sm_type  = extract_string (NULL);
}

bool
#ifdef __QNXTARGET__
cuda_remote_check_pending_sigint (remote_target *ops, ptid_t ptid)
#else /* __QNXTARGET__ */
cuda_remote_check_pending_sigint (remote_target *ops)
#endif /* __QNXTARDET__ */
{
  char *p;
  bool ret_val;
  cuda_packet_type_t packet_type = CHECK_PENDING_SIGINT;

  p = append_string ("qnv.", pktbuf.data (), false);
#ifndef __QNXTARGET__
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
#else
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);
  p = append_bin ((gdb_byte *) &ptid, p, sizeof (ptid), false);
#endif

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) &ret_val, sizeof (ret_val));
  return ret_val;
}

CUDBGResult
cuda_remote_api_finalize (remote_target *ops)
{
  char *p;
  CUDBGResult res;
  cuda_packet_type_t packet_type = API_FINALIZE;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), true);

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);

  extract_bin (pktbuf.data (), (gdb_byte *) &res, sizeof (res));
  return res;
}

void
cuda_remote_set_option (remote_target *ops)
{
  char *p;
  bool general_trace       = cuda_options_debug_general ();
  bool libcudbg_trace      = cuda_options_debug_libcudbg ();
  bool notifications_trace = cuda_options_debug_notifications ();
  bool notify_youngest     = cuda_options_notify_youngest ();
  unsigned stop_signal     = cuda_options_stop_signal ();

  cuda_packet_type_t packet_type = SET_OPTION;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type,         p, sizeof (cuda_packet_type_t), true);
  p = append_bin ((gdb_byte *) &general_trace,       p, sizeof (general_trace), true);
  p = append_bin ((gdb_byte *) &libcudbg_trace,      p, sizeof (libcudbg_trace), true);
  p = append_bin ((gdb_byte *) &notifications_trace, p, sizeof (notifications_trace), true);
  p = append_bin ((gdb_byte *) &notify_youngest,     p, sizeof (notify_youngest), true);
  p = append_string (stop_signal == GDB_SIGNAL_TRAP ? "SIGTRAP" : "SIGURG", p, false);

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);
}

void
cuda_remote_query_trace_message (remote_target *ops)
{
  char *p;
  cuda_packet_type_t packet_type = QUERY_TRACE_MESSAGE;

  if (!cuda_options_debug_general () &&
      !cuda_options_debug_libcudbg () &&
      !cuda_options_debug_notifications ())
    return;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);

  putpkt (ops, pktbuf.data ());
  getpkt (ops, &pktbuf, 1);
  p = extract_string (pktbuf.data ());
  while (strcmp ("NO_TRACE_MESSAGE", p) != 0)
    {
      fprintf (stderr, "%s\n", p);

      p = append_string ("qnv.", pktbuf.data (), false);
      p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);
      putpkt (ops, pktbuf.data ());
      getpkt (ops, &pktbuf, 1);
      p = extract_string (pktbuf.data ());
    }
  fflush (stderr);
}

#ifdef __QNXTARGET__
/* On QNX targets version is queried explicitly */
void
cuda_qnx_version_handshake (remote_target *ops)
{
  char *p;
  cuda_packet_type_t packet_type = VERSION_HANDSHAKE;

  p = append_string ("qnv.", pktbuf.data (), false);
  p = append_bin ((gdb_byte *) &packet_type, p, sizeof (packet_type), false);

  putpkt (ops, pktbuf.data ());
  if (getpkt (ops, &pktbuf, 1) == -1)
    {
      error (_("Server doesn't support CUDA.\n"));
    }

  p = extract_string (pktbuf.data ());
  cuda_version_handshake (p);
}
#endif /* __QNXTARGET__ */
