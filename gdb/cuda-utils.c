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

/* Utility functions for cuda-gdb */
#ifdef GDBSERVER
#include "common/gdb_locale.h"
#include "server.h"
#else
#include "common/common-defs.h"
#include "defs.h"
#include "inferior.h"
#include "gdb/signals.h"
#include "stack.h"
#include "gdbthread.h"
#include "cuda-options.h"
#include "objfiles.h"
#include "exceptions.h"
#include "utils.h"
#endif
#ifdef __APPLE__
#include <stddef.h>
#include <sys/sysctl.h>
#endif

#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/param.h>
#ifdef __QNXHOST__
# include <ftw.h>
#else
# include <dirent.h>
#endif
#include <unistd.h>
#include <string.h>

#include <fcntl.h>
#include "cuda-utils.h"
#include "cuda-defs.h"

#define RECORD_FORMAT_MASTER   "LOCK:%10d\n"
#define RECORD_FORMAT_DEVICE    "%4d:%10d\n"
#define RECORD_SIZE                       16
#define RECORD_MASTER                      0
#define RECORD_DEVICE(i)           ((i) + 1)
#define DEVICE_RECORD(i)           ((i) - 1)

int cuda_use_lockfile = 1;

static const char cuda_gdb_lock_file[] = "cuda-gdb.lock";
static char cuda_gdb_tmp_basedir[CUDA_GDB_TMP_BUF_SIZE] = {0};
static int cuda_gdb_lock_fd = -1;
static char* cuda_gdb_tmp_dir = NULL;
static uint64_t dev_mask = 0;

int
cuda_gdb_dir_create (const char *dir_name, uint32_t permissions,
                     bool override_umask, bool *dir_exists)
{
  int ret;
  mode_t old_umask = 0;

  /* Save the old umask and reset it */
  if (override_umask)
    old_umask = umask (0);

  ret = mkdir (dir_name, permissions);
  if ((ret < 0) && (errno == EEXIST))
    {
      /* Preexisting directory (may have lost a race to create it),
         report dir_exists and success */
      *dir_exists = true;
      ret = 0;
    }
  else
    *dir_exists = false;

  /* Restore the old umask */
  if (override_umask)
    umask (old_umask);

  return ret;
}

static void
cuda_gdb_tmpdir_create_basedir (void)
{
  int ret = 0;
  bool dir_exists = false;
  bool override_umask = true;

  if (getenv ("TMPDIR"))
    snprintf (cuda_gdb_tmp_basedir, sizeof (cuda_gdb_tmp_basedir),
              "%s/cuda-dbg", getenv ("TMPDIR"));
  else
    snprintf (cuda_gdb_tmp_basedir, sizeof (cuda_gdb_tmp_basedir),
              "/tmp/cuda-dbg");

  ret = cuda_gdb_dir_create (cuda_gdb_tmp_basedir,
                             S_IRWXU | S_IRWXG | S_IRWXO,
                             override_umask, &dir_exists);
  if (ret)
    error (_("Error creating temporary directory %s\n"),
           cuda_gdb_tmp_basedir);
}

static char *
cuda_gdb_get_tmp_basedir (void)
{
  if (cuda_gdb_tmp_basedir[0] == 0)
    cuda_gdb_tmpdir_create_basedir ();

  return cuda_gdb_tmp_basedir;
}

#ifdef __QNXHOST__
static int
fn_cleanup_dir (const char *path, const struct stat *stat, int flag, struct FTW *ftw)
{
  switch (flag)
    {
    case FTW_DP:
      rmdir (path);
      break;
    case FTW_F:
    case FTW_SL:
    case FTW_SLN:
      unlink (path);
      break;
    case FTW_DNR:
    case FTW_NS:
      /* Silently ignore read errors */
      break;
    }
  return 0;
}

void
cuda_gdb_tmpdir_cleanup_dir (const char *dirpath)
{
  const int maxDepth = 10;
  nftw (dirpath, fn_cleanup_dir, maxDepth, FTW_DEPTH);
}

#else /* __QNXHOST__*/

void
cuda_gdb_dir_cleanup_files (const char *dirpath)
{
  char path[CUDA_GDB_TMP_BUF_SIZE];
  DIR* dir = opendir (dirpath);
  struct dirent* dir_ent = NULL;

  if (!dir)
    return;

  while ((dir_ent = readdir (dir)))
    {
      if (!strcmp(dir_ent->d_name,".") ||
          !strcmp(dir_ent->d_name, ".."))
        continue;
      snprintf (path, sizeof (path), "%s/%s", dirpath, dir_ent->d_name);
      if (dir_ent->d_type == DT_DIR) {
        cuda_gdb_dir_cleanup_files (path);
        rmdir (path);
      }
      else
        unlink (path);
    };

  closedir (dir);
}

static void
cuda_gdb_tmpdir_cleanup_dir (const char *dirpath)
{
  cuda_gdb_dir_cleanup_files (dirpath);
  rmdir (dirpath);
}
#endif /* __QNXHOST__*/

void
cuda_gdb_tmpdir_cleanup_self (void *unused)
{
  if (!cuda_gdb_tmp_dir)
    return;

  cuda_gdb_tmpdir_cleanup_dir (cuda_gdb_tmp_dir);
  xfree (cuda_gdb_tmp_dir);
  cuda_gdb_tmp_dir = NULL;
}

static void
cuda_gdb_record_write (int record_idx, int pid)
{
  char record[CUDA_GDB_TMP_BUF_SIZE];
  int res;

  if (record_idx == 0)
    snprintf (record, CUDA_GDB_TMP_BUF_SIZE, RECORD_FORMAT_MASTER,
              pid);
  else
    snprintf (record, CUDA_GDB_TMP_BUF_SIZE, RECORD_FORMAT_DEVICE,
              DEVICE_RECORD (record_idx), pid);

  res = lseek (cuda_gdb_lock_fd, record_idx * RECORD_SIZE, SEEK_SET);
  if (res == -1)
    return;

  res = write (cuda_gdb_lock_fd, record, strlen (record));
  if (res == -1)
    return;
}

static int
cuda_gdb_record_read (int record_idx)
{

  char record[CUDA_GDB_TMP_BUF_SIZE] = {0};
  int res;
  char *colon = NULL;
  int rc = -1;

  res = lseek (cuda_gdb_lock_fd, record_idx * RECORD_SIZE, SEEK_SET);
  if (res == -1)
    return -1;

  res = read (cuda_gdb_lock_fd, record, RECORD_SIZE);
  if (res == -1)
    return -1;

  colon = strchr (record, ':');
  if (!colon || colon[1] == 0)
    return -1;

  if (sscanf(colon+1, "%d", &rc) != 1)
    return -1;

  return rc;
}

/* Returns true if lock was acquired and false if user decided not to acquire further locks */
static bool
cuda_gdb_record_set_lock (int record_idx, bool enable_lock)
{
  struct flock lock = {0};
  int e = 0;
  int pid = -1;

  lock.l_type = enable_lock ? F_WRLCK: F_UNLCK;
  lock.l_whence = SEEK_SET;
  lock.l_start = record_idx * RECORD_SIZE;
  lock.l_len = RECORD_SIZE;

  e = fcntl (cuda_gdb_lock_fd, F_SETLK, &lock);

  /* No further actions is necessary if lock was acquired successfully. */
  if (e == 0)
    return true;

  /* Raise an error if received an unexpected errno code */
  if (errno != EACCES && errno != EAGAIN)
    error (_("Internal error with the cuda-gdb lock file (errno=%d).\n"), errno);

  /* Ask the user if he want to continue */
  pid = cuda_gdb_record_read (record_idx);
#ifndef GDBSERVER
  current_top_target ()->terminal_ours ();
  if (nquery ("cuda-gdb failed to grab the lock file %s/%s.\n"
              "Another CUDA debug session (pid %d) could be in progress.\n"
              "Are you sure you want to continue? ", cuda_gdb_get_tmp_basedir (), cuda_gdb_lock_file, pid))
    {
      current_top_target ()->terminal_inferior ();
      return false;
    }
#endif

  if (record_idx != RECORD_MASTER)
    error (_("An instance of cuda-gdb(pid %d) is already using device %d.\n"
             "If you believe you are seeing this message in error, try "
             "deleting %s/%s.\n"), pid, DEVICE_RECORD (record_idx),
           cuda_gdb_get_tmp_basedir (), cuda_gdb_lock_file);
  else
    error (_("Another cuda-gdb instance is working with the lock file. Try again.\n"
             "If you believe you are seeing this message in error, try deleting %s/%s.\n"),
           cuda_gdb_get_tmp_basedir (), cuda_gdb_lock_file);
  return false;
}

static void
cuda_gdb_lock_file_initialize (void)
{
  uint32_t i;

  for (i = 0; i < CUDBG_MAX_DEVICES; i++)
    {
      cuda_gdb_record_write (RECORD_DEVICE(i), 0);
    }
}

void
cuda_gdb_record_remove_all (void *unused)
{
  int i;

  for (i = 0; i < CUDBG_MAX_DEVICES; i++)
    {
      if (dev_mask & (1 << i))
        {
          cuda_gdb_record_write (RECORD_DEVICE(i), 0);
          cuda_gdb_record_set_lock (RECORD_DEVICE(i), false);
          dev_mask &= ~(1 << i);
        }
    }

  if (cuda_gdb_lock_fd != -1)
    return;

  close (cuda_gdb_lock_fd);
  cuda_gdb_lock_fd = -1;
}

/* Check for the presence of the CUDA_VISIBLE_DEVICES variable. If it is
 * present, lock records */
static void
cuda_gdb_lock_file_create (void)
{
  struct stat st;
  char buf[CUDA_GDB_TMP_BUF_SIZE];
  char *visible_devices;
  uint32_t dev_id, num_devices = 0;
  int i;
  bool initialize_lock_file = false;
  bool grab_lock = true;
  mode_t old_umask;
  int my_pid = (int) getpid();

  /* Default == 1, can be overriden via a command-line option */
  if (cuda_use_lockfile == 0)
    return;

  snprintf (buf, sizeof (buf), "%s/%s",
              cuda_gdb_get_tmp_basedir (), cuda_gdb_lock_file);

  visible_devices = getenv ("CUDA_VISIBLE_DEVICES");

  if (stat (buf, &st) || !(S_ISREG(st.st_mode)))
    initialize_lock_file = true;

  /* Save the old umask and reset it */
  old_umask = umask (0);
  cuda_gdb_lock_fd = open (buf, O_CREAT | O_RDWR,
                           S_IRWXU | S_IRWXG | S_IRWXO);
  /* Restore the old umask */
  umask (old_umask);

  if (cuda_gdb_lock_fd == -1)
    error (_("Cannot open %s. \n"), buf);

  /* Register cleanup routine */
  /* No final cleanup chain at server side,
     cleanup function is called explicitly when server quits */
#ifndef GDBSERVER
  make_final_cleanup (cuda_gdb_record_remove_all, NULL);
#endif

  /* Get the mutex ("work") lock before doing anything */
  grab_lock = cuda_gdb_record_set_lock (RECORD_MASTER, true);
  if (!grab_lock)
    return;

  cuda_gdb_record_write (RECORD_MASTER, my_pid);

  if (initialize_lock_file)
    cuda_gdb_lock_file_initialize ();

  if (NULL == visible_devices)
    {
      /* Lock all devices */
      for (i = 0; i < CUDBG_MAX_DEVICES; i++)
        {
          grab_lock = cuda_gdb_record_set_lock (RECORD_DEVICE(i), true);
          if (!grab_lock) break;
          cuda_gdb_record_write (RECORD_DEVICE(i), my_pid);
          dev_mask |= 1 << i;
        }
    }
  else
    {
      /* Copy to local storage to prevent buffer overflows */
      strncpy (buf, visible_devices, CUDA_GDB_TMP_BUF_SIZE);

      visible_devices = buf;

      do
        {
          if (*visible_devices == ',')
            visible_devices++;

          if ((sscanf (visible_devices, "%u,", &dev_id) > 0) &&
              (++num_devices < CUDBG_MAX_DEVICES) &&
              (dev_id < CUDBG_MAX_DEVICES))
            {
              grab_lock = cuda_gdb_record_set_lock (RECORD_DEVICE(dev_id), true);
              if (!grab_lock) break;
              cuda_gdb_record_write (RECORD_DEVICE(dev_id), my_pid);
              dev_mask |= 1 << dev_id;
            }
          else
            break;
        } while ((visible_devices = strstr (visible_devices, ",")));
    }

    cuda_gdb_record_write (RECORD_MASTER, 0);
    cuda_gdb_record_set_lock (RECORD_MASTER, false);
}

static void
cuda_gdb_tmpdir_setup (void)
{
  char dirpath [CUDA_GDB_TMP_BUF_SIZE];
  int ret;
  bool dir_exists = false;
  bool override_umask = false;

  snprintf (dirpath, sizeof (dirpath), "%s/%u", cuda_gdb_get_tmp_basedir (),
            getpid ());

#ifdef __QNXHOST__
  cuda_gdb_tmpdir_cleanup_dir (dirpath); /* Try to remove the dir if it exists */
#endif /* __QNXHOST__ */
  ret = cuda_gdb_dir_create (dirpath, S_IRWXU | S_IRWXG | S_IXOTH, override_umask,
                             &dir_exists);
  if (ret)
    error (_("Error creating temporary directory %s\n"), dirpath);

#ifndef __QNXHOST__
  if (dir_exists)
    cuda_gdb_dir_cleanup_files (dirpath);
#endif /* __QNXHOST__ */

  cuda_gdb_tmp_dir = (char *) xmalloc (strlen (dirpath) + 1);
  strncpy (cuda_gdb_tmp_dir, dirpath, strlen (dirpath) + 1);

  /* No final cleanup chain at server side,
   * cleanup function is called explicitly when server quits */
#ifndef GDBSERVER
  make_final_cleanup (cuda_gdb_tmpdir_cleanup_self, NULL);
#endif
}

const char*
cuda_gdb_tmpdir_getdir (void)
{
  if (!cuda_gdb_tmp_dir)
    cuda_gdb_tmpdir_setup ();
  return cuda_gdb_tmp_dir;
}

static cuda_clock_t cuda_clock_ = 0;

cuda_clock_t
cuda_clock (void)
{
  return cuda_clock_;
}

void
cuda_clock_increment (void)
{
  ++cuda_clock_;
  if (cuda_clock_ == 0)
    warning (_("The internal clock counter used for cuda debugging wrapped around.\n"));
}

#ifndef GDBSERVER
static unsigned char *
cuda_nat_save_gdb_signal_handlers (void)
{
  unsigned char *sigs;
  int i,j;
  static int (*sighand_savers[])(int) =
    {signal_stop_state, signal_print_state, signal_pass_state};

  sigs = (unsigned char *) xmalloc (GDB_SIGNAL_LAST*ARRAY_SIZE(sighand_savers));

  for (i=0; i < ARRAY_SIZE(sighand_savers); i++)
    for (j=0; j < GDB_SIGNAL_LAST; j++)
      sigs[i*GDB_SIGNAL_LAST+j] = sighand_savers[i](j);

  return sigs;
}

static void
cuda_nat_restore_gdb_signal_handlers (unsigned char *sigs)
{
  int i,j;
  static int (*sighand_updaters[])(int,int) =
    {signal_stop_update, signal_print_update, signal_pass_update};

  for (i=0; i < ARRAY_SIZE(sighand_updaters); i++)
    for (j=0; j < GDB_SIGNAL_LAST; j++)
      sighand_updaters[i] (j, sigs[i*GDB_SIGNAL_LAST+j]);
}

static void cuda_nat_bypass_signals_cleanup (void *ptr)
{
  unsigned char *sigs = (unsigned char *) ptr;

  cuda_nat_restore_gdb_signal_handlers (sigs);
  xfree (ptr);
}

struct cleanup *
cuda_gdb_bypass_signals (void)
{
  unsigned char *sigs;
  unsigned cuda_stop_signal = cuda_options_stop_signal();
  int i;

  sigs = cuda_nat_save_gdb_signal_handlers ();
  for (i=0;i< GDB_SIGNAL_LAST; i++)
    {
      if ( i == cuda_stop_signal ||
           i == GDB_SIGNAL_TRAP  ||
           i == GDB_SIGNAL_KILL  ||
           i == GDB_SIGNAL_STOP  ||
           i == GDB_SIGNAL_CHLD  ||
           i >= GDB_SIGNAL_CUDA_UNKNOWN_EXCEPTION ) continue;
      signal_stop_update (i, 0);
      signal_pass_update (i, 1);
      signal_print_update (i, 1);
    }

  return make_cleanup (cuda_nat_bypass_signals_cleanup, sigs);
}

/* CUDA PTX registers cache */
struct cuda_ptx_cache_element {
  struct frame_id frame_id;
  cuda_coords_t coords;
  int dwarf_regnum;
  char data[16];
  int len;
};
typedef struct cuda_ptx_cache_element cuda_ptx_cache_element_t;
DEF_VEC_O(cuda_ptx_cache_element_t);

static VEC(cuda_ptx_cache_element_t) *cuda_ptx_register_cache = NULL;

/* Searches for cache entry containing given dwarf register for a given frame */
static cuda_ptx_cache_element_t *
cuda_ptx_cache_find_element (struct frame_id frame_id, int dwarf_regnum)
{
  int idx;
  struct cuda_ptx_cache_element *elem;
  cuda_coords_t coords;

  if (cuda_coords_get_current (&coords)) return NULL;

  for (idx =0;
       VEC_iterate (cuda_ptx_cache_element_t, cuda_ptx_register_cache, idx, elem);
       idx++)
    if (cuda_coords_equal (&coords, &elem->coords) &&
        frame_id_eq (frame_id, elem->frame_id) &&
        elem->dwarf_regnum == dwarf_regnum) return elem;

  return NULL;
}


/**
 *  Adds PTX register to a cache.
 *  PTX register cache is considered valid only for given lane within given frame
 */
void
cuda_ptx_cache_store_register (struct frame_info *frame, int dwarf_regnum, struct value *value)
{
  struct cuda_ptx_cache_element new_elem;
  struct cuda_ptx_cache_element *elem;

  /* If focus is not on device - return */
  if (cuda_coords_get_current (&new_elem.coords))
    return;

  /* Store information about cached register in temporary cache element*/
  new_elem.frame_id = get_frame_id (frame);
  new_elem.dwarf_regnum = dwarf_regnum;
  new_elem.len = TYPE_LENGTH(value_type(value));
  /* If element can not be cached - return */
  if (new_elem.len > sizeof(new_elem.data))
    return;
  memcpy (new_elem.data, value_contents_raw(value), new_elem.len);

  elem = cuda_ptx_cache_find_element (new_elem.frame_id, dwarf_regnum);
  if (elem)
    {
      elem->len = new_elem.len;
      memcpy (elem->data, new_elem.data, elem->len);
      return;
    }

  /* Add new element to the cache */
  VEC_safe_push (cuda_ptx_cache_element_t, cuda_ptx_register_cache, &new_elem);
}

/**
 * Retrieves previously cached value from PTX register cache.
 * \return either a previously cached value
 *         or struct value with optimized_out flag set
 */

struct value *
cuda_ptx_cache_get_register (struct frame_info *frame, int dwarf_regnum, struct type *type)
{
  struct cuda_ptx_cache_element *elem;
  struct value *retval;

  retval = allocate_value (type);

  elem = cuda_ptx_cache_find_element (get_frame_id (frame), dwarf_regnum);
  if (!elem || elem->len != TYPE_LENGTH(type) ||
      !cuda_options_variable_value_cache_enabled ())
    {
      VALUE_LVAL (retval) = not_lval;
      mark_value_bytes_optimized_out (retval, 0, value_length (retval));
      return retval;
    }

  mark_value_bytes_optimized_out (retval, 0, 0);
  set_value_cached (retval, 1);
  memcpy (value_contents_raw(retval), elem->data, elem->len);
  return retval;
}

void
cuda_ptx_cache_local_vars_iterator (const char *name, struct symbol *symbol, void *cb);

static void
cuda_ptx_cache_update_local_vars (void)
{
  struct frame_info *frame = NULL;
  const struct block *block = NULL;

  if (!cuda_focus_is_device() || find_thread_ptid (inferior_ptid)->executing) return;
  frame = get_current_frame();
  if (!frame) return;

  block = get_frame_block (frame, 0);
  if (!block) return;

  iterate_over_block_local_vars (block, cuda_ptx_cache_local_vars_iterator, frame);
}

/**
 * Refresh cuda ptx register cache
 * If cache is not empty but the focus was changed - clean up the cache
 * Then cache try to cache all local variables mapped to PTX/GPU registers.
 */
void
cuda_ptx_cache_refresh (void)
{
  cuda_coords_t coords;
  struct cuda_ptx_cache_element *elem;

  if (!VEC_empty(cuda_ptx_cache_element_t, cuda_ptx_register_cache))
    {
      elem = VEC_index (cuda_ptx_cache_element_t, cuda_ptx_register_cache, 0);
     /* If focus is still on the same lane - keep the cache intact */
     if ( (!cuda_coords_get_current (&coords) &&
          !cuda_coords_equal (&coords, &elem->coords)) ||
          !cuda_options_variable_value_cache_enabled ())
       {
         VEC_free(cuda_ptx_cache_element_t, cuda_ptx_register_cache);
         cuda_ptx_register_cache = 0;
       }
    }

  if (!cuda_options_variable_value_cache_enabled()) return;
  cuda_ptx_cache_update_local_vars ();
}


bool
cuda_managed_msymbol_p (struct bound_minimal_symbol bmsym)
{
  struct obj_section *section = MSYMBOL_OBJ_SECTION(bmsym.objfile, bmsym.minsym);
  struct objfile *obj= section ? section->objfile : NULL;
  struct gdbarch *arch = obj ? get_objfile_arch (obj) : NULL;

  if (arch == NULL) return false;

  return gdbarch_bfd_arch_info (arch)->arch == bfd_arch_m68k ?
  /* If this a device symbol MSYMBOL_TARGET_FLAG indicates if it is managed */
          (MSYMBOL_TARGET_FLAG_1 (bmsym.minsym) != 0) :
  /* Managed host symbols must be located in __nv_managed_data__ section */
          (section->the_bfd_section != NULL &&
          strcmp(section->the_bfd_section->name, "__nv_managed_data__") == 0 );
}


/* CUDA managed memory region list */
typedef struct {
  CORE_ADDR begin;
  CORE_ADDR end;
} memory_region_t;
DEF_VEC_O(memory_region_t);
static VEC(memory_region_t) *cuda_managed_memory_regions = NULL;
bool cuda_managed_memory_regions_populated = false;

/* Must be called right after inferior was suspended */
void
cuda_managed_memory_clean_regions (void)
{
  cuda_managed_memory_regions_populated = false;

  if (cuda_managed_memory_regions == NULL ||
      VEC_empty(memory_region_t, cuda_managed_memory_regions))
    return;
  VEC_free(memory_region_t, cuda_managed_memory_regions);
}

void
cuda_managed_memory_add_region (CORE_ADDR begin, CORE_ADDR end)
{
  memory_region_t new_reg = {begin, end};

  VEC_safe_push (memory_region_t, cuda_managed_memory_regions, &new_reg);
}

static void
cuda_managed_memory_populate_regions (void)
{
  CUDBGMemoryInfo regions[16];
  uint32_t regions_returned;
  uint64_t end;
  uint32_t cnt;
  uint64_t start_addr = 0;

  /* Check if information about managed memory regions has been queried already */
  if (cuda_managed_memory_regions_populated)
    return;

  do {
    cuda_api_get_managed_memory_region_info (start_addr, regions, ARRAY_SIZE(regions), &regions_returned);
    if (regions_returned == 0)
      return;
    /* Add fetched queries to the list and updated start address*/
    for (cnt = 0; cnt < regions_returned; cnt++)
      {
        end = regions[cnt].startAddress + regions[cnt].size;
        if (start_addr < end) start_addr = end;
        cuda_managed_memory_add_region ( regions[cnt].startAddress, end);
      }
  } while (regions_returned == ARRAY_SIZE (regions));
  cuda_managed_memory_regions_populated = true;
}

bool
cuda_managed_address_p (CORE_ADDR addr)
{
  int idx;
  memory_region_t *elem;

  cuda_managed_memory_populate_regions ();

  for (idx =0;
       VEC_iterate (memory_region_t, cuda_managed_memory_regions, idx, elem);
       idx++)
    if (elem->begin <= addr && elem->end > addr)
       return true;
  return false;
}

bool
cuda_is_value_managed_pointer (struct value *value)
{
  bool result = false;
  struct type *type = value ? value_type (value) : NULL;

  /* Sanity checks */
  if (type == NULL || TYPE_CODE (type) != TYPE_CODE_PTR)
    return result;

  TRY
    {
      result = cuda_managed_address_p (unpack_pointer (type, value_contents_for_printing (value)));
    }
  CATCH (e, RETURN_MASK_ALL)
    {
      if (e.reason != 0)
	return false;
    }
  END_CATCH

  return result;
}
#endif /* GDBSERVER */

#ifndef __APPLE__
static int
cuda_gdb_uid_from_pid (int pid)
{
  int uid = -1;
  FILE *procfile;
  char buffer[MAXPATHLEN], fname[MAXPATHLEN];

  /* Determine the uid by reading /proc/$pid/status */
  sprintf (fname, "/proc/%d/status", pid);
  procfile = fopen (fname, "r");
  if (procfile == NULL)
    return uid;

  while (fgets (buffer, MAXPATHLEN, procfile) != NULL)
    {
      if (strncmp (buffer, "Uid:\t", 5) != 0)
        continue;
      if (sscanf (buffer+5, "%d", &uid) != 1)
          uid = -1;
      break;
    }
  fclose (procfile);

  return uid;
}

#else

static int
cuda_gdb_uid_from_pid (int pid)
{
  int mib[4];
  struct kinfo_proc proc;
  size_t proc_size = sizeof (proc);

  mib[0] = CTL_KERN;
  mib[1] = KERN_PROC;
  mib[2] = KERN_PROC_PID;
  mib[3] = pid;

  if (sysctl (mib, 4, &proc, &proc_size, NULL, 0) == 0 &&
      proc_size > offsetof(struct kinfo_proc, kp_eproc.e_ucred.cr_uid) )
    return proc.kp_eproc.e_ucred.cr_uid;

  return -1;
}
#endif

bool
cuda_gdb_chown_to_pid_uid (int pid, const char *path)
{
  int uid;

  if (pid <= 0)
    return true;

  uid = cuda_gdb_uid_from_pid (pid);
  if (uid == -1)
    return true;

  return chown (path, uid, -1) == 0;
}

void
cuda_utils_initialize (void)
{
  static bool utils_initialized = false;

  /* Check if cuda utils were already initialized*/
  if (utils_initialized) return;

  /* Create the base temporary directory */
  cuda_gdb_tmpdir_create_basedir ();

  /* Create a lockfile to prevent multiple instances of cuda-gdb from
   * interfering with each other */
  cuda_gdb_lock_file_create ();

  /* Populate the temporary directory with a unique subdirectory for this
   * instance. */
  if (!cuda_gdb_tmp_dir)
    cuda_gdb_tmpdir_setup ();

  utils_initialized = true;
}

static bool cuda_host_address_resident_on_gpu = false;

void
cuda_set_host_address_resident_on_gpu (bool val)
{
  cuda_host_address_resident_on_gpu = val;
}

bool
cuda_is_host_address_resident_on_gpu (void)
{
  return cuda_host_address_resident_on_gpu;
}

static bool cuda_app_uses_uvm = false;

bool
cuda_is_uvm_used (void)
{
  return cuda_app_uses_uvm;
}

void
cuda_set_uvm_used (bool value)
{
  cuda_app_uses_uvm = value;
}
