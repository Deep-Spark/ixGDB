/* Copyright (C) 2009-2019 Free Software Foundation, Inc.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include "defs.h"
#include "progspace-and-thread.h"
#include "inferior.h"

/* See progspace-and-thread.h  */

void
switch_to_program_space_and_thread (program_space *pspace)
{
  inferior *inf;

  /* CUDA - focus */
  /* This is a limitation. We assume that there is only one program space (for
     now). If the focus is already set on a CUDA device, we keep it there.
     Otherwise, we let GDB behave normally. */
  if (cuda_focus_is_device ())
    {
      gdb_assert (number_of_program_spaces () <= 1);
      set_current_program_space (pspace);
      return;
    }

  inf = find_inferior_for_program_space (pspace);
  if (inf != NULL && inf->pid != 0)
    {
      thread_info *tp = any_live_thread_of_inferior (inf);

      if (tp != NULL)
	{
	  switch_to_thread (tp);
	  /* Switching thread switches pspace implicitly.  We're
	     done.  */
	  return;
	}
    }

  switch_to_no_thread ();
  set_current_program_space (pspace);
}
