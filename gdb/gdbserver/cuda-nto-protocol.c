/*
 * NVIDIA CUDA Debugger CUDA-GDB Copyright (C) 2017-2020 NVIDIA Corporation
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

#include "cuda-nto-protocol.h"

#include <sys/socket.h>
#include <string.h>

#define FRAME_CHAR '\x7e'
#define ESCAPE_CHAR '\x7d'

extern int pdebug_sockfd;
extern bool qnx_gdbserver_debug;

int mid; /* pdebug protocol message id */
char send_receive_buffer[MAX_PACKET_SIZE];

/* FOR DEBUGGING */
char *message_types[] =
{
  "DStMsg_connect",   /*  0  0x0 */
  "DStMsg_disconnect",    /*  1  0x1 */
  "DStMsg_select",    /*  2  0x2 */
  "DStMsg_mapinfo",   /*  3  0x3 */
  "DStMsg_load",      /*  4  0x4 */
  "DStMsg_attach",    /*  5  0x5 */
  "DStMsg_detach",    /*  6  0x6 */
  "DStMsg_kill",      /*  7  0x7 */
  "DStMsg_stop",      /*  8  0x8 */
  "DStMsg_memrd",     /*  9  0x9 */
  "DStMsg_memwr",     /* 10  0xA */
  "DStMsg_regrd",     /* 11  0xB */
  "DStMsg_regwr",     /* 12  0xC */
  "DStMsg_run",     /* 13  0xD */
  "DStMsg_brk",     /* 14  0xE */
  "DStMsg_fileopen",    /* 15  0xF */
  "DStMsg_filerd",    /* 16  0x10 */
  "DStMsg_filewr",    /* 17  0x11 */
  "DStMsg_fileclose",   /* 18  0x12 */
  "DStMsg_pidlist",   /* 19  0x13 */
  "DStMsg_cwd",     /* 20  0x14 */
  "DStMsg_env",     /* 21  0x15 */
  "DStMsg_base_address",    /* 22  0x16 */
  "DStMsg_protover",    /* 23  0x17 */
  "DStMsg_handlesig",   /* 24  0x18 */
  "DStMsg_cpuinfo",   /* 25  0x19 */
  "DStMsg_tidnames",    /* 26  0x1A */
  "DStMsg_procfsinfo",    /* 27  0x1B */
  "DStMsg_procfsstatus",    /* 28  0x1C */
  /* CUDA request */
  "DStMsg_cuda",    /* 29 0x1D */
  /* Room for new codes here.  */
  "unused",
  "unused",
  "DSrMsg_err",   /* 32  0x20 */
  "DSrMsg_ok",      /* 33  0x21 */
  "DSrMsg_okstatus",    /* 34  0x22 */
  "DSrMsg_okdata",    /* 35  0x23 */
  /* CUDA response */
  "DSrMsg_okcuda",    /* 36  0x24 */
  /* Room for new codes here.  */
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "unused",
  "DShMsg_notify"   /* 64  0x40 */
};

int
get_raw_pdebug_packet_size (char *src, int max_size)
{
  int length;

  length = 0;
  if (src[length++] != FRAME_CHAR)
    {
      error ("Illegal packet");
    }
  while (length < max_size && src[length++] != FRAME_CHAR)
    { }
  if (src[length - 1] != FRAME_CHAR)
    {
      error ("Illegal packet");
    }
  return length;
}

int
unpack_pdebug_packet (DScomm_t *packet, char *src)
{
  unsigned char modifier = 0;
  unsigned char c;
  unsigned char csum = 0;
  unsigned char *bp = packet->buf;

  src++; /* skip FRAME_CHAR */
  for (;;)
    {
      c = *src++;
      switch (c)
        {
        case ESCAPE_CHAR:
          modifier = 0x20;
          continue;
        case FRAME_CHAR:
          if (csum != 0xff)
            {
              error("Error unpacking a CUDA packet");
            }
          *--bp = '\0'; /* overwrite the checksum */
          return bp - packet->buf;
        default:
          c ^= modifier;
          csum += c;
          *bp++ = c;
          break;
        }
      modifier = 0;
    }
}

/* Assumes dest of at least MAX_PACKET_SIZE length */
int
pack_pdebug_packet (char *dest, const DScomm_t *packet, int length)
{
  char csum = 0;
  char *bp = dest;

  *bp++ = FRAME_CHAR;
  for (int i = 0; i < length; i++)
    {
      char c = packet->buf[i];

      csum += c;
      switch (c)
        {
        case FRAME_CHAR:
        case ESCAPE_CHAR:
          *bp++ = ESCAPE_CHAR;
          c ^= 0x20;
          break;
        }
      *bp++ = c;
    }

  csum ^= 0xff;
  switch (csum)
    {
    case FRAME_CHAR:
    case ESCAPE_CHAR:
      *bp++ = ESCAPE_CHAR;
      csum ^= 0x20;
      break;
    }
  *bp++ = csum;
  *bp++ = FRAME_CHAR;

  return bp - dest;
}

int
pack_cuda_packet (char *dest, char *src, int length)
{
  DScomm_t resp;

  if (length == 0)
    {
      length = strlen (src);
    }

  /* FIXME: always assumes little-endian */
  memset (&resp, 0, sizeof (DScomm_t));

  resp.pkt.hdr.cmd = DSrMsg_okcuda;
  resp.pkt.hdr.mid = mid++;
  resp.pkt.hdr.channel = SET_CHANNEL_DEBUG;
  memcpy (resp.pkt.cuda.data, src, length);

  return pack_pdebug_packet (dest, &resp, offsetof (DStMsg_cuda_t, data) + length);
}

void
putpkt_pdebug (DScomm_t *packet, int length)
{
  length = pack_pdebug_packet (send_receive_buffer, packet, length);
  if (write (pdebug_sockfd, send_receive_buffer, length) != length)
    {
      error ("Error writing a pdebug packet");
    }
}

void
getpkt_pdebug (DScomm_t *packet)
{
  int length;

  if (read (pdebug_sockfd, send_receive_buffer, MAX_PACKET_SIZE) == 0)
    {
      error ("Error reading a pdebug packet");
    }
  unpack_pdebug_packet (packet, send_receive_buffer);
}

int
qnx_write_inferior_memory (CORE_ADDR memaddr, const unsigned char *myaddr, int len)
{
  DScomm_t packet;

  memset (&packet, 0, sizeof (DScomm_t));

  packet.pkt.hdr.cmd = DStMsg_memwr;
  packet.pkt.hdr.mid = mid++;
  packet.pkt.hdr.channel = SET_CHANNEL_DEBUG;

  packet.pkt.memwr.addr = memaddr; /* FIXME: check if the endianness is always correct */
  memcpy (packet.pkt.memwr.data, myaddr, len);

  if (qnx_gdbserver_debug)
    {
      printf ("SRV => PDB %s:%x[%x]\n", message_types[packet.buf[0]], packet.buf[1], packet.buf[2]);
    }
  putpkt_pdebug (&packet, offsetof (DStMsg_memwr_t, data) + len);
  getpkt_pdebug (&packet);
  if (qnx_gdbserver_debug)
    {
      printf ("SRV <= PDB %s:%x[%x]\n", message_types[packet.buf[0]], packet.buf[1], packet.buf[2]);
    }

  switch (packet.pkt.hdr.cmd)
    {
    case DSrMsg_ok:
      return len;
    case DSrMsg_okstatus:
      return packet.pkt.okstatus.status;
    }

  return 0;
}
