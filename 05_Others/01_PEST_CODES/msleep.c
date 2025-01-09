#include <stdio.h>
#include <sys/time.h>
#include <sys/types.h>
/* ------------------------------------------------------------------ */
/* ------------------  Millisecond version of sleep  ---------------- */
/* ------------------------------------------------------------------ */
int msleep_(int *mS)
{
   struct timeval Timer;

   if ((*mS < 0) || (*mS > 3600000)) return -1;
   Timer.tv_sec  = *mS/1000;
   Timer.tv_usec = (*mS%1000)*1000;
   if (select(0,NULL,NULL,NULL,&Timer) < 0) return -1;
   return 0;
}
