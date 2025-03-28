C**********************************************************************C
C***********************  BeoPEST Public Data  ************************C
C**********************************************************************C
      MODULE BEOPEST
      IMPLICIT NONE
      LOGICAL BEOMASTER,BEOSLAVE,BEOLASTLOC
      INTEGER MAXNODE
      END
C**********************************************************************C
C***********************  BeoPEST private data  ***********************C
C**********************************************************************C
C  BeoPEST/TCP uses LIVE to track nodes
C  LIVE
C    -1 = idle
C     0 = offline
C     1 = running
C  BeoPEST/MPI uses IRDQ to store requests
      MODULE BEOPRIV
      IMPLICIT NONE
      ALLOCATABLE      LIVE,IRDQ,WALL,STRT,NUM,TEXT,PARBUF,OBSBUF
      LOGICAL          TCP,MPI
      INTEGER          NPAR,NOBS
      INTEGER          LIVE(:),IRDQ(:,:),NUM(:)
      DOUBLE PRECISION WALL(:),STRT(:),TOTIM,TSUM
      DOUBLE PRECISION PARBUF(:,:),OBSBUF(:,:)
      CHARACTER*256    TEXT(:)
      END
C**********************************************************************C
C********************  Initialize communications  *********************C
C**********************************************************************C
      SUBROUTINE BEOINIT(HOST,MXNODE)
      USE           BEOPEST
      INCLUDE       'beopest.inc'
#ifdef BEOMPI
      INCLUDE 'mpif.h'
      INTEGER IOS
#endif
      INTEGER          MXNODE,NODE
      DOUBLE PRECISION WALLTIME
      CHARACTER*10     CNN
      CHARACTER*(*)    HOST

      TCP = (MXNODE.gt.0)
      MPI = .not.TCP
C  Initialize TCP
      if (TCP) then
        MAXNODE = MXNODE
        call TCPOPEN(MAXNODE,TRIM(HOST)//CHAR(0),NODE)
C  Initialize MPI
      else
#ifdef BEOMPI
        call MPI_INIT(IOS)
        call MPI_COMM_RANK(MPI_COMM_WORLD,NODE   ,IOS)
        call MPI_COMM_SIZE(MPI_COMM_WORLD,MAXNODE,IOS)
        if (MAXNODE.eq.1) STOP 'Cannot run BeoPEST/MPI with one node'
#else
        STOP 'Compiled without MPI'
#endif
      endif

C  Mark master and slave
      BEOMASTER = (NODE.eq.0)
      BEOSLAVE  = (NODE.gt.0)
C  Allocate memory for tracking slave execution times
      if (BEOMASTER) then
        ALLOCATE(WALL(MAXNODE),STRT(MAXNODE),NUM(MAXNODE),TEXT(MAXNODE))
        NUM  = 0
        WALL = 0
      else
        MAXNODE = 1
      endif

C  Initialize MASTER array for tracking slaves
      if (BEOMASTER) then
        TSUM = 0
        TOTIM = WALLTIME()
        if (TCP) then
          ALLOCATE(LIVE(MAXNODE))
          LIVE = 0
        else
#ifdef BEOMPI
          ALLOCATE(IRDQ(2,MAXNODE))
          IRDQ = MPI_REQUEST_NULL
C         Directory name for each slave
          do NODE=1,MAXNODE
            WRITE(CNN,'(I10)') NODE
            TEXT(NODE) = TRIM(HOST)//TRIM(ADJUSTL(CNN))
          enddo
#endif
        endif
C  On MPI slaves change directory
      else if (MPI) then
        WRITE(CNN,'(I10)') NODE
        call SETDIR(TRIM(HOST)//TRIM(ADJUSTL(CNN))//CHAR(0))
      endif
      END
C**********************************************************************C
C***********************  Get number of nodes  ************************C
C**********************************************************************C
      SUBROUTINE BEONODE(KP0,KP1)
      USE     BEOPEST
      INCLUDE 'beopest.inc'
      INTEGER KP0,KP1

      KP0 = 1
      if (TCP) then
        call TCPNODE(LIVE,WALL,TEXT,KP1)
      else
        KP1 = MAXNODE-1
      endif
      END
C**********************************************************************C
C**********************  Terminate connections  ***********************C
C**********************************************************************C
      SUBROUTINE BEOFIN
      USE      BEOPEST
      INCLUDE 'beopest.inc'
#ifdef BEOMPI
      INCLUDE 'mpif.h'
      INTEGER IOS
#endif
      INTEGER K,KP0,KP1
      DOUBLE PRECISION WALLTIME
      CHARACTER*256 BEOTEXT

C  Master sends count zero (close) to slaves before closing TCP connections
      if (BEOMASTER) then
        call BEONODE(KP0,KP1)
        do K=KP0,KP1
          print '(A,I4,F8.3,I5,1X,A)','Node',
     |      K,WALL(K),NUM(K),TRIM(BEOTEXT(K))
        enddo
        TOTIM = WALLTIME()-TOTIM
        print '(A,F12.1)','Total CPU time     ',TSUM
        print '(A,F12.1)','Total elapsed time ',TOTIM
        print '(A,F12.3)','Speedup            ',TSUM/TOTIM
        if (TCP) then
          do K=KP0,KP1
            if (LIVE(K).ne.0) call TCPSEND(K,0,4,1)
          enddo
          call TCPCLOSE
          DEALLOCATE(LIVE)
        else
#ifdef BEOMPI
          do K=KP0,KP1
            call MPI_SEND(0,1,MPI_INTEGER,K,0,MPI_COMM_WORLD,IOS)
          enddo
          call MPI_Finalize(IOS)
          DEALLOCATE(IRDQ)
#endif
        endif
        DEALLOCATE(WALL,STRT,NUM,TEXT,PARBUF,OBSBUF)
C  Slave just closes connection
      else if (BEOSLAVE) then
        if (TCP) then
          call TCPCLOSE
        else
#ifdef BEOMPI
          call MPI_Finalize(IOS)
#endif
        endif
      endif
      END
C**********************************************************************C
C********************  Broadcast Parameter Counts  ********************C
C**********************************************************************C
      SUBROUTINE BEOCAST(NP,NT,CSUM)
      USE     BEOPEST
      INCLUDE 'beopest.inc'
#ifdef BEOMPI
      INCLUDE 'mpif.h'
#endif
      INTEGER NP,NT,CSUM,N,IOS

C  MPI Broadcast values
      if (MPI) then
#ifdef BEOMPI
        call MPI_BCAST(NP,1,MPI_INTEGER,0,MPI_COMM_WORLD,IOS)
        call MPI_BCAST(NT,1,MPI_INTEGER,0,MPI_COMM_WORLD,IOS)
        call MPI_BCAST(CSUM,1,MPI_INTEGER,0,MPI_COMM_WORLD,IOS)
#endif
C  Set values to send on master
      else if (BEOMASTER) then
        call TCPCAST(NP,NT,CSUM);
C  Receive values on slave
      else if (BEOSLAVE) then
        call TCPRECV(0,NP,4,1)
        call TCPRECV(0,NT,4,1)
        call TCPRECV(0,CSUM,4,1)
      endif
C  Set values for other routines to use
      NPAR = NP
      NOBS = NT

C  Allocate memory for buffers
      if (BEOMASTER) then
C       TCP manages memory internally - just need one buffer
        if (TCP) then
          N = 1
C       With MPI allocate memory for all nades
        else
          N = MAXNODE
        endif
C       Allocate memory
        ALLOCATE(PARBUF(NPAR,N),OBSBUF(NOBS,N),STAT=IOS)
        if (IOS.ne.0) call BEOFATAL('Cannot allocate buffer memory')
      endif
      END
C**********************************************************************C
C*******************  Push run from master to slave  ******************C
C**********************************************************************C
      SUBROUTINE BEOPUSH(IDST,PAR)
      INCLUDE 'beopest.inc'
#ifdef BEOMPI
      INCLUDE 'mpif.h'
      INTEGER IOS
#endif
      INTEGER          IDST,I
      DOUBLE PRECISION PAR(NPAR),WALLTIME

      STRT(IDST) = WALLTIME()
      if (TCP) then
C  Post asynchronous receive for parameters and targets
        call TCPPOST(IDST,0,8,NPAR)
        call TCPPOST(IDST,1,8,NOBS)
C  Send parameters to slave
        LIVE(IDST) = 1
        call TCPSEND(IDST,NPAR,4,1)
        call TCPSEND(IDST,PAR,8,NPAR)
      else
#ifdef BEOMPI
C  Post asynchronous receive for parameters and targets
        call MPI_IRECV(PARBUF(1,IDST),NPAR,MPI_DOUBLE_PRECISION,
     |     IDST,1,MPI_COMM_WORLD,IRDQ(1,IDST),IOS)
        call MPI_IRECV(OBSBUF(1,IDST),NOBS,MPI_DOUBLE_PRECISION,
     |     IDST,2,MPI_COMM_WORLD,IRDQ(2,IDST),IOS)
C  Send parameters to slave
        call MPI_SEND(NPAR,1,MPI_INTEGER,IDST,0,MPI_COMM_WORLD,IOS)
        call MPI_SEND(PAR,NPAR,MPI_DOUBLE_PRECISION,
     |    IDST,0,MPI_COMM_WORLD,IOS)
#endif
      endif
      END
C**********************************************************************C
C******************  Wait for parameters from master  *****************C
C**********************************************************************C
      FUNCTION BEOGET(PAR)
      INCLUDE 'beopest.inc'
#ifdef BEOMPI
      INCLUDE 'mpif.h'
      INTEGER STAT(MPI_STATUS_SIZE),IOS
#endif
      LOGICAL BEOGET
      INTEGER N,I
      DOUBLE PRECISION PAR(NPAR)

C  Get count:  0 => done
      if (TCP) then
        call TCPRECV(0,N,4,1)
      else
#ifdef BEOMPI
        call MPI_RECV(N,1,MPI_INTEGER,0,0,MPI_COMM_WORLD,STAT,IOS)
#endif
      endif

C  Get parameters from master
      if (N.le.0) then
C       Do nothing
      else if (TCP) then
        call TCPRECV(0,PAR,8,NPAR)
      else
#ifdef BEOMPI
        call MPI_RECV(PAR,NPAR,MPI_DOUBLE_PRECISION,
     |                0,0,MPI_COMM_WORLD,STAT,IOS)
#endif
      endif

C  Mark whether we got parameters or done
      BEOGET = (N.gt.0)
      END
C**********************************************************************C
C**********************  Send targets to master  **********************C
C**********************************************************************C
      SUBROUTINE BEOPUT(PAR,OBS)
      INCLUDE          'beopest.inc'
#ifdef BEOMPI
      INCLUDE 'mpif.h'
      INTEGER IOS,I
#endif
      DOUBLE PRECISION PAR(NPAR),OBS(NOBS)

C  Send observations to master
      if (TCP) then
        call TCPSEND(0,PAR,8,NPAR)
        call TCPSEND(0,OBS,8,NOBS)
      else
#ifdef BEOMPI
        call MPI_SEND(PAR,NPAR,MPI_DOUBLE_PRECISION,
     |                0,1,MPI_COMM_WORLD,IOS)
        call MPI_SEND(OBS,NOBS,MPI_DOUBLE_PRECISION,
     |                0,2,MPI_COMM_WORLD,IOS)
#endif
      endif
      END
C**********************************************************************C
C**********************  Check for completed run  *********************C
C**********************************************************************C
      FUNCTION BEOSTAT(IDST)
      INCLUDE          'beopest.inc'
#ifdef BEOMPI
      INCLUDE 'mpif.h'
      LOGICAL FLAG
      INTEGER STAT(MPI_STATUS_SIZE),K,IOS
#endif
      LOGICAL          NEW
      INTEGER          BEOSTAT,IDST,IFLG
      DOUBLE PRECISION WALLTIME
C  RETURNS
C   1 done
C   0 working
C  -1 failed

      NEW = .false.
      if (TCP) then
C  Running
        if (LIVE(IDST).gt.0) then
C         Check if results were received
          call TCPTEST(IDST,IFLG)
C         Node has completed
          if (IFLG.gt.0) then
            NEW = .true.
            LIVE(IDST) = -1
C         Mark node as failed
          else if (IFLG.lt.0) then
            LIVE(IDST) = 0
          endif
C         Mark results as received
          BEOSTAT = IFLG
C  Idle => done
        else if (LIVE(IDST).lt.0) then
          BEOSTAT = 1
C  Dead
        else
          BEOSTAT = -1
        endif
      else
#ifdef BEOMPI
C  Check for outstanding receives that have completed
        do K=1,2
          if (IRDQ(K,IDST).ne.MPI_REQUEST_NULL) then
            call MPI_TEST(IRDQ(K,IDST),FLAG,STAT,IOS)
            if (FLAG) then
              NEW = .true.
              IRDQ(K,IDST) = MPI_REQUEST_NULL
            endif
          endif
        enddo
C  Return status
        if (IRDQ(1,IDST).eq.MPI_REQUEST_NULL .and.
     |      IRDQ(2,IDST).eq.MPI_REQUEST_NULL) then
          BEOSTAT = 1
        else
          BEOSTAT = 0
        endif
#else
        BEOSTAT = 0
#endif
      endif
      if (BEOSTAT.gt.0 .and. NEW) then
        NUM(IDST) = NUM(IDST)+1
        WALL(IDST) = WALLTIME() - STRT(IDST)
        TSUM = TSUM + WALL(IDST)
      endif
      END
C**********************************************************************C
C****************************  Get run time  **************************C
C**********************************************************************C
      FUNCTION BEOWALL(IDST)
      INCLUDE          'beopest.inc'
      INTEGER          IDST
      DOUBLE PRECISION BEOWALL

      BEOWALL = WALL(IDST)
      END
C**********************************************************************C
C**********************  Check for completed run  *********************C
C**********************************************************************C
      SUBROUTINE BEOFATAL(MSG)
      INCLUDE      'beopest.inc'
#ifdef BEOMPI
      INCLUDE 'mpif.h'
      INTEGER IOS
#endif
      CHARACTER(*) MSG

#ifdef BEOMPI
      if (MPI) call MPI_Finalize(IOS)
#endif
      call FATAL(TRIM(MSG)//CHAR(0))
      END
C**********************************************************************C
C**************  Return wall time in seconds since 1970  **************C
C**********************************************************************C
      FUNCTION WALLTIME()
      IMPLICIT NONE
      DOUBLE PRECISION WALLTIME
      INTEGER  d(8),MDO(12),JY,JD
      DATA MDO /306,337,0,31,61,92,122,153,184,214,245,275/

C  Get system date and time (wall clock local time)
      call DATE_AND_TIME(values=d)
C  Calculate sidereal Julian day for dates in Gregorian Calendar
      if (d(2).lt.3) then
        JY = d(1)-1
      else
        JY = d(1)
      endif
      JD = 1721119 + d(3) + MDO(d(2)) + 365*JY + JY/4 - JY/100 + JY/400
C  1-Jan-1970 is JD 2440588
C  Time is adjusted to UTC to avoid daylight savings glitches
      WALLTIME = 86400*(JD-2440588) + 3600*d(5) + (d(6)-d(4))*60
     |         + d(7) + 1D-3*d(8)
      END
C**********************************************************************C
C***********************  Return name of node  ************************C
C**********************************************************************C
      FUNCTION BEOTEXT(K)
      USE     BEOPEST
      INCLUDE 'beopest.inc'
      INTEGER       K
      CHARACTER*256 BEOTEXT

      BEOTEXT = TEXT(K)
      END
C**********************************************************************C
C***********************  Return name of node  ************************C
C**********************************************************************C
      SUBROUTINE BEOSTORE(NODE,JOB,PARREG,OBSREG)
      USE     PESTDATA,  ONLY: PARREGFILE,OBSREGFILE
      USE     BEOPEST
      INCLUDE 'beopest.inc'
      INTEGER NODE,JOB,PARREG,OBSREG,IOS,K,I

C  Copy data from TCP buffer and store in register file
      if (TCP) then
        K = 1
        call TCPLOAD(NODE,0,PARBUF(1,K),8,NPAR)
        call TCPLOAD(NODE,1,OBSBUF(1,K),8,NOBS)
#ifdef BEOMPI
C  Set index to node
      else
        K = NODE
#endif
      endif
C  Store data to register files
      call STORE_PARALLEL_REGISTER(IOS,
     |  NPAR,PARREG,JOB,PARBUF(1,K),PARREGFILE)
      if (IOS.ne.0) call BEOFATAL('Cannot store returned PAR')
      call STORE_PARALLEL_REGISTER(IOS,
     |  NOBS,OBSREG,JOB,OBSBUF(1,K),OBSREGFILE)
      if (IOS.ne.0) call BEOFATAL('Cannot store returned OBJ')
      END
