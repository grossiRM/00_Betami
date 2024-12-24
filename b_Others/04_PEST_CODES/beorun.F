#ifdef BEO
C**********************************************************************C
C***  Slave version just runs those parameters requested by master  ***C
C**********************************************************************C
      SUBROUTINE RUNSLAVE
      USE PESTDATA
      IMPLICIT NONE
      ALLOCATABLE      PAR,OBS,NW,LL,LCINS,IIOBS,OBSN1,OBSN2,
     |                 PWORD,PARDEL,MRKDEL,A
      LOGICAL          BEOGET
      INTEGER          JFAIL,NP,NT,NS,ASIZE,NUML,NINSTR,IOS,ILINE,
     |                 INCPAR,NBLBMX,NCALL,MCALL,
     |                 NW(:),LL(:),LCINS(:),IIOBS(:),OBSN1(:),OBSN2(:)
      DOUBLE PRECISION PAR(:),OBS(:)
      CHARACTER        PARDEL(:),MRKDEL(:),A(:)
      CHARACTER*23     PWORD(:)
      CHARACTER*2000   BUF
      COMMON /MODCAL/  NCALL,MCALL

C  Get parameter count NP and target count NT from master
C  Use this as a sanity check
      call BEOCAST(NP,NT,NS)
      if (NP.ne.NPAR .or. NT.ne.NOBS) then
        print *,'NPAR Slave=',NPAR,' Master=',NP
        print *,'NOBS Slave=',NOBS,' Master=',NT
        call BEOFATAL('Par/Obs count mismatch')
      endif
C  Allocate memory for files, parameters and observations
      allocate(PARDEL(NTPLFLE),MRKDEL(NINSFLE),STAT=IOS)
      if (IOS.ne.0) call BEOFATAL('ALLOCATE del failed')
      allocate(PAR(NPAR),NW(NPAR),PWORD(NPAR),STAT=IOS)
      if (IOS.ne.0) call BEOFATAL('ALLOCATE par failed')
      allocate(OBS(NOBS),IIOBS(NOBS),OBSN1(NOBS),OBSN2(NOBS),STAT=IOS)
      if (IOS.ne.0) call BEOFATAL('ALLOCATE obs failed')
C  Read template and instruction files
      call IOCTL(JFAIL,NPAR,NTPLFLE,NINSFLE,NUML,ASIZE,NINSTR,NBLBMX,
     |  NW,TEMPFLE,INFLE,INSFLE,OUTFLE,PARDEL,MRKDEL,APAR,CLINE,ILINE)
      if (JFAIL.ne.0) call BEOFATAL('IOCTL failed')
C  Allocate memory for templates
      if (NUML.eq.0) NUML = 1
      allocate(A(ASIZE),LL(NUML),LCINS(NINSTR),STAT=IOS)
      if (IOS.ne.0) call BEOFATAL('ALLOCATE tpl/ins failed')
C  Read instruction file
      call READINS(LCINS,NINSTR,A,ASIZE,MRKDEL,JFAIL)
      if (IOS.ne.0) call BEOFATAL('READINS failed')

C  Wait for set of parameters
      MCALL = 0
      do while (BEOGET(PAR))
        MCALL = MCALL+1
C  Write model input files
        INCPAR = 0
        call INWRIT(JFAIL,NPAR,PRECIS,NOPNT,NTPLFLE,NW,PAR,SCALE,
     |    OFFSET,PARDEL,PWORD,INFLE,TEMPFLE,APAR,CLINE,MESSFILE,INCPAR,
     |    ITRANS,NOBS)
        if (JFAIL.ne.0) call BEOFATAL('INWRIT failed')
C  Run model
        call SYSTEM(TRIM(COMLIN(1)))
C  Read model output files
        call OUTRD(JFAIL,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,
     |             LL,OBSN1,OBSN2,IIOBS,OBS,AOBS,A,MRKDEL,CLINE,BUF)
        if (JFAIL.ne.0) call BEOFATAL('OUTRD failed')
C  Send results back
        call BEOPUT(PAR,OBS)
      enddo
C     print '(A,I6)','Total model calls:',MCALL
      END
C**********************************************************************C
C******  Routine to schedule manage slaves to do series of jobs  ******C
C**********************************************************************C
C  STAT is node status
C   -1 = offline
C    0 = idle
C  k>0 = running job k
C
C  MODE is the job status
C   -1 = not run
C    0 = run
C  k>0 = running on node k
      SUBROUTINE RUNMASTER(PARREG,OBSREG,NJOB,JFAIL)
      USE PESTDATA
      USE BEOPEST
      IMPLICIT NONE
      ALLOCATABLE      MODE,STAT,PAR
      INTEGER          PARREG,OBSREG,NJOB,JFAIL,BEOSTAT,NCALL,MCALL,
     |                 MODE(:),STAT(:),node,kjob,node0,node1,
     |                 idle,next,done,IOS
      DOUBLE PRECISION PAR(:),fast,BEOWALL
      CHARACTER*256    BEOTEXT
      COMMON /MODCAL/  NCALL,MCALL

C  Allocate memory for parameter and slave arrays
      allocate(PAR(NPAR),MODE(NJOB),STAT(MAXNODE),STAT=IOS)
      if (IOS.ne.0) call BEOFATAL('Cannot allocate memory in RUNMASTER')
C  Mark all slaves as dead
      STAT = -1
C  Mark all jobs as not run
      MODE = -1
C  kjob is the number of jobs completed
      kjob = 0
C  Launch jobs until all jobs are completed
      do while (kjob.lt.NJOB)

C       Poll all slaves to check for a change in status
        call BEONODE(node0,node1)
        do node=node0,node1
C         Slave is idle
          if (BEOSTAT(node).gt.0) then
C           Mark job as completed
            if (STAT(node).gt.0) then
              kjob = kjob+1
              MODE(STAT(node)) = 0
C             Store  parameters and observations
              call BEOSTORE(node,STAT(node),PARREG,OBSREG)
            endif
C           Mark slave as idle
            STAT(node) = 0
C         Slave has died
          else if (BEOSTAT(node).lt.0) then
C           Reschedule job
            if (STAT(node).gt.0) then
              print '(A,I4,2A)','FAILED job',STAT(node),
     |          ' on node ',TRIM(BEOTEXT(node))
              MODE(STAT(node)) = -1
            endif
C           Mark slave as offline
            STAT(node) = -1
          endif
        enddo

C  Scan slaves to find the fastest idle node
        idle = 0
        fast = 1d300
        do node=node0,node1
          if (STAT(node).eq.0 .and. BEOWALL(node).le.fast) then
            fast = BEOWALL(node)
            idle = node
          endif
        enddo
C  Find next unscheduled job
        do next=1,NJOB
          if (MODE(next).lt.0) goto 100
        enddo
        next = 0

C  Push next job to idle slave
100     if (idle.gt.0 .and. next.gt.0) then
          STAT(idle) = next
          MODE(next) = idle
C  Retrieve parameters
          call RETRIEVE_PARALLEL_REGISTER(IOS,
     |      NPAR,PARREG,next,PAR,PARREGFILE)
          if (IOS.ne.0) call BEOFATAL('Cannot retrieve PAR ')
C  Push to slave
          call BEOPUSH(idle,PAR)
C       Nothing to do right now so sleep a spell before trying again
        else
          call MSLEEP(5)
        endif
      enddo
C  Count model runs
      NCALL = NCALL+NJOB
C  Done - free MODE and STAT arrays
      deallocate(MODE,STAT)
      END
C**********************************************************************C
C**************  Get number of 'fast' slaves available  ***************C
C**********************************************************************C
      SUBROUTINE FASTSLAVES(NUMSLAVE)
      USE BEOPEST
      IMPLICIT NONE
      INTEGER          NUMSLAVE,node,node0,node1
      DOUBLE PRECISION fast,BEOWALL

C  Check for new nodes and get range of nodes
      call BEONODE(node0,node1)

C  Find fastest node (0 means new)
      fast = 1d300
      do node=node0,node1
        fast = MIN(BEOWALL(node),fast)
      enddo
      fast = max(fast,1D0)

C  Count nodes faster than 2*fast
      NUMSLAVE = 0
      do node=node0,node1
        if (BEOWALL(node).le.2*fast) NUMSLAVE = NUMSLAVE+1
      enddo
      END
C**********************************************************************C
C***********************  Read instruction file  **********************C
C**********************************************************************C
      SUBROUTINE READINS(LCINS,NINSTR,A,ASIZE,MRKDEL,JFAIL)
      USE PESTDATA
      IMPLICIT NONE
      INTEGER I,J,INS,ISUM,LCINS(NINSTR),NINSTR,ASIZE,NBLC,JFAIL
      CHARACTER MRKDEL(NINSFLE),A(ASIZE)

      INS  = 0
      ISUM = 0
      do I=1,NINSFLE
        call FFOPEN(JFAIL,INST,'r',INSFLE(I),78,CLINE)
        if (JFAIL.ne.0) RETURN
        read(INST,*)
        INS = INS+1
        CLINE(1:1) = CHAR(2)
        CLINE(2:2) = ' '
        CLINE(3:LEN(CLINE)) = OUTFLE(I)
        LCINS(INS) = ISUM+1
        do J=1,LEN_TRIM(CLINE)
          ISUM = ISUM+1
          A(ISUM) = CLINE(J:J)
        enddo
50      READ(INST,'(A)',END=180) CLINE
        call TABREM(CLINE)
        if (INDEX(CLINE,MRKDEL(I)).eq.0) call CMPRSS(CLINE)
        NBLC = LEN_TRIM(CLINE)
        if (NBLC.eq.0) goto 50
        INS = INS+1
        LCINS(INS) = ISUM+1
        do J=1,NBLC
          ISUM = ISUM+1
          A(ISUM) = CLINE(J:J)
        enddo
        goto 50
180     CLOSE(UNIT=INST)
      enddo
      END
#endif
