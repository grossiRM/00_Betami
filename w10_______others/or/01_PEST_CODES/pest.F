      PROGRAM PEST

#ifdef BEO
      USE BEOPEST
#endif
      IMPLICIT NONE

      LOGICAL LEXIST
      INTEGER I,IFAIL,RESTART
      CHARACTER*5   ATEMP
      CHARACTER*12 AVERSION
      CHARACTER*200 PEST_COMMAND_LINE,CTLFILE,JCOFIRSTFILE


C -- The version number is written to the screen.

        INCLUDE 'version.inc'
#ifdef PARALLEL
#ifdef BEO
      WRITE(6,'(3A)') ' BeoPEST Version ',TRIM(AVERSION),
     +  '. Watermark Numerical Computing.'
#else
#ifdef MPEST
	WRITE(6,1) TRIM(AVERSION)
1       FORMAT(' MPEST Version ',A,'. ',
     +  'Watermark Numerical Computing.')
#else
	WRITE(6,1) TRIM(AVERSION)
1       FORMAT(' Parallel PEST Version ',A,'. ',
     +  'Watermark Numerical Computing.')
#endif
#endif
#else
	WRITE(6,1) TRIM(AVERSION)
1       FORMAT(' PEST Version ',A,'. ',
     +  'Watermark Numerical Computing.')
#endif


C -- The command line is obtained.

#ifdef NO_CMDLINE
        WRITE(*,*)
2       WRITE(*,'(A)') ' ENTER COMMAND LINE ARGUMENTS > '
        READ(*,'(A)',ERR=2) PEST_COMMAND_LINE
#else
      CALL GET_COMMAND_LINE(PEST_COMMAND_LINE)
#endif

C -- The command line is parsed.

      CALL PARSE_COMMAND_LINE(IFAIL,PEST_COMMAND_LINE,CTLFILE,RESTART,
     +JCOFIRSTFILE)
      IF(IFAIL.NE.0) THEN
        WRITE(6,50)
#ifdef MPEST
50      FORMAT(/,' MPEST is run using the command:- ',/,/,
     +  '     MPEST pestfile [/r] [/j] [/i]',/,/,' where ',/,/,
     +  '     pestfile is the name of a PEST control file and ',
     +  '/r, /j, /i',/,
     +  '     are restart switches.')
        GO TO  9999
#else
50      FORMAT(/,' PEST is run using the command:- ',/,/,
     +  '     PEST pestfile [/r] [/j] [/s] [/d] [/i]',/,/,' where ',
     +  /,/,
     +  '     pestfile is the name of a PEST control file and ',
     +  '/r, /j, /s, /d, /i',/,
     +  '     are restart switches.')
        GO TO  9999
#endif
      END IF

#ifdef MPEST
        IF((RESTART.EQ.3).OR.(RESTART.EQ.4))THEN
          WRITE(6,50)
          GO TO 9999
        END IF
#endif

C -- An extension of ".pst" is added to the PEST control file if necessary.

      I=LEN_TRIM(CTLFILE)
      IF(I.GE.5) THEN
        ATEMP=CTLFILE(I-3:I)
        CALL LOWCAS(ATEMP)
        IF(ATEMP.EQ.'.pst') GO TO 110
      END IF
      CTLFILE(I+1:)='.pst'
110   CONTINUE

C -- The PEST input dataset is read.

      CALL READ_PEST_DATA(IFAIL,RESTART,CTLFILE)
      IF(IFAIL.GT.0)THEN
        CALL WRITE_PEST_MESSAGE(0)
        GO TO 9990
      END IF

#ifdef BEO
      if (BEOSLAVE) then
        call RUNSLAVE
      else
#endif
C -- The name of the JCO file to read Jacobian data for "/i" restart
C    option is provided.

      IF(RESTART.EQ.5)THEN
        WRITE(6,*)
149     WRITE(6,150,ADVANCE='NO')
150     FORMAT(' Enter name of JCO file for first iteration ',
     +  'sensitivities: ')
        READ(5,*) JCOFIRSTFILE
#ifndef UNIX
        CALL LOWCAS(JCOFIRSTFILE)
#endif
        INQUIRE(FILE=JCOFIRSTFILE,EXIST=LEXIST)
        IF(.NOT.LEXIST)THEN
          WRITE(6,151)
151       FORMAT('  *** File does not exist - try again ***')
          GO TO 149
        END IF
        CALL PUT_JCOFIRST(JCOFIRSTFILE)
      END IF

C -- Note, when returning from run_pest:-
C          ifail=0   Normal return
C          ifail=1   Return with error - new format.
C          ifail=2   Return with error - old format
C          ifail=-1  User requested immediate stop.

      CALL RUN_PEST(IFAIL)
      IF(IFAIL.GT.0)THEN
        CALL WRITE_PEST_MESSAGE(0)
      END IF
#ifdef BEO
      endif
#endif

C -- Now close PEST files. (Some of these may be open after PEST has
C    finished execution - particularly the run record file so that
C    an error message can be written to it.)

      CALL PEST_FILES_CLOSE()

C -- Memory from the PEST_DATA module is deallocated.

9990  CONTINUE
      CALL PEST_DATA_DEALLOCATE(IFAIL)
#ifdef BEO
C
      call BEOFIN
#endif

9999  CONTINUE

       END


      SUBROUTINE GET_COMMAND_LINE(COMMAND_LINE)

C -- Subroutine GET_COMMAND_LINE retreives any text supplied on the command line.

      IMPLICIT NONE
      CHARACTER (LEN=*), INTENT(OUT)  :: COMMAND_LINE

      INTEGER             :: IARGC
      INTEGER             :: LLEN,NARG,IB,I,NB,IBB
      CHARACTER (LEN=100) :: ARG(4)


      COMMAND_LINE=' '
#ifdef LAHEY
       CALL GETCL(COMMAND_LINE)
       GO TO 100
#else
       LLEN=LEN(COMMAND_LINE)
       NARG=IARGC()
       IF(NARG.EQ.0) GO TO 100
       IB=0
       DO I=1,MIN(NARG,4)
         CALL GETARG(I,ARG(I))
         NB=LEN_TRIM(ARG(I))
         IBB=MIN(IB+NB+1,LLEN)
         COMMAND_LINE(IB+1:IBB)= ARG(I)(1:NB)
         IB=IBB
         IF(IB.GE.LLEN) GO TO 100
       END DO
#endif

100   CONTINUE
      RETURN

      END



      SUBROUTINE PARSE_COMMAND_LINE(IFAIL,COMMAND_LINE,CTLFILE,RESTART,
     +JCOFIRSTFILE)

C -- Subroutine PARSE_COMMAND_LINE parses the PEST command line.
#ifdef BEO
      USE BEOPEST
#endif

      IMPLICIT NONE

      INTEGER, INTENT(OUT)              :: IFAIL
      CHARACTER (LEN=*), INTENT(INOUT)  :: COMMAND_LINE
      CHARACTER (LEN=*), INTENT(OUT)    :: CTLFILE,JCOFIRSTFILE
      INTEGER, INTENT(OUT)              :: RESTART

      INTEGER                           :: K,IR,IS,IJ,ID,I,IO
      INTEGER                           :: IH,IM,IL,NP
      CHARACTER (LEN=1)                 :: AA

      IFAIL=0
      IF(COMMAND_LINE.EQ.' ') GO TO 9000
      AA=' '
      COMMAND_LINE=ADJUSTL(COMMAND_LINE)
      CTLFILE=COMMAND_LINE
      IF(CTLFILE(1:1).EQ.'"')THEN
        AA='"'
      ELSE IF(CTLFILE(1:1).EQ.'''')THEN
        AA=''''
      END IF
      IF(AA.NE.' ') CTLFILE=CTLFILE(2:)
      I=INDEX(CTLFILE,AA)
      IF(I.LE.1) GO TO 9000
      CTLFILE=CTLFILE(1:I-1)
      RESTART=0
      IR=0
      IJ=0
      IS=0
      ID=0
      IO=0
      IH=0
      IM=0
      IR=INDEX(COMMAND_LINE,' /r ')
      IF(IR.EQ.0) IR=INDEX(COMMAND_LINE,' /R ')
      IJ=INDEX(COMMAND_LINE,' /j ')
      IF(IJ.EQ.0) IJ=INDEX(COMMAND_LINE,' /J ')
      IS=INDEX(COMMAND_LINE,' /s ')
      IF(IS.EQ.0) IS=INDEX(COMMAND_LINE,' /S ')
      ID=INDEX(COMMAND_LINE,' /d ')
      IF(ID.EQ.0) ID=INDEX(COMMAND_LINE,' /D ')
      IO=INDEX(COMMAND_LINE,' /i ')
      IF(IO.EQ.0) IO=INDEX(COMMAND_LINE,' /I ')
      IH=INDEX(COMMAND_LINE,' /h ')
      IF(IH.EQ.0) IH=INDEX(COMMAND_LINE,' /H ')
      IM=INDEX(COMMAND_LINE,' /m ')
      IF(IM.EQ.0) IM=INDEX(COMMAND_LINE,' /M ')
      IL=INDEX(COMMAND_LINE,' /l ')
      IF(IL.EQ.0) IL=INDEX(COMMAND_LINE,' /L ')

      IF(IR.NE.0)THEN
        RESTART=1
        COMMAND_LINE(IR+1:IR+2)='  '
        IR=1
      END IF
      IF(IJ.NE.0)THEN
        RESTART=2
        COMMAND_LINE(IJ+1:IJ+2)='  '
        IJ=1
      END IF
      IF(IS.NE.0)THEN
        RESTART=3
        COMMAND_LINE(IS+1:IS+2)='  '
        IS=1
      END IF
      IF(ID.NE.0)THEN
        RESTART=4
        COMMAND_LINE(ID+1:ID+2)='  '
        ID=1
      END IF
      IF(IO.NE.0)THEN
        RESTART=5
        COMMAND_LINE(IO+1:IO+2)='  '
        IO=1
      END IF
#ifdef BEO
      BEOLASTLOC = (IL.ne.0)
      if (IL.ne.0) COMMAND_LINE(IL+1:IL+2) = '  '
      if (IH.ne.0 .and. IM.ne.0) then
        STOP '/H and /M are mutually exclusive'
      else if (IH.ne.0 .or. IM.ne.0) then
        if (IH.ne.0) then
          NP = 8192
          IL = IH
        else
          NP = 0
          IL = IM
        endif
        COMMAND_LINE(IL+1:IL+2) = '  '
        do K=IL+4,LEN(COMMAND_LINE)
          if (COMMAND_LINE(K:K).ne.' ') goto 100
        enddo
        if (IH.ne.0) then
          STOP 'Missing host:port'
        else
          STOP 'Missing directory'
        endif
100     IL = K
        do K=IL+1,LEN(COMMAND_LINE)
          if (COMMAND_LINE(K:K).eq.' ') goto 199
        enddo
199     K = K-1
        call BEOINIT(COMMAND_LINE(IL:K),NP)
        COMMAND_LINE(IL:K) = '  '
      else
        BEOMASTER = .false.
        BEOSLAVE  = .false.
      endif
#endif

      IF(IR+IJ+IS+ID+IO.GT.1) GO TO 9000
      IF(INDEX(COMMAND_LINE,' /').NE.0) GO TO 9000

C -- The following is used to handle spaces in filenames because of idiosyncracies
C    in the command line argument command.

      IF(AA.EQ.' ')THEN
        CTLFILE=COMMAND_LINE
      END IF

#ifndef UNIX
      CALL LOWCAS(CTLFILE)
#endif

      RETURN

9000  IFAIL=1

      RETURN
      END


      SUBROUTINE PUT_JCOFIRST(AFILE)

      USE PESTDATA
      IMPLICIT NONE
      CHARACTER*(*) AFILE

      JCOFIRSTFILE=AFILE

      RETURN
      END
