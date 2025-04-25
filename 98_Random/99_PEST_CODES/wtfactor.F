        PROGRAM WTFACTOR

C -- PROGRAM WTFACTOR MULTIPLIES WEIGHTS FOR OBSERVATIONS AND PRIOR INFORMATION
C -- EQUATIONS BELONGING TO A GIVEN OBSERVATION GROUP BY A USER-SUPPLIED
C -- FACTOR.

        IMPLICIT NONE

        LOGICAL LEXIST
        INTEGER IFAIL,IERR,ILINE,IOK,IOBG,IFLAG,I,J,K,N,IEND
        INTEGER NBLNK
        INTEGER LW(10),RW(10)
        DOUBLE PRECISION FACTOR,WEIGHT
        CHARACTER*200 PESTFILE1,PESTFILE2
        CHARACTER*5 ALINE
        CHARACTER*12 AVERSION
        CHARACTER*30 AFACTOR,ATEMP,ATEMP1,ATEMP2,AOBSGP
        CHARACTER*300 CLINE,DLINE
#ifdef SYS_FUNCTION
        INTEGER NNN
        INTEGER SYSTEM
#endif

        INCLUDE 'version.inc'
        WRITE(6,10) TRIM(AVERSION)
10      FORMAT(' WTFACTOR Version ',A,'. ',
     +  'Watermark Numerical Computing.')
        WRITE(6,*)
#ifdef NO_CMDLINE
        WRITE(*,*)
1       WRITE(*,'(A)') ' ENTER COMMAND LINE ARGUMENTS > '
        READ(*,'(A)',ERR=1) CLINE
#else
        CALL PGETCL (CLINE)
#endif

        CALL SPACESUB(CLINE)
        CALL LINSPL(IFAIL,5,LW,RW,CLINE)
        IF(IFAIL.EQ.0) GO TO 9000
        CALL LINSPL(IFAIL,4,LW,RW,CLINE)
        IF(IFAIL.NE.0) GO TO 9000
        PESTFILE1=CLINE(LW(1):RW(1))
        PESTFILE2=CLINE(LW(4):RW(4))
        AOBSGP=CLINE(LW(2):RW(2))
        AFACTOR=CLINE(LW(3):RW(3))
        CALL REMCHAR(PESTFILE1,CHAR(211))
        CALL REMCHAR(PESTFILE2,CHAR(211))
        CALL REMCHAR(AOBSGP,CHAR(211))
        CALL REMCHAR(AFACTOR,CHAR(211))
        CALL LOWCAS(AOBSGP)
        CALL DREALRD(IFAIL,AFACTOR,FACTOR)
        IF(IFAIL.NE.0) THEN
          WRITE(6,35)
35        FORMAT(/,' Cannot read weight factor from command line.')
          GO TO 9900
        END IF
#ifdef CAPFILE
        CALL UPCAS(PESTFILE1)
        CALL UPCAS(PESTFILE2)
#endif
        IF(PESTFILE1.EQ.PESTFILE2)THEN
          WRITE(6,40)
40        FORMAT(/,' Input and output PEST control files cannot have ',
     +    'same name.')
          GO TO 9900
        END IF

        OPEN(UNIT=11,FILE=PESTFILE1,STATUS='OLD',IOSTAT=IERR)
        IF(IERR.NE.0) THEN
          WRITE(6,60) PESTFILE1(1:NBLNK(PESTFILE1))
60        FORMAT(/,' Cannot open PEST control file ',A,'.')
          GO TO 9900
        END IF
        OPEN(UNIT=20,FILE=PESTFILE2,IOSTAT=IERR)
        IF(IERR.NE.0)THEN
          WRITE(6,61) PESTFILE2(1:NBLNK(PESTFILE2))
61        FORMAT(/,' Cannot open file ',A,' for output.')
          GO TO 9900
        END IF
        IF(NBLNK(AOBSGP).GT.12)THEN
          WRITE(6,65) AOBSGP(1:NBLNK(AOBSGP))
65        FORMAT(/,' Observation group name "',A,
     +    '" greater than 12 characters in length.')
          GO TO 9900
        END IF
        DO 66 I=1,NBLNK(AOBSGP)
          IF(AOBSGP(I:I).EQ.' ')THEN
            WRITE(6,67)
67          FORMAT(/,' Observation group name cannot contain a blank ',
     +      'character.')
            GO TO 9900
          END IF
66      CONTINUE

C -- SOME VARIABLES ARE INITIALISED

        IFLAG=0
        IOK=0
        IEND=0
        IOBG=0
        ILINE=1

C -- THE FIRST PEST CONTROL FILE IS READ AND THE SECOND IS WRITTEN
C    UNTIL THE "* OBSERVATION GROUPS" SECTION IS ENCOUNTERED.

        WRITE(6,450) PESTFILE1(1:NBLNK(PESTFILE1)),
     +  PESTFILE2(1:NBLNK(PESTFILE2))
450     FORMAT(' Reading file ',A,' and writing file ',A,' ....')

        READ(11,'(A)',END=9200,ERR=9400) CLINE
        DLINE=CLINE
        CALL LOWCAS(DLINE)
        CALL SHIFTL(DLINE)
        IF(DLINE(1:4).NE.'pcf ')THEN
          WRITE(6,460) PESTFILE1(1:NBLNK(PESTFILE1))
460       FORMAT(/,' First line of file ',A,' should be "pcf".')
          GO TO 9900
        END IF
        WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
470     CONTINUE
          ILINE=ILINE+1
          READ(11,'(A)',ERR=9400,END=9200) CLINE
          CALL SHIFTL(CLINE)
          IF(CLINE(1:1).EQ.'*') THEN
            DLINE=CLINE
            CALL LOWCAS(DLINE)
            IF(INDEX(DLINE,'observation').NE.0) THEN
              IF(INDEX(DLINE,'group').NE.0)THEN
                IOBG=1
              END IF
            END IF
          END IF
          WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
          IF(IOBG.EQ.1) GO TO 480
        GO TO 470

C -- THE "* OBSERVATION GROUPS" SECTION OF THE PEST CONTROL FILE IS READ,
C -- LOOKING FOR THE PERTINENT OBSERVATION GROUP NAME.

480     CONTINUE
300     CONTINUE
          ILINE=ILINE+1
          READ(11,'(A)',ERR=9400,END=9500) CLINE
          CALL SHIFTL(CLINE)
          IF(CLINE(1:1).EQ.'*')THEN
            DLINE=CLINE
            CALL LOWCAS(DLINE)
            IF(INDEX(DLINE,'observation').NE.0)THEN
              IF(INDEX(DLINE,'data').NE.0)THEN
                WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
                GO TO 400
              END IF
            END IF
          END IF
          CALL LINSPL(IFAIL,1,LW,RW,CLINE)
          IF(IFAIL.NE.0) GO TO 9400
          ATEMP=CLINE(LW(1):RW(1))
          CALL LOWCAS(ATEMP)
          IF(ATEMP.EQ.AOBSGP) THEN
            IFLAG=1
            CALL LINSPL(IFAIL,2,LW,RW,CLINE)
            IF(IFAIL.EQ.0)THEN
              WRITE(6,310) AOBSGP(1:NBLNK(AOBSGP)),
     +        PESTFILE1(1:NBLNK(PESTFILE1))
310           FORMAT(/,' A covariance matrix has been ',
     +        'supplied for observation group "',A,
     +        '"',/,' in PEST control file ',A,'. WTFACTOR does not ',
     +        'permit this for ',/,' groups whose weights are to be ',
     +        'multiplied.')
              GO TO 9900
            END IF
          END IF
315       WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
        GO TO 300

400     CONTINUE
        IF(IFLAG.EQ.0)THEN
          WRITE(6,410) AOBSGP(1:NBLNK(AOBSGP)),
     +    PESTFILE1(1:NBLNK(PESTFILE1))
410       FORMAT(/,' Group "',A,'" not cited in ',
     +    '"* observation groups" section of ',/,
     +    ' PEST control file ',A)
          GO TO 9900
        END IF

C -- THE "* OBSERVATION DATA" SECTION OF THE EXISTING PEST CONTROL FILE
C    IS READ AND THE WEIGHT FACTOR IS APPLIED.

500     CONTINUE
          ILINE=ILINE+1
          READ(11,'(A)',ERR=9400,END=9500) CLINE
          IF(CLINE(1:1).EQ.'*')THEN
            DLINE=CLINE
            CALL LOWCAS(DLINE)
            IF(INDEX(DLINE,'command').NE.0)THEN
              WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
              GO TO 600
            END IF
          END IF
          CALL LINSPL(IFAIL,4,LW,RW,CLINE)
          IF(IFAIL.NE.0)THEN
            CALL WRTINT(ALINE,ILINE)
            WRITE(6,530) ALINE(1:NBLNK(ALINE)),
     +      PESTFILE1(1:NBLNK(PESTFILE1))
530         FORMAT(/,' Insufficient entries on line ',A,
     +      ' of file ',A)
            GO TO 9900
          END IF
          CALL DREALRD(IFAIL,CLINE(LW(3):RW(3)),WEIGHT)
          IF(IFAIL.NE.0) GO TO 9650
          ATEMP=CLINE(LW(4):RW(4))
          ATEMP1=ATEMP
          CALL LOWCAS(ATEMP1)
          IF(NBLNK(ATEMP1).GT.12) GO TO 9700
          IF(ATEMP1.EQ.AOBSGP)THEN
            WRITE(ATEMP2,'(1PG14.7)') WEIGHT*FACTOR
          ELSE
            ATEMP2=CLINE(LW(3):RW(3))
          END IF
          WRITE(20,550,ERR=9300) CLINE(1:RW(2)),
     +    ATEMP2(1:NBLNK(ATEMP2)),ATEMP(1:NBLNK(ATEMP))
550       FORMAT(1X,A,3X,A15,1X,A12)
        GO TO 500

C -- A SEARCH IS MADE FOR THE "* PRIOR INFORMATION" SECTION OF THE PEST
C    CONTROL FILE.

600     CONTINUE
          ILINE=ILINE+1
          READ(11,'(A)',ERR=9400,END=800) CLINE
          IF(CLINE(1:1).EQ.'*')THEN
            DLINE=CLINE
            CALL LOWCAS(DLINE)
            IF(INDEX(DLINE,'prior').NE.0)THEN
              WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
              GO TO 700
            END IF
          END IF
          WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
        GO TO 600

C -- THE "* PRIOR INFORMATION" SECTION IS NOW DEALT WITH

700     CONTINUE
        ILINE=ILINE+1
        READ(11,'(A)',ERR=9400,END=800) CLINE
        IF(CLINE.EQ.' ') THEN
          WRITE(20,'(A)',ERR=9300) ' '
          GO TO 700
        END IF
        CALL SHIFTL(CLINE)
        IF(CLINE(1:1).EQ.'*') THEN
          WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
          GO TO 810
        END IF
720     CONTINUE
          CALL WRTINT(ALINE,ILINE)
          ILINE=ILINE+1
          READ(11,'(A)',ERR=9400,END=750) DLINE
          CALL SHIFTL(DLINE)
          IF(DLINE(1:1).EQ.'*') THEN
            IEND=2
            GO TO 766
          END IF
          GO TO 760
750       IEND=1
          GO TO 766
760       CONTINUE
          IF(DLINE(1:1).EQ.'&')THEN
            WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
            CLINE=DLINE
            GO TO 720
          END IF
766       CONTINUE
          IF(CLINE.EQ.' ') THEN
            WRITE(20,'(A)',ERR=9300) ' '
            GO TO 850
          END IF
          N=NBLNK(CLINE)
          DO 770 I=N,1,-1
            IF(CLINE(I:I).EQ.' ') GO TO 780
770       CONTINUE
          WRITE(6,774) ALINE(1:NBLNK(ALINE)),
     +    PESTFILE1(1:NBLNK(PESTFILE1))
774       FORMAT(/,' Error in prior information equation at line ',A,
     +    ' of file ',A)
          GO TO 9900
780       ATEMP=CLINE(I+1:N)
          CALL LOWCAS(ATEMP)
          IF(ATEMP.NE.AOBSGP)THEN
            WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
            IF(IEND.EQ.1) THEN
              GO TO 800
            ELSE IF(IEND.EQ.2)THEN
              WRITE(20,'(A)',ERR=9300) DLINE(1:NBLNK(DLINE))
              GO TO 810
            END IF
            CLINE=DLINE
            GO TO 720
          END IF
          DO 790 J=I,1,-1
            IF(CLINE(J:J).NE.' ') GO TO 805
790       CONTINUE
          WRITE(6,774) ALINE(1:NBLNK(ALINE)),
     +    PESTFILE1(1:NBLNK(PESTFILE1))
          GO TO 9900
805       CONTINUE
          IF((J.EQ.1).AND.(CLINE(1:1).EQ.'&'))THEN
            WRITE(6,775) ALINE(1:NBLNK(ALINE)),
     +      PESTFILE1(1:NBLNK(PESTFILE1))
775         FORMAT(/,' WTFACTOR cannot alter the weight assigned to ',
     +      'a prior information equation ',/,' if the observation ',
     +      'group name and the prior information weight are on ',/,
     +      ' different lines; see line ',A,' of file ',A)
            GO TO 9900
          END IF
          DO 815 K=J,1,-1
            IF(CLINE(K:K).EQ.' ') GO TO 820
815       CONTINUE
          WRITE(6,774) ALINE(1:NBLNK(ALINE)),
     +    PESTFILE1(1:NBLNK(PESTFILE1))
          GO TO 9900
820       K=K+1
          ATEMP=CLINE(K:J)
          CALL DREALRD(IFAIL,ATEMP,WEIGHT)
          IF(IFAIL.NE.0)GO TO 9651
          WRITE(ATEMP2,'(1PG14.7)') WEIGHT*FACTOR
          WRITE(20,840) CLINE(1:K-1),ATEMP2(1:NBLNK(ATEMP2)),
     +    CLINE(I:N)
840       FORMAT(A,2X,A,2X,A)
850       IF(IEND.EQ.1)THEN
            GO TO 800
          ELSE IF(IEND.EQ.2)THEN
            WRITE(20,'(A)',ERR=9300) DLINE(1:NBLNK(DLINE))
            GO TO 810
          ELSE
            CLINE=DLINE
            GO TO 720
          END IF
        CONTINUE

C -- THE REMAINDER OF THE PEST CONTROL FILE IS WRITTEN.

810     CONTINUE
        ILINE=ILINE+1
        READ(11,'(A)',ERR=9400,END=800) CLINE
        WRITE(20,'(A)',ERR=9300) CLINE(1:NBLNK(CLINE))
        GO TO 810

800     CONTINUE

900        WRITE(6,920) PESTFILE2(1:NBLNK(PESTFILE2))
920        FORMAT(/,' File ',A,' written ok.')
        IOK=1
        GO TO 9990

9000    WRITE(6,9010)
9010    FORMAT(' WTFACTOR is run using the command:',/)
        WRITE(6,9020)
9020    FORMAT('     wtfactor pestfile1 obsgroup factor pestfile2',/)
        WRITE(6,9030)
9030    FORMAT(' where',/)
        WRITE(6,9050)
9050    FORMAT('     pestfile1 is the name of the input PEST ',
     +  'control file,')
        WRITE(6,9051)
9051    FORMAT('     obsgroup  is the name of an observation group,')
        WRITE(6,9052)
9052    FORMAT('     factor    is the multiplier for weights of ',
     +  'this group, and')
        WRITE(6,9060)
9060    FORMAT('     pestfile2 is the name of the output PEST ',
     +  'control file.')
        GO TO 9999

9200    WRITE(6,9210) PESTFILE1(1:NBLNK(PESTFILE1))
9210    FORMAT(/,' Unexpected end encountered to file ',A,
     +  ' while looking for ',/,' "* observation groups" section.')
        GO TO 9900
9300    WRITE(6,9310) PESTFILE2(1:NBLNK(PESTFILE2))
9310    FORMAT(/,' Error encountered while writing to file ',A)
        GO TO 9900
9400    CALL WRTINT(ALINE,ILINE)
        WRITE(6,9410) ALINE(1:NBLNK(ALINE)),
     +  PESTFILE1(1:NBLNK(PESTFILE1))
9410    FORMAT(/,' Error in line ',A,' of PEST control file ',A)
        GO TO 9900
9500    WRITE(6,9510) PESTFILE1(1:NBLNK(PESTFILE1))
9510    FORMAT(/,' Unexpected end encountered to file ',A,'.')
        GO TO 9900
9650    CALL WRTINT(ALINE,ILINE)
9651    WRITE(6,9660) ALINE(1:NBLNK(ALINE)),
     +  PESTFILE1(1:NBLNK(PESTFILE1))
9660    FORMAT(/,' Cannot read weight from line ',A,
     +  ' of file ',A)
        GO TO 9900
9700    CALL WRTINT(ALINE,ILINE)
        WRITE(6,9710) ALINE(1:NBLNK(ALINE)),
     +  PESTFILE1(1:NBLNK(PESTFILE1))
9710    FORMAT(/,' Observation group name greater than 12 characters ',
     +  'at line ',A,/,' of PEST control file ',A)
        go to 9900

9900    CONTINUE
9995    WRITE(6,9996)
9996    FORMAT(/,' Execution terminated.',/)
        GO TO 9990

9990    CONTINUE
        CLOSE(UNIT=11,IOSTAT=IERR)
        CLOSE(UNIT=20,IOSTAT=IERR)
        IF(IOK.EQ.0)THEN
          INQUIRE(FILE=PESTFILE2,EXIST=LEXIST)
#ifdef UNIX
#ifdef SYS_FUNCTION
          IF(LEXIST) NNN=SYSTEM('/bin/rm '//
     +    PESTFILE2(1:NBLNK(PESTFILE2)))
#else
          IF(LEXIST) CALL SYSTEM('/bin/rm '//
     +    PESTFILE2(1:NBLNK(PESTFILE2)))
#endif
#else
          IF(LEXIST) CALL SYSTEM('del "'//
     +    PESTFILE2(1:NBLNK(PESTFILE2))//'"')
#endif
        END IF

9999    END


        SUBROUTINE LINSPL(IFAIL,NUM,LW,RW,CLINE)

C -- SUBROUTINE LINSPL SPLITS A LINE INTO WHITESPACE-SEPARATED SUBSTRINGS

        INTEGER NBLNK
        INTEGER IFAIL,NW,NBLC,J,I
        INTEGER NUM
        INTEGER LW(NUM),RW(NUM)
        CHARACTER*(*) CLINE

        IFAIL=0
        NW=0
        NBLC=NBLNK(CLINE)
        IF(NBLC.EQ.0) THEN
          IFAIL=-1
          RETURN
        END IF
        J=0
5       IF(NW.EQ.NUM) RETURN
        DO 10 I=J+1,NBLC
        IF((CLINE(I:I).NE.' ').AND.(CLINE(I:I).NE.',')
     +  .AND.(ICHAR(CLINE(I:I)).NE.9)) GO TO 20
10      CONTINUE
        IFAIL=1
        RETURN
20      NW=NW+1
        LW(NW)=I
        DO 30 I=LW(NW)+1,NBLC
        IF((CLINE(I:I).EQ.' ').OR.(CLINE(I:I).EQ.',')
     +  .OR.(ICHAR(CLINE(I:I)).EQ.9)) GO TO 40
30      CONTINUE
        RW(NW)=NBLC
        IF(NW.LT.NUM) IFAIL=1
        RETURN
40      RW(NW)=I-1
        J=RW(NW)
        GO TO 5

        END


        SUBROUTINE LOWCAS(ASTRNG)

C -- SUBROUTINE LOWCAS CONVERTS A STRING TO LOWER CASE

        INTEGER NBLNK
        INTEGER I,J
        CHARACTER*(*) ASTRNG

        DO 10 I=1,NBLNK(ASTRNG)
        J=ICHAR(ASTRNG(I:I))
        IF((J.GE.65).AND.(J.LE.90)) ASTRNG(I:I)=CHAR(J+32)
10      CONTINUE
        RETURN
        END


        SUBROUTINE UPCAS(ASTRNG)

C -- SUBROUTINE UPCAS CONVERTS A STRING TO UPPER CASE

        INTEGER NBLNK
        INTEGER I,J
        CHARACTER*(*) ASTRNG

        DO 10 I=1,NBLNK(ASTRNG)
        J=ICHAR(ASTRNG(I:I))
        IF((J.GE.97).AND.(J.LE.122)) ASTRNG(I:I)=CHAR(J-32)
10      CONTINUE
        RETURN
        END



        SUBROUTINE SHIFTL(AA)

C -- SUBROUTINE SHIFTL REMOVES LEADING BLANKS FROM A STRING

        INTEGER L,I,J,II
        CHARACTER*(*) AA

        L=LEN(AA)
        DO 10 I=1,L
        IF((AA(I:I).NE.' ').AND.(ICHAR(AA(I:I)).NE.9)) GO TO 50
10      CONTINUE
        RETURN
50      IF(I.EQ.1) RETURN
        II=I-1
        DO 100 J=I,L
100     AA(J-II:J-II)=AA(J:J)
        DO 110 J=1,II
110     AA(L-J+1:L-J+1)=' '
        RETURN
        END



        SUBROUTINE WRTINT(ATEMP,IVAL)

C -- SUBROUTINE WRTINT WRITES AN INTEGER TO A STRING

        INTEGER IVAL
        CHARACTER*(*) ATEMP
        CHARACTER*8 AFMT

        AFMT='(I   )'
        WRITE(AFMT(3:5),'(I3)') LEN(ATEMP)
        WRITE(ATEMP,AFMT)IVAL
        CALL SHIFTL(ATEMP)
        RETURN
        END


        SUBROUTINE DREALRD(IFAIL,CLINE,RTEMP)

C -- Subroutine DREALRD reads a real number from a string.

        INTEGER IFAIL
        DOUBLE PRECISION RTEMP
        CHARACTER*8 AFMT
        CHARACTER*(*) CLINE

        IFAIL=0
        AFMT='(F   .0)'
        WRITE(AFMT(3:5),'(I3)') LEN(CLINE)
        READ(CLINE,AFMT,ERR=100) RTEMP
        RETURN

100     IFAIL=1
        RETURN
        END


