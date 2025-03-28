C     Last change:  JD   10 Apr 2000   11:03 am

C -- Following subroutine optwt is subroutine optwt_sl. This is a slow
C    version of the subroutine that was used before execution speedup was
C    employed at the cost of memory. Any changes made to the first of these
C    subroutines must also be made to the last.

      SUBROUTINE OPTWT(IFAIL,NXDIM,NSP4,NXROW,NPRIOR,NOBS,NPAR,NESPAR,
     +W1DIM,MAXITN,WFBEG,WFFAC,PHIMLIM,WFMAX,WFMIN,WFSOL,RTOL,
     +ITRANS,REFOBS,OVAL,OWGHT,NOBGNM,TMPWT,TMPOBS,
     +SC,W1,X,RHS,LHS,INFOCOUNT,IUNSTABLE,LAMSTR,LAMFAC,ICOVOBS,
     +JSTK,MAT1,MAT2,VEC1,VEC2,VEC3,VEC4,NOBSGP,IRGP,LHSFLAG,
     +LASTOBSROW,FIRSTREGROW,INOCOV,IOPTCALL,LINREG,WFSTART,
     +SVDMODE,MAXSING,LWORK,EIGTHRESH,WORKVEC4)

      USE PESTDATA, ONLY: XDIM1,XDIM2,XC,IXC,MAXCOMPDIM,NCOMPDIM,
     +                     ISTART_C,IFOUND_C,IREGADJ
      IMPLICIT NONE

      INTEGER IFAIL,NXROW,NPRIOR,NOBS,NPAR,NESPAR,NSP4,MAXITN,
     +NXDIM,INFOCOUNT,IUNSTABLE,IRESTART,IRST,IILAM,NNLAM,NESTMP,
     +ICOVOBS,NOBSGP,LHSFLAG,W1DIM
      INTEGER NOBGNM(NXROW),ITRANS(NPAR),JSTK(NPAR),IRGP(NOBSGP)
      DOUBLE PRECISION WFBEG,WFFAC,PHIMLIM,WFMIN,WFMAX,WFSOL,RTOL,
     +LAMSTR,LAMFAC,LAMBDA,RRTEMP
      DOUBLE PRECISION REFOBS(NXROW),OVAL(NXROW),
     +OWGHT(NXROW),TMPWT(NXROW),TMPOBS(NXROW),SC(NESPAR),W1(W1DIM),
     +X(XDIM1,XDIM2),RHS(nespar),LHS(NSP4,NESPAR),MAT1(NSP4,NESPAR),
     +MAT2(NSP4,NESPAR),VEC1(NESPAR),VEC2(NESPAR),VEC3(NESPAR),
     +VEC4(NESPAR)

      INTEGER ITN,IROW,IES,IPP,JES,I,INFO,IDIRN,IMODE,NST4,IREV,IHI,
     +ILO,LASTOBSROW,FIRSTREGROW,INOCOV,IOPTCALL,LINREG
      INTEGER LDU,LDVT,SVDMODE,LWORK,IICOUNT,MAXSING,J,K

      DOUBLE PRECISION WF,WFLAST,WFLAST1,PHIBEG,PHIRBEG,PHIMBEG,
     +PHI,PHILAST,PHILAST1,PHIM,PHIMLAST,PHIMLAST1,PHIR,PHIRLAST,
     +PHIRLAST1,RTEMP,WFTEMP,PHITEMP,PHIRTEMP,PHIMTEMP,PHIMHI,
     +WFHI,WFLO,PHIMLO,WFNEW,WFTEMP1,SUM1,WFSTART,RTEMP1,RTEMP_C
      DOUBLE PRECISION EIGMAX,EIGTHRESH
      DOUBLE PRECISION U(1,1),VT(1,1),WORKVEC4(LWORK)

      INTEGER ISTOP
      COMMON /SSTOP/ISTOP

      LDU=1
      LDVT=1
      IFAIL=0
      LHSFLAG=0
      IF(IOPTCALL.EQ.0)WFSTART=WFBEG

C --  If there are no non-zero measurement weights return.

      DO 20 IROW=1,NXROW
        IF(IRGP(NOBGNM(IROW)).EQ.0)THEN
          IF(OWGHT(IROW).GT.0.0D0) GO TO 10
        END IF
20    CONTINUE
      WFSOL=WFBEG
      RETURN
10    CONTINUE

C -- Before anything else is done, information is stored in some work
C    matrices for use in later calculations.

      IF(MAXCOMPDIM.LE.1)THEN
        DO IROW=1,NXROW
          IF(IRGP(NOBGNM(IROW)).NE.0)THEN
            TMPWT(IROW)=0.0D0
          ELSE
            TMPWT(IROW)=OWGHT(IROW)
          END IF
        END DO
      ELSE
        DO IROW=1,LASTOBSROW
          IF(IRGP(NOBGNM(IROW)).NE.0)THEN
            TMPWT(IROW)=0.0D0
          ELSE
            TMPWT(IROW)=OWGHT(IROW)
          END IF
        END DO
      END IF
      IES=0
      NESTMP=0
      DO 2400 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0)GO TO 2400
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 2400
        NESTMP=NESTMP+1
        IES=IES+1
        JES=0
        ISTART_C=1
        DO 2300 I=1,IPP
          IF(ITRANS(I).LT.0) GO TO 2300
          IF((ICOVOBS.EQ.0).AND.(JSTK(I).LT.0)) GO TO 2300
          JES=JES+1
          IF(JES.GT.IES) GO TO 2400
          RTEMP=0.0D0
          IF(MAXCOMPDIM.LE.1)THEN
            DO 2360 IROW=1,NXROW
              RTEMP=RTEMP+TMPWT(IROW)*X(IROW,IES)*X(IROW,JES)
2360        CONTINUE
          ELSE
             CALL DUAL_VECTOR_MUL(IFAIL,NCOMPDIM,1,LASTOBSROW,XC,IXC,
     +       TMPWT,IES,JES,RTEMP)
             IF(IFAIL.NE.0) GO TO 9980
          END IF
          MAT1(JES,IES)=RTEMP
2300    CONTINUE
2400  CONTINUE
c      DO IES=1,NESTMP-1
c        DO JES=IES+1,NESTMP
c          MAT1(JES,IES)=MAT1(IES,JES)
c        END DO
c      END DO

      IES=0
      ISTART_C=1
      DO 2250 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 2250
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 2250
        IES=IES+1
        RTEMP=0.0D0
        IF(MAXCOMPDIM.LE.1)THEN
          DO 2270 IROW=1,NXROW
            RTEMP=RTEMP+X(IROW,IES)*TMPWT(IROW)*
     +            (OVAL(IROW)-REFOBS(IROW))
2270      CONTINUE
        ELSE
          CALL SINGLE_VECTOR_MUL1(IFAIL,NCOMPDIM,1,LASTOBSROW,XC,IXC,
     +    IES,TMPWT,OVAL,REFOBS,RTEMP)
          IF(IFAIL.NE.0) GO TO 9980
          ISTART_C=IFOUND_C
        END IF
        VEC1(IES)=RTEMP
2250  CONTINUE

      IES=0
      ISTART_C=1
      DO 2001 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 2001
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 2001
        IES=IES+1
        IF(MAXCOMPDIM.LE.1)THEN
        CALL OBJCLC(2,VEC3(IES),NXROW,NPRIOR,NOBS,X(1,IES),OVAL,TMPWT,
     +  NOBGNM,NOBSGP,IRGP,SUM1)
        ELSE
          CALL SINGLE_VECTOR_MUL2(IFAIL,NCOMPDIM,1,LASTOBSROW,XC,IXC,
     +    IES,TMPWT,VEC3(IES))
          IF(IFAIL.NE.0) GO TO 9980
          ISTART_C=IFOUND_C
        END IF
2001  CONTINUE

      IF(MAXCOMPDIM.LE.1)THEN
        DO IROW=1,NXROW
          IF(IRGP(NOBGNM(IROW)).NE.0)THEN
            TMPWT(IROW)=OWGHT(IROW)
          ELSE
            TMPWT(IROW)=0.0D0
          END IF
        END DO
      ELSE
        DO IROW=FIRSTREGROW,NXROW
          IF(IRGP(NOBGNM(IROW)).NE.0)THEN
            TMPWT(IROW)=OWGHT(IROW)
          ELSE
            TMPWT(IROW)=0.0D0
          END IF
        END DO
      END IF

      IF((IOPTCALL.EQ.0).OR.(INOCOV.EQ.0).OR.(LINREG.EQ.0).
     +OR.(IREGADJ.NE.0))THEN
      IES=0
      DO 4400 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0)GO TO 4400
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 4400
        IES=IES+1
        JES=0
        ISTART_C=1
        DO 4300 I=1,IPP
          IF(ITRANS(I).LT.0) GO TO 4300
          IF((ICOVOBS.EQ.0).AND.(JSTK(I).LT.0)) GO TO 4300
          JES=JES+1
          IF(JES.GT.IES) GO TO 4400
          RTEMP=0.0D0
          IF(MAXCOMPDIM.LE.1)THEN
            DO 4360 IROW=1,NXROW
              RTEMP=RTEMP+TMPWT(IROW)*X(IROW,IES)*X(IROW,JES)
4360        CONTINUE
          ELSE
             CALL DUAL_VECTOR_MUL(IFAIL,NCOMPDIM,FIRSTREGROW,NXROW,XC,
     +       IXC,TMPWT,IES,JES,RTEMP)
             IF(IFAIL.NE.0) GO TO 9980
          END IF
          MAT2(JES,IES)=RTEMP
4300    CONTINUE
4400  CONTINUE
      ELSE
        RTEMP1=WFBEG/WFSTART
        DO IES=1,NESTMP
          DO JES=1,IES
            MAT2(JES,IES)=MAT2(JES,IES)*RTEMP1
          END DO
        END DO
      END IF
      IOPTCALL=1

c      DO IES=1,NESTMP-1
c        DO JES=IES+1,NESTMP
c          MAT2(JES,IES)=MAT2(IES,JES)
c        END DO
c      END DO
      IES=0
      ISTART_C=1
      DO 4250 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 4250
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 4250
        IES=IES+1
        RTEMP=0.0D0
        IF(MAXCOMPDIM.LE.1)THEN
          DO 4270 IROW=1,NXROW
            RTEMP=RTEMP+X(IROW,IES)*TMPWT(IROW)*
     +            (OVAL(IROW)-REFOBS(IROW))
4270      CONTINUE
        ELSE
          CALL SINGLE_VECTOR_MUL1(IFAIL,NCOMPDIM,FIRSTREGROW,
     +    NXROW,XC,IXC,IES,TMPWT,OVAL,REFOBS,RTEMP)
          IF(IFAIL.NE.0) GO TO 9980
          ISTART_C=IFOUND_C
        END IF
        VEC2(IES)=RTEMP
4250  CONTINUE

      IES=0
      ISTART_C=1
      DO 4000 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 4000
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 4000
        IES=IES+1
        IF(MAXCOMPDIM.LE.1)THEN
        CALL OBJCLC(2,VEC4(IES),NXROW,NPRIOR,NOBS,X(1,IES),OVAL,TMPWT,
     +  NOBGNM,NOBSGP,IRGP,SUM1)
        ELSE
          CALL SINGLE_VECTOR_MUL2(IFAIL,NCOMPDIM,FIRSTREGROW,NXROW,
     +    XC,IXC,IES,TMPWT,VEC4(IES))
          IF(IFAIL.NE.0) GO TO 9980
          ISTART_C=IFOUND_C
        END IF
4000  CONTINUE

C -- Initialise where necessary.

      IILAM=0
      NNLAM=5       !ARBITRARY
      LAMBDA=0.0D0
C      LAMBDA=LAMSTR

1     CONTINUE
      IRESTART=0
      IRST=0
      GO TO 6

5     CONTINUE

6     CONTINUE
      INFOCOUNT=0
      IUNSTABLE=0
      ITN=0
      IF(IRST.NE.1)THEN
        WF=WFBEG
        WFLAST=WFBEG
        WFLAST1=WFBEG
      ELSE
        WFLAST=WF
        WFLAST1=WF
        IRST=0
      END IF
      PHIMLAST1=1.0D300
      PHIMLAST=1.0D300
      PHIM=1.0D300
      PHILAST=1.0D300
      PHI=1.0D300
      PHIR=1.0D300
      PHIRLAST=1.0D300
      IMODE=0
      IREV=0

C -- For the initial "model" run, we use the weight factor supplied
C    from outside.

50    ITN=ITN+1

      PHILAST1=PHILAST
      PHILAST=PHI
      PHIMLAST1=PHIMLAST
      PHIMLAST=PHIM
      PHIRLAST1=PHIRLAST
      PHIRLAST=PHIR

C -- The new weights vector is formed. It is assumed that the owght vector
C    contains regularisation weights multiplied by WFBEG.

60    CONTINUE

      WFTEMP1=WF*WF/WFBEG/WFBEG
      DO 100 IROW=1,NXROW
        IF(IRGP(NOBGNM(IROW)).EQ.0)THEN
          TMPWT(IROW)=OWGHT(IROW)
        ELSE
          TMPWT(IROW)=OWGHT(IROW)*WFTEMP1
        END IF
100   CONTINUE

C -- Next the diagonal elements of the scaling matrix are calculated.

      IES=0
      DO 200 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 200
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 200
        IES=IES+1
        SC(IES)=VEC3(IES)+WFTEMP1*VEC4(IES)
        IF(SC(IES).LE.0.0D0) THEN
          SC(IES)=0.0D0
          GO TO 200
        END IF
        SC(IES)=SQRT(SC(IES))
        SC(IES)=1.0D0/SC(IES)
200   CONTINUE

      DO 220 IES=1,NESPAR
        IF(SC(IES).GT.0.0D0) GO TO 230
220   CONTINUE
      WFSOL=WFLAST
      RETURN
230   CONTINUE

C -- The RHS of the normal equations is calculated.

      IES=0
      DO 250 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 250
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 250
        IES=IES+1
        IF(SVDMODE.EQ.0)THEN
          RHS(IES)=(VEC1(IES)+WFTEMP1*VEC2(IES))*SC(IES)
        ELSE
          RHS(IES)=VEC1(IES)+WFTEMP1*VEC2(IES)
        END IF
250   CONTINUE

C -- Next the normal matrix is calculated.

      IES=0
      DO 400 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0)GO TO 400
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 400
        IES=IES+1
        JES=0
        DO 300 I=1,NPAR
          IF(ITRANS(I).LT.0) GO TO 300
          IF((ICOVOBS.EQ.0).AND.(JSTK(I).LT.0)) GO TO 300
          JES=JES+1
          IF(JES.GT.IES) GO TO 400
          IF(SVDMODE.EQ.0)THEN
            LHS(JES,IES)=(MAT1(JES,IES)+WFTEMP1*MAT2(JES,IES))
     +      *SC(IES)*SC(JES)
          ELSE
            LHS(JES,IES)=MAT1(JES,IES)+WFTEMP1*MAT2(JES,IES)
          END IF
300     CONTINUE
400   CONTINUE
      DO 405 IES=1,NESTMP-1
        DO 404 JES=IES+1,NESTMP
          LHS(JES,IES)=LHS(IES,JES)
404     CONTINUE
405   CONTINUE
      IF(LHSFLAG.EQ.1)THEN
        IES=0
        DO 3400 IPP=1,NPAR
          IF(ITRANS(IPP).LT.0)GO TO 3400
          IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 3400
          IES=IES+1
          JES=0
          DO 3300 I=1,NPAR
            IF(ITRANS(I).LT.0) GO TO 3300
            IF((ICOVOBS.EQ.0).AND.(JSTK(I).LT.0)) GO TO 3300
            JES=JES+1
            IF(JES.GT.IES) GO TO 3400
            MAT1(JES,IES)=(MAT1(JES,IES)+WFTEMP1*MAT2(JES,IES))
3300      CONTINUE
3400    CONTINUE
        DO 3405 IES=1,NESTMP-1
          DO 3404 JES=IES+1,NESTMP
            MAT1(JES,IES)=MAT1(IES,JES)
3404      CONTINUE
3405    CONTINUE
        RETURN
      END IF

C -- The Marquardt lambda is added if necessary.

425     IF(LAMBDA.NE.0.0D0) THEN
          RTEMP=0.0D0
          DO 440 IES=1,NESTMP
          IF(ABS(SC(IES)).GT.RTEMP) RTEMP=ABS(SC(IES))
440       CONTINUE
          RRTEMP=LAMBDA/RTEMP/RTEMP
          DO IES=1,NESTMP
            IF(SVDMODE.EQ.0)THEN
              LHS(IES,IES)=1.0D0+SC(IES)*SC(IES)*RRTEMP
            ELSE
              LHS(IES,IES)=LHS(IES,IES)+RRTEMP
            END IF
          END DO
        END IF

      NST4=NESPAR
      IF(SVDMODE.EQ.0)THEN
        CALL DPOFA(LHS,NSP4,NST4,INFO)
      ELSE
        CALL DGESVD('O','N',NST4,NST4,LHS,NSP4,W1,U,LDU,VT,
     +  LDVT,WORKVEC4,LWORK,INFO)
      END IF
      IF(INFO.NE.0)THEN
        INFOCOUNT=INFOCOUNT+1
        PHI=-1.0D300
        PHIM=1.0D300
        IF(ITN.EQ.1)THEN
          IF(IRESTART.EQ.0)THEN
            IRESTART=1
            IRST=1
            WF=WFMIN
            GO TO 5
          END IF
          IDIRN=1
          GO TO 1500
        ELSE IF(ITN.EQ.2)THEN
          IF(IDIRN.EQ.-1)THEN
            WFSOL=WFLAST
            GO TO 3000
          ELSE
            IF(PHILAST.GT.-1.0D300)THEN
              WFSOL=WFLAST
              GO TO 3000
            ELSE
              GO TO 1550
            END IF
          END IF
        ELSE
          IF(IREV.EQ.1)THEN
            WFTEMP=WFLAST1
            WFLAST1=WFLAST
            WFLAST=WFTEMP
            PHITEMP=PHILAST1
            PHILAST1=PHILAST
            PHILAST=PHITEMP
            PHIMTEMP=PHIMLAST1
            PHIMLAST1=PHIMLAST
            PHIMLAST=PHIMTEMP
            PHIRTEMP=PHIRLAST1
            PHIRLAST1=PHIRLAST
            PHIRLAST=PHIRTEMP
            IREV=0
          END IF
          IF(IMODE.EQ.0) THEN
            IF(IDIRN.EQ.-1)THEN
              WFSOL=WFLAST
              GO TO 3000
            ELSE
              IF(PHILAST.GT.-1.0D300)THEN
                WFSOL=WFLAST
                GO TO 3000
              ELSE
                GO TO 1550
              END IF
            END IF
          ELSE
            WFSOL=WFLAST
            GO TO 3000
          END IF
        END IF
      END IF


C the above strategy assumes that increasing the regularisation weight
C is better for the solution numerically. This is really the main condition on the
C regularisation information.

C -- NEXT WE SOLVE FOR THE UPDATING DIRECTION

      IF(SVDMODE.EQ.0)THEN
        DO 500 IES=1,NESPAR
          W1(IES)=RHS(IES)
500     CONTINUE
        CALL DPOSL(LHS,NSP4,NST4,W1)
        DO 510 IES=1,NESPAR
          W1(IES)=W1(IES)*SC(IES)
510     CONTINUE
      ELSE
         IICOUNT=1
         EIGMAX=W1(1)
         IF(EIGMAX.LE.0.0D0)THEN
           WFSOL=WFLAST
           GO TO 3000
         END IF
         DO I=2,NST4
           IF(ABS(W1(I)/EIGMAX).LT.EIGTHRESH)EXIT
           IF(IICOUNT.GE.MAXSING)EXIT
           IICOUNT=IICOUNT+1
         END DO
         DO I=1,NST4
           RTEMP=0.0D0
           DO J=1,NST4
             DO K=1,IICOUNT
               RTEMP=RTEMP+LHS(I,K)*LHS(J,K)/W1(K)*RHS(J)
             END DO
           END DO
           WORKVEC4(I)=RTEMP
         END DO
         DO I=1,NST4
           W1(I)=WORKVEC4(I)
         END DO
      END IF

8700  CONTINUE

C -- CHECK TO SEE IF WE SHOULD STOP

      CALL STOPRESS(0)
      IF((ISTOP.EQ.2).OR.(ISTOP.EQ.1)) RETURN

C -- THE OBJECTIVE FUNCTION AND ITS TWO COMPONENTS ARE NEXT CALCULATED
C    UNDER THE LINEARITY ASSUMPTION.

      DO 600 IROW=1,NXROW
        RTEMP=0.0D0
        IES=0
        ISTART_C=1
        DO 580 IPP=1,NPAR
          IF(ITRANS(IPP).LT.0) GO TO 580
          IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 580
          IES=IES+1
          IF(MAXCOMPDIM.LE.1)THEN
            RTEMP=RTEMP+X(IROW,IES)*W1(IES)
          ELSE
            CALL GET_VALUE(NCOMPDIM,XC,IXC,RTEMP_C,IROW,IES)
            ISTART_C=IFOUND_C
            RTEMP=RTEMP+RTEMP_C*W1(IES)
          END IF
580     CONTINUE
        TMPOBS(IROW)=RTEMP+REFOBS(IROW)
600   CONTINUE

C -- THE NEW OBJECTIVE FUNCTION COMPONENTS ARE CALCULATED.
      CALL OBJCLC(1,PHI,NXROW,NPRIOR,NOBS,TMPOBS,OVAL,TMPWT,NOBGNM,
     +NOBSGP,IRGP,SUM1)
      CALL OBJCLC(888888,PHIR,NXROW,NPRIOR,NOBS,TMPOBS,
     +OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,PHIM)
      PHIR=PHIR/WFBEG/WFBEG

C -- NEXT A DECISION IS MADE ON WHAT THE NEXT WEIGHTING FACTOR SHOULD BE.

      IF(IMODE.NE.0) GO TO 2000

      IF(PHILAST.LT.-1.0D300)THEN
        IF(PHIM.GE.PHIMLIM)THEN
          WFSOL=WF
          GO TO 3000
        ELSE
          IDIRN=1
          GO TO 1550
        END IF
      END IF

      IF(ITN.EQ.1)THEN
        IF(PHIM.GT.PHIMLIM)THEN
          IDIRN=-1
          GO TO 1600
        ELSE
          IDIRN=1
          GO TO 1550
        END IF
      ELSE IF(ITN.EQ.2)THEN
        IF(IDIRN.EQ.-1)THEN
          IF(PHIM.GT.PHIMLIM)THEN
            IF(PHIM.LT.PHIMLAST-RTOL*PHIMLAST)THEN
              GO TO 1600
            ELSE
              WFSOL=WF
              GO TO 3000
            END IF
          ELSE
            IF(PHIM.LT.PHIMLAST)THEN
              IF(PHIMLAST.GE.PHIMLIM)THEN
                IF(PHIR.GE.PHIRLAST-RTOL*PHIRLAST)THEN
                  GO TO 2000
                ELSE
                  GO TO 1600
                END IF
              ELSE
                IF(ABS(PHIR-PHIRLAST).LT.0.5*RTOL*(PHIR+PHIRLAST))THEN
                  IF(PHIR.LE.PHIRLAST)THEN
                    WFSOL=WF
                  ELSE
                    WFSOL=WFLAST
                  END IF
                  GO TO 3000
                END IF
                IF(PHIR.GE.PHIRLAST)THEN
                  IREV=1
                  GO TO 1500
                ELSE
                  GO TO 1600
                END IF
              END IF
            ELSE
              WFSOL=WFLAST
              GO TO 3000
            END IF
          END IF
        ELSE IF(IDIRN.EQ.1)THEN
          IF(PHIM.LT.PHIMLAST)THEN
            WFSOL=WF
            GO TO 3000
          ELSE
            IF(PHIM.LT.PHIMLIM)THEN
              IF(PHIR.LT.PHIRLAST)THEN
                GO TO 1550
              ELSE
                IDIRN=-1
                IREV=-1
                GO TO 1650
              END IF
            ELSE
              IF(PHIMLAST.LE.PHIMLIM)THEN
                GO TO 2000
              ELSE
                WFSOL=WFLAST  !we should not be here
                GO TO 3000
              END IF
            END IF
          END IF
        END IF
      ELSE IF(ITN.GT.2)THEN
        IF(IREV.EQ.1)THEN
          WFTEMP=WFLAST1
          WFLAST1=WFLAST
          WFLAST=WFTEMP
          PHITEMP=PHILAST1
          PHILAST1=PHILAST
          PHILAST=PHITEMP
          PHIMTEMP=PHIMLAST1
          PHIMLAST1=PHIMLAST
          PHIMLAST=PHIMTEMP
          PHIRTEMP=PHIRLAST1
          PHIRLAST1=PHIRLAST
          PHIRLAST=PHIRTEMP
          IREV=0
        END IF
        IF(IDIRN.EQ.1)THEN
          IF(ABS(PHIM-PHIMLAST).LT.0.5*RTOL*(PHIMLAST+PHIM)*0.2)THEN  !0.2 ARBITRARY
            IUNSTABLE=1
            WFSOL=WF
            GO TO 3000
          END IF
          IF(PHIM.LT.PHIMLAST)THEN
            WFSOL=WF
            GO TO 3000
          ELSE
            IF(PHIM.GE.PHIMLIM)THEN
              IF(PHIMLAST.LE.PHIMLIM)THEN
                GO TO 2000
              ELSE
                WFSOL=WFLAST  !WE SHOULD NOT BE HERE
                GO TO 3000
              END IF
            ELSE
              IF(PHIR.LT.PHIRLAST)THEN
                GO TO 1550
              ELSE
                WFSOL=WFLAST
                GO TO 3000
              END IF
            END IF
          END IF
        ELSE IF(IDIRN.EQ.-1)THEN
          IF(PHIM.GT.PHIMLAST-0.5*RTOL*(PHIMLAST+PHIM))THEN
            IF(PHIR.GT.PHIRLAST-0.5*RTOL*(PHIRLAST+PHIR))THEN
              IF(PHIR.LE.PHIRLAST)THEN
                WFSOL=WF
              ELSE
                WFSOL=WFLAST
              END IF
              GO TO 3000
            ELSE
              GO TO 1600
            END IF
          ELSE
            IF(PHIM.GT.PHIMLIM)THEN
              GO TO 1600
            ELSE
              IF(PHIMLAST.GE.PHIMLIM)THEN
                GO TO 2000
              ELSE
                IF(PHIR.LT.PHIRLAST-0.5*RTOL*(PHIR+PHIRLAST))THEN
                  GO TO 1600
                ELSE
                  IF(PHIR.LT.PHIRLAST)THEN
                    WFSOL=WF
                  ELSE
                    WFSOL=WFLAST
                  END IF
                  GO TO 3000
                END IF
              END IF
            END IF
          END IF
        END IF
      END IF

      WRITE(*,1450)
1450  FORMAT(/,' *** Programming error - contact programmer ***',/)
      STOP

1500  WFLAST1=WFLAST  !upwards from start
      WFLAST=WF
      IF(IRESTART.EQ.0)THEN
        WF=WFBEG*WFFAC
      ELSE
        WF=WFMIN*WFFAC
      END IF
      IF(WF.GT.WFMAX)THEN
        WFSOL=WFMAX
        GO TO 3000
      ELSE
        GO TO 50
      END IF
1550  WFLAST1=WFLAST  !continue upwards
      WFLAST=WF
      WF=WF*WFFAC
      IF(WF.GT.WFMAX)THEN
        WFSOL=WFMAX
        GO TO 3000
      ELSE
        GO TO 50
      END IF
1600  WFLAST1=WFLAST  !continue downwards
      WFLAST=WF
      WF=WF/WFFAC
      IF(WF.LT.WFMIN)THEN
        WFSOL=WFMIN
        GO TO 3000
      ELSE
        GO TO 50
      END IF
1650  WFLAST1=WFLAST  !downwards from start
      WFLAST=WF
      IF(IRESTART.EQ.0)THEN
        WF=WFBEG/WFFAC
      ELSE
        WF=WFMIN/WFFAC
      END IF
      IF(WF.LT.WFMIN)THEN
        WFSOL=WFMIN
        GO TO 3000
      ELSE
        GO TO 50
      END IF

c -- The following code is executed only if phimlim is subtended.

2000  CONTINUE
      IMODE=1

c -- Of the three runs that we have, the weight factors are found that
c    correspond to those that most closely subtend phimlim.

      IF(PHIM.GE.PHIMLIM)THEN
        IHI=1
        PHIMHI=PHIM
        IF((PHIMLAST.GE.PHIMLIM).AND.(PHIMLAST.LT.PHIMHI))THEN
          IHI=2
          PHIMHI=PHIMLAST
        END IF
        IF((PHIMLAST1.GE.PHIMLIM).AND.(PHIMLAST1.LT.PHIMHI))THEN
          IHI=3
          PHIMHI=PHIMLAST1
        END IF
      ELSE IF(PHIMLAST.GE.PHIMLIM)THEN
        IHI=2
        PHIMHI=PHIMLAST
        IF((PHIMLAST1.GE.PHIMLIM).AND.(PHIMLAST1.LT.PHIMHI))THEN
          IHI=3
          PHIMHI=PHIMLAST1
        END IF
      ELSE IF(PHIMLAST1.GE.PHIMLIM)THEN
        IHI=3
        PHIMHI=PHIMLAST1
      ELSE
        GO TO 2500
      END IF
      IF(PHIMHI.GT.1.0D299) GO TO 2500
      IF(IHI.EQ.1)THEN
        WFHI=WF
      ELSE IF(IHI.EQ.2)THEN
        WFHI=WFLAST
      ELSE
        WFHI=WFLAST1
      END IF
      IF(PHIM.LT.PHIMLIM)THEN
        ILO=1
        PHIMLO=PHIM
        IF((PHIMLAST.LT.PHIMLIM).AND.(PHIMLAST.GT.PHIMLO))THEN
          ILO=2
          PHIMLO=PHIMLAST
        END IF
        IF((PHIMLAST1.LT.PHIMLIM).AND.(PHIMLAST1.GT.PHIMLO))THEN
          ILO=3
          PHIMLO=PHIMLAST1
        END IF
      ELSE IF(PHIMLAST.LT.PHIMLIM)THEN
        ILO=2
        PHIMLO=PHIMLAST
        IF((PHIMLAST1.LT.PHIMLIM).AND.(PHIMLAST1.GT.PHIMLO))THEN
          ILO=3
          PHIMLO=PHIMLAST1
        END IF
      ELSE IF(PHIMLAST1.LT.PHIMLIM)THEN
        ILO=3
        PHIMLO=PHIMLAST1
      ELSE
        GO TO 2500
      END IF
      IF(PHIMLO.LT.-1.0D299) GO TO 2500
      IF(ILO.EQ.1)THEN
        WFLO=WF
      ELSE IF(ILO.EQ.2)THEN
        WFLO=WFLAST
      ELSE
        WFLO=WFLAST1
      END IF

c -- Now calculate the new weight factor.

      IF(PHIMHI.EQ.PHIMLO)THEN
        WFSOL=WFHI
        GO TO 3000
      END IF
      WFNEW=WFLO+(PHIMLIM-PHIMLO)/(PHIMHI-PHIMLO)*(WFHI-WFLO)

c -- Termination criteria for search.

      IF((ABS(PHIMHI-PHIMLIM).LT.RTOL*PHIMLIM).OR.
     +   (ABS(PHIMLO-PHIMLIM).LT.RTOL*PHIMLIM))THEN
         WFSOL=WFNEW
         GO TO 3000
      END IF
      IF(ABS(PHIMHI-PHIMLO).LT.0.5*RTOL*(PHIMHI+PHIMLO))THEN
        WFSOL=WFNEW
        GO TO 3000
      END IF
      IF(ITN.GT.MAXITN)THEN
        WFSOL=WFNEW
        GO TO 3000
      END IF
c      WFLAST1=WFLAST
c      WFLAST=WF
c      WF=WFNEW
      WFLAST1=WFHI
      PHIMLAST1=PHIMHI
      WFLAST=WFLO
      PHIMLAST=PHIMLO
      WF=WFNEW
      ITN=ITN+1
      GO TO 60

2500  RTEMP=ABS(PHIM-PHIMLIM)
      WFSOL=WF
      IF(ABS(PHIMLAST-PHIMLIM).LT.RTEMP)THEN
        RTEMP=ABS(PHIMLAST-PHIMLIM)
        WFSOL=WFLAST
      END IF
      IF(ABS(PHIMLAST1-PHIMLIM).LT.RTEMP)THEN
        WFSOL=WFLAST1
      END IF
      GO TO 3000

3000  CONTINUE
      IF(ABS(WFSOL-WFMAX).LT.1.0D-5*WFMAX)THEN
        IF(IILAM.GE.NNLAM)RETURN
        IF(LAMSTR.EQ.0.0D0) RETURN
        IF(IILAM.EQ.0)THEN
          LAMBDA=LAMSTR
        ELSE
          LAMBDA=LAMBDA*LAMFAC
        END IF
        IILAM=IILAM+1
        GO TO 1
      END IF

      LHSFLAG=1
      WF=WFSOL
      GO TO 60

9980  IFAIL=1
      RETURN

#ifdef PESTMOD
      END SUBROUTINE OPTWT
#else
      END
#endif


C -- The following subroutine is a version of the above subroutine that was used
C    before execution speedup was accomplished at the cost of memory.

      SUBROUTINE OPTWT_SL(IFAIL,NXDIM,NSP4,NXROW,NPRIOR,NOBS,NPAR,
     +NESPAR,W1DIM,RHSDIM,MAXITN,WFBEG,WFFAC,PHIMLIM,WFMAX,WFMIN,
     +WFSOL,RTOL,ITRANS,REFOBS,OVAL,OWGHT,NOBGNM,TMPWT,TMPOBS,
     +SC,W1,X,RHS,LHS,INFOCOUNT,IUNSTABLE,LAMSTR,LAMFAC,ICOVOBS,
     +JSTK,NOBSGP,IRGP,
     +IPCG,LHSD1,LHSD2,NPCGDIM,cgr1,cgr2,cgv,cgw,cgy,
     +cgprecon,cgshift,cgnout,cgitnlim,cgeps,cgrtol,cgistop,cgitn,
     +cganorm,cgacond,cgrnorm,cgxnorm,lhsvdim,lhsvec,
     +WORKVEC1,WORKVEC2,WORKVEC3,
     +SVDMODE,MAXSING,LWORK,EIGTHRESH,WORKVEC4)

      USE PESTDATA, ONLY: XDIM1,XDIM2,XC,IXC,MAXCOMPDIM,NCOMPDIM,
     +                     ISTART_C,IFOUND_C,
     +                     LSQRMODE,LSQR_ATOL,LSQR_BTOL,LSQR_CONLIM,
     +                     LSQR_ITNLIM,IREGADJ
      IMPLICIT NONE


      INTEGER IFAIL,NXROW,NPRIOR,NOBS,NPAR,NESPAR,NSP4,MAXITN,
     +NXDIM,INFOCOUNT,IUNSTABLE,IRESTART,IRST,IILAM,NNLAM,NESTMP,
     +ICOVOBS,NOBSGP,W1DIM,RHSDIM
      INTEGER IPCG,LHSD1,LHSD2,LHSIND,NPCGDIM
      INTEGER NOBGNM(NXROW),ITRANS(NPAR),JSTK(NPAR),IRGP(NOBSGP)
      DOUBLE PRECISION WFBEG,WFFAC,PHIMLIM,WFMIN,WFMAX,WFSOL,RTOL,
     +LAMSTR,LAMFAC,LAMBDA,RRTEMP
      DOUBLE PRECISION REFOBS(NXROW),OVAL(NXROW),
     +OWGHT(NXROW),TMPWT(NXROW),TMPOBS(NXROW),SC(NESPAR),W1(W1DIM),
     +X(XDIM1,XDIM2),RHS(RHSDIM)
      DOUBLE PRECISION LHS(LHSD1,LHSD2)

      INTEGER ITN,IROW,IES,IPP,JES,I,INFO,IDIRN,IMODE,NST4,IREV,IHI,
     +ILO
      DOUBLE PRECISION WF,WFLAST,WFLAST1,PHIBEG,PHIRBEG,PHIMBEG,
     +PHI,PHILAST,PHILAST1,PHIM,PHIMLAST,PHIMLAST1,PHIR,PHIRLAST,
     +PHIRLAST1,RTEMP,WFTEMP,PHITEMP,PHIRTEMP,PHIMTEMP,PHIMHI,
     +WFHI,WFLO,PHIMLO,WFNEW,WFTEMP1,SUM1,RTEMP_C

C -- PCG STUFF

      logical cgprecon
      integer cgnout,cgitnlim,cgistop,cgitn
      integer lhsvdim
      double precision cgshift,cgeps,cgrtol,cganorm,cgacond,cgrnorm,
     +cgxnorm
      double precision cgr1(NPCGDIM),cgr2(NPCGDIM),cgv(NPCGDIM),
     +cgw(NPCGDIM),cgy(NPCGDIM)
      double precision lhsvec(lhsvdim)

C -- LSQR STUFF

      LOGICAL WANTSE
      INTEGER LSQR_LENIW,LSQR_LENRW
      INTEGER LSQR_ISTOP,LSQR_ITN
      INTEGER LSQR_IW(1)
      DOUBLE PRECISION LSQR_ANORM,LSQR_ACOND,LSQR_RNORM,LSQR_ARNORM,
     +LSQR_XNORM
      DOUBLE PRECISION WORKVEC1(NESPAR),WORKVEC2(NESPAR),
     +WORKVEC3(1)

C -- SVD MODE 2 STUFF

      INTEGER LDU,LDVT,SVDMODE,LWORK,IICOUNT,MAXSING,J,K,IESS,ITEMP,
     +NUMEIG
      DOUBLE PRECISION EIGMAX,EIGTHRESH
      DOUBLE PRECISION U(1,1),VT(1,1),WORKVEC4(LWORK)

      INTEGER ISTOP
      COMMON /SSTOP/ISTOP

      EXTERNAL CPROD

C -- Initialise where necessary.

      IFAIL=0
      IILAM=0
      NNLAM=5       !ARBITRARY
      LAMBDA=0.0D0
C      LAMBDA=LAMSTR
      LDU=1
      LDVT=1

1     CONTINUE
      IRESTART=0
      IRST=0
      GO TO 6

5     CONTINUE

6     CONTINUE
      INFOCOUNT=0
      IUNSTABLE=0
      ITN=0
      IF(IRST.NE.1)THEN
        WF=WFBEG
        WFLAST=WFBEG
        WFLAST1=WFBEG
      ELSE
        WFLAST=WF
        WFLAST1=WF
        IRST=0
      END IF
      PHIMLAST1=1.0D300
      PHIMLAST=1.0D300
      PHIM=1.0D300
      PHILAST=1.0D300
      PHI=1.0D300
      PHIR=1.0D300
      PHIRLAST=1.0D300
      IMODE=0
      IREV=0

C --  If there are no non-zero measurement weights return.

      DO 20 IROW=1,NXROW
        IF(IRGP(NOBGNM(IROW)).EQ.0)THEN
          IF(OWGHT(IROW).GT.0.0D0) GO TO 10
        END IF
20    CONTINUE
      WFSOL=WFBEG
      RETURN
10    CONTINUE

C -- For the initial "model" run, we use the weight factor supplied
C    from outside.


50    ITN=ITN+1

      PHILAST1=PHILAST
      PHILAST=PHI
      PHIMLAST1=PHIMLAST
      PHIMLAST=PHIM
      PHIRLAST1=PHIRLAST
      PHIRLAST=PHIR

C -- The new weights vector is formed. It is assumed that the owght vector
C    contains regularisation weights multiplied by WFBEG.

60    CONTINUE

      WFTEMP1=WF*WF/WFBEG/WFBEG
      DO 100 IROW=1,NXROW
        IF(IRGP(NOBGNM(IROW)).EQ.0)THEN
          TMPWT(IROW)=OWGHT(IROW)
        ELSE
          TMPWT(IROW)=OWGHT(IROW)*WFTEMP1
        END IF
100   CONTINUE

C -- If we are using LSQR, the following code is run.

      IF(LSQRMODE.NE.0)THEN
         IF(IILAM.EQ.0) LAMBDA=LAMSTR
         NESTMP=0
         DO IPP=1,NPAR
           IF(ITRANS(IPP).LT.0)CYCLE
           IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) CYCLE
           NESTMP=NESTMP+1
         END DO
         WANTSE=.FALSE.
         LSQR_LENIW=1
         LSQR_LENRW=NXROW
         DO IROW=1,NXROW
           TMPOBS(IROW)=SQRT(TMPWT(IROW))*(OVAL(IROW)-REFOBS(IROW))
         END DO
         CALL LSQR(NXROW,NESTMP,CPROD,LAMBDA,WANTSE,
     +             LSQR_LENIW,LSQR_LENRW,LSQR_IW,TMPWT,
     +             TMPOBS,WORKVEC1,WORKVEC2,W1,WORKVEC3,
     +             LSQR_ATOL,LSQR_BTOL,LSQR_CONLIM,LSQR_ITNLIM,
     +             0,
     +             LSQR_ISTOP,LSQR_ITN,LSQR_ANORM,LSQR_ACOND,
     +             LSQR_RNORM,LSQR_ARNORM,LSQR_XNORM,0)
            GO TO 8700
      END IF

C -- Next the diagonal elements of the scaling matrix are calculated.

      IF(SVDMODE.EQ.0)THEN
        IES=0
        ISTART_C=1
        DO 200 IPP=1,NPAR
          IF(ITRANS(IPP).LT.0) GO TO 200
          IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 200
          IES=IES+1
          IF(MAXCOMPDIM.LE.1)THEN
          CALL OBJCLC(2,SC(IES),NXROW,NPRIOR,NOBS,X(1,IES),OVAL,TMPWT,
     +    NOBGNM,NOBSGP,IRGP,SUM1)
          ELSE
            CALL SINGLE_VECTOR_MUL2(IFAIL,NCOMPDIM,1,NXROW,XC,IXC,
     +      IES,TMPWT,SC(IES))
            IF(IFAIL.NE.0) GO TO 9980
            ISTART_C=IFOUND_C
          END IF
          IF(SC(IES).LE.0.0D0) THEN
            SC(IES)=0.0D0
            GO TO 200
          END IF
          SC(IES)=SQRT(SC(IES))
          SC(IES)=1.0D0/SC(IES)
200     CONTINUE

        DO 220 IES=1,NESTMP
          IF(SC(IES).GT.0.0D0) GO TO 230
220     CONTINUE
        WFSOL=WFLAST
        RETURN
230     CONTINUE
      END IF

C -- The RHS of the normal equations is calculated.

      IES=0
      ISTART_C=1
      DO 250 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 250
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 250
        IES=IES+1
        RTEMP=0.0D0
        IF(MAXCOMPDIM.LE.1)THEN
          DO 270 IROW=1,NXROW
            RTEMP=RTEMP+X(IROW,IES)*TMPWT(IROW)*
     +            (OVAL(IROW)-REFOBS(IROW))
270       CONTINUE
        ELSE
          CALL SINGLE_VECTOR_MUL1(IFAIL,NCOMPDIM,1,NXROW,XC,IXC,
     +    IES,TMPWT,OVAL,REFOBS,RTEMP)
          IF(IFAIL.NE.0) GO TO 9980
          ISTART_C=IFOUND_C
        END IF
        IF(SVDMODE.EQ.0)THEN
          RHS(IES)=RTEMP*SC(IES)
        ELSE
          RHS(IES)=RTEMP
        END IF
250   CONTINUE

C -- The case where SVDMODE is 2 is now handled. Note that this is somewhat inefficient
C    due to the need to continually reformulate LHS.

      IF(SVDMODE.EQ.2)THEN
        ISTART_C=1
        IES=0
        IESS=0
        DO IPP=1,NPAR
          IF(ITRANS(IPP).LT.0)THEN
            IF(ITRANS(IPP).LT.-100001)IESS=IESS+1
            CYCLE
          END IF
          IES=IES+1
          IESS=IESS+1
          IF(MAXCOMPDIM.LE.1)THEN
            DO IROW=1,NXROW
              LHS(IROW,IES)=X(IROW,IESS)*SQRT(TMPWT(IROW))
            END DO
          ELSE
            CALL GET_VECTOR(NCOMPDIM,NXROW,XC,IXC,
     +      LHS(1,IES),IESS)
            ISTART_C=IFOUND_C
            DO IROW=1,NXROW
              LHS(IROW,IES)=LHS(IROW,IES)*SQRT(TMPWT(IROW))
            END DO
          END IF
        END DO
        NESTMP=0
        DO IPP=1,NPAR
          IF(ITRANS(IPP).LT.0)CYCLE
          IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) CYCLE
          NESTMP=NESTMP+1
        END DO
        NST4=NESTMP
        CALL DGESVD('N','O',NXROW,NST4,LHS,NXROW,W1,U,LDU,
     +  VT,LDVT,WORKVEC4,LWORK,INFO)
        IF(INFO.NE.0)THEN
          DO I=1,MIN(NXROW,NST4)
            W1(I)=0.0D0
          END DO
        ELSE
          DO I=1,MIN(NXROW,NST4)
            W1(I)=W1(I)*W1(I)
          END DO
        END IF
        IF(LAMBDA.NE.0)THEN
          DO I=1,MIN(NST4,NXROW)
            W1(I)=W1(I)+LAMBDA
          END DO
        END IF

C -- Low eigenvalues are now removed.

        IICOUNT=1
        EIGMAX=W1(1)
        IF(EIGMAX.LE.0.0D0) THEN
          WFSOL=WFLAST
          GO TO 3000
        END IF
        ITEMP=MIN(NST4,NXROW)
        DO I=2,ITEMP
          IF(ABS(W1(I)/EIGMAX).LT.EIGTHRESH)EXIT
          IF(IICOUNT.GE.MAXSING)EXIT
          IICOUNT=IICOUNT+1
        END DO
        NUMEIG=IICOUNT

C -- Next the solution is computed.

        DO I=1,IICOUNT
          RTEMP=0.0D0
          DO K=1,NST4
            RTEMP=RTEMP+LHS(I,K)*RHS(K)
          END DO
          WORKVEC4(I)=RTEMP/W1(I)
        END DO
        DO I=1,NST4
          RTEMP=0.0D0
          DO K=1,IICOUNT
            RTEMP=RTEMP+LHS(K,I)*WORKVEC4(K)
          END DO
          W1(I)=RTEMP
        END DO
        GO TO 8700
      END IF

C -- Next the normal matrix is calculated.

      IES=0
      NESTMP=0
      LHSIND=0
      DO 400 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0)GO TO 400
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 400
        NESTMP=NESTMP+1
        IES=IES+1
        JES=0
        ISTART_C=1
        DO 300 I=1,IPP
          IF(ITRANS(I).LT.0) GO TO 300
          IF((ICOVOBS.EQ.0).AND.(JSTK(I).LT.0)) GO TO 300
          JES=JES+1
          RTEMP=0.0D0
          IF(MAXCOMPDIM.LE.1)THEN
            DO 360 IROW=1,NXROW
              RTEMP=RTEMP+TMPWT(IROW)*X(IROW,IES)*X(IROW,JES)
360         CONTINUE
          ELSE
             CALL DUAL_VECTOR_MUL(IFAIL,NCOMPDIM,1,NXROW,XC,
     +       IXC,TMPWT,IES,JES,RTEMP)
             IF(IFAIL.NE.0) GO TO 9980
          ENDIF
          IF(IPCG.EQ.0)THEN
            LHS(JES,IES)=RTEMP*SC(IES)*SC(JES)
          ELSE
            IF(JES.LE.IES)THEN
              LHSIND=LHSIND+1
              LHSVEC(LHSIND)=RTEMP*SC(IES)*SC(JES)
            END IF
          END IF
300     CONTINUE
400   CONTINUE
      IF(IPCG.EQ.0)THEN
        DO IES=1,NESTMP-1
          DO JES=IES+1,NESTMP
            LHS(JES,IES)=LHS(IES,JES)
          END DO
        END DO
      END IF

C -- The Marquardt lambda is added if necessary.

425   IF(LAMBDA.NE.0.0D0) THEN
        RTEMP=0.0D0
        DO 440 IES=1,NESTMP
        IF(ABS(SC(IES)).GT.RTEMP) RTEMP=ABS(SC(IES))
440     CONTINUE
        RRTEMP=LAMBDA/RTEMP/RTEMP
        LHSIND=0
        DO 445 IES=1,NESTMP
        IF(IPCG.EQ.0)THEN
          LHS(IES,IES)=1.0D0+SC(IES)*SC(IES)*RRTEMP
        ELSE
          LHSIND=LHSIND+IES
          LHSVEC(LHSIND)=1.0D0+SC(IES)*SC(IES)*RRTEMP
        END IF
445     CONTINUE
      END IF

      NST4=NESTMP
      IF(IPCG.EQ.1)THEN
        INFO=0
         call cgsolve(NST4,RHS,cgr1,cgr2,cgv,cgw,W1,cgy,
     +   cgprecon,cgshift,cgnout,cgitnlim,cgeps,cgrtol,cgistop,cgitn,
     +   cganorm,cgacond,cgrnorm,cgxnorm,lhsvdim,lhsvec)
         IF(CGISTOP.GE.4)INFO=1
      ELSE
          CALL DPOFA(LHS,NSP4,NST4,INFO)
      END IF

      IF(INFO.NE.0)THEN
        INFOCOUNT=INFOCOUNT+1
        PHI=-1.0D300
        PHIM=1.0D300
        IF(ITN.EQ.1)THEN
          IF(IRESTART.EQ.0)THEN
            IRESTART=1
            IRST=1
            WF=WFMIN
            GO TO 5
          END IF
          IDIRN=1
          GO TO 1500
        ELSE IF(ITN.EQ.2)THEN
          IF(IDIRN.EQ.-1)THEN
            WFSOL=WFLAST
            GO TO 3000
          ELSE
            IF(PHILAST.GT.-1.0D300)THEN
              WFSOL=WFLAST
              GO TO 3000
            ELSE
              GO TO 1550
            END IF
          END IF
        ELSE
          IF(IREV.EQ.1)THEN
            WFTEMP=WFLAST1
            WFLAST1=WFLAST
            WFLAST=WFTEMP
            PHITEMP=PHILAST1
            PHILAST1=PHILAST
            PHILAST=PHITEMP
            PHIMTEMP=PHIMLAST1
            PHIMLAST1=PHIMLAST
            PHIMLAST=PHIMTEMP
            PHIRTEMP=PHIRLAST1
            PHIRLAST1=PHIRLAST
            PHIRLAST=PHIRTEMP
            IREV=0
          END IF
          IF(IMODE.EQ.0) THEN
            IF(IDIRN.EQ.-1)THEN
              WFSOL=WFLAST
              GO TO 3000
            ELSE
              IF(PHILAST.GT.-1.0D300)THEN
                WFSOL=WFLAST
                GO TO 3000
              ELSE
                GO TO 1550
              END IF
            END IF
          ELSE
            WFSOL=WFLAST
            GO TO 3000
          END IF
        END IF
      END IF


C the above strategy assumes that increasing the regularisation weight
C is better for the solution numerically. This is really the main condition on the
C regularisation information.

C -- NEXT WE SOLVE FOR THE UPDATING DIRECTION

      IF(IPCG.EQ.0)THEN
        DO 500 IES=1,NESPAR
          W1(IES)=RHS(IES)
500     CONTINUE
        CALL DPOSL(LHS,NSP4,NST4,W1)
      END IF
      DO 510 IES=1,NESPAR
        W1(IES)=W1(IES)*SC(IES)
510   CONTINUE
8700  CONTINUE

C -- CHECK TO SEE IF WE SHOULD STOP

      CALL STOPRESS(0)
      IF((ISTOP.EQ.2).OR.(ISTOP.EQ.1)) RETURN

C -- THE OBJECTIVE FUNCTION AND ITS TWO COMPONENTS ARE NEXT CALCULATED
C    UNDER THE LINEARITY ASSUMPTION.

      DO 600 IROW=1,NXROW
        RTEMP=0.0D0
        IES=0
        ISTART_C=1
        DO 580 IPP=1,NPAR
          IF(ITRANS(IPP).LT.0) GO TO 580
          IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 580
          IES=IES+1
          IF(MAXCOMPDIM.LE.1)THEN
            RTEMP=RTEMP+X(IROW,IES)*W1(IES)
          ELSE
            CALL GET_VALUE(NCOMPDIM,XC,IXC,RTEMP_C,IROW,IES)
            ISTART_C=IFOUND_C
            RTEMP=RTEMP+RTEMP_C*W1(IES)
          END IF
580     CONTINUE
        TMPOBS(IROW)=RTEMP+REFOBS(IROW)
600   CONTINUE

C -- THE NEW OBJECTIVE FUNCTION COMPONENTS ARE CALCULATED.

      CALL OBJCLC(1,PHI,NXROW,NPRIOR,NOBS,TMPOBS,OVAL,TMPWT,NOBGNM,
     +NOBSGP,IRGP,SUM1)
      CALL OBJCLC(888888,PHIR,NXROW,NPRIOR,NOBS,TMPOBS,
     +OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,PHIM)
      PHIR=PHIR/WFBEG/WFBEG

C -- NEXT A DECISION IS MADE ON WHAT THE NEXT WEIGHTING FACTOR SHOULD BE.

      IF(ABS(PHIM-PHIMLIM).LT.RTOL*PHIMLIM)THEN
        WFSOL=WF
        GO TO 3000
      END IF

      IF(IMODE.NE.0) GO TO 2000

      IF(PHILAST.LT.-1.0D300)THEN
        IF(PHIM.GE.PHIMLIM)THEN
          WFSOL=WF
          GO TO 3000
        ELSE
          IDIRN=1
          GO TO 1550
        END IF
      END IF

      IF(ITN.EQ.1)THEN
        IF(PHIM.GT.PHIMLIM)THEN
          IDIRN=-1
          GO TO 1600
        ELSE
          IDIRN=1
          GO TO 1550
        END IF
      ELSE IF(ITN.EQ.2)THEN
        IF(IDIRN.EQ.-1)THEN
          IF(PHIM.GT.PHIMLIM)THEN
            IF(PHIM.LT.PHIMLAST-RTOL*PHIMLAST)THEN
              GO TO 1600
            ELSE
              WFSOL=WF
              GO TO 3000
            END IF
          ELSE
            IF(PHIM.LT.PHIMLAST)THEN
              IF(PHIMLAST.GE.PHIMLIM)THEN
                IF(PHIR.GE.PHIRLAST-RTOL*PHIRLAST)THEN
                  GO TO 2000
                ELSE
                  GO TO 1600
                END IF
              ELSE
                IF(ABS(PHIR-PHIRLAST).LT.0.5*RTOL*(PHIR+PHIRLAST))THEN
                  IF(PHIR.LE.PHIRLAST)THEN
                    WFSOL=WF
                  ELSE
                    WFSOL=WFLAST
                  END IF
                  GO TO 3000
                END IF
                IF(PHIR.GE.PHIRLAST)THEN
                  IREV=1
                  GO TO 1500
                ELSE
                  GO TO 1600
                END IF
              END IF
            ELSE
              WFSOL=WFLAST
              GO TO 3000
            END IF
          END IF
        ELSE IF(IDIRN.EQ.1)THEN
          IF(PHIM.LT.PHIMLAST)THEN
            WFSOL=WF
            GO TO 3000
          ELSE
            IF(PHIM.LT.PHIMLIM)THEN
              IF(PHIR.LT.PHIRLAST)THEN
                GO TO 1550
              ELSE
                IDIRN=-1
                IREV=-1
                GO TO 1650
              END IF
            ELSE
              IF(PHIMLAST.LE.PHIMLIM)THEN
                GO TO 2000
              ELSE
                WFSOL=WFLAST  !we should not be here
                GO TO 3000
              END IF
            END IF
          END IF
        END IF
      ELSE IF(ITN.GT.2)THEN
        IF(IREV.EQ.1)THEN
          WFTEMP=WFLAST1
          WFLAST1=WFLAST
          WFLAST=WFTEMP
          PHITEMP=PHILAST1
          PHILAST1=PHILAST
          PHILAST=PHITEMP
          PHIMTEMP=PHIMLAST1
          PHIMLAST1=PHIMLAST
          PHIMLAST=PHIMTEMP
          PHIRTEMP=PHIRLAST1
          PHIRLAST1=PHIRLAST
          PHIRLAST=PHIRTEMP
          IREV=0
        END IF
        IF(IDIRN.EQ.1)THEN
          IF(ABS(PHIM-PHIMLAST).LT.0.5*RTOL*(PHIMLAST+PHIM)*0.2)THEN  !0.2 ARBITRARY
            IUNSTABLE=1
            WFSOL=WF
            GO TO 3000
          END IF
          IF(PHIM.LT.PHIMLAST)THEN
            WFSOL=WF
            GO TO 3000
          ELSE
            IF(PHIM.GE.PHIMLIM)THEN
              IF(PHIMLAST.LE.PHIMLIM)THEN
                GO TO 2000
              ELSE
                WFSOL=WFLAST  !WE SHOULD NOT BE HERE
                GO TO 3000
              END IF
            ELSE
              IF(PHIR.LT.PHIRLAST)THEN
                GO TO 1550
              ELSE
                WFSOL=WFLAST
                GO TO 3000
              END IF
            END IF
          END IF
        ELSE IF(IDIRN.EQ.-1)THEN
          IF(PHIM.GT.PHIMLAST-0.5*RTOL*(PHIMLAST+PHIM))THEN
            IF(PHIR.GT.PHIRLAST-0.5*RTOL*(PHIRLAST+PHIR))THEN
              IF(PHIR.LE.PHIRLAST)THEN
                WFSOL=WF
              ELSE
                WFSOL=WFLAST
              END IF
              GO TO 3000
            ELSE
              GO TO 1600
            END IF
          ELSE
            IF(PHIM.GT.PHIMLIM)THEN
              GO TO 1600
            ELSE
              IF(PHIMLAST.GE.PHIMLIM)THEN
                GO TO 2000
              ELSE
                IF(PHIR.LT.PHIRLAST-0.5*RTOL*(PHIR+PHIRLAST))THEN
                  GO TO 1600
                ELSE
                  IF(PHIR.LT.PHIRLAST)THEN
                    WFSOL=WF
                  ELSE
                    WFSOL=WFLAST
                  END IF
                  GO TO 3000
                END IF
              END IF
            END IF
          END IF
        END IF
      END IF

      WRITE(*,1450)
1450  FORMAT(/,' *** Programming error - contact programmer ***',/)
      STOP

1500  WFLAST1=WFLAST  !upwards from start
      WFLAST=WF
      IF(IRESTART.EQ.0)THEN
        WF=WFBEG*WFFAC
      ELSE
        WF=WFMIN*WFFAC
      END IF
      IF(WF.GT.WFMAX)THEN
        WFSOL=WFMAX
        GO TO 3000
      ELSE
        GO TO 50
      END IF
1550  WFLAST1=WFLAST  !continue upwards
      WFLAST=WF
      WF=WF*WFFAC
      IF(WF.GT.WFMAX)THEN
        WFSOL=WFMAX
        GO TO 3000
      ELSE
        GO TO 50
      END IF
1600  WFLAST1=WFLAST  !continue downwards
      WFLAST=WF
      WF=WF/WFFAC
      IF(WF.LT.WFMIN)THEN
        WFSOL=WFMIN
        GO TO 3000
      ELSE
        GO TO 50
      END IF
1650  WFLAST1=WFLAST  !downwards from start
      WFLAST=WF
      IF(IRESTART.EQ.0)THEN
        WF=WFBEG/WFFAC
      ELSE
        WF=WFMIN/WFFAC
      END IF
      IF(WF.LT.WFMIN)THEN
        WFSOL=WFMIN
        GO TO 3000
      ELSE
        GO TO 50
      END IF

c -- The following code is executed only if phimlim is subtended.

2000  CONTINUE
      IMODE=1

c -- Of the three runs that we have, the weight factors are found that
c    correspond to those that most closely subtend phimlim.

      IF(PHIM.GE.PHIMLIM)THEN
        IHI=1
        PHIMHI=PHIM
        IF((PHIMLAST.GE.PHIMLIM).AND.(PHIMLAST.LT.PHIMHI))THEN
          IHI=2
          PHIMHI=PHIMLAST
        END IF
        IF((PHIMLAST1.GE.PHIMLIM).AND.(PHIMLAST1.LT.PHIMHI))THEN
          IHI=3
          PHIMHI=PHIMLAST1
        END IF
      ELSE IF(PHIMLAST.GE.PHIMLIM)THEN
        IHI=2
        PHIMHI=PHIMLAST
        IF((PHIMLAST1.GE.PHIMLIM).AND.(PHIMLAST1.LT.PHIMHI))THEN
          IHI=3
          PHIMHI=PHIMLAST1
        END IF
      ELSE IF(PHIMLAST1.GE.PHIMLIM)THEN
        IHI=3
        PHIMHI=PHIMLAST1
      ELSE
        GO TO 2500
      END IF
      IF(PHIMHI.GT.1.0D299) GO TO 2500
      IF(IHI.EQ.1)THEN
        WFHI=WF
      ELSE IF(IHI.EQ.2)THEN
        WFHI=WFLAST
      ELSE
        WFHI=WFLAST1
      END IF
      IF(PHIM.LT.PHIMLIM)THEN
        ILO=1
        PHIMLO=PHIM
        IF((PHIMLAST.LT.PHIMLIM).AND.(PHIMLAST.GT.PHIMLO))THEN
          ILO=2
          PHIMLO=PHIMLAST
        END IF
        IF((PHIMLAST1.LT.PHIMLIM).AND.(PHIMLAST1.GT.PHIMLO))THEN
          ILO=3
          PHIMLO=PHIMLAST1
        END IF
      ELSE IF(PHIMLAST.LT.PHIMLIM)THEN
        ILO=2
        PHIMLO=PHIMLAST
        IF((PHIMLAST1.LT.PHIMLIM).AND.(PHIMLAST1.GT.PHIMLO))THEN
          ILO=3
          PHIMLO=PHIMLAST1
        END IF
      ELSE IF(PHIMLAST1.LT.PHIMLIM)THEN
        ILO=3
        PHIMLO=PHIMLAST1
      ELSE
        GO TO 2500
      END IF
      IF(PHIMLO.LT.-1.0D299) GO TO 2500
      IF(ILO.EQ.1)THEN
        WFLO=WF
      ELSE IF(ILO.EQ.2)THEN
        WFLO=WFLAST
      ELSE
        WFLO=WFLAST1
      END IF

c -- Now calculate the new weight factor.

      IF(PHIMHI.EQ.PHIMLO)THEN
        WFSOL=WFHI
        GO TO 3000
      END IF
      WFNEW=WFLO+(PHIMLIM-PHIMLO)/(PHIMHI-PHIMLO)*(WFHI-WFLO)

c -- Termination criteria for search.

      IF(ABS(PHIMHI-PHIMLIM).LT.RTOL*PHIMLIM)THEN
        WFSOL=WFHI
        GO TO 3000
      END IF
      IF(ABS(PHIMLO-PHIMLIM).LT.RTOL*PHIMLIM)THEN
        WFSOL=WFLO
        GO TO 3000
      END IF
      IF((ABS(PHIMHI-PHIMLIM).LT.RTOL*PHIMLIM).OR.
     +   (ABS(PHIMLO-PHIMLIM).LT.RTOL*PHIMLIM))THEN
         WFSOL=WFNEW
         GO TO 3000
      END IF
      IF(ABS(PHIMHI-PHIMLO).LT.0.5*RTOL*(PHIMHI+PHIMLO))THEN
        WFSOL=WFNEW
        GO TO 3000
      END IF
      IF(ITN.GT.MAXITN)THEN
        WFSOL=WFNEW
        GO TO 3000
      END IF
c      WFLAST1=WFLAST
c      WFLAST=WF
c      WF=WFNEW
      WFLAST1=WFHI
      PHIMLAST1=PHIMHI
      WFLAST=WFLO
      PHIMLAST=PHIMLO
      WF=WFNEW
      ITN=ITN+1
      GO TO 60

2500  RTEMP=ABS(PHIM-PHIMLIM)
      WFSOL=WF
      IF(ABS(PHIMLAST-PHIMLIM).LT.RTEMP)THEN
        RTEMP=ABS(PHIMLAST-PHIMLIM)
        WFSOL=WFLAST
      END IF
      IF(ABS(PHIMLAST1-PHIMLIM).LT.RTEMP)THEN
        WFSOL=WFLAST1
      END IF
      GO TO 3000

3000  CONTINUE
      IF(ABS(WFSOL-WFMAX).LT.1.0D-5*WFMAX)THEN
        IF(IILAM.GE.NNLAM)RETURN
        IF(LAMSTR.EQ.0.0D0) RETURN
        IF(IILAM.EQ.0)THEN
          LAMBDA=LAMSTR
        ELSE
          LAMBDA=LAMBDA*LAMFAC
        END IF
        IILAM=IILAM+1
        GO TO 1
      END IF

      RETURN

9980  IFAIL=1
      RETURN

#ifdef PESTMOD
      END SUBROUTINE OPTWT_SL
#else
      END
#endif
