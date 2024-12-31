        SUBROUTINE RUN_PEST(IPFAIL)

        USE SVD_PARM_DATA
        USE PESTDATA
#ifdef BEO
        USE BEOPEST
#endif
        IMPLICIT NONE

C --- Maximum length of line search.

        INTEGER, PARAMETER :: MAXSEARCH=100

C --- File width
        CHARACTER*2000 BUF

C -- Declarations for logical variables.

#ifdef PESTMOD
        LOGICAL LOPENED
#else
        LOGICAL SAME,LOPENED
#endif
        LOGICAL LEXIST,CGPRECON
        LOGICAL WANTSE
        LOGICAL EQUALS

C -- Temporary declaration of variables used in LSQR. We'll put them here for
C    the moment because some of these will need to mature.

        INTEGER LSQR_LENIW,LSQR_LENRW,LSQR_ISTOP,LSQR_ITN
        INTEGER LSQR_NOUT,HOLDFLAG
        INTEGER RHSDIM,IEXTJACMODE,IPASS1,IFLAG_X,IFLAG_NOCOV
        INTEGER NLHSDIM1,NLHSDIM2,NRMDIM,NRMLODIM,NMLODIM1,NMLODIM2
        INTEGER LSQR_IW(1)
        DOUBLE PRECISION LSQR_ANORM,LSQR_ACOND,LSQR_RNORM,LSQR_ARNORM,
     +  LSQR_XNORM
        DOUBLE PRECISION LSQR_RW(1)
        CHARACTER*200 JCOFILE
        EXTERNAL CPROD

C -- Declarations for integer variables.

        INTEGER ICOUNT1,ICOUNT2,IITEMP,MIPP,MIES,ITEMP1,NOBSNZERO
        INTEGER IIRST,ITEMP,DONTCOPY,IIIRST,IVEC,MJES,MJESS
        INTEGER JWT,JCOV,NM,NM1,MXOBSCOV,IGROUP,TOTWT,N,NN,
     +          IPSTART,MATDIM1,NCOVDIM1,NCOVDIM2,NCOVDIM3,NWDIM,
     +          LT1LAST,LT2LAST,LTNEG,LKTN,ILTFACHI,ILTFACLO,
     +          IUNSTABLE,LTFLAGX
        INTEGER IPMXCHG,IPMX1CHG,IPMX2CHG
        INTEGER LITN,LTLAST,LTFLAG,KITN,LTREV,LJTN,
     +          LTBIS,LT1,LT2,LT3,LTFLOTN,LTFHITN,LLTN,LTCOUNT
        INTEGER IREG,REGITN,NXDIM,INFOCOUNT,JJREG,JIRST
        INTEGER JCALL,JACMS
        INTEGER LSCH1,LDOSCH,LSEARCH,LRUN,LR,SENSTO,LMRUN,FRFLAG,
     +          ILAMPL,ILAMMAX,SITN,SJMM,NOPAR,ITNJMM,SIMM,LINESAVE,
     +          ISUBTEND,ICONVERGE,REPEATRUN
        INTEGER IAUI,NOAUI,IAFIN,NANJPPL0,NANJPPL1,
     +          NANJPPL2,IAPMXCHG,IAPMX1CHG,IAPMX2CHG,IFRAC,ICC,
     +          IMINSENS,IMAXSENS,FIRSTAUI,IPHOLD,
     +          IAHOLD,IISPACE,ISOLMSG
        INTEGER NVECBND,JJ,ICOVOBS,ISTKFLG,IWARN1,
     +          IRST1JAC,IIFLAG
        INTEGER NREGOBS,IGPNM,IPIVOTREG,IMAXTOT,K,
     +          NLHSDIM,LHSIND,LHSVDIM,NPCGDIM,
     +          LHSD1,LHSD2,LHSVD,LPOS,INUM,LHSFLAG,LASTOBSROW,
     +          INOCOV,IOPTCALL,FIRSTREGROW,ITEMP2,
     +          NCONDUNIT,NORIGWGHT,ISENSCOUNT,IWCOUNT,
     +          IRCOUNT,IG,ITEMP3,ITEMP4
        INTEGER PIRESTART,PTUNIT,SRST,NUMTIED,JNESPAR,DIRESTART,
     +          DTUNIT,DRST,IDCOUNT,NORMRETURN,JTIMES,NOWRITE
        INTEGER SVDUNIT,IICOUNT,NOLAMBDA,LWORK,JFAIL,TEMPJACDIM
        INTEGER JSVDA,WORK_CDIM,NREGADJPAR
        INTEGER I,J,I2OR3,IPP,IOPT,IES,JPP0,JPP1,JPP0LT,
     +          JPP1LT,ITN,NOCHNG,ILOCK,LOOPT,NESTMP,IMM,NJPPL0,
     +          NJPPL1,JPP0LO,JPP1LO,IROW,
     +          IFIN,IESS,JES,JESS,ISWTCH,IOBS,NUMREL,NUMFAC,
     +          ICHNG,NREV,NREVLO,OPTSWH,NOWRLM,JMM,JPP2,DECNOF,
     +          JPP2LT,JPP2LO,NJPPL2,NUML,NBLBMX,
     +          IRST,IIOPT,MATSTO,NEPL23,ISTOP,
     +          DDCALL,MMCALL,NSCALL,SOPDIM,NOEFF,NSLAVE,NRUN,
     +          IRUN,NNRUN,II,NSRUN,IFLTYP,LDSIN,LDSOU,
     +          iperr,jperr,IIRUN,IWAIT,IIIWAIT,ISHELL,IPRLL,IPASS,
     +          NOBSGP1,NSCALLP,MMCALLP,DDCALLP,DERFLAG
        INTEGER ICOVF,ICORF,IEIGF
        INTEGER PDFLAG,IFIN1,IPRED,JPREDICT,
     +          PNOBGNM,JJPRED,ICOUNT,KPRED
        INTEGER ILINE,IERR,NCALL,INFO,NSP4,NST4,NINSTR,ASIZE,MCALL
        INTEGER IFAIL,IPFAIL,CGNOUT,CGISTOP,CGITN
        INTEGER NNSEARCH,IMCOVFLAG,ICOUNTREGOBS,ICOUNTREGNZ,NUMEIG,
     +          NEWNUMEIG,NANNUMEIG,NUMEIGLST,NOBSREGUL,KREG,NUMAVO
        INTEGER MSRUN,MLAMRUN,SWITCHFLAG,SBACKFLAG,LAMFLAG
        INTEGER W1DIM
        INTEGER DIM_DI_PAR,DIM_DI_OBS,DIM_DI_IES,DIM_DI_PARENT
        INTEGER LDU,LDVT,W2DIM,NDIM4
        INTEGER JACUPDATEFLAG,JACUPCOUNT
        INTEGER ISENREUSE,ISENREUSEFLAG,SENMXREUSE,IGP,MAXIPP
        INTEGER SLAVESTOP
        INTEGER REIFLAG,IGTARG,ISPLIT,WV5DIM,SPLITFLAG,JSPLIT,JA2
        INTEGER JFRAC
        INTEGER MANYLAM,NUMLAMORIG
        INTEGER TEMPDIM1,TEMPDIM2,IRR,IESTOT,IWORKTEMP,NNEIG,LWORKTEMP
        INTEGER JFLAG
        INTEGER WORKVEC3DIM
        INTEGER REPEATFLAG
        INTEGER FAILUREFLAG,FAILURECOUNT,SUCCESSCOUNT
        INTEGER FLAG5,DAPARRECL,DAOBSRECL,WORKVEC7DIM
        INTEGER NMLODIM2_FLAG
        INTEGER PARETO_FIN,PAR_TERM_COUNT,NUM_WEIGHT_ADJUST,IWTFLAG,
     +          PAR_NUM_PREVIOUS

#ifdef UNIX
        INTEGER UNLINK
#ifdef SYS_FUNCTION
        INTEGER NNN
        INTEGER SYSTEM
#endif
#endif

C -- Declarations for integer arrays.

        INTEGER PHOLD(:),EHOLD(:),ISENS(:)
        INTEGER LL(:),LCINS(:),NW(:),OBSN1(:),OBSN2(:),IDBND(:),
     +          IIOBS(:),INCPAR(:),JSTK(:),IRGP(:),PRIORPAR(:)

C -- Declarations for real arrays.

        REAL             ORIGWGHT(:),IWSENS(:)

C -- Declarations for double precision variables.

        DOUBLE PRECISION SUM1,RCOND,EIGMAX
        DOUBLE PRECISION RRRTEMP,NLAMBDA,NRELPMX,NFACPMX,RTEMPMAX,
     +                   NRELPREDSTP,NABSPREDSTP,NINITSCHFAC,
     +                   NMULSCHFAC,BESTFAC
        DOUBLE PRECISION LTFAC,LTPHILO,LTPHIHI,
     +                   LTFACLO,LTRTEMP1,LTRTEMP2,LTFAC1,LTFACMIN,
     +                   LTFACMAX,LTPRED1,LTFACHI,PD12,RMAXFAC,
     +                   RMINFAC,RDISTLO,RDISTHI
        DOUBLE PRECISION LTF1,LTF2,LTF3,LTP1,LTP2,LTP3,LTF2F1,
     +                   LTF1F3,LTF3F2,LTA,LTB,LTC
        DOUBLE PRECISION WF,PHIR2,PHIM,WFSOL,WFTEMP,PHIML,PHIRL,
     +                   PHIMLO,PHIRLO,PHIR,TEMPHIM,TEMPHIR,NEWPHIM,
     +                   NEWPHIR,WFLO,PD1RFAC,PHIMLIMKP
        DOUBLE PRECISION LAMKPL,LAMKPH,SLAMBDA
        DOUBLE PRECISION AFRACPSI,ALOFRAC,NAROL,NAPSI,NANDMXL0,
     +                   NANDMXL1,NANDMXL2,NANLAM,AMXCHG,AMX1CHG,
     +                   AMX2CHG,MAXSENS,MINSENS,FIRSTPSI,PSIPI,
     +                   ALAMBDA,NAPHIM,NAPHIR
        DOUBLE PRECISION RMAXTOT,RSUM2,RMAX2,RSUM1,SEOGPPIV,
     +                   RCOUNT,WFSTART,GEOMAVSENS,RTEMPG,TOTPHIREG,
     +                   BESTFRAC,RTEMP_C
        DOUBLE PRECISION RTEMP,PSI,LAMBDA,RRTEMP,BETA,
     +                   ADMX0,ADMX1,ROSTAR,S,RO,DMX0LT,DMX1LT,ROLAST,
     +                   TEMPSI,DMX0,DMX1,PSILO,PSIL,GRDNRM,DIRNRM,
     +                   TMPSIL,NROL,NDMXL0,NDMXL1,RLO,DMX0LO,DMX1LO,
     +                   NEWPSI,RO0,RO1,S0,S1,MXCHG,LAMSTR,
     +                   LAMLST,LAMLO,NEWLAM,FRCPSI,MX1CHG,MX2CHG,
     +                   DMAX2,DMX2,ADMX2,S2,RO2,DMX2LT,DMX2LO,NDMXL2,
     +                   PSI1,TVAL,TTVAL,RRRTMP,TTTVAL
        DOUBLE PRECISION PDEN,PNUM,PLAMDASH,RTEMP1
        DOUBLE PRECISION PRTEMP,PPA,PPB,PPC,PROOT1,PROOT2,
     +                   PRTEMP1,PRTEMP2,PRTEMP3,PRTEMP4
        DOUBLE PRECISION NEWPRED,TEMPRD,PRED1,TMPRL,RPTEMP,
     +                   PREDLO,REFPRED,PREDL,POVAL,POWGHT,PWTKP
        DOUBLE PRECISION CGSHIFT,CGEPS,CGANORM,CGACOND,CGRNORM,
     +                   CGXNORM
        DOUBLE PRECISION ALPHA,NEWALPHA,NANALPH,ALPHLST,AVOWGHTKP
        DOUBLE PRECISION E_PRED, EW_PRED, RES_PRED,E_PREDLO,RES_PREDLO,
     +                   EW_PREDLO,NEWE_PRED,NEWEW_PRED,NEWRES_PRED
        DOUBLE PRECISION RTEMPLAM
        DOUBLE PRECISION SENMAX,PHIAB
        DOUBLE PRECISION TPSISUB,TOTGPHIMLIM,GPHIMLIM,FACADJUST,DTEMP
        DOUBLE PRECISION OLDARAT,RTEMP2
        DOUBLE PRECISION PROJTOT,WEIGHTPROJTOT,TOTWEIGHTSUM,
     +                   REGWEIGHTNUL,WEIGHTMAX,WEIGHTMIN,
     +                   WEIGHTRANGE,WEIGHTRATIO,WFLAST,REGSENMAX
        DOUBLE PRECISION PAR_WTFAC,PAR_WTFAC_KP

C -- Declarations for double precision arrays.

        DOUBLE PRECISION REFOBS(:),W2(:),ORGVAL(:),TMPVAL(:),RHS(:),
     +                   W1(:),LHS(:,:),GRAD(:),
     +                   TMPOBS(:),NEWVAL(:),NEWOBS(:),SC(:),
     +                   PSILOL(:),LSTVAL(:),NORM(:,:),NORMLO(:,:),
     +                   GHOLD(:),PSISUB(:),NRM(:,:),NRMLO(:,:),
     +                   SEOGP(:),OWGHTLO(:),WORKVEC1(:),WORKVEC2(:),
     +                   WORKVEC3(:),WORKVEC4(:),WORK_C(:)
        DOUBLE PRECISION WORKVEC5(:)
        DOUBLE PRECISION PHIMLOL(:),PHIRLOL(:),PRDLOL(:),PN(:),
     +                   PNTEMP(:),PC0(:)
        DOUBLE PRECISION W1KP(:),LTTMPOBS(:,:),LTPHI(:),LTPRED(:),
     +                   LTFACC(:),RWORK1(:),RWORK2(:)
        DOUBLE PRECISION NAVAL(:),NAOBS(:),SENS(:)
        DOUBLE PRECISION CGR1(:),CGR2(:),CGV(:),CGW(:),CGY(:),LHSVEC(:)
        DOUBLE PRECISION OWGHTKP(:)
        DOUBLE PRECISION SCREUSE(:)
        DOUBLE PRECISION GFAC(:),OLDGFAC(:)
        DOUBLE PRECISION TEMPSVD(:,:),WORKVEC6(:)
        DOUBLE PRECISION WORKVEC7(:)
        DOUBLE PRECISION PARPSISUB(:)

C -- Declarations for character variables.

        CHARACTER*1   PPOSNEG,PREDHL
        CHARACTER*5   ABOUND
        CHARACTER*7   APRECS,AATEMP,AAMAXMIN
        CHARACTER*8   FMT
        CHARACTER*9   ATEMP1,APOINT
        CHARACTER*12  ATEMPG,ATEMP3
        CHARACTER*12  ATEMP12
        CHARACTER*16  ANUM
        CHARACTER*20  PAOBS
        CHARACTER*20  ATEMP20
        CHARACTER*80  AEXPL
        CHARACTER*100 CASEFL
        CHARACTER*200 FLENME1,JUPFILE
        CHARACTER*200 PTFILE,DTFILE,RSDFILE
        CHARACTER*200 MODFLE
        CHARACTER*200 FLENME
        CHARACTER*200 PAR_PHIFILE,PAR_PARFILE
        CHARACTER*200 AFILE

C -- Declarations for character arrays.

        CHARACTER*80 MSG(2)

        CHARACTER     PARDEL(:),MRKDEL(:),A(:)
        CHARACTER*12  APAR1(:),APAR2(:)
        CHARACTER*23  PWORD(:)

C -- Declarations for arrays involved in parallelisation.

        INTEGER PARREG,OBSREG
#ifdef PARALLEL
        INTEGER IDET(:),ISTATS(:),ISTATR(:),ISTRTME(:),
     +          IRUNTME(:),JRUN(:),ITRIAL(:),IORDER(:)
        DOUBLE PRECISION REGLAM(:)
        CHARACTER*80  ASLDIR(:)
        CHARACTER*200 SLAVEGROUP(:)
        CHARACTER*200 SINFLE(:,:),SOUFLE(:,:),SCOM(:),MSINFLE(:),
     +                MSOUFLE(:)
        CHARACTER*30  ASLAVE(:)
        CHARACTER*80  MANFLE,SREADFLE,OREADFLE,PREADFLE,SFINFLE,
     +                MREADFLE
        CHARACTER*4   ATEMP
#endif

! -- Allocatable statements.

        ALLOCATABLE::REFOBS,RHS,W1,W2,LHS,ORGVAL,TMPVAL,
     +               GRAD,TMPOBS,NEWVAL,NEWOBS,SC,PSILOL,LSTVAL,
     +               PARDEL,MRKDEL,A,LL,LCINS,NW,OBSN1,
     +               OBSN2,IIOBS,PWORD,NORM,NORMLO,
     +               PHOLD,GHOLD,EHOLD,PSISUB,INCPAR,
     +               RWORK1,RWORK2,APAR1,APAR2,
     +               NRM,NRMLO,JSTK,IRGP,SEOGP,OWGHTLO,WORKVEC1,
     +               WORKVEC2,WORKVEC3,WORKVEC4,WORK_C,ORIGWGHT,
     +               PRIORPAR,IWSENS,IDBND
        ALLOCATABLE::WORKVEC5
        ALLOCATABLE::PN,PNTEMP,PC0,PRDLOL
        ALLOCATABLE::W1KP,LTTMPOBS,LTPHI,LTPRED,LTFACC
        ALLOCATABLE::PHIMLOL,PHIRLOL
        ALLOCATABLE::ISENS,NAVAL,NAOBS,SENS
        ALLOCATABLE::CGR1,CGR2,CGV,CGW,CGY,LHSVEC
        ALLOCATABLE::OWGHTKP
        ALLOCATABLE::SCREUSE
        ALLOCATABLE::GFAC,OLDGFAC
        ALLOCATABLE::TEMPSVD,WORKVEC6
        ALLOCATABLE::WORKVEC7
        ALLOCATABLE::PARPSISUB


#ifdef PARALLEL
        ALLOCATABLE::ASLAVE,ASLDIR,IDET,ISTATS,IORDER,
     +               ISTATR,ISTRTME,IRUNTME,JRUN,SINFLE,SOUFLE,SCOM,
     +               ITRIAL,REGLAM,MSINFLE,MSOUFLE,SLAVEGROUP
#endif

C -- Common blocks.

        COMMON /FLENME/FLENME
        COMMON /MODCAL/NCALL,MCALL,JCALL
        COMMON /CASE/CASEFL
        COMMON /SSTOP/ISTOP
        common /pperr/ iperr,jperr
        COMMON /WAITINT/IWAIT

C -- Delete any old "pest.stp" files.

        CALL DELFILE1(JFAIL,'pest.stp',cline)
        IF(JFAIL.NE.0) GO TO 9891

C -- Delete any old message files.

        CALL DELFILE1(JFAIL,'pest.mmf',cline)
        IF(JFAIL.NE.0) GO TO 9891

C -- Variables are initialised.

C -- First some MPEST variables.

        MSRUN=5          ! Maximum number of lambda runs (MPEST).
                         ! Maybe inform later with number of available nodes
#ifdef MPEST
        MESSFILE=0
#endif

C -- Other variables

        CLINE=' '
        BUF=' '
        ERRMSG=' '
        ERRMSG_ADD=' '
        PTUNIT=IPTF
        DTUNIT=IPTF
        IPFAIL=0
        IREG=0
        IPRED=0
        NCALL=0
        MCALL=0
        JCALL=0
        DDCALL=0
        MMCALL=0
        NSCALL=0
        NSCALLP=0
        MMCALLP=0
        DDCALLP=0
        ICHNG=0
        IOPT=0
        IRST=0
        SRST=0
        DRST=0
        IPERR=0
        JPERR=0
        IPRLL=0
        IIRST=0
        DONTCOPY=0
        IIIRST=0
        LTLAST=0
        LITN=0
        LTBIS=0
        LTCOUNT=0
        JACMS=0
        LDOSCH=0
        REPEATRUN=1
        LSEARCH=0
        LSCH1=0
        MATSTO=0
        SENSTO=0
        ILAMMAX=100000
        JIRST=0
        IFRAC=0
        NOAUI=0
        IAUI=0
        ALOFRAC=1.0D300
        FIRSTAUI=0
        IAFIN=0
        ISOLMSG=0
        ICOVOBS=0
        IWARN1=0
        IRST1JAC=0
        ISTKFLG=0
        IPIVOTREG=-999
        NJPPL0=0
        NJPPL1=0
        NJPPL2=0
        NREV=0
        S=0.0D0
        DMX0LT=0.0D0
        DMX1LT=0.0D0
        ROLAST=0.0D0
        DMX0=0.0D0
        DMX1=0.0D0
        DMX2=0.0D0
        PSIL=0.0D0
        NROL=0
        NDMXL0=0
        NDMXL1=0
        RLO=0.0D0
        DMX0LO=0.0D0
        DMX1LO=0.0D0
        NEWPSI=0.0D0
        LAMLST=0.0D0
        DMX2LT=0.0D0
        DMX2LO=0.0D0
        NDMXL2=0
        PHIM=0.0D0
        PHIR=0.0D0
        PHIML=0.0D0
        PHIRL=0.0D0
        TEMPHIM=0.0D0
        TEMPHIR=0.0D0
        NEWPHIM=0.0D0
        NEWPHIR=0.0D0
        WFSOL=0.0D0
        LHSFLAG=0
        IOPTCALL=0
        GEOMAVSENS=0.0D0
        IF(NPHISTP.GT.1)THEN
          NPHISTP=NPHISTP-1
        ELSE
          NPHISTP=1
        END IF
        SVDUNIT=ISDF
        IF(LSQRWRITE.EQ.0)THEN
          LSQR_NOUT=0
        ELSE
          LSQR_NOUT=ILSQ
        END IF
        NOLAMBDA=0
        TEMPSI=1.0D300
        NWDIM=MAX(NPAR,NXROW)
        IF(RLAMBDA1.LE.0.0D0)THEN
          RLAMBDA1=0.0D0
          NUMLAM=1
        END IF
        PIRESTART=0
        DIRESTART=0
        IF(NSEARCH.GT.MAXSEARCH)NSEARCH=MAXSEARCH
        NNSEARCH=0
        IMCOVFLAG=0
        ICOUNTREGOBS=0
        ICOUNTREGNZ=0
        NUMEIG=0
        NEWNUMEIG=0
        NANNUMEIG=0
        NUMEIGLST=0
        ALPHA=0.0D0
        NEWALPHA=0.0D0
        NANALPH=0.0D0
        ALPHLST=0.0D0
        E_PRED=0.0D0
        EW_PRED=0.0D0
        RES_PRED=0.0D0
        E_PREDLO=0.0D0
        EW_PREDLO=0.0D0
        RES_PREDLO=0.0D0
        NEWE_PRED=0.0D0
        NEWEW_PRED=0.0D0
        NEWRES_PRED=0.0D0
        NOWRITE=0
        SWITCHFLAG=0
        SBACKFLAG=0
        LAMFLAG=0
        ISTOP=0
        IFLAG_NOCOV=0
        IPASS1=0
        IFLAG_X=0
        PR_INDEX=0
        NPR_NONZERO=0
        IF(MEMSAVE.EQ.-1)MEMSAVE=0
        IF(LSQRMODE.NE.0) MEMSAVE=1
        IF(SVDMODE.EQ.2) MEMSAVE=1
        LDU=1
        LDVT=1
        JACUPDATEFLAG=0
        IF(DOSENREUSE.EQ.'senreuse')THEN
          IF(SENMAXREUSE.EQ.0)THEN
            DOSENREUSE='nosenreuse'
          END IF
        END IF
        SLAVESTOP=0
        REIFLAG=0
        IGTARG=0
        SPLITFLAG=0
        IF(SPLITSWH.LE.0.0D0) SPLITFLAG=1
        JSPLIT=0
        JFRAC=0
        MANYLAM=0
        NUMLAMORIG=NUMLAM
        JACUPCOUNT=0
        PDFLAG=0
        REPEATFLAG=0

C -- Congugate gradient variables are initialised.

        cgnout=icgn
        cgprecon=.false.
        cgshift=0.0d0
        cgeps=1.0d-35
        NCONDUNIT=ICND

C -- Variables pertaining to parameter holding are initialised.

        NLAMBDA=0.0D0
        NRELPMX=0.0D0
        NRELPREDSTP=0.0D0
        NABSPREDSTP=0.0D0
        NINITSCHFAC=0.0D0
        NMULSCHFAC=0.0D0
        NNSEARCH=0
        NFACPMX=0.0D0
        NVECBND=0
        REGWEIGHTNUL=SQRT(1.0D7)
        REGSINGTHRESH=1.0D0/SQRT(REGSINGTHRESH)
        WFLAST=1.0D0
        FAILUREFLAG=0
        FAILURECOUNT=0
        SUCCESSCOUNT=0

        PAR_WTFAC=1.0D0
        PARETO_FIN=0
        PAR_TERM_COUNT=0

        IF(RESTART.EQ.1)THEN
          IRST=1
        ELSE IF(RESTART.EQ.3)THEN
          IRST=1
          SRST=1
        ELSE IF (RESTART.EQ.4)THEN
          IRST=1
          DRST=1
        ELSE IF(RESTART.EQ.2)THEN
          IRST=2
          IIIRST=2
        END IF
        IF(RESTART.NE.0) SVDA_SAVE=0

        NMLODIM2_FLAG=0

C -- Parallel run register unit numbers are initialized.

        PARREG=IPPP
        OBSREG=IPPO

#ifdef PARALLEL
        IPRLL=1
#endif
        IF((SRST.EQ.1).AND.(IPRLL.EQ.0))THEN
          WRITE(ERRMSG,112)
112       FORMAT('"/s" switch can only be used for restart with ',
     +    'Parallel PEST.')
          GO TO 9890
        END IF
        IF((DRST.EQ.1).AND.(IPRLL.NE.0))THEN
          WRITE(ERRMSG,113)
113       FORMAT('"/d" switch cannot be used for restart with ',
     +    'Parallel PEST.')
           GO TO 9890
        END IF
        IF(RESTART.EQ.2)THEN
          IF((IREGADJ.EQ.4).OR.(IREGADJ.EQ.5))THEN
            WRITE(ERRMSG,109)
109         FORMAT('"/j" switch cannot be used for restart when ',
     +      'IREGADJ control variable is set to 4 or 5.')
            GO TO 9890
          END IF
        END IF
        IF(SVDA.EQ.1)THEN
          IF(RESTART.EQ.2)THEN
            WRITE(ERRMSG,111)
111         FORMAT('PEST cannot restart with /j switch when ',
     +      'SVD-assist functionality is operative.')
            GO TO 9890
          END IF
        END IF

        N=LEN_TRIM(CTLFILE)
        DO I=N,1,-1
          IF(CTLFILE(I:I).EQ.'.') GO TO 15
        END DO
        GO TO 17
15      IF(I.EQ.N) THEN
          CASEFL=CTLFILE(1:N-1)
          GO TO 18
        END IF
#ifdef UNIX
        IF(CTLFILE(I+1:I+1).EQ.'/') GO TO 17
#else
        IF(CTLFILE(I+1:I+1).EQ.'\') GO TO 17
#endif
        CASEFL=CTLFILE(1:I-1)
        GO TO 18
17      CASEFL=TRIM(CTLFILE)
18      CONTINUE
        J=LEN_TRIM(CASEFL)

C -- IF PEST IS NOT RESTARTING AN OLD RUN, WE DELETE ANY JACOBIAN FILES
C    AND ANY RESTART FILES.


       JCOFILE=CASEFL
       JCOFILE(J+1:J+4)='.jco'
       IF(IRST.EQ.0)THEN
         CASEFL(J+1:J+4)='.jac'
         CALL DELFILE1(JFAIL,CASEFL(1:J+4),CLINE)
         IF(JFAIL.NE.0) GO TO 9891

         CASEFL(J+1:J+4)='.rst'
         CALL DELFILE1(JFAIL,CASEFL(1:J+4),CLINE)
         IF(JFAIL.NE.0) GO TO 9891

         CASEFL(J+1:J+4)='.jst'
         CALL DELFILE1(JFAIL,CASEFL(1:J+4),CLINE)
         IF(JFAIL.NE.0) GO TO 9891

         CASEFL(J+1:J+4)='.jco'
         IF(CASEFL.NE.JCOFIRSTFILE)THEN
           CALL DELFILE1(JFAIL,CASEFL(1:J+4),CLINE)
           IF(JFAIL.NE.0) GO TO 9891
         END IF

         IF(IJCO.NE.0)THEN
           DO I=1,100
             CALL WRITINT(AATEMP,I)
             CASEFL(J+1:)='.jco'//'.'//TRIM(AATEMP)
             IF(CASEFL.NE.JCOFIRSTFILE)THEN
               CALL DELFILE1(JFAIL,CASEFL,CLINE)
             END IF
           END DO
         END IF
         CASEFL(J+1:)=' '

         CASEFL(J+1:J+4)='.rei'
         CALL DELFILE1(JFAIL,CASEFL(1:J+4),CLINE)
         IF(JFAIL.NE.0) GO TO 9891
         IF(IREI.NE.0)THEN
           DO I=1,100
             CALL WRITINT(AATEMP,I)
             CASEFL(J+1:)='.rei'//'.'//TRIM(AATEMP)
             CALL DELFILE1(JFAIL,CASEFL,CLINE)
           END DO
         END IF
         CASEFL(J+1:)=' '

       END IF

       PTFILE=CASEFL
       PTFILE(J+1:J+4)='.prf'
       IF((IRST.EQ.0).OR.((IRST.EQ.1).AND.(IPRLL.EQ.0)))THEN
         CALL DELFILE1(JFAIL,PTFILE(1:J+4),CLINE)
         IF(JFAIL.NE.0) GO TO 9891
       END IF
       DTFILE=CASEFL
       DTFILE(J+1:J+4)='.drf'
       IF((IRST.EQ.0).OR.((IRST.EQ.1).AND.(IPRLL.NE.0)))THEN
         CALL DELFILE1(JFAIL,DTFILE(1:J+4),CLINE)
         IF(JFAIL.NE.0) GO TO 9891
       END IF

C -- If PEST is not restarting we delete any old Pareto data files.

       PAR_PHIFILE=CASEFL
       PAR_PHIFILE(J+1:J+4)='.pod'
       PAR_PARFILE=CASEFL
       PAR_PARFILE(J+1:J+4)='.ppd'
       IF(IRST.EQ.0)THEN
         CALL DELFILE1(JFAIL,PAR_PHIFILE(1:J+4),CLINE)
         IF(JFAIL.NE.0) GO TO 9891
         CALL DELFILE1(JFAIL,PAR_PARFILE(1:J+4),CLINE)
         IF(JFAIL.NE.0) GO TO 9891
       END IF

C -- We also delete any old iteration specific parameter value files.

       IF(IRST.EQ.0)THEN
         ICOUNT=0
         I=-1
         DO
           I=I+1
           CALL WRITINT(ANUM,I)
           AFILE=TRIM(CASEFL)//'.par'//'.'//TRIM(ANUM)
           INQUIRE(FILE=AFILE,EXIST=LEXIST)
           IF(LEXIST)THEN
             ICOUNT=0
             CALL DELFILE1(JFAIL,AFILE,CLINE)
           ELSE
             ICOUNT=ICOUNT+1
             IF(ICOUNT.EQ.10) GO TO 29
           END IF
         END DO
       END IF
29     CONTINUE
       IF(PESTMODE.EQ.4)THEN
         AFILE=TRIM(CASEFL)//'.par'
         CALL DELFILE1(JFAIL,AFILE,CLINE)
       END IF
       AFILE=' '

#ifdef PARALLEL
#ifdef BEO
C      Set NPAR, NOBS and CHECKSUM for sanity check on slaves
       call BEOCAST(NPAR,NOBS,0)
#endif
#ifndef MPEST
        CALL GETSECS(II,1)
#endif
        CALL SLAVDAT1(JFAIL,NSLAVE,IFLTYP,CLINE,IPRLL,LDOSCH,REPEATRUN)
        IF(LDOSCH.EQ.-9999)THEN
          IF(NUMLAM.EQ.1)THEN
            WRITE(ERRMSG,19)
19          FORMAT('If PARLAM is set to -9999 in the Parallel PEST ',
     +      'run management file, NUMLAM in the control data section ',
     +      'of the PEST control file must be set greater than one.')
            GO TO 9890
          END IF
          LDOSCH=-NUMLAM
          MSRUN=NUMLAM
          MANYLAM=1
        END IF
#ifdef MPEST
        REPEATRUN=0
        NSLAVE=1
        IF(LDOSCH.EQ.-99999999)LDOSCH=1
        IF(IWAIT.EQ.-99999999)IWAIT=10
        IFLTYP=0
#else
        IF(JFAIL.NE.0) GO TO 9891
        IF(NSLAVE.EQ.1) LDOSCH=0
#endif
        ALLOCATE(ASLAVE(NSLAVE),ASLDIR(NSLAVE),IDET(NSLAVE),
     +  ISTATS(NSLAVE),ISTRTME(NSLAVE),IRUNTME(NSLAVE),JRUN(NSLAVE),
     +  SCOM(NSLAVE),IORDER(NSLAVE),SLAVEGROUP(NSLAVE),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200
        MANFLE=CASEFL(:LEN_TRIM(CASEFL))//'.rmr'
        MREADFLE='pest.rdy'
        SREADFLE='pslave.rdy'
        PREADFLE='param.rdy'
        OREADFLE='observ.rdy'
        SFINFLE='pslave.fin'
        SCOM='Not yet known.'
        WRITE(6,*)
#ifdef MPEST
        ASLDIR(1)=' '
        ASLAVE(1)=' '
        SLAVEGROUP(1)=' '
        IRUNTME(1)=1
#else
        call pestwait(iwait*6)
        CALL SLAVDAT2(JFAIL,NSLAVE,IRUNTME,IDET,ASLAVE,ASLDIR,CLINE,
     +  AFILE,SREADFLE,OREADFLE,PREADFLE,SFINFLE,MANFLE,MREADFLE,SCOM,
     +  SLAVEGROUP)
        IF(JFAIL.NE.0) GO TO 9891
#endif
#endif


#ifdef UNIX
        ILINE=UNLINK('jacob.runs')
        FLENME='jacob.runs'
        CALL FFOPEN(JFAIL,IJRF,'w',' ',6,CLINE)
        IF(JFAIL.NE.0) GO TO 9891
#endif

        FLENME=CTLFILE

        IPRED=0
        IF(PESTMODE.EQ.1)THEN
          IPRED=0
        ELSE IF(PESTMODE.EQ.2)THEN
          IPRED=1
        ELSE IF(PESTMODE.EQ.3)THEN
          IREG=1
        END IF
        IF(PESTMODE.EQ.4)THEN
          WRITE(6,20231)
20231     FORMAT(/,' PEST is running in Pareto mode.')
          IF((NOPTMAX.EQ.0).OR.(NOPTMAX.EQ.-1).OR.(NOPTMAX.EQ.-2))THEN
            WRITE(6,20232) NOPTMAX
20232       FORMAT(' Mode changed to estimation as NOPTMAX set ',
     +      'to',I3,'.')
            PESTMODE=0
          END IF
        ELSE
          IF(IREG.EQ.1)THEN
            WRITE(6,2023)
2023        FORMAT(/,' PEST is running in regularisation mode.')
          ELSE
            IF(IPRED.EQ.0)THEN
              WRITE(6,2021)
2021          FORMAT(/,' PEST is running in parameter estimation mode.')
            ELSE
              WRITE(6,2022)
2022          FORMAT(/,' PEST is running in predictive analysis mode.')
            END IF
          END IF
        END IF

        IF(IPRED.NE.0) LAMFORGIVE=0

C -- Stopping according to an objective function threshold is permitted
C    only in parameter estimation mode.

        IF(IPRED.NE.0)THEN
          PHISTOPTHRESH=-1.1D100
          PHIABANDON=-1.1D100
        END IF
        IF(LASTRUN.LE.0)THEN
          LASTRUN=0
        ELSE IF(LASTRUN.GE.1) THEN
          LASTRUN=1
        END IF

C -- An existing RSD file is deleted if PEST is not restarting.

        RSDFILE=CASEFL
        RSDFILE(J+1:J+4)='.rsd'
        IF(IRST.EQ.0)THEN
          CALL DELFILE1(JFAIL,RSDFILE(1:J+4),CLINE)
          IF(JFAIL.NE.0) GO TO 9890
        END IF

C -- The value of the IRES variable is adjusted if necessary.

        IF(PESTMODE.EQ.2) IRES=0
        IF((IRES.NE.0).AND.(IRES.NE.1))THEN
          IF((IREG.EQ.0).AND.(SVDMODE.EQ.0).AND.(SVDA.EQ.0))THEN
            IRES=0
          ELSE
            IRES=1
          END IF
        END IF

        IF((IRST.ne.0).AND.(RSTFLE.EQ.0)) THEN
          CALL STPERR(28,0,' ',0,' ',CLINE)
          GO TO 9891
        END IF
        IF(RSTFLE.EQ.0) GO TO 2035
        FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rst'
        IF(IRST.NE.0) THEN
          CALL FFOPEN(JFAIL,-IRSF,'r',' ',22,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          READ(IRSF,ERR=9550,END=9600) IIOPT,JNESPAR,JJPRED,JJREG,JSVDA
          IF(JNESPAR.NE.NESPAR) GO TO 9550
          IF(JJPRED.NE.IPRED) THEN
            IF(JJPRED.EQ.0) THEN
              CALL STPERR(139,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
            IF(JJPRED.NE.0) THEN
              CALL STPERR(138,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
          END IF
          IF(JSVDA.NE.SVDA)THEN
            WRITE(ERRMSG,2029)
2029          FORMAT('SVD-Assist setting different in this run from ',
     +        'that used in previous run.')
            GO TO 9890
          END IF
          IF(JJREG.NE.IREG)THEN
            IF(JJREG.EQ.0) THEN
              CALL STPERR(149,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
            IF(JJREG.NE.0) THEN
              CALL STPERR(148,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
          END IF
          IF(IIOPT.EQ.0) THEN
            CLOSE(UNIT=IRSF)
            IRST=0
            SRST=0
            DRST=0
          ELSE IF(IIOPT.LT.0) THEN
            IF(IRST.EQ.1)THEN
              CALL STPERR(29,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
          END IF
        ELSE
          CALL FFOPEN(JFAIL,-IRSF,'w',' ',25,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(IRSF,ERR=9300) IOPT,NESPAR,IPRED,IREG,SVDA
          CLOSE(UNIT=IRSF)
        END IF

C -- THE RUN RECORD FILE IS OPENED.

2035    FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rec'
        IF(IRST.EQ.0) THEN
#ifdef UNIX
          J=UNLINK(FLENME(1:LEN_TRIM(FLENME)))
#endif
          CALL FFOPEN(JFAIL,IREC,'w',' ',6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          J=LEN_TRIM(CASEFL)
          WRITE(FMT,2011) MAX(27-J/2,2)
2011      FORMAT('(T',I2,',A)')
          WRITE(IREC,FMT,ERR=9350)'PEST RUN RECORD: CASE '//
     +    CASEFL(:LEN_TRIM(CASEFL))
        ELSE IF(IRST.EQ.1)THEN
          CALL FFOPEN(JFAIL,IREC,'a',' ',27,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(IREC,2040,ERR=9350)
2040      FORMAT(/,/,T31,'.....RESTART.....')
        ELSE IF(IRST.EQ.2)THEN
          CALL FFOPEN(JFAIL,IREC,'a',' ',27,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(IREC,2041,ERR=9350)
2041      FORMAT(/,/,T15,'.....RESTART: UPGRADING PARAMETERS ',
     +    'AGAIN.....')
        END IF
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif

C -- THE PARAMETER SENSITIVITY RECORD FILE IS OPENED.

        FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.sen'
        IF(IRST.EQ.0)THEN
          CALL FFOPEN(JFAIL,ISNS,'w',' ',6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          J=LEN_TRIM(CASEFL)
          WRITE(FMT,2051) MAX(27-J/2,1)
2051      FORMAT('(T',I2,',A)')
          WRITE(ISNS,FMT,ERR=9700) 'PARAMETER SENSITIVITIES: CASE '//
     +    CASEFL(:LEN_TRIM(CASEFL))
        ELSE IF(IRST.EQ.1) THEN
          CALL FFOPEN(JFAIL,ISNS,'a',' ',6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(ISNS,2040,ERR=9700)
        ELSE IF(IRST.EQ.2)THEN
          CALL FFOPEN(JFAIL,ISNS,'a',' ',6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(ISNS,2041,ERR=9700)
        END IF
#ifdef FLUSHFILE
        CALL FLUSH(ISNS)
#endif

        FLENME=CTLFILE
        IF(MAXCOMPDIM.LT.0) MAXCOMPDIM=0
        NOBSGP1=0

        IF(PRECIS.EQ.1)THEN
          APRECS='double'
        ELSE IF(PRECIS.EQ.0)THEN
          APRECS='single'
        END IF
        IF(NOPNT.EQ.0)THEN
          APOINT='point'
        ELSE IF (NOPNT.EQ.1)THEN
          APOINT='nopoint'
        END IF

#ifdef PARALLEL
        IF(NUMCOM.GT.1)THEN
          CALL STPERR(158,0,' ',0,' ',CLINE)
          GO TO 9891
        END IF
        IF(JACFILE.NE.0)THEN
          CALL STPERR(159,0,' ',0,' ',CLINE)
          GO TO 9891
        END IF
#endif

        LAMBDA=RLAMBDA1
        DMAX2=FACPARMAX
        IF(FACPARMAX.GT.0.0)FACPARMAX=LOG10(FACPARMAX)
        IF(DOAUI.EQ.'aui')THEN
          IF((IREG.EQ.1).OR.(IPRED.NE.0))THEN
            CALL STPERR(175,0,' ',0,' ',CLINE)
            GO TO 9891
          END IF
        END IF
        ICOVF=ICOV
        ICORF=ICOR
        IEIGF=IEIG
        ICOV=1
        ICOR=1
        IEIG=1

C -- IF NECESSARY, DEFAULT VALUES FOR USER-INTERVENTION VARIABLES
C    ARE CALCULATED.

        IF(DOAUI.EQ.'noaui')THEN
          MAXAUI=0
        ELSE
          IF(AUISECT.EQ.0)THEN
            MAXAUI=NESPAR*0.75
            NAUINOACCEPT=MAXAUI*0.75
            IF(NAUINOACCEPT.LT.3)NAUINOACCEPT=3
            IF(DOAUI.EQ.'auid')then
              AUISENSRAT=1.0
              AUINUMFREE=1
              AUIPHIRATACCEPT=0.99
            END IF
          END IF
        END IF

C -- ARRAY STORAGE IS ALLOCATED

        ALLOCATE(ORGVAL(NPAR),TMPVAL(NPAR),
     +  NEWVAL(NPAR),PSILOL(NPHISTP),
     +  LSTVAL(NPAR),PWORD(NPAR),NW(NPAR),JSTK(NPAR),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

        IF(MEMSAVE.EQ.0)THEN
          ALLOCATE(APAR1(NPAR),APAR2(NPAR),STAT=IERR)
        ELSE
          ALLOCATE(APAR1(1),APAR2(1),STAT=IERR)
        END IF
        IF(IERR.NE.0) GO TO 9200

        W1DIM=MAX(NXROW,NPAR)
        ALLOCATE(REFOBS(NXROW),W1(W1DIM),
     +  TMPOBS(NXROW),NEWOBS(NXROW),
     +  OBSN1(NOBS),OBSN2(NOBS),IIOBS(NOBS),PSISUB(NOBSGP+1),
     +  IRGP(NOBSGP),SEOGP(NOBSGP),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200
        ALLOCATE(GFAC(NOBSGP),OLDGFAC(NOBSGP),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200
        GFAC=1.0D0
        OLDGFAC=1.0D0
        ALLOCATE(GHOLD(NPARGP),IDBND(NPARGP),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200
        IF((LSQRMODE.GT.0).OR.(SVDMODE.GT.0).OR.(NOPTMAX.EQ.0)
     +    .OR.(NOPTMAX.EQ.-2))THEN
          HOLDFLAG=0
          ALLOCATE(ISENS(1),NAVAL(1),SENS(1),
     +    PHOLD(1),EHOLD(1),NAOBS(1),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
        ELSE
          HOLDFLAG=1
          ALLOCATE(ISENS(NPAR),NAVAL(NPAR),SENS(NPAR),
     +    PHOLD(NPAR),EHOLD(NESPAR),NAOBS(NXROW),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
        END IF

        IF(LSQRMODE.EQ.1)THEN
          RHSDIM=1
        ELSE
          RHSDIM=NESPAR
        END IF
        ALLOCATE(RHS(RHSDIM),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

        ALLOCATE(GRAD(NESPAR),SC(NESPAR),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

        ALLOCATE(PARDEL(NTPLFLE),MRKDEL(NINSFLE),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

        IF(IPRED.GT.0)THEN
          ALLOCATE (PN(NPAR),PNTEMP(NPAR),PC0(NXROW),STAT=IERR)
        ELSE
          ALLOCATE (PN(1),PNTEMP(1),PC0(1),STAT=IERR)
        END IF
        IF(IERR.NE.0) GO TO 9200

        ALLOCATE (PHIMLOL(NPHISTP),PHIRLOL(NPHISTP),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

        WORK_CDIM=1
        IF(MAXCOMPDIM.GT.1) WORK_CDIM=NXROW
        ALLOCATE(WORK_C(WORK_CDIM),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

        IF(CONJGRAD.EQ.1)THEN
          NPCGDIM=NESPAR
        ELSE
          NPCGDIM=1
        END IF
        ALLOCATE(CGR1(NPCGDIM),CGR2(NPCGDIM),CGV(NPCGDIM),
     +  CGW(NPCGDIM),CGY(NPCGDIM),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

        IF(DOSENREUSE.EQ.'senreuse')THEN
          ALLOCATE(SCREUSE(NPAR),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
        END IF

C -- IF PRIOR INFORMATION IS PRESENT AND WE ARE USING LSQR, PRIOR INFORMATION INDEXING
C    IS IMPLEMENTED.

        IF(LSQRMODE.EQ.1)THEN
          IF(NPRIOR.NE.0)THEN
            PR_INDEX=1
            NPR_NONZERO=0
            IF(MAXCOMPDIM.LE.1)THEN
              DO IES=1,NESPAR
                DO IROW=NOBS+1,NXROW
                  IF(X(IROW,IES).GT.0.0D0) NPR_NONZERO=NPR_NONZERO+1
                END DO
              END DO
            ELSE
              DO I=1,NCOMPDIM
                INUM=IXC(I)
                IES=(INUM-1)/NXROW+1
                IROW=INUM-(IES-1)*NXROW
                IF(IROW.GT.NOBS) NPR_NONZERO=NPR_NONZERO+1
              END DO
            END IF
            ALLOCATE(IXPR(NPR_NONZERO),IXPC(NPR_NONZERO),
     +      IESTRANS(NESPAR),STAT=IERR)
            IF(IERR.NE.0) GO TO 9200
            I=0
            IF(MAXCOMPDIM.LE.1)THEN
              DO IES=1,NESPAR
                DO IROW=NOBS+1,NXROW
                  IF(X(IROW,IES).GT.0.0) THEN
                    I=I+1
                    IXPC(I)=IES
                    IXPR(I)=IROW
                  END IF
                END DO
              END DO
            ELSE
              DO J=1,NCOMPDIM
                INUM=IXC(J)
                IES=(INUM-1)/NXROW+1
                IROW=INUM-(IES-1)*NXROW
                IF(IROW.GT.NOBS)THEN
                  I=I+1
                  IXPC(I)=IES
                  IXPR(I)=IROW
                END IF
              END DO
            END IF
            DO IES=1,NESPAR
              IESTRANS(IES)=IES
            END DO
          ELSE
            ALLOCATE(IXPR(1),IXPC(1),IESTRANS(1),STAT=IERR)
            IF(IERR.NE.0) GO TO 9200
          END IF
        ELSE
          ALLOCATE(IXPR(1),IXPC(1),IESTRANS(1),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
        END IF

C -- ARRAY INITIALISATION

        LSTVAL=0
        DO 10 IPP=1,NPAR
10      ORGVAL(IPP)=PVAL(IPP)
        LSTVAL(1)=1.0D300

        ISWTCH=0
        I2OR3=0
        JA2=0
        DO 20 J=1,NPARGP
        IF((FORCEN(J).EQ.1).OR.(FORCEN(J).EQ.4)) THEN
          DO 25 IPP=1,NPAR
          IF(IPARGP(IPP).NE.J) GO TO 25
          IF(ITRANS(IPP).LT.0) GO TO 25
          IF(APAR(IPP)(1:3).EQ.'iw_')GO TO 25
          ISWTCH=1
          I2OR3=2
          GO TO 21
25        CONTINUE
        ELSE IF(FORCEN(J).EQ.2)THEN
          JA2=1
        END IF
20      CONTINUE
21      CONTINUE
        IF(ISWTCH.EQ.1)THEN
          IF(JACFILE.NE.0)THEN
            DO 5021 I=1,NPAR
              IF(DERCOM(I).NE.0) THEN
                IF(ITRANS(I).GE.0) GO TO 5023
              END IF
5021        CONTINUE
            ISWTCH=0
5023        CONTINUE
          END IF
        END IF
        IF(PESTMODE.EQ.4) ISWTCH=0

! -- Array initialization for a derivatives-calculation variable.

        DO I=1,NPARGP
          IF(DERINCLB(I).LE.0.0)THEN
            IDBND(I)=0
          ELSE
            IDBND(I)=1
          END IF
          IF(INCTYP(I).EQ.2) THEN
            IDBND(I)=0
            DERINCLB(I)=0.0D0
          END IF
        END DO

C -- WE NOW CHECK WHETHER ANY PARAMETERS ARE REGULARISATION WEIGHT ADJUSTMENT
C    FACTOR PARAMETERS.

        NREGADJPAR=0
        DO IPP=1,NPAR
          IF(APAR(IPP)(1:3).EQ.'iw_')THEN
            IF(IREG.EQ.0)THEN
              WRITE(ERRMSG,114)
114           FORMAT('At least one parameter defined in PEST control ',
     +        'file is an inverse weight ("iw_") parameter. ',
     +        'This is not allowed unless PEST runs in regularisation ',
     +        'mode.')
              GO TO 9890
            END IF
            ATEMPG=APAR(IPP)(4:)
            DO I=1,NOBSGP
              IF(OBGNME(I).EQ.ATEMPG)THEN
                NREGADJPAR=NREGADJPAR+1
                OFFSET(IPP)=I
                SCALE(IPP)=-1.0D37
                GO TO 120
              END IF
            END DO
            WRITE(ERRMSG,115)
115         FORMAT('At least one parameter defined in PEST control ',
     +      'file is an inverse weight parameter. However its ',
     +      'corresponding observation group has not been defined.')
            GO TO 9890
120         CONTINUE
          END IF
        END DO

C -- If running in Pareto mode and SVD-assist is being undertaken, multiple BPA
C    files are requested.

        IF(PESTMODE.EQ.4)THEN
          IF(SVDA.EQ.1)THEN
            SVDA_MULBPA=1
          END IF
        END IF

C -- SVD-assist data is read.

        IF(SVDA.EQ.1)THEN
          CALL INITIALISE_BASE(IFAIL,SVDA_SCALADJ)
          IF(IFAIL.NE.0) GO TO 9890
          IF(IRST.EQ.0)THEN
          IF(SVDA_MULBPA.EQ.0)THEN
            CALL BASE_PARDEF(IFAIL,IREC,0,NREGADJPAR,-1,SVDA_SCALADJ)
          ELSE
            CALL BASE_PARDEF(IFAIL,IREC,0,NREGADJPAR,0,SVDA_SCALADJ)
          END IF
          ELSE
          CALL BASE_PARDEF(IFAIL,IREC,2,NREGADJPAR,0,SVDA_SCALADJ)
          END IF
          IF(IFAIL.NE.0) GO TO 9890
        ELSE
          BNPARGP=1
          ALLOCATE(BPARGPNME(1),SENFACTOR(1),STAT=IERR)
        END IF

C -- Pareto data files are opened.

        IF(PESTMODE.EQ.4)THEN
          ALLOCATE(PARPSISUB(NOBSGP+1),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
          FLENME=PAR_PHIFILE
          IF(IRST.EQ.0) THEN
            CALL FFOPEN(JFAIL,IPOD,'w',' ',6,CLINE)
            IF(JFAIL.NE.0) THEN
              WRITE(ERRMSG,2061) TRIM(FLENME)
2061          FORMAT('Cannot open file ',A,' to record Pareto ',
     +        'objective function data.')
              GO TO 9890
            END IF
            DO I=1,NOBSGP
              ATEMP12=ADJUSTL(OBGNME(I))
              WRITE(IPOD,2045,ADVANCE='NO') TRIM(ATEMP12)
2045          FORMAT(1X,A12,1X)
            END DO
            IF(NOBS_REPORT.GT.0)THEN
              DO I=1,NOBS_REPORT
                ATEMP20=ADJUSTL(AOBS(IOBS_REPORT(I)))
                WRITE(IPOD,2046,ADVANCE='NO') TRIM(ATEMP20)
2046            FORMAT(1X,A20,1X)
              END DO
            END IF
            WRITE(IPOD,'(A)') ' '
#ifdef FLUSHFILE
            CALL FLUSH(IPOD)
#endif
          ELSE
            INQUIRE(FILE=FLENME,EXIST=LEXIST)
            IF(.NOT.LEXIST)THEN
              WRITE(ERRMSG,2042) TRIM(FLENME)
              GO TO 9890
            END IF
            CALL FFOPEN(JFAIL,IPOD,'a',' ',6,CLINE)
            IF(JFAIL.NE.0) THEN
              WRITE(ERRMSG,2042) TRIM(FLENME)
2042          FORMAT('Cannot open file ',A,' to continue recording ',
     +        'of Pareto objective function data.')
              GO TO 9890
            END IF
          END IF
          FLENME=PAR_PARFILE
          PAR_NUM_PREVIOUS=0
          IF(IRST.NE.0)THEN
            CALL FFOPEN(JFAIL,-IPPD,'r',' ',6,CLINE)
            IF(IFAIL.NE.0) GO TO 20442
            READ(IPPD,ERR=20442,END=20442) ITEMP
            READ(IPPD,ERR=20442,END=20442) (RTEMP,I=1,ITEMP)
            READ(IPPD,ERR=20442,END=20442) (RTEMP,I=1,ITEMP)
            READ(IPPD,ERR=20442,END=20442) (ATEMP12,I=1,ITEMP)
            READ(IPPD,ERR=20442,END=20442) ITEMP
            READ(IPPD,ERR=20442,END=20442) (ATEMP12,I=1,ITEMP)
            READ(IPPD,ERR=20442,END=20442) ITEMP
            IF(ITEMP.NE.0)THEN
              READ(IPPD,ERR=20442,END=20442) (ATEMP20,I=1,ITEMP)
            END IF
            DO II=10,100
              INQUIRE(UNIT=II,OPENED=LOPENED)
              IF(.NOT.LOPENED)THEN
                INQUIRE(FILE='t###.###',EXIST=LEXIST)
                IF(LEXIST)THEN
                  CALL DELFILE1(JFAIL,'t###.###',cline)
                END IF
                OPEN(UNIT=II,FILE='t###.###',FORM='UNFORMATTED',
     +          IOSTAT=IERR)
                IF(IERR.NE.0) GO TO 20442
                DO
                  READ(IPPD,ERR=20444,END=20444)
     +            (PARPSISUB(I),I=1,NOBSGP)
                  IF(NOBS_REPORT.NE.0)THEN
                    READ(IPPD,ERR=20444,END=20444)
     +              (REFOBS(IOBS_REPORT(I)),I=1,NOBS_REPORT)
                  END IF
                  IF(SVDA.EQ.0)THEN
                    READ(IPPD,ERR=20444,END=20444) (PVAL_O(I),I=1,NPAR)
                  ELSE
                    READ(IPPD,ERR=20444,END=20444) (BPVAL(I),I=1,BNPAR)
                  END IF
                  WRITE(II,ERR=20442)
     +            (PARPSISUB(I),I=1,NOBSGP)
                  IF(NOBS_REPORT.NE.0)THEN
                    WRITE(II,ERR=20442)
     +              (REFOBS(IOBS_REPORT(I)),I=1,NOBS_REPORT)
                  END IF
                  IF(SVDA.EQ.0)THEN
                    WRITE(II,ERR=20442) (PVAL_O(I),I=1,NPAR)
                  ELSE
                    WRITE(II,ERR=20442) (BPVAL(I),I=1,BNPAR)
                  END IF
                  PAR_NUM_PREVIOUS=PAR_NUM_PREVIOUS+1
                END DO
20444           CONTINUE
                INQUIRE(UNIT=II,OPENED=LOPENED)
                IF(LOPENED)THEN
                  IF(PAR_NUM_PREVIOUS.EQ.0)THEN
                    CLOSE(UNIT=II,STATUS='DELETE')
                  ELSE
                    CLOSE(UNIT=II)
                  END IF
                END IF
                CLOSE(UNIT=IPPD)
                GO TO 20448
              END IF
            END DO
20442       WRITE(ERRMSG,20445)
            GO TO 9890
          END IF
20448     CONTINUE
          CALL FFOPEN(JFAIL,-IPPD,'w',' ',6,CLINE)
          IF(JFAIL.NE.0) THEN
            WRITE(ERRMSG,2043) TRIM(FLENME)
2043        FORMAT('Cannot open binary file ',A,' to record Pareto ',
     +      'parameter data.')
            GO TO 9890
          END IF
          IF(SVDA.EQ.0)THEN
            WRITE(IPPD) NPAR
          ELSE
            WRITE(IPPD) BNPAR
          END IF
          IF(SVDA.EQ.0)THEN
            WRITE(IPPD) (SCALE(I),I=1,NPAR)
            WRITE(IPPD) (OFFSET(I),I=1,NPAR)
            WRITE(IPPD) (APAR(I),I=1,NPAR)
          ELSE
            WRITE(IPPD) (BSCALE(I),I=1,BNPAR)
            WRITE(IPPD) (BOFFSET(I),I=1,BNPAR)
            WRITE(IPPD) (BAPAR(I),I=1,BNPAR)
          END IF
          WRITE(IPPD) NOBSGP
          WRITE(IPPD) (OBGNME(I),I=1,NOBSGP)
          WRITE(IPPD) NOBS_REPORT
          IF(NOBS_REPORT.NE.0)THEN
            WRITE(IPPD) (AOBS(IOBS_REPORT(I)),I=1,NOBS_REPORT)
          END IF
          IF(PAR_NUM_PREVIOUS.NE.0)THEN
            OPEN(UNIT=II,FILE='t###.###',STATUS='OLD',
     +      FORM='UNFORMATTED',ERR=20446)
            DO I=1,PAR_NUM_PREVIOUS
              READ(II,ERR=20446,END=20446)
     +        (PARPSISUB(J),J=1,NOBSGP)
              IF(NOBS_REPORT.NE.0)THEN
                READ(II,ERR=20446,END=20446)
     +          (REFOBS(IOBS_REPORT(J)),J=1,NOBS_REPORT)
              END IF
              IF(SVDA.EQ.0)THEN
                READ(II,ERR=20446,END=20446) (PVAL_O(J),J=1,NPAR)
              ELSE
                READ(II,ERR=20446,END=20446) (BPVAL(J),J=1,BNPAR)
              END IF
              WRITE(IPPD)(PARPSISUB(J),J=1,NOBSGP)
              IF(NOBS_REPORT.NE.0)THEN
                WRITE(IPPD)(REFOBS(IOBS_REPORT(J)),J=1,NOBS_REPORT)
              END IF
              IF(SVDA.EQ.0)THEN
                WRITE(IPPD) (PVAL_O(J),J=1,NPAR)
              ELSE
                WRITE(IPPD) (BPVAL(J),J=1,BNPAR)
              END IF
            END DO
            CLOSE(UNIT=11,STATUS='DELETE')
            GO TO 20447
20446       CONTINUE
            WRITE(ERRMSG,20445)
20445       FORMAT('Cannot retrieve Pareto parameter values ',
     +      'from previous PEST run.')
            GO TO 9890
20447       CONTINUE
          END IF
#ifdef FLUSHFILE
          CALL FLUSH(IPPD)
#endif
        END IF

C -- IF PEST IS RUNNING IN PREDICTIVE ANALYSIS MODE, THE PRESENCE OF THE "PREDICT"
C    OBSERVATION GROUP IS VERIFIED.

        IF(NOBSGP1.EQ.-999)THEN
          OBGNME(1)='no_name'
          NUMCOV=0
        END IF
        JPREDICT=0
        KPRED=0
        IF(IPRED.GT.0)THEN
          DO 36 I=1,NOBSGP
            IF(OBGNME(I)(1:12).EQ.'predict     ')THEN
              JPREDICT=I
              IF(COVIND(I)) THEN
                CALL STPERR(171,0,' ',0,' ',CLINE)
                GO TO 9891
              END IF
              GO TO 24
            END IF
36        CONTINUE
24        IF(JPREDICT.EQ.0)THEN
            CALL STPERR(135,1,' ',0,' ',CLINE)
            GO TO 9891
          END IF
        ELSE
          DO 37 I=1,NOBSGP
            IF(OBGNME(I)(1:12).EQ.'predict     ')THEN
              KPRED=I
            END IF
37        CONTINUE
        END IF

        DO I=1,NOBSGP
          IRGP(I)=0
          SEOGP(I)=-1.0D0
        END DO
        NREGOBS=0
        IF(IREG.EQ.1)THEN
          DO I=1,NOBSGP
            IF(OBGNME(I)(1:5).EQ.'regul')THEN
              IRGP(I)=1
              NREGOBS=NREGOBS+1
            END IF
          END DO
          IF(NREGOBS.EQ.0)THEN
            CALL STPERR(145,1,' ',0,' ',CLINE)
            GO TO 9891
          END IF
        END IF
5020    CONTINUE

C -- MEMORY IS NOW ALLOCATED FOR ARRAYS USED BY PARAMETERS WHICH ADJUST WEIGHTS.

        IF(NREGADJPAR.GT.0)THEN
          ITEMP1=NREGADJPAR
        ELSE
          ITEMP1=1
        END IF
        ALLOCATE(PRIORPAR(ITEMP1),IWSENS(ITEMP1),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200
        PRIORPAR=0

C -- THE OBSERVATION DATA SECTION OF THE PEST CONTROL FILE IS PROCESSED

        IF(IPRED.GT.0)THEN
          ITEMP=0
          DO 41 I=1,NOBS
            IF(NOBGNM(I).EQ.JPREDICT)THEN
              PWTKP=SQRT(OWGHT(I))
              OWGHT(I)=0.0D0
              ITEMP=ITEMP+1
              ITEMP1=I
            END IF
41        CONTINUE
          IF(ITEMP.EQ.0) THEN
            CALL STPERR(136,1,' ',0,' ',CLINE)
            GO TO 9891
          ELSE IF(ITEMP.GT.1)THEN
            CALL STPERR(137,1,' ',0,' ',CLINE)
            GO TO 9891
          END IF
          IF(PREDNOISE.NE.0)THEN
            IF(PWTKP.LE.0.0D0)THEN
              WRITE(ERRMSG,42)
42            FORMAT('If PREDNOISE is not set to zero, the weight ',
     +        'assigned to the prediction in the "observation data" ',
     +        'section of the PEST control file must be positive.')
              GO TO 9890
            END IF
          END IF

C -- OBSERVATIONS ARE RE-ARRANGED WITH THE PREDICTION PLACED AT END

          IF(ITEMP1.NE.NOBS)THEN
            IF(JACFILE.EQ.1)THEN
              CALL STPERR(163,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
            POVAL=OVAL(ITEMP1)
            POWGHT=OWGHT(ITEMP1)
            PAOBS=AOBS(ITEMP1)
            PNOBGNM=NOBGNM(ITEMP1)
            DO 26 I=ITEMP1,NOBS-1
              OVAL(I)=OVAL(I+1)
              OWGHT(I)=OWGHT(I+1)
              AOBS(I)=AOBS(I+1)
              NOBGNM(I)=NOBGNM(I+1)
26          CONTINUE
            OVAL(NOBS)=POVAL
            OWGHT(NOBS)=POWGHT
            AOBS(NOBS)=PAOBS
            NOBGNM(NOBS)=PNOBGNM
            IF(JACFILE.EQ.2)THEN
              DO I=1,DI_NDEP
                II=DI_OBS(I)
                IF(II.GT.ITEMP1)THEN
                  DI_OBS(I)=II-1
                ELSE IF(II.EQ.ITEMP1)THEN
                  DI_OBS(I)=NOBS
                END IF
              END DO
            END IF
          END IF
        ELSE
          IF(KPRED.NE.0)THEN
            ICOUNT=0
            DO 39 I=1,NOBS
              IF(NOBGNM(I).EQ.KPRED)THEN
                IF(ICOUNT.EQ.0)THEN
                  ICOUNT=I
                ELSE
                  ICOUNT=-1
                END IF
              END IF
39          CONTINUE
            IF(ICOUNT.GT.0) THEN
              KPRED=ICOUNT
            ELSE
              KPRED=0
            END IF
          END IF
        END IF

        MODFLE=COMLIN(1)

C -- TEMPLATE AND INSTRUCTION FILES ARE PROCESSED.

        CALL IOCTL(JFAIL,NPAR,NTPLFLE,NINSFLE,NUML,ASIZE,NINSTR,NBLBMX,
     +  NW,TEMPFLE,INFLE,INSFLE,OUTFLE,PARDEL,MRKDEL,APAR,CLINE,ILINE)
        IF(JFAIL.NE.0) GO TO 9891
        IF(NUML.EQ.0)NUML=1

        ALLOCATE(A(ASIZE),LL(NUML),LCINS(NINSTR),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

C -- SOME PREDICTIVE ANALYSIS DATA IS PROCESSED.

        IF(IPRED.NE.0)THEN
          IF(NPREDMAXMIN.EQ.1)THEN
            PPOSNEG='n'
            PREDHL='h'
          ELSE IF(NPREDMAXMIN.EQ.-1)THEN
            PPOSNEG='p'
            PREDHL='l'
          ELSE
            GO TO 9800
          END IF
          NPREDSTP=NPREDSTP-1
          ALLOCATE (PRDLOL(NPREDSTP),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
          ALLOCATE(W1KP(NPAR),LTTMPOBS(NOBS,MAXSEARCH),
     +    LTPHI(0:MAXSEARCH),LTPRED(0:MAXSEARCH),LTFACC(0:MAXSEARCH),
     +    STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
        ELSE
          ALLOCATE (PRDLOL(1),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
          ALLOCATE(W1KP(NPAR),LTTMPOBS(1,1),
     +    LTPHI(0:1),LTPRED(0:1),LTFACC(0:1),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
        END IF

C -- SOME REGULARISATION DATA IS PROCESSED.

       IF(IREG.EQ.1)THEN
          REGITN=1000
          WF=WFINIT
          IF(WF.LT.WFMIN)WF=WFMIN
          IF(WF.GT.WFMAX)WF=WFMAX
          PD1RFAC=PHIMACCEPT/PHIMLIM
          PHIMLIMKP=PHIMLIM
        END IF

C -- IF SENSITIVITY RE-USE IS OPERATIVE SOME PROCESSING IS CARRIED OUT.

        IF(DOSENREUSE.EQ.'senreuse')THEN
          IF(SENMAXREUSE.LT.0) SENMAXREUSE=NESPAR/2
          IF(SENMAXREUSE.GT.NESPAR-1)THEN
            WRITE(ERRMSG,50171)
50171       FORMAT('SENMAXREUSE must not be set higher than ',
     +      'the number of adjustable parameters minus 1 in ',
     +      '"sensitivity reuse" section of PEST control file.')
            GO TO 9890
          END IF
          IF(IPRED.EQ.1)THEN
            IF(SENPREDWEIGHT.LT.0.0D0)THEN
              RTEMP=0.0D0
              DO IROW=1,NOBS-1
                IF(OWGHT(IROW).GT.RTEMP)RTEMP=OWGHT(IROW)
              END DO
              IF(NXROW.GT.NOBS)THEN
                IF(SENPIEXCLUDE.EQ.'no')THEN
                  DO IROW=NOBS+1,NXROW
                    IF(OWGHT(IROW).GT.RTEMP) RTEMP=OWGHT(IROW)
                  END DO
                END IF
              END IF
              SENPREDWEIGHT=RTEMP
            ELSE
              SENPREDWEIGHT=SENPREDWEIGHT*SENPREDWEIGHT
            END IF
          END IF
          IF((IRST.NE.0).AND.(NOPTMAX.NE.0))THEN
            IF(IIOPT.GT.1)THEN
              FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.jac'
              CALL FFOPEN(JFAIL,-IRSJ,'r',' ',131,CLINE)
              IF(JFAIL.NE.0) GO TO 9891
              READ(IRSJ,ERR=9750,END=9750) ITEMP,JJPRED,JJREG
              IF(JJPRED.NE.IPRED) THEN
                IF(JJPRED.EQ.0) THEN
                  CALL STPERR(139,0,' ',0,' ',CLINE)
                  GO TO 9891
                END IF
                IF(JJPRED.NE.0) THEN
                  CALL STPERR(138,0,' ',0,' ',CLINE)
                  GO TO 9891
                END IF
              END IF
              IF(JJREG.NE.IREG)THEN
                IF(JJREG.EQ.0) THEN
                  CALL STPERR(149,0,' ',0,' ',CLINE)
                  GO TO 9891
                END IF
                IF(JJREG.NE.0) THEN
                  CALL STPERR(148,0,' ',0,' ',CLINE)
                  GO TO 9891
                END IF
              END IF
              IF(MAXCOMPDIM.LE.1)THEN
                X=0.0D0                     ! AN ARRAY
                READ(IRSJ,ERR=9750,END=9750) ICOUNT
                IF(ICOUNT.NE.0)THEN
                  DO I=1,ICOUNT
                    READ(IRSJ,ERR=9750,END=9750) IROW,IES,X(IROW,IES)
                  END DO
                END IF
              ELSE
                READ(IRSJ,ERR=9750,END=9750)NCOMPDIM
                IF(NCOMPDIM.GE.MAXCOMPDIM-3) GO TO 9970
                IF(NCOMPDIM.NE.0)THEN
                  DO I=1,NCOMPDIM
                    READ(IRSJ,ERR=9750,END=9750) IXC(I),XC(I)
                  END DO
                END IF
              END IF
              CLOSE(UNIT=IRSJ)
            END IF
          END IF
        END IF

C -- WE NOW FIND OUT HOW MANY ELEMENTS OF THE ORIGWGHT ARRAY ARE REQUIRED AND
C    ALLOCATE MEMORY FOR THAT.

        IF(NREGADJPAR.GT.0)THEN
          NORIGWGHT=0
          DO IPP=1,NPAR
            IF(SCALE(IPP).LT.-1.0D35)THEN
              ITEMP=NINT(OFFSET(IPP))
              DO IOBS=1,NXROW
                IF(NOBGNM(IOBS).EQ.ITEMP)THEN
                  NORIGWGHT=NORIGWGHT+1
                END IF
              END DO
            END IF
          END DO
        ELSE
          NORIGWGHT=1
        END IF
        ALLOCATE(ORIGWGHT(NORIGWGHT),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

C -- WE NOW IDENTIFY WHICH OBSERVATIONS/PRIOR INFORMATION EQUATIONS ARE USED
C    FOR WEIGHTS AGJUSTMENT PARAMETERS AND STORE THE RESULT IN A VECTOR.
C    NOTE THAT WE ARE ASSUMING THAT EACH SUCH ITEM OF PRIOR INFORMATION CITES
C    ONLY ONE PARAMETER AND THAT EACH SUCH PARAMETER OCCURS IN ONLY ONE ITEM OF
C    PRIOR INFORMATION.

        IF(NREGADJPAR.GT.0)THEN
          ISTART_C=1
          IF(NPRIOR.GT.0)THEN
            ICOUNT=0
            IES=0
            DO IPP=1,NPAR
              IF(ITRANS(IPP).GE.0)IES=IES+1
              IF(SCALE(IPP).LT.-1.0D35)THEN
                ICOUNT=ICOUNT+1
                IF(MAXCOMPDIM.LE.1)THEN
                  DO IOBS=NOBS+1,NXROW
                    IF(X(IOBS,IES).NE.0.0D0)THEN
                      PRIORPAR(ICOUNT)=IOBS
                      GO TO 5017
                    END IF
                  END DO
                ELSE
                  CALL GET_VECTOR(NCOMPDIM,NXROW,XC,IXC,WORK_C,IES)
                  ISTART_C=IFOUND_C
                  DO IOBS=NOBS+1,NXROW
                    IF(WORK_C(IOBS).NE.0.0D0)THEN
                      PRIORPAR(ICOUNT)=IOBS
                      GO TO 5017
                    END IF
                  END DO
                END IF
5017            CONTINUE
              END IF
            END DO
          END IF
        END IF

C -- THE LOCATION IN THE JACOBIAN MATRIX OF THE LAST ROW REGULARISATION
C    INFORMATION IS NOW DETERMINED.

        IF(IREG.EQ.0)THEN
          LASTOBSROW=0
          FIRSTREGROW=0
        ELSE
          DO IROW=NXROW,1,-1
            IF(IRGP(NOBGNM(IROW)).EQ.0)THEN
              LASTOBSROW=IROW
              GO TO 5102
            END IF
          END DO
5102      CONTINUE
          DO IROW=1,NXROW
            IF(IRGP(NOBGNM(IROW)).NE.0)THEN
              FIRSTREGROW=IROW
              GO TO 5103
            END IF
          END DO
5103      CONTINUE
        END IF

C -- MEMORY IS NOW ALLOCATED FOR A WEIGHTS MEMORY ARRAY THAT KEEPS A RECORD
C    OF WEIGHTS USED WHEN BEST PARAMETERS WERE FOUND (APPLICABLE ONLY IF
C    WORKING IN REGULARISATION MODE WHERE INTER-GROUP WEIGHTS ADJUSTMENT
C    IS POSSIBLE).

        IF((IREG.NE.0).AND.(IREGADJ.NE.0))THEN
          ALLOCATE(OWGHTLO(NXROW),STAT=IERR)
        ELSE
          ALLOCATE(OWGHTLO(1),STAT=IERR)
        END IF
        IF(IERR.NE.0) GO TO 9200
        OWGHTLO=0.0D0       ! Initialization
        OWGHTLO(1)=-1.1E35

        NOBSREGUL=0
        IF(IREG.NE.0)THEN
          IF((IREGADJ.EQ.3).OR.(IREGADJ.EQ.4).OR.
     +       (IREGADJ.EQ.5))THEN
            DO IROW=1,NXROW
              J=NOBGNM(IROW)
              IF(IRGP(J).NE.0) NOBSREGUL=NOBSREGUL+1
            END DO
            ALLOCATE(OWGHTKP(NOBSREGUL),STAT=IERR)
            IF(IERR.NE.0) GO TO 9200
          ELSE
            ALLOCATE(OWGHTKP(1),STAT=IERR)
            IF(IERR.NE.0) GO TO 9200
          END IF
        ELSE IF(PESTMODE.EQ.4)THEN
          NUM_WEIGHT_ADJUST=0
          IWTFLAG=0
          DO I=1,NXROW
            IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
              NUM_WEIGHT_ADJUST=NUM_WEIGHT_ADJUST+1
              IF(OWGHT(I).NE.0.0D0) IWTFLAG=1
            END IF
          END DO
          IF(NUM_WEIGHT_ADJUST.EQ.0)THEN
            WRITE(ERRMSG,5016)
5016        FORMAT('Observation group for Pareto weights adjustment ',
     +      'has no members.')
            GO TO 9890
          END IF
          IF(IWTFLAG.EQ.0)THEN
            IF(.NOT.COVIND(PAR_NOBGNM)) THEN
              WRITE(ERRMSG,50161)
50161         FORMAT('Observation group for Pareto weights adjustment ',
     +        'has no members which are have a non-zero weight.')
              GO TO 9890
            END IF
          END IF
          ALLOCATE(OWGHTKP(NUM_WEIGHT_ADJUST),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
        ELSE
          ALLOCATE(OWGHTKP(1),STAT=IERR)
          IF(IERR.NE.0) GO TO 9200
        END IF

C -- MEMORY IS ALLOCATED FOR NESPAR*NESPAR ARRAYS DEPENDING ON MEMSAVE
C    AND CONJGRAD.

        IF(CONJGRAD.EQ.1)THEN
          NLHSDIM1=1
          NLHSDIM2=1
          LHSIND=0
          DO 5101 I=1,NESPAR
            LHSVDIM=LHSVDIM+I
5101      CONTINUE
        ELSE IF(LSQRMODE.EQ.1)THEN
          NLHSDIM1=1
          NLHSDIM2=1
          LHSIND=0
          LHSVDIM=1
        ELSE IF(SVDMODE.EQ.2)THEN
          NLHSDIM1=NXROW
          NLHSDIM2=NESPAR
          LHSIND=0
          LHSVDIM=1
        ELSE
          NLHSDIM1=NESPAR
          NLHSDIM2=NESPAR
          LHSVDIM=1
        END IF
        IF(MEMSAVE.EQ.0)THEN
          IF((IREG.NE.0).OR.(SVDMODE.NE.0).OR.(LSQRMODE.NE.0))THEN
              IFLAG_NOCOV=1
              NMLODIM2=1
            IF((IREG.EQ.1).AND.((SVDMODE.EQ.0).OR.(SVDMODE.EQ.1)))THEN
              NMLODIM1=NESPAR
            ELSE
              NMLODIM1=1
            END IF
          ELSE
            NMLODIM1=NESPAR
            NMLODIM2=NESPAR
          END IF
        ELSE
          NMLODIM1=1
          NMLODIM2=1
        END IF
        IF(NOPTMAX.EQ.0)THEN
          NMLODIM1=1
          NMLODIM2=1
          NLHSDIM2=1
          LHSVDIM=1
        END IF
        IF(NOPTMAX.EQ.-2)THEN
          NMLODIM1=1
          NLHSDIM1=1
          NMLODIM2_FLAG=NMLODIM2
          NMLODIM2=1
        END IF
        ALLOCATE(NORM(NMLODIM1,NMLODIM1),NORMLO(NMLODIM2,NMLODIM2),
     +  LHS(NLHSDIM1,NLHSDIM2),LHSVEC(LHSVDIM),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200
        IF(NOPTMAX.EQ.-2)THEN
          NMLODIM2=NMLODIM2_FLAG
        ELSE
          NMLODIM2_FLAG=0
        END IF
        NORMLO=0.0D0

        IF(IREG.NE.0)THEN
          ICOV=ICOVF
          ICOR=ICORF
          IEIG=IEIGF
        END IF
        IF(MEMSAVE.EQ.1)THEN
          ICOVF=0
          ICORF=0
          IEIGF=0
        END IF
        INOCOV=1
        IF((ICOVF.EQ.1).OR.(ICORF.EQ.1).OR.(IEIGF.EQ.1))INOCOV=0
        IF((ICOVF.NE.0).OR.(ICORF.NE.0).OR.(IEIGF.NE.0))THEN
          ITEMP=NESPAR
          ITEMP2=NESPAR
        ELSE
          ITEMP=1
          IF((IREG.NE.0).AND.(MEMSAVE.EQ.0))THEN
            ITEMP2=NESPAR
          ELSE
            ITEMP2=1
          END IF
        END IF
        IF((IREG.NE.0).AND.(MEMSAVE.EQ.0))THEN
          ITEMP1=NESPAR
          ITEMP3=NESPAR
          ITEMP4=NESPAR
        ELSE
          ITEMP1=1
          ITEMP3=1
          ITEMP4=1
        END IF
        IF(SVDMODE.EQ.1)THEN
          ITEMP2=NESPAR
          ITEMP4=8*NESPAR
          LWORK=ITEMP4
          IF(RLAMBDA1.NE.0.0D0)THEN
            ITEMP3=NESPAR
          ELSE
            NOLAMBDA=1
          END IF
        END IF
        IF(LSQRMODE.EQ.1)THEN
          ITEMP1=NESPAR
        END IF
        IF(SVDMODE.EQ.2)THEN
          LWORK=MAX(3*MIN(NXROW,NESPAR)+MAX(NXROW,NESPAR),
     +              5*MIN(NXROW,NESPAR))
          LWORK=NINT(FLOAT(LWORK)*1.3)       ! ARBITRARY
          ITEMP4=LWORK
          ITEMP3=MIN(NESPAR,NXROW)
        END IF
        ITEMP4=MAX(6*NESPAR,ITEMP4)
        NDIM4=ITEMP4
        IF(NOPTMAX.EQ.0)THEN
          ITEMP=1
          ITEMP1=1
          ITEMP2=1
          ITEMP3=1
          ITEMP4=1
        ELSE IF (NOPTMAX.EQ.-2)THEN
          ITEMP=1
          ITEMP2=1
        END IF
        ALLOCATE(NRM(ITEMP2,ITEMP2),NRMLO(ITEMP,ITEMP),
     +  WORKVEC1(ITEMP1),WORKVEC2(ITEMP1),WORKVEC3(ITEMP3),
     +  WORKVEC4(ITEMP4),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200
        WORKVEC3DIM=ITEMP3

        ISPLIT=0
        JSPLIT=0
        FLAG5=0
        DO I=1,NPARGP
          IF(SPLITTHRESH(I).GT.0.0D0)THEN
            JSPLIT=1
            IF(SPLITACTION(I).EQ.3)THEN
              ISPLIT=1
            END IF
          END IF
          IF((FORCEN(I).EQ.4).OR.(FORCEN(I).EQ.5)) FLAG5=1
        END DO
        WV5DIM=1
        IF((ISPLIT.EQ.1).OR.(FLAG5.EQ.1))WV5DIM=NOBS
        ALLOCATE(WORKVEC5(WV5DIM),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

#ifdef PARALLEL
        WORKVEC7DIM=MAX(NOBS,NPAR)
#else
        WORKVEC7DIM=1
#endif
        ALLOCATE(WORKVEC7(WORKVEC7DIM),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

C -- FILES PERTAINING TO THE SOLUTION PROCESS ARE OPENED.

        IF(CONJGRAD.EQ.1)THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.cg'
          IF(IRST.EQ.0)THEN
            CALL FFOPEN(JFAIL,CGNOUT,'w',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(CGNOUT,2053,ERR=9710) CASEFL(:LEN_TRIM(CASEFL))
2053        FORMAT(' RECORD FROM CONJUGATE GRADIENT SOLVER: CASE ',A)
          ELSE IF(IRST.EQ.1) THEN
            CALL FFOPEN(JFAIL,CGNOUT,'a',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(CGNOUT,2040,ERR=9710)
          ELSE IF(IRST.EQ.2)THEN
            CALL FFOPEN(JFAIL,CGNOUT,'a',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(CGNOUT,2041,ERR=9710)
          END IF
        ELSE IF(SVDMODE.NE.0)THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.svd'
          IF(IRST.EQ.0)THEN
            CALL FFOPEN(JFAIL,SVDUNIT,'w',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(SVDUNIT,2057,ERR=9710) CASEFL(:LEN_TRIM(CASEFL))
2057        FORMAT(' RECORD OF SVD CALCULATIONS: CASE ',A)
          ELSE IF(IRST.EQ.1) THEN
            CALL FFOPEN(JFAIL,SVDUNIT,'a',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(SVDUNIT,2040,ERR=9710)
          ELSE IF(IRST.EQ.2)THEN
            CALL FFOPEN(JFAIL,SVDUNIT,'a',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(SVDUNIT,2041,ERR=9710)
          END IF
        ELSE IF(LSQRMODE.EQ.1)THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.lsq'
          IF(IRST.EQ.0)THEN
            IF(LSQR_NOUT.NE.0)THEN
              CALL FFOPEN(JFAIL,LSQR_NOUT,'w',' ',6,CLINE)
              IF(JFAIL.NE.0) GO TO 9891
              WRITE(LSQR_NOUT,2058,ERR=9710) CASEFL(:LEN_TRIM(CASEFL))
2058          FORMAT(' RECORD OF LSQR CALCULATIONS: CASE ',A)
            END IF
          ELSE IF(IRST.EQ.1) THEN
            IF(LSQR_NOUT.NE.0)THEN
              CALL FFOPEN(JFAIL,LSQR_NOUT,'a',' ',6,CLINE)
              IF(JFAIL.NE.0) GO TO 9891
              WRITE(LSQR_NOUT,2040,ERR=9710)
            END IF
          ELSE IF(IRST.EQ.2)THEN
            IF(LSQR_NOUT.NE.0)THEN
              CALL FFOPEN(JFAIL,LSQR_NOUT,'a',' ',6,CLINE)
              IF(JFAIL.NE.0) GO TO 9891
              WRITE(LSQR_NOUT,2041,ERR=9710)
            END IF
          END IF
        ELSE
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.cnd'
          IF(IRST.EQ.0)THEN
            CALL FFOPEN(JFAIL,NCONDUNIT,'w',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(NCONDUNIT,2052,ERR=9710) CASEFL(:LEN_TRIM(CASEFL))
2052        FORMAT(' RECORD OF "NORMAL MATRIX" CONDITION NUMBERS: ',
     +      'CASE ',A)
          ELSE IF(IRST.EQ.1) THEN
            CALL FFOPEN(JFAIL,NCONDUNIT,'a',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(NCONDUNIT,2040,ERR=9710)
          ELSE IF(IRST.EQ.2)THEN
            CALL FFOPEN(JFAIL,NCONDUNIT,'a',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(NCONDUNIT,2041,ERR=9710)
          END IF
        END IF

C -- SOME PROCESSING OF COVARIANCE MATRIX DATA IS UNDERTAKEN.

        MXOBSCOV=0
        TOTWT=0
        DO 6530 IGROUP=1,NOBSGP
          IF(.NOT.COVIND(IGROUP)) GO TO 6530
          ICOUNT=0
          DO 6520 I=1,NXROW
            IF(NOBGNM(I).EQ.IGROUP) THEN
              ICOUNT=ICOUNT+1
              TOTWT=TOTWT+1
            END IF
6520      CONTINUE
          IF(ICOUNT.GT.MXOBSCOV)MXOBSCOV=ICOUNT
6530    CONTINUE
        IF(MXOBSCOV.EQ.0)NUMCOV=0
        NM=MAX(MXOBSCOV,1)
        N=MAX(NUMCOV,1)
        NM1=MAX(TOTWT,1)
        ALLOCATE(RWORK1(NM1),RWORK2(NM1),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

C -- THE GENERAL WORK VECTOR W2 IS ALLOCATED.

        W2DIM=MAX(NXROW,NPAR,6*MXOBSCOV)
        ALLOCATE(W2(W2DIM),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200

C -- IF COVARIANCE MATRIX IS SUPPLIED FOR AN OBSERVATION GROUP, THAT GROUP
C    MUST PERTAIN TO OBSERVATIONS OR PRIOR INFORMATION, BUT NOT BOTH.

        IF(NUMCOV.GT.0)THEN
          IF(NPRIOR.GT.0)THEN
            DO 6601 I=NOBS+1,NXROW
              IGROUP=NOBGNM(I)
              DO 6602 JCOV=1,NUMCOV
                IF(IGROUP.EQ.COVGP(JCOV))THEN
                  DO 6603 J=1,NOBS
                    IF(NOBGNM(J).EQ.IGROUP) GO TO 9980
6603              CONTINUE
                END IF
6602          CONTINUE
6601        CONTINUE
          END IF
        END IF

C -- If an non-regularisation observation group has a covariance matrix
C    supplied, then a flag is set for later use.

        IMCOVFLAG=0
        IF(NUMCOV.GT.0)THEN
          DO I=1,NUMCOV
            J=COVGP(I)
            IF(IRGP(J).EQ.0) IMCOVFLAG=IMCOVFLAG+1
          END DO
        END IF

C -- If running in regularisation mode we also count the number of
C    observations and prior information equations which are members of
C    regularisation groups. Of these, we count the number for which the
C    "observed value" is non-zero.

       ICOUNTREGOBS=0
       ICOUNTREGNZ=0
       IF(IREG.NE.0)THEN
         DO I=1,NXROW
           J=NOBGNM(I)
           IF(IRGP(J).EQ.0)CYCLE
           ICOUNTREGOBS=ICOUNTREGOBS+1
           IF(OVAL(I).NE.0.0D0) ICOUNTREGNZ=ICOUNTREGNZ+1
         END DO
       END IF

C -- A check is made on whether we are using individual group target
C    measurement objective functions.

       DO I=1,NOBSGP
         IF(GTARG(I).GT.-1.0D35) IGTARG=1
       END DO

C -- A possible error condition is checked for.

        IF(IGTARG.EQ.1)THEN
          DO I=1,NOBSGP
            IF(IRGP(I).EQ.0)THEN
              ICOUNT=0
              DO IROW=1,NXROW
                IF(NOBGNM(IROW).EQ.I) ICOUNT=ICOUNT+1
              END DO
              IF(ICOUNT.EQ.0)THEN
                WRITE(ERRMSG,6609)
6609            FORMAT('If group-specific target measurement ',
     +          'objective function values ',
     +          'are supplied, then every non-regularisation ',
     +          'observation group must possess at least one member.')
                GO TO 9890
              END IF
            END IF
          END DO
        END IF

C -- THE NUMBER OF TIED PARAMETERS IS EVALUATED.

        NUMTIED=0
        DO I=1,NPAR
          IF((ITRANS(I).LT.0).AND.(ITRANS(I).GT.-100000))
     +    NUMTIED=NUMTIED+1
        END DO

C -- SOME PRELIMINARY WORK IS DONE PRIOR TO OPTIMISATION

        IFLAG_X=0

        IF(HOLDFLAG.NE.0)THEN
          DO 7580 IPP=1,NPAR
            ISENS(IPP)=0
7580      CONTINUE
          DO 7581 IPP=1,NPAR
            SENS(IPP)=0.0D0
7581      CONTINUE
        END IF

        NUMREL=0
        NUMFAC=0
        DO 22 IPP=1,NPAR
        IF((ITRANS(IPP).NE.1).AND.(ITRANS(IPP).NE.0)) GO TO 22
        IF(PARCHGLIM(IPP).EQ.0) THEN
          NUMREL=NUMREL+1
        ELSE IF(PARCHGLIM(IPP).EQ.1) THEN
          NUMFAC=NUMFAC+1
        END IF
22      CONTINUE

        DO 16 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 16
        IF(APAR(IPP)(1:3).EQ.'iw_') GO TO 16
        IF((SCALE(IPP).NE.1.0D0).OR.(OFFSET(IPP).NE.0.0D0))THEN
          RTEMP=PARUBND(IPP)*SCALE(IPP)+OFFSET(IPP)
          RRTEMP=PARLBND(IPP)*SCALE(IPP)+OFFSET(IPP)
          RRRTMP=PVAL(IPP)*SCALE(IPP)+OFFSET(IPP)
          J=1
        ELSE
          RTEMP=PARUBND(IPP)
          RRTEMP=PARLBND(IPP)
          RRRTMP=PVAL(IPP)
          J=0
        END IF
        CALL WRTSIG(IERR,RRRTMP,PWORD(IPP),NW(IPP),PRECIS,TTTVAL,
     +  NOPNT)
        IF(IERR.NE.0) THEN
          CALL STPERR(33+IERR,1,APAR(IPP),0,' ',CLINE)
          GO TO 9891
        END IF
        CALL WRTSIG(IERR,RTEMP,PWORD(IPP),NW(IPP),PRECIS,TVAL,NOPNT)
        IF(IERR.NE.0) THEN
          CALL STPERR(111+IERR,1,APAR(IPP),0,' ',CLINE)
          GO TO 9891
        END IF
        CALL WRTSIG(IERR,RRTEMP,PWORD(IPP),NW(IPP),PRECIS,TTVAL,
     +  NOPNT)
        IF(IERR.NE.0) THEN
          CALL STPERR(111+IERR,1,APAR(IPP),0,' ',CLINE)
          GO TO 9891
        END IF
        IF(J.EQ.1) THEN
          PARUBND(IPP)=(TVAL-OFFSET(IPP))/SCALE(IPP)
          PARLBND(IPP)=(TTVAL-OFFSET(IPP))/SCALE(IPP)
          PVAL(IPP)=(TTTVAL-OFFSET(IPP))/SCALE(IPP)
        ELSE
          PARUBND(IPP)=TVAL
          PARLBND(IPP)=TTVAL
          PVAL(IPP)=TTTVAL
        END IF
16      CONTINUE

C -- SOME ARRAY DIMENSIONS ARE ASSIGNED

        SOPDIM=1

#ifdef PARALLEL

C -- INPUT AND OUTPUT FILENAMES ARE STORED

       ALLOCATE(SINFLE(NTPLFLE,NSLAVE),SOUFLE(NINSFLE,NSLAVE),
     + STAT=IERR)
       IF(IERR.NE.0) GO TO 9200
       LDSIN=NTPLFLE
       LDSOU=NINSFLE
#ifdef MPEST
       ALLOCATE(MSINFLE(NTPLFLE),MSOUFLE(NINSFLE),STAT=IERR)
       IF(IERR.NE.0) GO TO 9200
       DO J=1,NTPLFLE
         SINFLE(J,1)=INFLE(J)
       END DO
       DO J=1,NINSFLE
         SOUFLE(J,1)=OUTFLE(J)
       END DO
#else
       ALLOCATE(MSINFLE(1),MSOUFLE(1),STAT=IERR)
       IF(IERR.NE.0) GO TO 9200
       IF(IFLTYP.EQ.-1)THEN
         DO 2362 I=1,NSLAVE
           DO 2359 J=1,NTPLFLE
             SINFLE(J,I)=INFLE(J)(1:LEN_TRIM(INFLE(J)))
2359       CONTINUE
           DO 2358 J=1,NINSFLE
             SOUFLE(J,I)=OUTFLE(J)(1:LEN_TRIM(OUTFLE(J)))
2358       CONTINUE
2362     CONTINUE
       ELSE IF(IFLTYP.EQ.0)THEN
         DO 2400 I=1,NSLAVE
           DO 2350 J=1,NTPLFLE
             SINFLE(J,I)=ASLDIR(I)(1:LEN_TRIM(ASLDIR(I)))//
     +       INFLE(J)(1:LEN_TRIM(INFLE(J)))
2350       CONTINUE
           DO 2355 J=1,NINSFLE
             SOUFLE(J,I)=ASLDIR(I)(1:LEN_TRIM(ASLDIR(I)))//
     +       OUTFLE(J)(1:LEN_TRIM(OUTFLE(J)))
2355       CONTINUE
2400     CONTINUE
       ELSE
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rmf'
         CALL SLAVDAT3(JFAIL,NSLAVE,NTPLFLE,NINSFLE,LDSIN,LDSOU,SINFLE,
     +   SOUFLE,CLINE)
         IF(JFAIL.NE.0) GO TO 9891
       END IF
#endif

       IF(IPRLL.EQ.0) THEN
         SCOM(1)=MODFLE(1:LEN_TRIM(MODFLE))
       END IF

       IF((LDOSCH.EQ.1).OR.(LDOSCH.LT.0))THEN
         IF(RLAMBDA1.EQ.0.0D0)THEN
           LDOSCH=2
         END IF
       END IF
#ifdef MPEST
       IF(LDOSCH.EQ.2)LDOSCH=0
#else
       CALL WRITSLV2(JFAIL,NSLAVE,NTPLFLE,NINSFLE,LDSIN,LDSOU,LDOSCH,
     +  REPEATRUN,ASLAVE,SINFLE,SOUFLE,SCOM,AFILE,CLINE)
        IF(JFAIL.NE.0) GO TO 9891

        IF(IPRLL.EQ.0)THEN
          CALL FFOPEN(JFAIL,IPCM,'w','p###.cmd',6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(IPCM,'(A)') MODFLE(1:LEN_TRIM(MODFLE))
          CLOSE(UNIT=IPCM)
          CALL PESTWAIT(IWAIT)
        END IF
#endif

C -- THE MAXIMUM NUMBER OF SIMULTANEOUS MODEL RUNS IS CALCULATED

        NSRUN=0
        DO 8100 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 8100
        IF(SCALE(IPP).LT.-1.0D35) GO TO 8100
        I=IPARGP(IPP)
        IF(FORCEN(I).EQ.2)THEN
          NSRUN=NSRUN+1
        ELSE IF((FORCEN(I).EQ.3).OR.(FORCEN(I).EQ.1)) THEN
          NSRUN=NSRUN+2
        ELSE IF((FORCEN(I).EQ.4).OR.(FORCEN(I).EQ.5)) THEN
          NSRUN=NSRUN+4
        ENDIF
8100    CONTINUE
        IF(NSRUN.LT.NSLAVE)NSRUN=NSLAVE   !required for parallel lambda search
        ALLOCATE(ISTATR(NSRUN),ITRIAL(NSRUN),INCPAR(NSRUN),
     +  REGLAM(NSRUN),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200
        DAPARRECL=8*MAXRECLNUM
        DAOBSRECL=8*MAXRECLNUM
        OPEN(UNIT=PARREG,FILE=PARREGFILE,STATUS='REPLACE',
     +  FORM='UNFORMATTED',ACCESS='DIRECT',RECL=DAPARRECL,IOSTAT=IERR)
        IF(IERR.NE.0) THEN
          WRITE(ERRMSG,8101) TRIM(PARREGFILE)
8101      FORMAT('Cannot open parameter parallel run queue file ',
     +    A,'.')
          GO TO 9890
        END IF
        OPEN(UNIT=OBSREG,FILE=OBSREGFILE,STATUS='REPLACE',
     +  FORM='UNFORMATTED',ACCESS='DIRECT',RECL=DAOBSRECL,IOSTAT=IERR)
        IF(IERR.NE.0) THEN
          WRITE(ERRMSG,8102) TRIM(OBSREGFILE)
8102      FORMAT('Cannot open observation parallel run queue file ',
     +    A,'.')
          GO TO 9890
        END IF
#else
        ALLOCATE(INCPAR(1),STAT=IERR)
        IF(IERR.NE.0) GO TO 9200
#endif

#ifdef PARALLEL
       DO 2100 I=1,NSLAVE
         ISTATS(I)=0
2100   CONTINUE
       NNRUN=0
       IF((NOPTMAX.EQ.-1).OR.(NOPTMAX.EQ.-2)) LDOSCH=0
       IF((IPRED.EQ.1).AND.(NSEARCH.GT.1))LDOSCH=0
#endif

C -- THE JSTK VECTOR IS INITIALIZED

        DO 2105 IPP=1,NPAR
          JSTK(IPP)=0
2105    CONTINUE

        IF(IRST.NE.0) THEN
          GO TO 110
        END IF

C -- INPUT DATA IS WRITTEN TO THE RUN RECORD FILE

        NCOVDIM1=1
        IF(NUMCOV.GT.1) NCOVDIM1=NUMCOV
        NCOVDIM2=1
        NCOVDIM3=1
        IF(MXOBSCOV.GT.1)NCOVDIM2=MXOBSCOV
        IF(TOTWT.GT.1)NCOVDIM3=TOTWT
        CALL WRTALL(JFAIL,IREC,NPAR,NESPAR,NPARGP,NOBS,NPRIOR,NUMCOM,
     +  JACFILE,NXROW,NTPLFLE,NINSFLE,ITRANS,SCALE,OFFSET,PARCHGLIM,
     +  IPARGP,PVAL,PARLBND,PARUBND,APAR,OVAL,OWGHT,X,AOBS,IDBND,INCTYP,
     +  FORCEN,DERMTHD,DERINC,DERINCMUL,DERINCLB,PARGNME,RLAMBDA1,
     +  RLAMFAC,PHIRATSUF,PHIREDLAM,NUMLAM,RELPARMAX,DMAX2,FACORIG,
     +  PHIREDSWH,PHIREDSTP,NPHINORED,NPHISTP,NOPTMAX,RELPARSTP,NRELPAR,
     +  NUMREL,NUMFAC,TEMPFLE,INFLE,INSFLE,OUTFLE,CLINE,PRECIS,NOPNT,
     +  NOBSGP,NOBGNM,OBGNME,JPREDICT,NPREDMAXMIN,PD0,PD1,PD2,
     +  ABSPREDLAM,RELPREDLAM,ABSPREDSWH,RELPREDSWH,NPREDNORED,
     +  ABSPREDSTP,RELPREDSTP,NPREDSTP,NSEARCH,INITSCHFAC,MULSCHFAC,
     +  IREG,PHIMLIM,PHIMACCEPT,FRACPHIM,WFINIT,WFMIN,WFMAX,WFFAC,WFTOL,
     +  REGITN,DERCOM,COMLIN,COMJAC,EXTDERFLE,MESSFILE,NCOVDIM1,NUMCOV,
     +  COVGP,NOBSCOV,COVAR,MAXAUI,AUISTARTOPT,NOAUIPHIRAT,AUIRESTITN,
     +  AUISENSRAT,AUIHOLDMAXCHG,AUINUMFREE,AUIPHIRATSUF,
     +  AUIPHIRATACCEPT,NAUINOACCEPT,UPVECBEND,IBOUNDSTICK,ICOVOBS,
     +  MEMSAVE,CONJGRAD,CGRTOL,CGITNLIM,LINREG,SVDMODE,
     +  MAXSING,EIGTHRESH,EIGWRITE,DI_DERFILE,NOPTSWITCH,BNPARGP,
     +  BPARGPNME,SENFACTOR,PWTKP,IGTARG)
        IF(JFAIL.NE.0) GO TO 9891
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif

C -- IF ANY COVARIANCE MATRICES ARE PROVIDED, THE ROTATION MATRICES ARE
C    NOW CALCULATED.

        JWT=0
        IF(NUMCOV.NE.0)THEN
          DO 6650 JCOV=1,NUMCOV
            IF(NOBSCOV(JCOV).LE.0) GO TO 6650
            IGROUP=COVGP(JCOV)
            IF(NOBSCOV(JCOV).EQ.1)THEN
              W1(1)=COVAR(JCOV)%DVAL(1,1)
              COVAR(JCOV)%DVAL(1,1)=1.0D0
            ELSE
              NM=MXOBSCOV
              CALL DGESVD('O','N',NOBSCOV(JCOV),NOBSCOV(JCOV),
     +        COVAR(JCOV)%DVAL,NOBSCOV(JCOV),W1,U,LDU,VT,LDVT,
     +        W2,W2DIM,INFO)
              IF(INFO.NE.0)THEN
                CALL STPERR(169,1,OBGNME(IGROUP),0,' ',CLINE)
                GO TO 9891
              END IF
              CALL VECTOR_REVERSE(NOBSCOV(JCOV),W1)                    ! For compatibility with older versions
              CALL MATRIX_REVERSE(NOBSCOV(JCOV),NOBSCOV(JCOV),         ! For compatibility with older versions
     +        NOBSCOV(JCOV),COVAR(JCOV)%DVAL)
            END IF
            J=0
            DO 6640 I=1,NXROW
              IF(NOBGNM(I).EQ.IGROUP)THEN
                J=J+1
                JWT=JWT+1
                IF(W1(J).LE.0.0D0) THEN
                  CALL STPERR(169,1,OBGNME(IGROUP),0,' ',CLINE)
                  GO TO 9891
                END IF
                OWGHT(I)=1.0/W1(J)
              END IF
6640        CONTINUE
6650      CONTINUE
        END IF

C -- AFFECTED OBSERVATION WEIGHTS WERE RE-CALCULATED FROM RESPECTIVE EIGENVALUES
C    ABOVE. NOW THE OBSERVATIONS ARE ALSO ROTATED.

        IF(NUMCOV.NE.0)THEN
          DO 6700 JCOV=1,NUMCOV
            IF(NOBSCOV(JCOV).LE.0) GO TO 6700
            IGROUP=COVGP(JCOV)
            I=0
            DO 6670 J=1,NXROW
              IF(NOBGNM(J).EQ.IGROUP)THEN
                I=I+1
                W1(I)=OVAL(J)
              END IF
6670        CONTINUE
            CALL ROTATE(1,NOBSCOV(JCOV),NOBSCOV(JCOV),1,1,1,
     +      COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,NXROW,NOBGNM)
            I=0
            DO 6680 J=1,NXROW
              IF(NOBGNM(J).EQ.IGROUP)THEN
                I=I+1
                OVAL(J)=W1(I)
              END IF
6680        CONTINUE
6700      CONTINUE
        END IF

C -- IF THERE IS ANY PRIOR INFORMATION PRESENT, AND ITS WEIGHTS ARE CORRELATED,
C    THEN THE APPROPRIATE PART OF THE X VECTOR IS ROTATED.


        MATDIM1=NXROW

        IF(NUMCOV.NE.0)THEN
          IF(NPRIOR.NE.0)THEN
            IPSTART=NXROW-NPRIOR+1
            DO 6750 JCOV=1,NUMCOV
              IGROUP=COVGP(JCOV)
              DO 6720 I=IPSTART,NXROW
                IF(NOBGNM(I).EQ.IGROUP)THEN
                  CALL ROTATE(2,NOBSCOV(JCOV),NOBSCOV(JCOV),MATDIM1,
     +            NESPAR,IPSTART,COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,
     +            NXROW,NOBGNM)
                  GO TO 6750
                END IF
6720          CONTINUE
6750        CONTINUE
          END IF
        END IF

C -- A CHECK IS MADE FOR THE PRESENCE OF MEMBERS OF A REGULARISATION
C    OBSERVATION GROUP

        IF(IREG.EQ.1)THEN
          ICOUNT=0
          RTEMP=0.0D0
          DO 5030 I=1,NXROW
            IGPNM=NOBGNM(I)
            IF(IRGP(IGPNM).NE.0)THEN
              ICOUNT=ICOUNT+1
              RTEMP=RTEMP+ABS(OWGHT(I))
              IF(RTEMP.GT.0.0D0) GO TO 5032
            END IF
5030      CONTINUE
          IF(ICOUNT.EQ.0)THEN
            FLENME=CTLFILE
            CALL STPERR(146,1,' ',0,' ',CLINE)
            GO TO 9891
          END IF
          IF(RTEMP.EQ.0.0D0)THEN
            FLENME=CTLFILE
            CALL STPERR(147,1,' ',0,' ',CLINE)
            GO TO 9891
          END IF
        END IF
5032    CONTINUE

C -- IF IREGADJ IS EQUAL TO 3, 4 OR 5 WE KEEP A RECORD OF ORIGINAL WEIGHTS.

        AVOWGHTKP=0.0D0
        NUMAVO=0
        IF(IREG.NE.0)THEN
          IF(IREGADJ.EQ.3)THEN
            KREG=0
            DO IROW=1,NXROW
              J=NOBGNM(IROW)
              IF(IRGP(J).NE.0)THEN
                KREG=KREG+1
                IF(OWGHT(IROW).GT.0.0D0)THEN
                  OWGHTKP(KREG)=SQRT(OWGHT(IROW))
                  NUMAVO=NUMAVO+1
                  AVOWGHTKP=AVOWGHTKP+OWGHTKP(KREG)
                ELSE
                  OWGHTKP(KREG)=0.0D0
                END IF
              END IF
            END DO
            IF(NUMAVO.GT.0)THEN
              IF(AVOWGHTKP.GT.0.0D0)THEN
                AVOWGHTKP=NUMAVO/AVOWGHTKP
                KREG=0
                DO IROW=1,NXROW
                  J=NOBGNM(IROW)
                  IF(IRGP(J).NE.0)THEN
                    KREG=KREG+1
                    IF(OWGHT(IROW).GT.0.0D0)THEN
                      OWGHTKP(KREG)=OWGHTKP(KREG)*AVOWGHTKP
                    END IF
                  END IF
                END DO
              END IF
            ENDIF
          ELSE IF((IREGADJ.EQ.4).OR.(IREGADJ.EQ.5))THEN
            KREG=0
            DO IROW=1,NXROW
              J=NOBGNM(IROW)
              IF(IRGP(J).NE.0)THEN
                KREG=KREG+1
                OWGHTKP(KREG)=OWGHT(IROW)
              END IF
            END DO
          END IF
        END IF

C -- If Pareto weights adjustment is undertaken, pertinent initial weights are stored.

        IF(PESTMODE.EQ.4)THEN
          J=0
          DO I=1,NXROW
            IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
              J=J+1
              OWGHTKP(J)=OWGHT(I)
            END IF
          END DO
        END IF

        WRITE(6,30) CASEFL(:LEN_TRIM(CASEFL))
30      FORMAT(/' PEST run record: case ',A)
        WRITE(6,31) CASEFL(:LEN_TRIM(CASEFL))//'.rec'
31      FORMAT(' (See file ',A,' for full details.)')
        WRITE(6,32) MODFLE(:LEN_TRIM(MODFLE))
32      FORMAT(/,' Model command line: ',/,1X,A)
#ifdef PARALLEL
#ifndef MPEST
        WRITE(IRMR,2010)
2010    FORMAT(/,' RUNNING MODEL FOR FIRST TIME  -----> ')
#endif
        WRITE(6,2012)
2012    FORMAT(/,' RUNNING MODEL FOR FIRST TIME .....')
#else
        WRITE(6,33)
33      FORMAT(/,' Running model .....')
#endif


        IF(PESTMODE.EQ.4)THEN
          PAR_WTFAC=PAR_WTFAC_START
          ATEMP12=OBGNME(PAR_NOBGNM)
          WRITE(6,7552) TRIM(ATEMP12),PAR_WTFAC
7552      FORMAT(/,T4,'Pareto weight factor for observation group "',A,
     +    '"'T62,'= ',1PG12.5,/)
          PAR_WTFAC_KP=PAR_WTFAC
          PAR_WTFAC=PAR_WTFAC*PAR_WTFAC
          J=0
          DO I=1,NXROW
            IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
              J=J+1
              OWGHT(I)=OWGHTKP(J)*PAR_WTFAC
            END IF
          END DO
        END IF

C -- A RECORD IS KEPT OF ORIGINAL WEIGHTS (IF DOING ADAPTIVE REGULARISATION).
C    (NOTE THAT THIS MUST BE DONE AFTER OBSERVATION ROTATION)

        ICOUNT=0
        IF(NREGADJPAR.GT.0)THEN
          DO IPP=1,NPAR
            IF(SCALE(IPP).LT.-1.0D35)THEN
              ITEMP=NINT(OFFSET(IPP))
              DO IOBS=1,NXROW
                IF(NOBGNM(IOBS).EQ.ITEMP)THEN
                  ICOUNT=ICOUNT+1
                  ORIGWGHT(ICOUNT)=OWGHT(IOBS)
                END IF
              END DO
            END IF
          END DO
        END IF

C -- BEFORE THE FIRST MODEL RUN, REGULARISATION WEIGHTS ARE ADJUSTED.

        IF(IREG.EQ.1)THEN
          DO 5120 IROW=1,NXROW
            IGPNM=NOBGNM(IROW)
            IF(IRGP(IGPNM).NE.0) OWGHT(IROW)=OWGHT(IROW)*WFINIT*WFINIT
5120      CONTINUE
        END IF

C -- PEST RUNS THE MODEL FOR THE FIRST TIME

        I=-1000
        J=0
        INCPAR(1)=0
        IF(NOPTMAX.NE.0)THEN
          IF(SVDA.EQ.1)THEN
            IF(SVDA_SUPDERCALC.EQ.1)THEN
              CALL GET_BASE_REFOBS(IFAIL,REFOBS)
              IF(IFAIL.EQ.0)THEN
                WRITE(6,5121)
5121            FORMAT(' No need to run model - model results ',
     +          'obtained from base PEST run.',/)
                GO TO 2211
              END IF
            END IF
          END IF
        END IF

#ifdef PARALLEL
        CALL STORE_PARALLEL_REGISTER
     +  (IFAIL,NPAR,PARREG,1,PVAL,PARREGFILE)
        IF(IFAIL.NE.0) GO TO 9890
#ifdef BEO
       if (BEOMASTER) then
         call RUNMASTER(PARREG,OBSREG,1,JFAIL)
         if (JFAIL.ne.0) goto 9891
       else
#endif
#ifdef MPEST
        FLENME=TRIM(CASEFL)//'.lambda_runs'
        CALL FFOPEN(JFAIL,IRMR,'w',' ',6,CLINE)
        IF(JFAIL.NE.0) GO TO 9891
        WRITE(IRMR,'(I5)') 1
        CLOSE(UNIT=IRMR)
        FLENME=TRIM(CASEFL)//'.jacobian_runs'
        CALL DELFILE1(JFAIL,FLENME,CLINE)
        IF(JFAIL.NE.0) GO TO 9891
        CALL DORUNS_M(JFAIL,NSLAVE,1,
     +  ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,LL,OBSN1,
     +  OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,
     +  NPAR,PRECIS,NOPNT,NTPLFLE,NW,SCALE,
     +  OFFSET,PARDEL,PWORD,TEMPFLE,APAR,
     +  OUTFLE,INSFLE,LDSIN,LDSOU,SINFLE,
     +  SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,MSINFLE,MSOUFLE,
     +  MODFLE,WORKVEC7DIM,WORKVEC7)
        IF(JFAIL.EQ.2) GO TO 9890
        IF(JFAIL.NE.0) GO TO 9891
#else
        PIRESTART=0
        CALL DORUNS(JFAIL,NSLAVE,1,ISTATS,ISTATR,
     +  ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,LL,OBSN1,
     +  OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,ISTRTME,
     +  IRUNTME,JRUN,NPAR,PRECIS,NOPNT,NTPLFLE,NW,SCALE,
     +  OFFSET,PARDEL,PWORD,TEMPFLE,APAR,OREADFLE,PREADFLE,
     +  MANFLE,ASLAVE,NNRUN,OUTFLE,INSFLE,LDSIN,LDSOU,SINFLE,
     +  SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,IDET,SREADFLE,MREADFLE,
     +  SCOM,PIRESTART,PTUNIT,PTFILE,IOPT,REPEATRUN,0,SLAVEGROUP,
     +  WORKVEC7DIM,WORKVEC7)
        IF(JFAIL.NE.0) GO TO 9891
#endif
#ifdef BEO
       endif
#endif
        IF(ISTOP.EQ.2)THEN
          IFIN=10
          GO TO 6000
        ELSE IF(ISTOP.EQ.1) THEN
          IPFAIL=-1
          GO TO 9891
        END IF
        CALL RETRIEVE_PARALLEL_REGISTER
     +  (IFAIL,NOBS,OBSREG,1,REFOBS,OBSREGFILE)
        IF(IFAIL.NE.0) GO TO 9890
#else
        REPEATFLAG=0
        CALL MODEL(JFAIL,NPAR,NOBS,ASIZE,NINSTR,NTPLFLE,NINSFLE,NUML,
     +  NBLBMX,SCALE,OFFSET,PVAL,NW,PWORD,APAR,REFOBS,AOBS,I,W2,J,
     +  PRECIS,NOPNT,LCINS,LL,OBSN1,OBSN2,IIOBS,INSFLE,TEMPFLE,INFLE,
     +  OUTFLE,PARDEL,MRKDEL,A,CLINE,BUF,COMLIN(1),J,RTEMP,MESSFILE,
     +  INCPAR(1),ITRANS,REPEATFLAG,0)
        IF(JFAIL.NE.0) GO TO 9891
#endif

2211    CONTINUE

C -- IF ANY COVARIANCE MATRICES HAVE BEEN SUPPLIED, THEN THE REFERENCE
C    OBSERVATIONS ARE ROTATED.

        IF(NUMCOV.GT.0)THEN
          DO 6800 JCOV=1,NUMCOV
            IF(NOBSCOV(JCOV).LE.0) GO TO 6800
            IGROUP=COVGP(JCOV)
            I=0
            DO 6810 J=1,NOBS
              IF(NOBGNM(J).EQ.IGROUP)THEN
                I=I+1
                W1(I)=REFOBS(J)
              END IF
6810        CONTINUE
            CALL ROTATE(1,NOBSCOV(JCOV),NOBSCOV(JCOV),1,1,1,
     +      COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,NXROW,NOBGNM)
            I=0
            DO 6820 J=1,NXROW
              IF(NOBGNM(J).EQ.IGROUP)THEN
                I=I+1
                REFOBS(J)=W1(I)
              END IF
6820        CONTINUE
6800      CONTINUE
        END IF

        IF(IPRED.GT.0)REFPRED=REFOBS(NOBS)+E_PRED

        IF(NPRIOR.NE.0) CALL PRRCLC(JFAIL,NPAR,NOBS,NXROW,NESPAR,PVAL,
     +  ITRANS,REFOBS,X,CLINE,SCALE,NREGADJPAR,PRIORPAR)
        IF(JFAIL.NE.0) GO TO 9891
        CALL OBJCLC(1,PSI,NXROW,NPRIOR,NOBS,REFOBS,OVAL,OWGHT,NOBGNM,
     +  NOBSGP,IRGP,SUM1)
        IF(PREDNOISE.GT.0) PSI=PSI+RES_PRED
        IF(IREG.EQ.1)THEN
          CALL OBJCLC(888888,PHIR2,NXROW,NPRIOR,NOBS,REFOBS,
     +    OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,PHIM)
          PHIR2=PHIR2/WF/WF
        END IF
        DO 50 IPP=1,NPAR
50      PVAL_O(IPP)=PVAL(IPP)
        DO 60 IROW=1,NXROW
60      MODVAL_O(IROW)=REFOBS(IROW)
        PSILO=PSI
        LOOPT=0
        IF(IREG.EQ.1)THEN
          PHIMLO=PHIM
          PHIRLO=PHIR2
          WFLO=WF
        END IF

        IF(IPRED.GT.0)THEN
          IF(PSILO.LE.PD1)THEN
            PREDLO=REFPRED
          ELSE
            IF(PREDHL.EQ.'l')THEN
              PREDLO=1.0D300
            ELSE
              PREDLO=-1.0D300
            END IF
          END IF
          RES_PREDLO=RES_PRED
          E_PREDLO=E_PRED
          EW_PREDLO=EW_PRED
        END IF

        DO 65 I=1,NPHISTP
          PSILOL(I)=1.0D300
          PHIMLOL(I)=1.0D300
          PHIRLOL(I)=1.0D300
65      CONTINUE

        IF(IPRED.GT.0)THEN
          DO 66 I=1,NPREDSTP
            IF(PREDHL.EQ.'l')THEN
              PRDLOL(I)=1.0D300
            ELSE
              PRDLOL(I)=-1.0D300
            END IF
66        CONTINUE
        END IF

        IF(PESTMODE.NE.4)THEN
        CALL PRMSAV(JFAIL,NPAR,APAR,PVAL_O,CLINE,APRECS,APOINT,SCALE,
     +  OFFSET,-1)
        IF(JFAIL.NE.0) GO TO 9891
        END IF
C        IF(SVDA.EQ.1)SVDA_SAVE=1  Commenting this out removes the "save-even-if-phi-worse" bug.

        OPTSWH=-1
        WRITE(IREC,80,ERR=9350)
80      FORMAT(T30,'OPTIMISATION RECORD')
        WRITE(IREC,90,ERR=9350)
90      FORMAT(/,/,' INITIAL CONDITIONS: ')
        IF(IREG.EQ.1)THEN
          WRITE(IREC,5051,ERR=9350) WF
          WRITE(6,5051) WF
5051      FORMAT('   Current regularisation weight factor',T62,'= ',
     +    1PG12.5)
        END IF
        IF(PESTMODE.EQ.4)THEN
          ATEMP12=OBGNME(PAR_NOBGNM)
          WRITE(IREC,7552) TRIM(ATEMP12),PAR_WTFAC_KP
        END IF
        IF(IREG.NE.0)THEN
C          WRITE(IREC,89,ERR=9350) PSI
C          WRITE(6,89) PSI
          WRITE(IREC,5035,ERR=9350) PHIM
          WRITE(6,5035) PHIM
5035      FORMAT('   Current value of measurement objective function',
     +    T62,'= ',1PG12.5)
          WRITE(IREC,5060,ERR=9350) PHIR2
          WRITE(6,5060) PHIR2
5060      FORMAT('   Current value of regularisation objective ',
     +    'function ',T62,'= ',1PG12.5)
          WRITE(IREC,*,ERR=9350)
          WRITE(6,*,ERR=9350)
        END IF
        IF(((NOBSGP.GT.1).AND.(IPRED.EQ.0)).OR.
     +     ((NOBSGP.GT.2).AND.(IPRED.NE.0))
     +      .OR.(NPRIOR.NE.0))THEN
          WRITE(IREC,89,ERR=9350) PSI
          WRITE(6,89) PSI
89        FORMAT('   Sum of squared weighted residuals (ie phi)',
     +    T62,'= ',1PG12.5)
          IF(PREDNOISE.GT.0)THEN
            WRITE(IREC,88,ERR=9350) RES_PRED
            WRITE(6,88) RES_PRED
88          FORMAT('   Contribution to phi from predictive error'
     +      T62,'= ',1PG12.5)
          END IF
          DO 92 I=1,NOBSGP
            IF(I.NE.JPREDICT)THEN
              CALL OBJCLC(-I,PSISUB(I),NXROW,NPRIOR,NOBS,REFOBS,OVAL,
     +        OWGHT,NOBGNM,NOBSGP,IRGP,SUM1)
              WRITE(IREC,93,ERR=9350) OBGNME(I)(1:LEN_TRIM(OBGNME(I))),
     +        PSISUB(I)
              WRITE(6,93) OBGNME(I)(1:LEN_TRIM(OBGNME(I))),PSISUB(I)
93            FORMAT('   Contribution to phi from observation group "',
     +        A,'"',T62,'= ',1PG12.5)
            END IF
92        CONTINUE
        ELSE
          WRITE(IREC,91,ERR=9350) PSI
          WRITE(6,91) PSI
91        FORMAT  ('    Sum of squared weighted residuals (ie phi) = ',
     +    1PG12.5)
          IF(PREDNOISE.GT.0)THEN
            WRITE(IREC,87,ERR=9350) RES_PRED
            WRITE(6,87) RES_PRED
87          FORMAT('    Predictive noise contribution              = ',
     +      1PG12.5)
          END IF
        END IF
        IF(NPRIOR.NE.0)THEN
          CALL OBJCLC(-999,PSISUB(NOBSGP+1),NXROW,NPRIOR,NOBS,REFOBS,
     +    OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,SUM1)
          IF(PSISUB(NOBSGP+1).GT.-1.0D299)THEN
            WRITE(IREC,94,ERR=9350) PSISUB(NOBSGP+1)
            WRITE(6,94) PSISUB(NOBSGP+1)
94          FORMAT('   Contribution to phi from ungrouped prior ',
     +      'information',T62,'= ',1PG12.5)
          END IF
        END IF
        IF(IPRED.NE.0)THEN
          IF((NOBSGP.GT.2).OR.(NPRIOR.NE.0))THEN
            WRITE(IREC,96,ERR=9350) REFOBS(NOBS)
            WRITE(6,96) REFOBS(NOBS)
96          FORMAT('   Prediction',T62,'= ',1PG12.5)
            IF(PREDNOISE.NE.0)THEN
              WRITE(IREC,86,ERR=9350) E_PRED
              WRITE(6,86) E_PRED
86            FORMAT('   Predictive error',T62,'= ',1PG12.5)
            END IF
          ELSE
            WRITE(IREC,97,ERR=9350) REFOBS(NOBS)
            WRITE(6,97) REFOBS(NOBS)
97          FORMAT('    Prediction                                 = ',
     +      11PG12.5)
            IF(PREDNOISE.NE.0)THEN
              WRITE(IREC,85,ERR=9350) E_PRED
              WRITE(6,85) E_PRED
85            FORMAT('    Predictive error                           = ',
     +        1PG12.5)
            END IF
          END IF
        END IF

        WRITE(IREC,*,ERR=9350)
        CALL PRMWRT(JFAIL,IREC,NPAR,APAR,PVAL,LSTVAL,CLINE)
        IF(JFAIL.NE.0) GO TO 9891
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif

        IF(PESTMODE.EQ.4)THEN
          J=0
          DO I=1,NXROW
            IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
              J=J+1
              RTEMP=OWGHT(I)
              OWGHT(I)=OWGHTKP(J)
              OWGHTKP(J)=RTEMP
            END IF
          END DO
          CALL OBJCLC(-PAR_NOBGNM,PARPSISUB(PAR_NOBGNM),NXROW,NPRIOR,
     +    NOBS,REFOBS,OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,SUM1)
          J=0
          DO I=1,NXROW
            IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
              J=J+1
              RTEMP=OWGHT(I)
              OWGHT(I)=OWGHTKP(J)
              OWGHTKP(J)=RTEMP
            END IF
          END DO
          DO I=1,NOBSGP
            IF(I.NE.PAR_NOBGNM)THEN
              WRITE(ATEMP12,'(1PG12.6)') PSISUB(I)
              PARPSISUB(I)=PSISUB(I)
            ELSE
              WRITE(ATEMP12,'(1PG12.6)') PARPSISUB(I)
            END IF
            ATEMP12=ADJUSTL(ATEMP12)
            WRITE(IPOD,'(1X,A12,1X)',ADVANCE='NO') TRIM(ATEMP12)
          END DO
          IF(NOBS_REPORT.NE.0)THEN
            DO I=1,NOBS_REPORT
              WRITE(ATEMP20,'(1PG13.6)') REFOBS(IOBS_REPORT(I))
              ATEMP20=ADJUSTL(ATEMP20)
              WRITE(IPOD,'(1X,A20,1X)',ADVANCE='NO') TRIM(ATEMP20)
            END DO
          END IF
          WRITE(IPOD,'(A)') ' '
          WRITE(IPPD) (PARPSISUB(I),I=1,NOBSGP)
          IF(NOBS_REPORT.NE.0)THEN
            WRITE(IPPD) (REFOBS(IOBS_REPORT(I)),I=1,NOBS_REPORT)
          END IF
          IF(SVDA.EQ.0)THEN
            WRITE(IPPD) (PVAL(I),I=1,NPAR)
            CALL PRMSAV(JFAIL,NPAR,APAR,PVAL,CLINE,APRECS,APOINT,SCALE,
     +      OFFSET,IOPT)
            IF(JFAIL.NE.0) GO TO 9891
          ELSE
            IF(IOPT.EQ.0)THEN
              WRITE(IPPD) (BPVAL(I),I=1,BNPAR)
            END IF
          END IF
#ifdef FLUSHFILE
          CALL FLUSH(IPPD)
          CALL FLUSH(IPOD)
#endif
        END IF

        IF(PESTMODE.NE.4)THEN
          IF((IPRED.EQ.0).AND.(PSI.EQ.0.0D0)) THEN
            IFIN=2
            GO TO 6000
          END IF
          IF(IREG.EQ.0)THEN
            IF(PSI.LE.PHISTOPTHRESH)THEN
              IFIN=14
              GO TO 6000
            END IF
            IF(PHIABANDON(0).GT.0.0D0)THEN
              IF(PSI.GE.PHIABANDON(0))THEN
                IFIN=15
                PHIAB=PHIABANDON(0)
                GO TO 6000
              END IF
            END IF
          ELSE
            IF(PHIM.LE.PHISTOPTHRESH)THEN
              IFIN=14
              GO TO 6000
            END IF
            IF(PHIABANDON(0).GT.0.0D0)THEN
              IF(PHIM.GE.PHIABANDON(0))THEN
                IFIN=15
                PHIAB=PHIABANDON(0)
                GO TO 6000
              END IF
            END IF
          END IF
        END IF
#ifndef PARALLEL
        IF(ISTOP.EQ.2) THEN
          IFIN=10
          GO TO 6000
        ELSE IF(ISTOP.EQ.1) THEN
          IPFAIL=-1
          GO TO 9891
        END IF
#endif

C -- IF REQUIRED THE JUPITER _PE AND _SS FILES ARE OPENED.

        IF((JUP_PAR_FILES.NE.0).AND.(J_OUTNAM.NE.' '))THEN
          JUPFILE=TRIM(J_OUTNAM)//'._pe'
          CALL FFOPEN(JFAIL,J_PE,'w',JUPFILE,6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(J_PE,104,ADVANCE='NO')
104       FORMAT(1X,'"ITERATION"',1X)
          DO I=1,NPAR
            IF(ITRANS(I).LT.0) CYCLE
            WRITE(J_PE,103,ADVANCE='NO') APAR(I)
103         FORMAT('"',A,'"',1X)
          END DO
          WRITE(J_PE,'(A)') ' '
          JUPFILE=TRIM(J_OUTNAM)//'._ss'
          CALL FFOPEN(JFAIL,J_SS,'w',JUPFILE,6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(J_SS,106)
106       FORMAT(1X,'"ITERATION" "SSWR-(DEPENDENTS ONLY)"',
     +     ' "SSWR-(PRIOR INFORMATION ONLY)"',
     +     ' "SSWR-(TOTAL OBSERVATIONS AND PRIOR INFORMATION)"')
        END IF

        IF(NOPTMAX.EQ.0)THEN
          IFIN=3
          GO TO 6000
        END IF
        IF(PESTMODE.NE.4)THEN
          IF(IREG.EQ.0)THEN
            IF(PSI.LE.PHISTOPTHRESH)THEN
              IFIN=14
              GO TO 6000
            END IF
            IF(PHIABANDON(0).GT.0.0D0)THEN
              IF(PSI.GE.PHIABANDON(0))THEN
                IFIN=15
                PHIAB=PHIABANDON(0)
                GO TO 6000
              END IF
            END IF
          ELSE
            IF(PHIM.LE.PHISTOPTHRESH)THEN
              IFIN=14
              GO TO 6000
            END IF
            IF(PHIABANDON(0).GT.0.0D0)THEN
              IF(PHIM.GE.PHIABANDON(0))THEN
                IFIN=15
                PHIAB=PHIABANDON(0)
                GO TO 6000
              END IF
            END IF
          END IF
        END IF
        NORM(1,1)=-1.0D30
        NORMLO(1,1)=-1.0D30

        JPP0LT=-1
        JPP1LT=-1
        JPP2LT=-1
        JPP0LO=-1
        JPP1LO=-1
        JPP2LO=-1
        NREVLO=1
        LAMLO=RLAMBDA1

C -- THE START OF A NEW OPTIMISATION ITERATION

100     IOPT=IOPT+1
        JACUPCOUNT=0
        LINESAVE=0
        SUCCESSCOUNT=0
        FAILURECOUNT=0

C -- THE REI FILE IS WRITTEN.

        IF(PESTMODE.NE.4)THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rei'
          CALL FFOPEN(JFAIL,IGNF,'w',' ',6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          CALL REIWRITE(JFAIL,IGNF,IOPT,IREG,NXROW,IPRED,NOBS,
     +    NOBSGP,NUMCOV,OWGHT,NOBGNM,NOBSCOV,AOBS,OVAL,MODVAL_O,OBGNME,
     +    COVGP)
          IF(JFAIL.NE.0) GO TO 9300
        END IF
        IF(IREI.NE.0)THEN
          CALL WRITINT(AATEMP,IOPT-1)
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rei'//'.'//TRIM(AATEMP)
          CALL FFOPEN(JFAIL,IGNF,'w',' ',6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          CALL REIWRITE(JFAIL,IGNF,IOPT,IREG,NXROW,IPRED,NOBS,
     +    NOBSGP,NUMCOV,OWGHT,NOBGNM,NOBSCOV,AOBS,OVAL,MODVAL_O,OBGNME,
     +    COVGP)
          IF(JFAIL.NE.0) GO TO 9300
        END IF
        REIFLAG=0

C -- IS THIS AN ITERATION FOR SENSITIVITY REUSE?

        ISENREUSE=0
        IF(DOSENREUSE.EQ.'senreuse')THEN
          ISENREUSE=1
          ITEMP=(IOPT-1)/SENALLCALCINT
          ITEMP=1+ITEMP*SENALLCALCINT
          IF(ITEMP.EQ.IOPT)THEN
            ISENREUSE=0
          ELSE
            SENMXREUSE=SENMAXREUSE
            IES=0
            DO IPP=1,NPAR
              IF(ITRANS(IPP).GE.0)THEN
                IES=IES+1
                ITEMP=NOBS
                IF(IPRED.NE.0) ITEMP=NOBS-1
                RTEMP=0.0D0
                IF(IREG.NE.0)THEN
                  DO IROW=1,ITEMP
                    IGP=NOBGNM(IROW)
                    IF(IRGP(IGP).EQ.0)RTEMP=RTEMP+
     +              X(IROW,IES)*X(IROW,IES)*OWGHT(IROW)
                  END DO
                ELSE
                  DO IROW=1,ITEMP
                    RTEMP=RTEMP+X(IROW,IES)*X(IROW,IES)*OWGHT(IROW)
                  END DO
                END IF
                IF(NPRIOR.GT.0)THEN
                  IF(SENPIEXCLUDE.EQ.'no')THEN
                    IF(IREG.NE.0)THEN
                      DO IROW=NOBS+1,NXROW
                        IGP=NOBGNM(IROW)
                        IF(IRGP(IGP).EQ.0)RTEMP=RTEMP+
     +                  X(IROW,IES)*X(IROW,IES)*OWGHT(IROW)
                      END DO
                    ELSE
                      DO IROW=NOBS+1,NXROW
                        RTEMP=RTEMP+X(IROW,IES)*X(IROW,IES)*OWGHT(IROW)
                      END DO
                    END IF
                  END IF
                END IF
                IF(IPRED.NE.0)THEN
                  RTEMP=RTEMP+X(NOBS,IES)*X(NOBS,IES)*SENPREDWEIGHT
                END IF
                SCREUSE(IPP)=RTEMP
              ELSE
                SCREUSE(IPP)=-1.0D300
              END IF
            END DO
            SENMAX=0.0D0
            DO IPP=1,NPAR
              IF(SCREUSE(IPP).GT.SENMAX) SENMAX=SCREUSE(IPP)
            END DO
            IF(SENMAX.EQ.0.0D0)THEN
              ISENREUSE=0
              GO TO 2171
            END IF
            RTEMP=SENMAX*SENRELTHRESH*SENRELTHRESH
            DO IPP=1,NPAR
              IF(SCREUSE(IPP).GT.RTEMP)SCREUSE(IPP)=-1.0D300
            END DO
2172        CONTINUE
            ICOUNT=0
            SENMAX=0.0D0
            DO IPP=1,NPAR
              IF(SCREUSE(IPP).GT.-1.0D299)THEN
                ICOUNT=ICOUNT+1
                IF(SCREUSE(IPP).GE.SENMAX)THEN
                  SENMAX=SCREUSE(IPP)
                  MAXIPP=IPP
                END IF
              END IF
            END DO
            IF(ICOUNT.GT.SENMXREUSE)THEN
              SCREUSE(MAXIPP)=-1.0D300
              GO TO 2172
            END IF
          END IF
2171      CONTINUE
        END IF

C -- THE JUPITER _PE FILE IS UPDATED.

        IF((JUP_PAR_FILES.NE.0).AND.(J_OUTNAM.NE.' '))THEN
          WRITE(J_PE,101,ADVANCE='NO') IOPT-1
101       FORMAT(4X,I5,3X)
          DO I=1,NPAR
            IF(ITRANS(I).LT.0) CYCLE
            WRITE(J_PE,102,ADVANCE='NO') PVAL(I)
102         FORMAT(1PG14.7,1X)
          END DO
          WRITE(J_PE,'(A)') ' '
        END IF

        BESTFRAC=1.0D30
        DECNOF=0
        NEPL23=0
        IF(RSTFLE.EQ.0) GO TO 129

C -- IF NECESSARY, BINARY FILES ARE READ FOR RESTART INFORMATION

110     CONTINUE
        DECNOF=0
        IF(IRST.EQ.0) THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rst'
          FLENME1=CASEFL(:LEN_TRIM(CASEFL))//'.jst'
          INQUIRE(FILE=FLENME,EXIST=LEXIST)
          IF(LEXIST)THEN
            IF(DONTCOPY.EQ.0)THEN
              CALL DELFILE1(JFAIL,FLENME1,CLINE)
              IF(JFAIL.NE.0) GO TO 9891
#ifdef UNIX
#ifdef SYS_FUNCTION
              NNN=SYSTEM('/bin/cp '//FLENME(1:LEN_TRIM(FLENME))//' '//
     +        FLENME1(1:LEN_TRIM(FLENME1))//' > /dev/null')
#else
              CALL SYSTEM('/bin/cp '//FLENME(1:LEN_TRIM(FLENME))//' '//
     +        FLENME1(1:LEN_TRIM(FLENME1))//' > /dev/null')
#endif
#else
#ifdef LAHEY
#ifdef AT
              CALL COPYFILE(FLENME,FLENME1)
#else
              CALL SYSTEM('copy "'//FLENME(1:LEN_TRIM(FLENME))//'" "'//
     +        FLENME1(1:LEN_TRIM(FLENME1))//'" > nul')
#endif
#else
#ifdef SYS_FUNCTION
              NNN=SYSTEM('copy "'//FLENME(1:LEN_TRIM(FLENME))//'" "'//
     +        FLENME1(1:LEN_TRIM(FLENME1))//'" > nul')
#else
              CALL SYSTEM('copy "'//FLENME(1:LEN_TRIM(FLENME))//'" "'//
     +        FLENME1(1:LEN_TRIM(FLENME1))//'" > nul')
#endif
#endif
#endif
            ELSE
              DONTCOPY=0
            END IF
          END IF
          CALL FFOPEN(JFAIL,-IRSF,'w',' ',25,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(IRSF,ERR=9300) IOPT,NESPAR,IPRED,IREG,SVDA
          WRITE(IRSF,ERR=9300)I2OR3,JPP0LT,JPP1LT,LOOPT,NJPPL0,
     +    NJPPL1,JPP0LO,JPP1LO,NPHISTP,ISWTCH,ICHNG,NREV,NREVLO,
     +    OPTSWH,JPP2LT,JPP2LO,NJPPL2,PRECIS,NOPNT,KPRED,SPLITFLAG,
     +    ISPLIT,JSPLIT,JA2
          WRITE(IRSF,ERR=9300) NCALL,JCALL,MMCALL,DDCALL,MMCALLP,
     +    DDCALLP
          WRITE(IRSF,ERR=9300)PSI,LAMBDA,RELPARMAX,FACPARMAX,S,DMX0LT,
     +    DMX1LT,ROLAST,DMX0,DMX1,PSILO,PSIL,NROL,NDMXL0,NDMXL1,RLO,
     +    DMX0LO,DMX1LO,NEWPSI,LAMLST,LAMLO,DMAX2,DMX2,DMX2LT,DMX2LO,
     +    NDMXL2,UPVECBEND,IBOUNDSTICK,ICOVOBS,BESTFRAC,SWITCHFLAG,
     +    SBACKFLAG,LAMFLAG,RLAMFAC
          WRITE(IRSF,ERR=9300)IMCOVFLAG,ICOUNTREGOBS,ICOUNTREGNZ,
     +    NUMEIG,NEWNUMEIG,NUMEIGLST,ALPHA,NEWALPHA,ALPHLST
          WRITE(IRSF,ERR=9300) JACFILE,NUMCOM,MESSFILE,JACMS
          WRITE(IRSF,ERR=9300) RLAMFAC_ADJUST,RLAMFACEXP
          WRITE(IRSF,ERR=9300) RHSDIM,IPASS1,IFLAG_X,IFLAG_NOCOV,
     +    BNPRIORCOUNT,NEWBASEDER
          WRITE(IRSF,ERR=9300) FAILURECOUNT,SUCCESSCOUNT,
     +    FAILUREFLAG,LAMFORGIVE,REPEATFLAG,WORKVEC3DIM
          IF(LSQRMODE.NE.0)THEN
            WRITE(IRSF,ERR=9300) PR_INDEX,NPR_NONZERO
            IF(NPR_NONZERO.NE.0)THEN
              WRITE(IRSF,ERR=9300) (IXPR(I),I=1,NPR_NONZERO),
     +        (IXPC(I),I=1,NPR_NONZERO)
              IF(NESPAR.NE.0)THEN
                WRITE(IRSF,ERR=9300) (IESTRANS(I),I=1,NESPAR)
              END IF
            END IF
          END IF
          WRITE(IRSF,ERR=9300) (PVAL(I),I=1,NPAR),(PVAL_O(I),I=1,NPAR),
     +    (LSTVAL(I),I=1,NPAR),(DERCOM(I),I=1,NPAR)
          WRITE(IRSF,ERR=9300) (ITRANS(I),I=1,NPAR),(JSTK(I),I=1,NPAR),
     +    (PSILOL(I),I=1,NPHISTP)
          WRITE(IRSF,ERR=9300) (OFFSET(I),I=1,NPAR)
          WRITE(IRSF,ERR=9300) (REFOBS(I),I=1,NXROW),
     +    (MODVAL_O(I),I=1,NXROW)
          WRITE(IRSF,ERR=9300) (OVAL(I),I=1,NXROW),(OWGHT(I),I=1,NXROW)
          WRITE(IRSF,ERR=9300) MAXCOMPDIM
          IF(NPRIOR.NE.0)THEN
            IF(MAXCOMPDIM.LE.1)THEN
              ICOUNT=0
              DO IES=1,NESPAR
                DO IROW=NOBS+1,NXROW
                  IF(X(IROW,IES).NE.0.0D0) ICOUNT=ICOUNT+1
                END DO
              END DO
              WRITE(IRSF,ERR=9300) ICOUNT
              DO IES=1,NESPAR
                DO IROW=NOBS+1,NXROW
                  IF(X(IROW,IES).NE.0.0D0) THEN
                    WRITE(IRSF,ERR=9300) IROW,IES,X(IROW,IES)
                  END IF
                END DO
              END DO
            ELSE
              WRITE(IRSF,ERR=9300) NCOMPDIM
              DO I=1,NCOMPDIM
                WRITE(IRSF,ERR=9300) IXC(I),XC(I)
              END DO
            END IF
          END IF
          IF(NMLODIM2_FLAG.EQ.0)THEN
          WRITE(IRSF,ERR=9300) ((NORMLO(I,J),I=1,NMLODIM2),J=1,NMLODIM2)
          ELSE
          DTEMP=NORMLO(1,1)
          WRITE(IRSF,ERR=9300) ((DTEMP,I=1,NMLODIM2),J=1,NMLODIM2)
          END IF
          IF(NUMCOV.GT.0)THEN
            WRITE(IRSF,ERR=9300) NCOVDIM1,NCOVDIM2,NCOVDIM3
            DO 7000 JCOV=1,NUMCOV
              IF(NOBSCOV(JCOV).GT.0)THEN
                WRITE(IRSF,ERR=9300)
     +          ((COVAR(JCOV)%DVAL(I,J),I=1,NOBSCOV(JCOV)),
     +          J=1,NOBSCOV(JCOV))
              END IF
7000        CONTINUE
          END IF
          IF(IPRED.NE.0) THEN
            WRITE(IRSF,ERR=9300) PDFLAG,NPREDSTP,NPREDMAXMIN,
     +      NPREDNORED,
     +      IFIN1,IPRED,JPREDICT,PNOBGNM,PREDNOISE
            WRITE(IRSF,ERR=9300) NEWPRED,TEMPRD,PRED1,TMPRL,RPTEMP,
     +      PREDLO,ABSPREDSTP,RELPREDSTP,REFPRED,PREDL,ABSPREDSWH,
     +      RELPREDSWH,POVAL,POWGHT,E_PRED,EW_PRED,RES_PRED,
     +      E_PREDLO,EW_PREDLO,RES_PREDLO,NEWE_PRED,NEWEW_PRED,
     +      NEWRES_PRED
            WRITE(IRSF,ERR=9300) PDEN,PNUM,PLAMDASH,PD0,RTEMP1,PD1,
     +      PROOT1,PROOT2,PD2,ABSPREDLAM,RELPREDLAM
            WRITE(IRSF,ERR=9300) (PRDLOL(I),I=1,NPREDSTP),
     +      (PC0(I),I=1,NXROW)
            WRITE(IRSF,ERR=9300) (PN(I),I=1,NPAR),(PNTEMP(I),I=1,NPAR)
            WRITE(IRSF,ERR=9300)  PREDHL,PAOBS,PPOSNEG
          END IF
          IF(IREG.NE.0)THEN
            WRITE(IRSF,ERR=9300) WF,PHIR2,PHIM,PHIR,PHIML,PHIRL,
     +      PHIMLO,PHIRLO,TEMPHIM,TEMPHIR,NEWPHIM,NEWPHIR,WFLO,WFSOL,
     +      IREGADJ,IPIVOTREG,NOBSREGUL,WFLAST
            WRITE(IRSF,ERR=9300) (PHIMLOL(I),I=1,NPHISTP),
     +      (PHIRLOL(I),I=1,NPHISTP)
            WRITE(IRSF,ERR=9300) (SEOGP(I),I=1,NOBSGP),
     +      (IRGP(I),I=1,NOBSGP)
            IF(IREGADJ.GT.0)THEN
              WRITE(IRSF,ERR=9300) (OWGHTLO(I),I=1,NXROW)
              IF((IREGADJ.EQ.3).OR.(IREGADJ.EQ.4).OR.
     +           (IREGADJ.EQ.5))THEN
                WRITE(IRSF,ERR=9300) (OWGHTKP(I),I=1,NOBSREGUL)
              END IF
            END IF
            IF(NREGADJPAR.GT.0)THEN
              WRITE(IRSF,ERR=9300) GEOMAVSENS
              WRITE(IRSF,ERR=9300) (ORIGWGHT(I),I=1,NORIGWGHT)
              WRITE(IRSF,ERR=9300) (PRIORPAR(I),I=1,NREGADJPAR)
              WRITE(IRSF,ERR=9300) (IWSENS(I),I=1,NREGADJPAR)
            END IF
            WRITE(IRSF,ERR=9300) IGTARG
            WRITE(IRSF,ERR=9300) (GFAC(I),I=1,NOBSGP)
            WRITE(IRSF,ERR=9300) (OLDGFAC(I),I=1,NOBSGP)
          END IF
          IF(SVDA.NE.0)THEN
            WRITE(IRSF,ERR=9300) BNSTMP
            WRITE(IRSF,ERR=9300) (BITRANS(I),I=1,BNPAR)
            WRITE(IRSF,ERR=9300) (BPVAL(I),I=1,BNPAR),
     +                         (SVAL(I),I=1,BNPAR),
     +                         (SVALUE(I),I=1,NPAR)
            WRITE(IRSF,ERR=9300) ((TBX(J,I),I=1,BNESPAR),J=1,NPAR)
          END IF
          WRITE(IRSF,ERR=9300)DOSENREUSE
          WRITE(IRSF,ERR=9300)SENPIEXCLUDE
          WRITE(IRSF,ERR=9300)SENRELTHRESH,SENPREDWEIGHT,SENMAXREUSE,
     +    SENALLCALCINT,ISENREUSE
          IF(DOSENREUSE.EQ.'senreuse')THEN
            WRITE(IRSF,ERR=9300) (SCREUSE(IPP),IPP=1,NPAR)
          END IF
          IF(PESTMODE.EQ.4)THEN
            WRITE(IRSF,ERR=9300) PARETO_FIN,PAR_TERM_COUNT
            WRITE(IRSF,ERR=9300) PAR_WTFAC,PAR_WTFAC_KP
            WRITE(IRSF,ERR=9300) (PARPSISUB(I),I=1,NOBSGP)
            WRITE(IRSF,ERR=9300) (OWGHTKP(I),I=1,NUM_WEIGHT_ADJUST)
          END IF
          CLOSE(UNIT=IRSF)
        ELSE
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rst'
          IOPT=IIOPT
          IF(IIOPT.LT.0) THEN
            CLOSE(UNIT=IRSF)
            GO TO 4251
          END IF
          READ(IRSF,ERR=9550,END=9600)I2OR3,JPP0LT,JPP1LT,LOOPT,
     +    NJPPL0,NJPPL1,JPP0LO,JPP1LO,NPHISTP,ISWTCH,ICHNG,NREV,
     +    NREVLO,OPTSWH,JPP2LT,JPP2LO,NJPPL2,PRECIS,NOPNT,KPRED,
     +    SPLITFLAG,ISPLIT,JSPLIT,JA2
          READ(IRSF,ERR=9550,END=9600) NCALL,JCALL,MMCALL,DDCALL,
     +    MMCALLP,DDCALLP
          READ(IRSF,ERR=9550,END=9600)PSI,LAMBDA,RELPARMAX,FACPARMAX,S,
     +    DMX0LT,DMX1LT,ROLAST,DMX0,DMX1,PSILO,PSIL,NROL,NDMXL0,
     +    NDMXL1,RLO,DMX0LO,DMX1LO,NEWPSI,LAMLST,LAMLO,DMAX2,DMX2,
     +    DMX2LT,DMX2LO,NDMXL2,UPVECBEND,IBOUNDSTICK,ICOVOBS,BESTFRAC,
     +    SWITCHFLAG,SBACKFLAG,LAMFLAG,RLAMFAC
          READ(IRSF,ERR=9550,END=9600)IMCOVFLAG,ICOUNTREGOBS,
     +    ICOUNTREGNZ,NUMEIG,NEWNUMEIG,NUMEIGLST,ALPHA,NEWALPHA,ALPHLST
          READ(IRSF,ERR=9550,END=9600) JACFILE,NUMCOM,MESSFILE,JACMS
          READ(IRSF,ERR=9550,END=9600) RLAMFAC_ADJUST,RLAMFACEXP
          READ(IRSF,ERR=9550,END=9600) RHSDIM,IPASS1,IFLAG_X,
     +    IFLAG_NOCOV,BNPRIORCOUNT,NEWBASEDER
          READ(IRSF,ERR=9550,END=9600) FAILURECOUNT,SUCCESSCOUNT,
     +    FAILUREFLAG,LAMFORGIVE,REPEATFLAG,WORKVEC3DIM
          IF(LSQRMODE.NE.0)THEN
            READ(IRSF,ERR=9550,END=9600) PR_INDEX,NPR_NONZERO
            IF(NPR_NONZERO.NE.0)THEN
              READ(IRSF,ERR=9550,END=9600) (IXPR(I),I=1,NPR_NONZERO),
     +        (IXPC(I),I=1,NPR_NONZERO)
              IF(NESPAR.NE.0)THEN
                READ(IRSF,ERR=9550,END=9600) (IESTRANS(I),I=1,NESPAR)
              END IF
            END IF
          END IF
          READ(IRSF,ERR=9550,END=9600) (PVAL(I),I=1,NPAR),
     +    (PVAL_O(I),I=1,NPAR),(LSTVAL(I),I=1,NPAR),
     +    (DERCOM(I),I=1,NPAR)
          READ(IRSF,ERR=9550,END=9600) (ITRANS(I),I=1,NPAR),
     +    (JSTK(I),I=1,NPAR),
     +    (PSILOL(I),I=1,NPHISTP)
          READ(IRSF,ERR=9550,END=9600) (OFFSET(I),I=1,NPAR)
          READ(IRSF,ERR=9550,END=9600) (REFOBS(I),I=1,NXROW),
     +    (MODVAL_O(I),I=1,NXROW)
          READ(IRSF,ERR=9550,END=9600) (OVAL(I),I=1,NXROW),
     +    (OWGHT(I),I=1,NXROW)
          READ(IRSF,ERR=9550,END=9600) ITEMP
          IF((ITEMP.GT.1).AND.(MAXCOMPDIM.LE.1)) GO TO 9060
          IF((ITEMP.LE.1).AND.(MAXCOMPDIM.GT.1)) GO TO 9060
          IF(NPRIOR.NE.0)THEN
            IF(MAXCOMPDIM.LE.1)THEN
              DO IES=1,NESPAR
                DO IROW=NOBS+1,NXROW
                  X(IROW,IES)=0.0D0
                END DO
              END DO
              READ(IRSF,ERR=9550,END=9600) ICOUNT
              IF(ICOUNT.GT.0)THEN
                DO I=1,ICOUNT
                  READ(IRSF,ERR=9550,END=9600) IROW,IES,X(IROW,IES)
                END DO
              END IF
            ELSE
              READ(IRSF,ERR=9550,END=9600) NCOMPDIM
              IF(NCOMPDIM.GE.MAXCOMPDIM-3) GO TO 9970
              DO I=1,NCOMPDIM
                READ(IRSF,ERR=9550,END=9600) IXC(I),XC(I)
              END DO
            END IF
          END IF
          IF(NMLODIM2_FLAG.EQ.0)THEN
          READ(IRSF,ERR=9550,END=9600) ((NORMLO(I,J),I=1,NMLODIM2),
     +    J=1,NMLODIM2)
          ELSE
          READ(IRSF,ERR=9550,END=9600) ((DTEMP,I=1,NMLODIM2),
     +    J=1,NMLODIM2)
          NORMLO(1,1)=-1.0D30
          END IF
          IF(NUMCOV.GT.0)THEN
            READ(IRSF,ERR=9550,END=9600) NCOVDIM1,NCOVDIM2,NCOVDIM3
            DO 7010 JCOV=1,NUMCOV
              IF(NOBSCOV(JCOV).GT.0)THEN
                READ(IRSF,ERR=9550,END=9600)
     +          ((COVAR(JCOV)%DVAL(I,J),I=1,NOBSCOV(JCOV)),
     +          J=1,NOBSCOV(JCOV))
              END IF
7010        CONTINUE
          END IF
          IF(IPRED.NE.0)THEN
            READ(IRSF,ERR=9550,END=9600) PDFLAG,NPREDSTP,NPREDMAXMIN,
     +      NPREDNORED,IFIN1,IPRED,JPREDICT,PNOBGNM,PREDNOISE
            READ(IRSF,ERR=9550,END=9600) NEWPRED,TEMPRD,PRED1,TMPRL,
     +      RPTEMP,PREDLO,ABSPREDSTP,RELPREDSTP,REFPRED,PREDL,
     +      ABSPREDSWH,RELPREDSWH,POVAL,POWGHT,E_PRED,EW_PRED,RES_PRED,
     +      E_PREDLO,EW_PREDLO,RES_PREDLO,NEWE_PRED,NEWEW_PRED,
     +      NEWRES_PRED
            READ(IRSF,ERR=9550,END=9600) PDEN,PNUM,PLAMDASH,PD0,RTEMP1,
     +      PD1,PROOT1,PROOT2,PD2,ABSPREDLAM,RELPREDLAM
            READ(IRSF,ERR=9550,END=9600) (PRDLOL(I),I=1,NPREDSTP),
     +      (PC0(I),I=1,NXROW)
            READ(IRSF,ERR=9550,END=9600) (PN(I),I=1,NPAR),
     +      (PNTEMP(I),I=1,NPAR)
            READ(IRSF,ERR=9550,END=9600)  PREDHL,PAOBS,PPOSNEG
          END IF
          IF(IREG.NE.0)THEN
            READ(IRSF,ERR=9550,END=9600) WF,PHIR2,PHIM,PHIR,PHIML,
     +      PHIRL,PHIMLO,PHIRLO,TEMPHIM,TEMPHIR,NEWPHIM,NEWPHIR,WFLO,
     +      WFSOL,IREGADJ,IPIVOTREG,NOBSREGUL,WFLAST
            READ(IRSF,ERR=9550,END=9600) (PHIMLOL(I),I=1,NPHISTP),
     +      (PHIRLOL(I),I=1,NPHISTP)
            READ(IRSF,ERR=9550,END=9600) (SEOGP(I),I=1,NOBSGP),
     +      (IRGP(I),I=1,NOBSGP)
            IF(IREGADJ.GT.0)THEN
              READ(IRSF,ERR=9550,END=9600) (OWGHTLO(I),I=1,NXROW)
              IF((IREGADJ.EQ.3).OR.(IREGADJ.EQ.4).OR.
     +           (IREGADJ.EQ.5))THEN
                READ(IRSF,ERR=9550,END=9600) (OWGHTKP(I),I=1,NOBSREGUL)
              END IF
            END IF
            IF(NREGADJPAR.GT.0)THEN
              READ(IRSF,ERR=9550,END=9600) GEOMAVSENS
              READ(IRSF,ERR=9550,END=9600) (ORIGWGHT(I),I=1,NORIGWGHT)
              READ(IRSF,ERR=9550,END=9600) (PRIORPAR(I),I=1,NREGADJPAR)
              READ(IRSF,ERR=9550,END=9600) (IWSENS(I),I=1,NREGADJPAR)
            END IF
            READ(IRSF,ERR=9550,END=9600) IGTARG
            READ(IRSF,ERR=9550,END=9600) (GFAC(I),I=1,NOBSGP)
            READ(IRSF,ERR=9550,END=9600) (OLDGFAC(I),I=1,NOBSGP)
          END IF
          IF(SVDA.NE.0)THEN
            READ(IRSF,ERR=9550,END=9600) BNSTMP
            READ(IRSF,ERR=9550,END=9600) (BITRANS(I),I=1,BNPAR)
            READ(IRSF,ERR=9550,END=9600) (BPVAL(I),I=1,BNPAR),
     +                         (SVAL(I),I=1,BNPAR),
     +                         (SVALUE(I),I=1,NPAR)
            READ(IRSF,ERR=9550,END=9600)
     +      ((TBX(J,I),I=1,BNESPAR),J=1,NPAR)
          END IF
          READ(IRSF,ERR=9550,END=9600)DOSENREUSE
          READ(IRSF,ERR=9550,END=9600)SENPIEXCLUDE
          READ(IRSF,ERR=9550,END=9600)SENRELTHRESH,SENPREDWEIGHT,
     +    SENMAXREUSE,SENALLCALCINT,ISENREUSE
          IF(DOSENREUSE.EQ.'senreuse')THEN
            READ(IRSF,ERR=9550,END=9600) (SCREUSE(IPP),IPP=1,NPAR)
          END IF
          IF(PESTMODE.EQ.4)THEN
            READ(IRSF,ERR=9550,END=9600) PARETO_FIN,PAR_TERM_COUNT
            READ(IRSF,ERR=9550,END=9600) PAR_WTFAC,PAR_WTFAC_KP
            READ(IRSF,ERR=9550,END=9600) (PARPSISUB(I),I=1,NOBSGP)
            READ(IRSF,ERR=9550,END=9600)
     +      (OWGHTKP(I),I=1,NUM_WEIGHT_ADJUST)
          END IF
          CLOSE(UNIT=IRSF)

4251      CONTINUE
          IF(IRST.EQ.2)THEN
            FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.jac'
            CALL FFOPEN(JFAIL,-IRSF,'r',' ',131,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            READ(IRSF) ITEMP
            CLOSE(UNIT=IRSF)
            IF(ITEMP.EQ.IOPT) GO TO 4252
            FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.jst'
            CALL FFOPEN(JFAIL,-IRSF,'r',' ',22,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            READ(IRSF,ERR=9550,END=9600) IOPT,NESPAR,JJPRED,JJREG,JSVDA
            IF(JJPRED.NE.IPRED) THEN
              IF(JJPRED.EQ.0) THEN
                CALL STPERR(139,0,' ',0,' ',CLINE)
                GO TO 9891
              END IF
              IF(JJPRED.NE.0) THEN
                CALL STPERR(138,0,' ',0,' ',CLINE)
                GO TO 9891
              END IF
            END IF
            IF(JSVDA.NE.SVDA)THEN
              WRITE(ERRMSG,2029)
              GO TO 9890
            END IF
            IF(JJREG.NE.IREG)THEN
              IF(JJREG.EQ.0) THEN
                CALL STPERR(149,0,' ',0,' ',CLINE)
                GO TO 9891
              END IF
              IF(JJREG.NE.0) THEN
                CALL STPERR(148,0,' ',0,' ',CLINE)
                GO TO 9891
              END IF
            END IF
            READ(IRSF,ERR=9550,END=9600)I2OR3,JPP0LT,JPP1LT,LOOPT,
     +      NJPPL0,NJPPL1,JPP0LO,JPP1LO,NPHISTP,ISWTCH,ICHNG,NREV,
     +      NREVLO,OPTSWH,JPP2LT,JPP2LO,NJPPL2,PRECIS,NOPNT,KPRED,
     +      SPLITFLAG,ISPLIT,JSPLIT,JA2
            READ(IRSF,ERR=9550,END=9600) ITEMP,ITEMP,ITEMP,ITEMP,ITEMP,
     +      ITEMP
            READ(IRSF,ERR=9550,END=9600)PSI,LAMBDA,RELPARMAX,FACPARMAX,
     +      S,DMX0LT,DMX1LT,ROLAST,DMX0,DMX1,PSILO,PSIL,NROL,NDMXL0,
     +      NDMXL1,RLO,DMX0LO,DMX1LO,NEWPSI,LAMLST,LAMLO,DMAX2,DMX2,
     +      DMX2LT,DMX2LO,NDMXL2,UPVECBEND,IBOUNDSTICK,ICOVOBS,BESTFRAC,
     +      SWITCHFLAG,SBACKFLAG,LAMFLAG,RLAMFAC
            READ(IRSF,ERR=9550,END=9600)IMCOVFLAG,ICOUNTREGOBS,
     +      ICOUNTREGNZ,NUMEIG,NEWNUMEIG,NUMEIGLST,ALPHA,NEWALPHA,
     +      ALPHLST
            READ(IRSF,ERR=9550,END=9600) JACFILE,NUMCOM,MESSFILE,JACMS
            READ(IRSF,ERR=9550,END=9600) RLAMFAC_ADJUST,RLAMFACEXP
            READ(IRSF,ERR=9550,END=9600) RHSDIM,IPASS1,IFLAG_X,
     +      IFLAG_NOCOV,BNPRIORCOUNT,NEWBASEDER
            READ(IRSF,ERR=9550,END=9600) FAILURECOUNT,SUCCESSCOUNT,
     +      FAILUREFLAG,LAMFORGIVE,REPEATFLAG,WORKVEC3DIM
            IF(LSQRMODE.NE.0)THEN
              READ(IRSF,ERR=9550,END=9600) PR_INDEX,NPR_NONZERO
              IF(NPR_NONZERO.NE.0)THEN
                READ(IRSF,ERR=9550,END=9600) (IXPR(I),I=1,NPR_NONZERO),
     +          (IXPC(I),I=1,NPR_NONZERO)
                IF(NESPAR.NE.0)THEN
                  READ(IRSF,ERR=9550,END=9600) (IESTRANS(I),I=1,NESPAR)
                END IF
              END IF
            END IF
            READ(IRSF,ERR=9550,END=9600) (PVAL(I),I=1,NPAR),
     +      (PVAL_O(I),I=1,NPAR),(LSTVAL(I),I=1,NPAR),
     +      (DERCOM(I),I=1,NPAR)
            READ(IRSF,ERR=9550,END=9600) (ITRANS(I),I=1,NPAR),
     +      (JSTK(I),I=1,NPAR),
     +      (PSILOL(I),I=1,NPHISTP)
            READ(IRSF,ERR=9550,END=9600) (OFFSET(I),I=1,NPAR)
            READ(IRSF,ERR=9550,END=9600) (REFOBS(I),I=1,NXROW),
     +      (MODVAL_O(I),I=1,NXROW)
            READ(IRSF,ERR=9550,END=9600) (OVAL(I),I=1,NXROW),
     +      (OWGHT(I),I=1,NXROW)
            READ(IRSF,ERR=9550,END=9600) ITEMP
            IF((ITEMP.GT.1).AND.(MAXCOMPDIM.LE.1)) GO TO 9060
            IF((ITEMP.LE.1).AND.(MAXCOMPDIM.GT.1)) GO TO 9060
            IF(NPRIOR.NE.0)THEN
              IF(MAXCOMPDIM.LE.1)THEN
                DO IES=1,NESPAR
                  DO IROW=NOBS+1,NXROW
                    X(IROW,IES)=0.0D0
                  END DO
                END DO
                READ(IRSF,ERR=9550,END=9600) ICOUNT
                IF(ICOUNT.GT.0)THEN
                  DO I=1,ICOUNT
                    READ(IRSF,ERR=9550,END=9600) IROW,IES,X(IROW,IES)
                  END DO
                END IF
              ELSE
                READ(IRSF,ERR=9550,END=9600) NCOMPDIM
                IF(NCOMPDIM.GE.MAXCOMPDIM-3) GO TO 9970
                DO I=1,NCOMPDIM
                  READ(IRSF,ERR=9550,END=9600) IXC(I),XC(I)
                END DO
              END IF
            END IF
            IF(NMLODIM2_FLAG.EQ.0)THEN
            READ(IRSF,ERR=9550,END=9600) ((NORMLO(I,J),I=1,NMLODIM2),
     +      J=1,NMLODIM2)
            ELSE
            READ(IRSF,ERR=9550,END=9600) ((DTEMP,I=1,NMLODIM2),
     +      J=1,NMLODIM2)
            NORMLO(1,1)=-1.0D30
            END IF
            IF(NUMCOV.GT.0)THEN
              READ(IRSF,ERR=9550,END=9600) NCOVDIM1,NCOVDIM2,NCOVDIM3
              DO 7020 JCOV=1,NUMCOV
              IF(NOBSCOV(JCOV).GT.0)THEN
                READ(IRSF,ERR=9550,END=9600)
     +          ((COVAR(JCOV)%DVAL(I,J),I=1,NOBSCOV(JCOV)),
     +          J=1,NOBSCOV(JCOV))
              END IF
7020          CONTINUE
            END IF
            IF(IPRED.NE.0)THEN
              READ(IRSF,ERR=9550,END=9600) PDFLAG,NPREDSTP,NPREDMAXMIN,
     +        NPREDNORED,IFIN1,IPRED,JPREDICT,PNOBGNM,PREDNOISE
              READ(IRSF,ERR=9550,END=9600) NEWPRED,TEMPRD,PRED1,TMPRL,
     +        RPTEMP,PREDLO,ABSPREDSTP,RELPREDSTP,REFPRED,PREDL,
     +        ABSPREDSWH,RELPREDSWH,POVAL,POWGHT,E_PRED,EW_PRED,
     +        RES_PRED,
     +        E_PREDLO,EW_PREDLO,RES_PREDLO,NEWE_PRED,NEWEW_PRED,
     +        NEWRES_PRED
              READ(IRSF,ERR=9550,END=9600) PDEN,PNUM,PLAMDASH,PD0,
     +        RTEMP1,PD1,PROOT1,PROOT2,PD2,ABSPREDLAM,RELPREDLAM
              READ(IRSF,ERR=9550,END=9600) (PRDLOL(I),I=1,NPREDSTP),
     +        (PC0(I),I=1,NXROW)
              READ(IRSF,ERR=9550,END=9600) (PN(I),I=1,NPAR),
     +        (PNTEMP(I),I=1,NPAR)
              READ(IRSF,ERR=9550,END=9600)  PREDHL,PAOBS,PPOSNEG
            END IF
            IF(IREG.NE.0)THEN
              READ(IRSF,ERR=9550,END=9600) WF,PHIR2,PHIM,PHIR,PHIML,
     +        PHIRL,PHIMLO,PHIRLO,TEMPHIM,TEMPHIR,NEWPHIM,NEWPHIR,WFLO,
     +        WFSOL,IREGADJ,IPIVOTREG,NOBSREGUL,WFLAST
              READ(IRSF,ERR=9550,END=9600) (PHIMLOL(I),I=1,NPHISTP),
     +        (PHIRLOL(I),I=1,NPHISTP)
              READ(IRSF,ERR=9550,END=9600) (SEOGP(I),I=1,NOBSGP),
     +        (IRGP(I),I=1,NOBSGP)
              IF(IREGADJ.GT.0)THEN
                READ(IRSF,ERR=9550,END=9600) (OWGHTLO(I),I=1,NXROW)
              IF((IREGADJ.EQ.3).OR.(IREGADJ.EQ.4).OR.
     +           (IREGADJ.EQ.5))THEN
                  READ(IRSF,ERR=9550,END=9600)
     +            (OWGHTKP(I),I=1,NOBSREGUL)
                END IF
              END IF
              IF(NREGADJPAR.GT.0)THEN
                READ(IRSF,ERR=9550,END=9600) GEOMAVSENS
                READ(IRSF,ERR=9550,END=9600)
     +          (ORIGWGHT(I),I=1,NORIGWGHT)
                READ(IRSF,ERR=9550,END=9600)
     +          (PRIORPAR(I),I=1,NREGADJPAR)
                READ(IRSF,ERR=9550,END=9600)
     +          (IWSENS(I),I=1,NREGADJPAR)
              END IF
              READ(IRSF,ERR=9550,END=9600) IGTARG
              READ(IRSF,ERR=9550,END=9600) (GFAC(I),I=1,NOBSGP)
              READ(IRSF,ERR=9550,END=9600) (OLDGFAC(I),I=1,NOBSGP)
            END IF
           IF(SVDA.NE.0)THEN
              READ(IRSF,ERR=9550,END=9600) BNSTMP
              READ(IRSF,ERR=9550,END=9600) (BITRANS(I),I=1,BNPAR)
              READ(IRSF,ERR=9550,END=9600) (BPVAL(I),I=1,BNPAR),
     +                           (SVAL(I),I=1,BNPAR),
     +                           (SVALUE(I),I=1,NPAR)
              READ(IRSF,ERR=9550,END=9600)
     +        ((TBX(J,I),I=1,BNESPAR),J=1,NPAR)
            END IF
            READ(IRSF,ERR=9550,END=9600)DOSENREUSE
            READ(IRSF,ERR=9550,END=9600)SENPIEXCLUDE
            READ(IRSF,ERR=9550,END=9600)SENRELTHRESH,SENPREDWEIGHT,
     +      SENMAXREUSE,SENALLCALCINT,ISENREUSE
            IF(DOSENREUSE.EQ.'senreuse')THEN
              READ(IRSF,ERR=9550,END=9600) (SCREUSE(IPP),IPP=1,NPAR)
            END IF
            IF(PESTMODE.EQ.4)THEN
              READ(IRSF,ERR=9550,END=9600) PARETO_FIN,PAR_TERM_COUNT
              READ(IRSF,ERR=9550,END=9600) PAR_WTFAC,PAR_WTFAC_KP
              READ(IRSF,ERR=9550,END=9600) (PARPSISUB(I),I=1,NOBSGP)
              READ(IRSF,ERR=9550,END=9600)
     +        (OWGHTKP(I),I=1,NUM_WEIGHT_ADJUST)
            END IF
            CLOSE(UNIT=IRSF)
            DONTCOPY=1
4252        CONTINUE
            IIRST=2
          END IF

C -- IF ANY PARAMETERS ARE STUCK TO THEIR BOUNDS AND WE ARE DOING A /R
C    RESTART THE JACOBIAN MATRIX MUST BE READ LATER.

          IF(IRST.EQ.1)THEN
            IF(IBOUNDSTICK.NE.0)THEN
              DO 4255 I=1,NPAR
                IF(JSTK(I).NE.0)THEN
                  IRST1JAC=1
                  GO TO 4254
                END IF
4255          CONTINUE
4254          CONTINUE
            END IF
          END IF
         IRST=0
        END IF
129     CONTINUE
        CALL WRITINT(ATEMP12,IOPT)
        WRITE(IREC,131,ERR=9350) TRIM(ATEMP12)
131     FORMAT(/,/,' OPTIMISATION ITERATION NO.',T36,': ',A)
        IF(CONJGRAD.EQ.1)THEN
          WRITE(CGNOUT,142,ERR=9710)
142       FORMAT(/,/,1X,78('-'))
          WRITE(CGNOUT,143,ERR=9710) IOPT
143       FORMAT(' OPTIMISATION ITERATION NO.',T36,': ',I4)
#ifdef FLUSHFILE
          CALL FLUSH(CGNOUT)
#endif
        ELSE IF(SVDMODE.NE.0)THEN
          WRITE(SVDUNIT,142,ERR=9710)
          WRITE(SVDUNIT,143,ERR=9710) IOPT
#ifdef FLUSHFILE
          CALL FLUSH(SVDUNIT)
#endif
        ELSE IF(LSQRMODE.NE.0)THEN
          IF(LSQR_NOUT.NE.0)THEN
            WRITE(LSQR_NOUT,142,ERR=9710)
            WRITE(LSQR_NOUT,143,ERR=9710) IOPT
#ifdef FLUSHFILE
            CALL FLUSH(LSQR_NOUT)
#endif
          END IF
        ELSE
          WRITE(NCONDUNIT,142,ERR=9710)
          WRITE(NCONDUNIT,143,ERR=9710) IOPT
#ifdef FLUSHFILE
          CALL FLUSH(NCONDUNIT)
#endif
        END IF

#ifdef PARALLEL
#ifndef MPEST
        WRITE(IRMR,128) IOPT
#endif
#endif
#ifdef UNIX
        WRITE(IJRF,127) IOPT
127     FORMAT(/,/,' OPTIMISATION ITERATION NO. ',I3,' ----->')
#endif
        WRITE(6,131) TRIM(ATEMP12)
        CALL WRITINT(ATEMP12,NCALL)
        WRITE(6,132) TRIM(ATEMP12)
c        WRITE(6,133) PSI
        WRITE(IREC,132,ERR=9350) TRIM(ATEMP12)
132     FORMAT(T5,'Model calls so far',T36,': ',A)
        IF(JACFILE.NE.0)THEN
          CALL WRITINT(ATEMP12,JCALL)
          WRITE(6,4120) TRIM(ATEMP12)
          WRITE(IREC,4120,ERR=9350) TRIM(ATEMP12)
4120      FORMAT(T5,'Derivative model calls so far',T36,': ',A)
        END IF
C        IF((((NOBSGP.EQ.1).AND.(IPRED.EQ.0)).OR.
C     +     ((NOBSGP.EQ.2).AND.(IPRED.NE.0))).AND.(NPRIOR.EQ.0))THEN

C -- If PEST is running in Pareto mode the weight factor for this iteration is evaluated.

        PARETO_FIN=0
        IF(PESTMODE.EQ.4)THEN
          J=0
          DO I=1,IOPT
            IF(I.LT.1+NUM_ITER_START)THEN
              CONTINUE
            ELSE
              IF(J.EQ.0)THEN
                J=1
                II=I
              ELSE
                IF(J.LT.NUM_WTFAC_INC)THEN
                  IF(I-II.GE.NUM_ITER_GEN)THEN
                    J=J+1
                    II=I
                  END IF
                END IF
              END IF
            END IF
            IF(J.EQ.NUM_WTFAC_INC)THEN
              IF(I-II+1.EQ.NUM_ITER_FIN) PARETO_FIN=1
            ELSE IF(J.EQ.NUM_WTFAC_INC-1)THEN
              IF(NUM_ITER_FIN.EQ.0)THEN
                IF(I-II+1.EQ.NUM_ITER_GEN) PARETO_FIN=1
              END IF
            END IF
          END DO
          PAR_WTFAC=PAR_WTFAC_START+(PAR_WTFAC_FIN-PAR_WTFAC_START)*
     +    FLOAT(J)/FLOAT(NUM_WTFAC_INC)
          ATEMP12=OBGNME(PAR_NOBGNM)
          WRITE(6,7551) TRIM(ATEMP12),PAR_WTFAC
7551      FORMAT(/,T5,'Pareto weight factor for observation group "',A,
     +    '"'T62,': ',1PG12.5,/)
          WRITE(IREC,7551,ERR=9350) TRIM(ATEMP12),PAR_WTFAC
          PAR_WTFAC=PAR_WTFAC*PAR_WTFAC
          J=0
          DO I=1,NXROW
            IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
              J=J+1
              OWGHT(I)=OWGHTKP(J)*PAR_WTFAC
            END IF
          END DO
          CALL OBJCLC(1,PSI,NXROW,NPRIOR,NOBS,REFOBS,OVAL,OWGHT,
     +    NOBGNM,NOBSGP,IRGP,SUM1)
        END IF

        IF((NOBSGP.EQ.1).AND.(IPRED.EQ.0).AND.(NPRIOR.EQ.0))THEN
          WRITE(IREC,133,ERR=9350) PSI
          WRITE(6,133) PSI
133       FORMAT(T5,'Starting phi for this iteration: ',1PG12.5)
        ELSE
          IF(IREG.EQ.1)THEN
            WRITE(IREC,5052,ERR=9350) WF
            WRITE(6,5052) WF
5052        FORMAT(T5,'Current regularisation weight factor',T62,': ',
     +      1PG12.5)
          END IF
          IF(IREG.EQ.0)THEN
            WRITE(IREC,137,ERR=9350) PSI
            WRITE(6,137) PSI
137         FORMAT(T5,'Starting phi for this iteration',T62,': ',
     +      1PG12.5)
          END IF
          IF(PREDNOISE.GT.0)THEN
            WRITE(IREC,138,ERR=9350) RES_PRED
            WRITE(6,138) RES_PRED
138         FORMAT(T5,'Contribution to phi from predictive error',
     +      T62,': ',1PG12.5)
          END IF
        END IF

        IF(IREG.NE.0) THEN
          CALL OBJCLC(888888,PHIR2,NXROW,NPRIOR,NOBS,REFOBS,
     +    OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,PHIM)
          PHIR2=PHIR2/WF/WF
          PHIR=PHIR2
          WRITE(IREC,5036,ERR=9350) PHIM
          WRITE(6,5036) PHIM
5036      FORMAT(T5,'Current value of measurement objective function',
     +    T62,': ',1PG12.5)
          WRITE(IREC,5061,ERR=9350) PHIR2
          WRITE(6,5061) PHIR2
5061      FORMAT(T5,'Current value of regularisation objective ',
     +    'function ',T62,': ',1PG12.5)
          IF(IREGADJ.NE.0)THEN
            WRITE(IREC,5062)
            WRITE(6,5062)
5062        FORMAT(T5,'Note: regularisation objective function is not ',
     +      'comparable between',/,T5,'iterations because of ',
     +      'IREGADJ regularisation weights adjustment.')
          END IF
          WRITE(IREC,*,ERR=9350)
          WRITE(6,*)
          WRITE(IREC,137,ERR=9350) PSI
          WRITE(6,137) PSI
        END IF
        IF(((NOBSGP.GT.1).AND.(IPRED.EQ.0)).OR.
     +     ((NOBSGP.GT.2).AND.(IPRED.NE.0)).OR.(NPRIOR.NE.0))THEN
          DO 1256 I=1,NOBSGP
            IF(I.NE.JPREDICT)THEN
              CALL OBJCLC(-I,PSISUB(I),NXROW,NPRIOR,NOBS,REFOBS,OVAL,
     +        OWGHT,NOBGNM,NOBSGP,IRGP,SUM1)
              WRITE(IREC,1257,ERR=9350)
     +        OBGNME(I)(1:LEN_TRIM(OBGNME(I))),PSISUB(I)
              WRITE(6,1257) OBGNME(I)(1:LEN_TRIM(OBGNME(I))),PSISUB(I)
1257          FORMAT(T5,'Contribution to phi from observation group "',
     +        A,'"',T62,': ',1PG12.5)
            END IF
1256      CONTINUE
        END IF
        IF(NPRIOR.NE.0)THEN
          CALL OBJCLC(-999,PSISUB(NOBSGP+1),NXROW,NPRIOR,NOBS,REFOBS,
     +    OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,SUM1)
          IF(PSISUB(NOBSGP+1).GT.-1.0D299)THEN
            WRITE(IREC,1258,ERR=9350) PSISUB(NOBSGP+1)
            WRITE(6,1258) PSISUB(NOBSGP+1)
1258        FORMAT(T5,'Contribution to phi from ungrouped prior ',
     +      'information',T62,': ',1PG12.5)
          END IF
        END IF
        IF(IPRED.NE.0)THEN
          IF(PREDHL.EQ.'l')THEN
            AAMAXMIN='Minimum'
          ELSE
            AAMAXMIN='Maximum'
          ENDIF
          IF(PSILO.GT.PD1)THEN
              WRITE(IREC,1261) AAMAXMIN
              WRITE(6,1261) AAMAXMIN
1261          FORMAT(T5,A,' prediction so far for phi less than PD1',
     +        T62,': min phi > PD1')
          ELSE
            WRITE(IREC,1263)AAMAXMIN,PREDLO
            WRITE(6,1263)AAMAXMIN,PREDLO
1263        FORMAT(T5,A,' prediction so far for phi less than PD1',
     +      T62,': ',1PG12.5)
            IF(PREDNOISE.GT.0)THEN
              WRITE(IREC,1268) E_PREDLO
              WRITE(6,1268) E_PREDLO
1268          FORMAT(T5,'Optimised predictive error',T62,': ',1PG12.5)
            END IF
          END IF
        END IF
        IF(IGTARG.NE.0)THEN
          IF(IOPT.GT.1)THEN
            WRITE(6,5752)
            WRITE(IREC,5752,ERR=9350)
5752        FORMAT(/,'    Measurement objective functions with ',
     +      'original weights:-')
            DO I=1,NOBSGP
              IF(IRGP(I).EQ.0)THEN
                WRITE(IREC,5297,ERR=9350)
     +          TRIM(OBGNME(I)),PSISUB(I)/GFAC(I)
                WRITE(6,5297) TRIM(OBGNME(I)),PSISUB(I)/GFAC(I)
5297            FORMAT('    Measurement phi for observation group "',
     +          A,'"',T62,': ',1PG12.5)
              END IF
            END DO
          END IF
        END IF

#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif

C -- THE JUPITER _SS FILE IS UPDATED.

        IF((JUP_PAR_FILES.NE.0).AND.(J_OUTNAM.NE.' '))THEN
          CALL OBJCLC(4,PSIPI,NXROW,NPRIOR,NOBS,REFOBS,OVAL,OWGHT,
     +    NOBGNM,NOBSGP,IRGP,SUM1)
          WRITE(J_SS,105)IOPT-1,PSI-PSIPI,PSIPI,PSI
105       FORMAT(1X,I10,10X,1PG14.7,15X,1PG14.7,15X,1PG14.7)
        END IF

C -- IF PARAMETER BOUNDS GLUING IS OPERATIONAL, THAT IS NOW DONE.

        IF(IBOUNDSTICK.NE.0)THEN
        DO 1274 IPP=1,NPAR
          IF((ITRANS(IPP).GE.0).OR.(ITRANS(IPP).LT.-100001)) THEN
          IF((PVAL(IPP).LE.PARLBND(IPP)).OR.
     +       (PVAL(IPP).GE.PARUBND(IPP)))
     +       THEN
            IF(JSTK(IPP).GE.0)JSTK(IPP)=JSTK(IPP)+1
            IF((JSTK(IPP).GE.IBOUNDSTICK).AND.(ICOVOBS.EQ.0))THEN
              JSTK(IPP)=-1
              IF(SAME(PVAL(IPP),PARLBND(IPP),1.0D-5))THEN
                ABOUND='lower'
              ELSE
                ABOUND='upper'
              END IF
              WRITE(6,139) APAR(IPP)(1:LEN_TRIM(APAR(IPP))),
     +        ABOUND(1:LEN_TRIM(ABOUND))
              WRITE(IREC,139) APAR(IPP)(1:LEN_TRIM(APAR(IPP))),
     +        ABOUND(1:LEN_TRIM(ABOUND))
139           FORMAT(T5,'Parameter "',A,'" permanently glued to its ',
     +        A,' bound.')
#ifdef FLUSHFILE
              CALL FLUSH(IREC)
#endif
            END IF
          ELSE
            JSTK(IPP)=0
          END IF
          END IF
1274    CONTINUE
        END IF
7550    CONTINUE

        IF(IPRED.GT.0)THEN
          IF(PREDHL.EQ.'l')THEN
            NEWPRED=1.0D300
          ELSE
            NEWPRED=-1.0D300
          END IF
        END IF

        NEWPSI=1.0D300
        NEWPHIM=1.0D300
        NEWPHIR=1.0D300
        JMM=0
        IF(IAUI.EQ.0)THEN
          IF(IOPT.EQ.1) THEN
            LAMBDA=RLAMBDA1
          ELSE
            LAMBDA=LAMLST
            IF(NREV.EQ.0) LAMBDA=LAMBDA/RLAMFAC
            IF((LAMBDA.NE.0.0D0).AND.(LAMBDA.LT.1.0D-20))
     +      LAMBDA=1.0D-20
          END IF
          ALAMBDA=LAMBDA
        ELSE
          LAMBDA=ALAMBDA
        END IF
        IF(DOAUI.EQ.'aui')THEN
          IF(ALAMBDA.GT.1.0D8)ALAMBDA=1.0D8
          IF(LAMBDA.GT.1.0D8)LAMBDA=1.0D8
        END IF
        IF(PESTMODE.EQ.4)THEN
          IF(LAMBDA.EQ.0.0D0)THEN
            IF(RLAMBDA1.NE.0.0D0)LAMBDA=RLAMBDA1
          END IF
        END IF

        PSIL=PSI
        PHIML=PHIM
        PHIRL=PHIR

        IF(IPRED.GT.0)PREDL=REFPRED

C -- If automatic lambda change factor adjustment is operation, this is now done.

        IF(MANYLAM.NE.0)THEN
          LAMBDA=RLAMBDA1
          IF(LAMBDA.EQ.0.0)THEN
            LAMBDA=1.0E-5
          END IF
          IF(RLAMFAC_ADJUST.EQ.0)THEN
            RLAMFAC_ADJUST=1
            RLAMFACEXP=-RLAMFAC
          END IF
        ELSE
          IF(LAMBDA.LT.1.0D-8)LAMBDA=1.0D-8
        END IF

        IF(LAMBDA.NE.0.0D0)THEN
          IF(RLAMFAC_ADJUST.EQ.1)THEN
            IF(LAMBDA.GT.1.0)THEN
              RLAMFAC=LAMBDA**(1.0D0/RLAMFACEXP)
            ELSE IF(LAMBDA.LT.1.0)THEN
              RLAMFAC=(1.0D0/LAMBDA)**(1.0D0/RLAMFACEXP)
            ELSE
              RLAMFAC=2.0                   ! arbitrary
            END IF
            IF(RLAMFAC.LT.2.0)RLAMFAC=2.0
          END IF
        END IF

        J=0
        JJ=0
        DO 130 IPP=1,NPAR
        IF(ITRANS(IPP).LT.-100001) THEN
          ITRANS(IPP)=ITRANS(IPP)+200000
          J=J+1
          IF(JSTK(IPP).LT.0)JJ=JJ+1
        END IF
130     CONTINUE
        IF(J-JJ.GT.0)THEN
          WRITE(IREC,140,ERR=9350)
          WRITE(6,140)
140       FORMAT(T5,'All frozen parameters freed.')
#ifdef FLUSHFILE
          CALL FLUSH(IREC)
#endif
        END IF
        MATSTO=1
        SENSTO=1
        IF(IAUI.NE.0) GO TO 302

C -- BEFORE THE SENSITIVITY MATRIX IS CALCULATED, REFERENCE OBSERATIONS
C    ARE UNROTATED SO THAT THEY CAN BE USED AS A BASIS FOR FINITE-DIFFERENCE
C    DERIVATIVES CALCULATIONS.

        IF(IIRST.NE.2)THEN
        IF(NUMCOV.GT.0)THEN
          DO 6850 JCOV=1,NUMCOV
            IF(NOBSCOV(JCOV).LE.0) GO TO 6850
            IGROUP=COVGP(JCOV)
            I=0
            DO 6860 J=1,NXROW
              IF(NOBGNM(J).EQ.IGROUP)THEN
                I=I+1
                W1(I)=REFOBS(J)
              END IF
6860        CONTINUE
            CALL ROTATE(-1,NOBSCOV(JCOV),NOBSCOV(JCOV),1,1,1,
     +      COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,NXROW,NOBGNM)
            I=0
            DO 6870 J=1,NXROW
              IF(NOBGNM(J).EQ.IGROUP)THEN
                I=I+1
                REFOBS(J)=W1(I)
              END IF
6870        CONTINUE
6850      CONTINUE
        END IF
        END IF

C -- THE SENSITIVITY MATRIX IS NEXT CALCULATED

        J=0
        ISENREUSEFLAG=0
        DO 145 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 145
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 145
        IF(SCALE(IPP).LT.-1.0D35) GO TO 145
        IF(ISENREUSE.NE.0)THEN
          IF(SCREUSE(IPP).GE.0.0D0) THEN
            ISENREUSEFLAG=ISENREUSEFLAG+1
            GO TO 145
          END IF
        END IF
        I=IPARGP(IPP)
        IF(FORCEN(I).EQ.2) THEN
          J=J+1
        ELSE IF(FORCEN(I).EQ.3) THEN
          J=J+2
        ELSE IF(FORCEN(I).EQ.1) THEN
          J=J+I2OR3-1
        ELSE IF(FORCEN(I).EQ.4)THEN
          IF(I2OR3.EQ.2)THEN
            J=J+1
          ELSE IF(I2OR3.EQ.3)THEN
            J=J+4
          END IF
        ELSE IF(FORCEN(I).EQ.5)THEN
          J=J+4
        END IF
        CALL WRITINT(ATEMP1,J)
145     CONTINUE
        IF((SVDA_SUPDERCALC.EQ.1).AND.(IOPT.EQ.1))THEN
          WRITE(6,1512)
1512      FORMAT  (/,'    Calculating Jacobian matrix from ',
     +    'base parameter sensitivities.....')
        ELSE
          IF(JACFILE.EQ.0)THEN
            CALL WRITINT(ATEMP1,J)
            WRITE(6,151) TRIM(ATEMP1)
151         FORMAT(/,'    Calculating Jacobian matrix: running ',
     +      'model ',A,' times .....')
            IF(ISENREUSEFLAG.NE.0)THEN
              WRITE(6,1515)
1515          FORMAT('    (Sensitivity re-use operational this ',
     +        'iteration.)')
              WRITE(IREC,15151)
15151         FORMAT(/,4X,'Sensitivity re-use is operational during ',
     +        'this iteration.')
              WRITE(IREC,15152) TRIM(ATEMP1)
15152         FORMAT(4X,'Model is run ',a,' times for derivatives ',
     +        'calculation.')
            END IF
          ELSE
            WRITE(6,1511)
1511        FORMAT(/,'    Calculating Jacobian matrix .....')
          END IF
        END IF
        JTIMES=J
#ifdef PARALLEL
#ifndef MPEST
        WRITE(IRMR,151) TRIM(ATEMP1)
#endif
#endif
        NRUN=J
        IF((IIRST.EQ.2).OR.(IRST1JAC.EQ.1))THEN
          IF(IIRST.EQ.2)THEN
            WRITE(6,152)
152         FORMAT('    Jacobian already calculated.')
#ifdef PARALLEL
#ifndef MPEST
            WRITE(IRMR,152)
#endif
#endif
          END IF
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.jac'
          CALL FFOPEN(JFAIL,-IRSF,'r',' ',131,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          READ(IRSF,ERR=9750,END=9750) ITEMP,JJPRED,JJREG
          IF(JJPRED.NE.IPRED) THEN
            IF(JJPRED.EQ.0) THEN
              CALL STPERR(139,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
            IF(JJPRED.NE.0) THEN
              CALL STPERR(138,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
          END IF
          IF(JJREG.NE.IREG)THEN
            IF(JJREG.EQ.0) THEN
              CALL STPERR(149,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
            IF(JJREG.NE.0) THEN
              CALL STPERR(148,0,' ',0,' ',CLINE)
              GO TO 9891
            END IF
          END IF
          IF(IIRST.EQ.2)THEN
            IF(ITEMP.NE.IOPT)THEN
              CALL STPERR(127,1,' ',0,' ',CLINE)
              GO TO 9891
            END IF
          END IF
          IF(MAXCOMPDIM.LE.1)THEN
            X=0.0D0                     ! AN ARRAY
            READ(IRSF,ERR=9750,END=9750) ICOUNT
            IF(ICOUNT.NE.0)THEN
              DO I=1,ICOUNT
                READ(IRSF,ERR=9750,END=9750) IROW,IES,X(IROW,IES)
              END DO
            END IF
          ELSE
            READ(IRSF,ERR=9750,END=9750)NCOMPDIM
            IF(NCOMPDIM.GE.MAXCOMPDIM-3) GO TO 9970
            IF(NCOMPDIM.NE.0)THEN
              DO I=1,NCOMPDIM
                READ(IRSF,ERR=9750,END=9750) IXC(I),XC(I)
              END DO
            END IF
          END IF
          READ(IRSF,ERR=9750,END=9750) WFSOL
          CLOSE(UNIT=IRSF)

C -- THE SENSITIVITY MATRIX IS NOW ROTATED.
C    HOWEVER THIS IS ONLY DONE IF OBSERVATION COVARIANCE MATRICES ARE PROVIDED.

          IF(NUMCOV.NE.0)THEN
            MATDIM1=NXROW
            NM=MXOBSCOV
            IPSTART=1
            DO 6913 JCOV=1,NUMCOV
              IGROUP=COVGP(JCOV)
              IF(NPRIOR.NE.0)THEN
                DO 6914 I=NOBS+1,NXROW
                  IF(NOBGNM(I).EQ.IGROUP) GO TO 6913
6914            CONTINUE
              END IF
              DO 6915 I=IPSTART,NXROW
                IF(NOBGNM(I).EQ.IGROUP)THEN
                  CALL ROTATE(2,NOBSCOV(JCOV),NOBSCOV(JCOV),MATDIM1,
     +            NESPAR,IPSTART,COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,
     +            NXROW,NOBGNM)
                  GO TO 6913
                END IF
6915          CONTINUE
6913        CONTINUE
          END IF

          JIRST=1
          IRST1JAC=0
          IF(IIRST.EQ.2) THEN
            IIRST=0
            IF(IREG.EQ.1)THEN
C            IF((IREG.EQ.1).AND.(IREGADJ.NE.0))THEN
              GO TO 6907
            ELSE
              GO TO 302
            END IF
          END IF
        END IF
C -- Note that the above code is repeated below.

C -- If we are using SVD assist, a new parcalc template file is written.

        IF((SVDA.EQ.1).AND.(IOPT.GT.1))THEN
          IF(SVDA_MULBPA.EQ.0)THEN
            CALL BASE_PARDEF(IFAIL,IREC,0,NREGADJPAR,-1,
     +      SVDA_SCALADJ)
          ELSE
            CALL BASE_PARDEF(IFAIL,IREC,0,NREGADJPAR,IOPT-1,
     +      SVDA_SCALADJ)
          END IF
          IF(IFAIL.NE.0) GO TO 9890
        END IF

C -- Now derivatives are calculated.

        IES=0
        IRUN=0
        IIRUN=0
        ISTART_C=1
        DERFLAG=0

C -- If we start with the "/i" switch, we now read the nominated JCO file.

        IF((RESTART.EQ.5).AND.(IOPT.EQ.1))THEN
          CALL READ_JCO_FIRST_FILE(IFAIL)
          IF(IFAIL.NE.0) GO TO 9890
          GO TO 2013
        END IF

C -- If we are using SVDA and super parameter derivatives can be calculated
C    from base parameter derivatives, this is now done.

        IF(SVDA.NE.0)THEN
          IF(SVDA_SUPDERCALC.NE.0)THEN
            IF(IOPT.EQ.1)THEN
              CALL BASE_SUPDERCALC(IFAIL,SVDA_SCALADJ,1)
              IF(IFAIL.EQ.1) THEN
                 GO TO 9970
              ELSE IF(IFAIL.NE.0)THEN
                GO TO 9890
              END IF
              GO TO 2013
            END IF
          END IF
          IF(JACFILE.NE.0)THEN
            IF(IOPT.GT.1)THEN
#ifndef PARALLEL
              NXDIM=NXROW
              TEMPJACDIM=MAX(NXROW,NPAR)
              IF(JACFILE.EQ.2)THEN
                 DIM_DI_PAR=DI_NPAR
                 DIM_DI_OBS=DI_NDEP
                 DIM_DI_IES=DI_NPAR
                 DIM_DI_PARENT=DI_NPAR
              ELSE
                 DIM_DI_PAR=1
                 DIM_DI_OBS=1
                 DIM_DI_IES=1
                 DIM_DI_PARENT=1
              END IF
              IEXTJACMODE=1
              CALL EXTJAC(JFAIL,NXDIM,NOBS,NESPAR,NPAR,
     +        DIM_DI_PAR,DIM_DI_OBS,DIM_DI_IES,DIM_DI_PARENT,
     +        PVAL,ORGVAL,ITRANS,
     +        SCALE,W2,X,COMJAC,EXTDERFLE,CLINE,APAR,AOBS,DERCOM,PRECIS,
     +        NOPNT,NTPLFLE,NW,OFFSET,PARDEL,PWORD,INFLE,TEMPFLE,
     +        MESSFILE,
     +        DI_NSKIP,DI_NDEP,DI_NPAR,DI_ORIENTATION,DI_DERFORMAT,
     +        DI_PAR,
     +        DI_IES,DI_PARENT,DI_OBS,JACFILE,TEMPJACDIM,NUMTIED,W1,
     +        IEXTJACMODE)
              IF(JFAIL.EQ.1) GO TO 9891
              IF(JFAIL.EQ.2) GO TO 9890
#endif

              CALL PROCESS_BASE_PAR_DERIV(IFAIL)
              IF(IFAIL.NE.0) GO TO 9890
              CALL BASE_SUPDERCALC(IFAIL,SVDA_SCALADJ,2)
              IF(IFAIL.EQ.1) THEN
                GO TO 9970
              ELSE IF(IFAIL.NE.0)THEN
                GO TO 9890
              END IF
              GO TO 2013
            END IF
          END IF
        END IF

C -- If necessary the derivatives restart file is opened.

        IDCOUNT=0
        IF(DRST.EQ.1)THEN
          DIRESTART=2
        ELSE
          DIRESTART=1
        END IF
        IF(RSTFLE.EQ.0) DIRESTART=0
        IF(IPRLL.NE.0) DIRESTART=0
        IF(DIRESTART.EQ.2)THEN
          INQUIRE(FILE=DTFILE,EXIST=LEXIST)
          IF(.NOT.LEXIST)THEN
            DIRESTART=1
            DRST=0
          ELSE
            JPERR=1
            CALL FFOPEN(JFAIL,-DTUNIT,'r',DTFILE,22,CLINE)
            JPERR=0
            IF(IPERR.NE.0)THEN
              DIRESTART=1
              DRST=0
            ELSE
              ITEMP=0
              READ(DTUNIT,IOSTAT=IERR) ITEMP
              IF(ITEMP.NE.IOPT)THEN
                CLOSE(UNIT=DTUNIT,IOSTAT=IERR)
                DIRESTART=1
                DRST=0
              END IF
            END IF
          END IF
        END IF
        IF(DIRESTART.EQ.1)THEN
          CALL DELFILE1(JFAIL,DTFILE,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          JPERR=1
          CALL FFOPEN(JFAIL,-DTUNIT,'w',DTFILE,25,CLINE)
          JPERR=0
          IF(IPERR.NE.0)THEN
            DRST=0
            DIRESTART=0
            GO TO 217
          END IF
          WRITE(DTUNIT,IOSTAT=IERR) IOPT
#ifdef FLUSHFILE
          CALL FLUSH(DTUNIT)
#endif
217       CONTINUE
        END IF

C -- Now derivatives are calculated.

C -- But first, if we are reading external derivatives, we zero all existing derivatives
C    except for prior information.
        IF(JACFILE.NE.0)THEN
          IF(MAXCOMPDIM.GT.1)THEN
            CALL ZERO_MATRIX(IFAIL,NCOMPDIM,XC,IXC,NOBS)
          ELSE
            DO IES=1,NESPAR
              DO IROW=1,NOBS
                X(IROW,IES)=0.0D0
              END DO
            END DO
          END IF
        END IF

        DO 200 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 200
        IES=IES+1
        IF((JACFILE.NE.0).AND.(DERCOM(IPP).EQ.0)) GO TO 200
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 200
        IF(SCALE(IPP).LT.-1.0D35)THEN
          ITEMP=NINT(OFFSET(IPP))
          DO IOBS=1,NXROW
C Note - we are assuming that no
C iw_ variable is written to a template file (and hence affects any observations)
C and thus that it is only cited in prior information.
            IF(NOBGNM(IOBS).EQ.ITEMP)THEN
              IF(ITRANS(IPP).EQ.0)THEN
                IF(MAXCOMPDIM.LE.1)THEN
                  X(IOBS,IES)=-1.0/PVAL(IPP)/PVAL(IPP)*
     +            (REFOBS(IOBS)-OVAL(IOBS))
                ELSE
                  RTEMP=-1.0/PVAL(IPP)/PVAL(IPP)*
     +            (REFOBS(IOBS)-OVAL(IOBS))
                  CALL STORE_VALUE(IFAIL,NCOMPDIM,XC,IXC,RTEMP,
     +            IOBS,IES)
                  IF(IFAIL.NE.0) GO TO 9970
                  ISTART_C=IFOUND_C
                END IF
              ELSE
                IF(MAXCOMPDIM.LE.1)THEN
                  X(IOBS,IES)=-2.30259/PVAL(IPP)*
     +            (REFOBS(IOBS)-OVAL(IOBS))
                ELSE
                  RTEMP=-2.30259/PVAL(IPP)*
     +            (REFOBS(IOBS)-OVAL(IOBS))
                  CALL STORE_VALUE(IFAIL,NCOMPDIM,XC,IXC,RTEMP,
     +            IOBS,IES)
                  IF(IFAIL.NE.0) GO TO 9970
                  ISTART_C=IFOUND_C
                END IF
              END IF
            ELSE
              IF(MAXCOMPDIM.LE.1)THEN
                IF(IOBS.LE.NOBS)THEN
                  X(IOBS,IES)=0.0D0
                END IF
              END IF
            END IF
          END DO
          GO TO 200
        END IF

C -- Now we check whether these derivatives are available in a derivatives
C    restart file.

        IF(DIRESTART.EQ.2)THEN
194       CONTINUE
          READ(DTUNIT,END=197,ERR=197) ITEMP,ITEMP1,ITEMP2
          IF(ITEMP.GT.IES) GO TO 197
          IF(MAXCOMPDIM.LE.1)THEN
            READ(DTUNIT,END=197,ERR=197) (X(IOBS,IES),IOBS=1,NOBS)
            IF(ITEMP.LT.IES) GO TO 194
            IDCOUNT=IDCOUNT+1
            NCALL=ITEMP1
            IIRUN=ITEMP2
            GO TO 200
          ELSE
            READ(DTUNIT,END=197,ERR=197) (WORK_C(IOBS),IOBS=1,NOBS)
            IF(ITEMP.LT.IES) GO TO 194
            ISTART_C=1
            DO IOBS=1,NOBS
              CALL STORE_VALUE(IFAIL,NCOMPDIM,XC,IXC,WORK_C(IOBS),
     +        IOBS,IES)
              IF(IFAIL.NE.0) GO TO 9970
              ISTART_C=IFOUND_C
            END DO
            IDCOUNT=IDCOUNT+1
            NCALL=ITEMP1
            IIRUN=ITEMP2
            GO TO 200
          END IF
        END IF

196     CONTINUE

        IF(ISENREUSE.NE.0)THEN
          IF(SCREUSE(IPP).GE.0.0D0) GO TO 1961
        END IF

        IF(MAXCOMPDIM.LE.1)THEN
          CALL DERCLC(JFAIL,DERFLAG,NSCALL,MMCALL,NSCALLP,MMCALLP,
     +    SOPDIM,I2OR3,IPP,NPAR,NOBS,NPARGP,ASIZE,NINSTR,NTPLFLE,
     +    NINSFLE,NUML,NBLBMX,PRECIS,NOPNT,PVAL,PARLBND,PARUBND,ITRANS,
     +    SCALE,OFFSET,IPARGP,W2,NW,PWORD,APAR,AOBS,DERINC,IDBND,
     +    INCTYP,DERINCLB,FORCEN,DERMTHD,DERINCMUL,REFOBS,
     +    X(1,IES),W1,LCINS,LL,OBSN1,OBSN2,IIOBS,INSFLE,TEMPFLE,INFLE,
     +    OUTFLE,PARDEL,MRKDEL,A,CLINE,BUF,COMLIN(DERCOM(IPP)),
     +    NRUN,IRUN,1,INCPAR,
     +    MESSFILE,IIRUN,NORMRETURN,WV5DIM,WORKVEC5,SPLITFLAG)
          IF(JFAIL.NE.0) GO TO 9891
        ELSE
          IF((SPLITFLAG.NE.0).AND.(JSPLIT.NE.0))THEN
            IF((SPLITTHRESH(IPARGP(IPP)).NE.0.0D0).AND.
     +         (SPLITACTION(IPARGP(IPP)).EQ.3))THEN
              IF(NCOMPDIM.EQ.0)THEN
                DO I=1,NOBS
                  WORK_C(I)=0.0D0
                END DO
              ELSE
                ISTART_C=1
                CALL GET_VECTOR(NCOMPDIM,NXROW,XC,IXC,WORK_C,IES)
                ISTART_C=IFOUND_C
              END IF
            END IF
          END IF
          CALL DERCLC(JFAIL,DERFLAG,NSCALL,MMCALL,NSCALLP,MMCALLP,
     +    SOPDIM,I2OR3,IPP,NPAR,NOBS,NPARGP,ASIZE,NINSTR,NTPLFLE,
     +    NINSFLE,NUML,NBLBMX,PRECIS,NOPNT,PVAL,PARLBND,PARUBND,ITRANS,
     +    SCALE,OFFSET,IPARGP,W2,NW,PWORD,APAR,AOBS,DERINC,IDBND,
     +    INCTYP,DERINCLB,FORCEN,DERMTHD,DERINCMUL,REFOBS,
     +    WORK_C,W1,LCINS,LL,OBSN1,OBSN2,IIOBS,INSFLE,TEMPFLE,INFLE,
     +    OUTFLE,PARDEL,MRKDEL,A,CLINE,BUF,COMLIN(DERCOM(IPP)),
     +    NRUN,IRUN,1,INCPAR,
     +    MESSFILE,IIRUN,NORMRETURN,WV5DIM,WORKVEC5,SPLITFLAG)
          IF(JFAIL.NE.0) GO TO 9891
          ISTART_C=1
          DO IOBS=1,NOBS
            CALL STORE_VALUE(IFAIL,NCOMPDIM,XC,IXC,WORK_C(IOBS),
     +      IOBS,IES)
            IF(IFAIL.NE.0) GO TO 9970
            ISTART_C=IFOUND_C
          END DO
        END IF

1961    CONTINUE

C -- If necessary, values are stored in the mid-run restart file.

        IF(DIRESTART.EQ.1)THEN
          IF(NORMRETURN.EQ.1)THEN
            WRITE(DTUNIT,IOSTAT=IERR) IES,NCALL,IIRUN
            IF(MAXCOMPDIM.LE.1)THEN
              WRITE(DTUNIT,IOSTAT=IERR) (X(IOBS,IES),IOBS=1,NOBS)
            ELSE
              WRITE(DTUNIT,IOSTAT=IERR) (WORK_C(IOBS),IOBS=1,NOBS)
            END IF
#ifdef FLUSHFILE
            CALL FLUSH(DTUNIT)
#endif
          END IF
        END IF

C -- Has a stop condition been encountered?

        IF(ISTOP.EQ.2)THEN
          IF(DIRESTART.NE.0) CLOSE(UNIT=DTUNIT,IOSTAT=IERR)
          IFIN=10
          GO TO 6000
        ELSE IF(ISTOP.EQ.1) THEN
          IF(DIRESTART.NE.0) CLOSE(UNIT=DTUNIT,IOSTAT=IERR)
          IPFAIL=-1
          GO TO 9891
        END IF
        GO TO 200

197     CONTINUE
        CALL WRITINT(ATEMP3,IDCOUNT)
        WRITE(6,198) TRIM(ATEMP3)
198     FORMAT('    Derivatives for ',A,' parameters found ',
     + 'in restart file.')
        NOWRITE=0
        IF(IDCOUNT.EQ.NESPAR) NOWRITE=1
#ifdef INTEL
        IF((IDCOUNT.NE.0).AND.(IDCOUNT.NE.NESPAR))THEN
          WRITE(6,2561)
2561      FORMAT('    - number of runs completed...')
          WRITE(6,'(a)',ADVANCE='NO') '     '
        END IF
#endif
        CLOSE(UNIT=DTUNIT,STATUS='DELETE',IOSTAT=IERR)
        IF(IERR.NE.0)THEN
          DIRESTART=0
          DRST=0
        ELSE
          DIRESTART=1
          JPERR=1
          CALL FFOPEN(JFAIL,-DTUNIT,'w',DTFILE,25,CLINE)
          JPERR=0
          IF(IPERR.NE.0) THEN
            DIRESTART=0
            DRST=0
            GO TO 218
          END IF
          WRITE(DTUNIT,ERR=195) IOPT
          IF(IES.GT.1)THEN
            DO I=1,IES-1
              WRITE(DTUNIT,ERR=195) I,NCALL,IIRUN
              IF(MAXCOMPDIM.LE.1)THEN
                WRITE(DTUNIT,ERR=195) (X(IOBS,I),IOBS=1,NOBS)
              ELSE
                CALL GET_VECTOR(NCOMPDIM,NXROW,XC,IXC,WORK_C,I)
                WRITE(DTUNIT,ERR=195) (WORK_C(IOBS),IOBS=1,NOBS)
              END IF
            END DO
          END IF
#ifdef FLUSHFILE
          CALL FLUSH(DTUNIT)
#endif
218       CONTINUE
          GO TO 196
        END IF
195     CLOSE(UNIT=DTUNIT,IOSTAT=IERR)
        DIRESTART=0
        DRST=0
        GO TO 196

200     CONTINUE

C -- Close the derivatives restart file if necessary.

        IF(DIRESTART.NE.0) THEN
          INQUIRE(UNIT=DTUNIT,OPENED=LOPENED)
          IF(LOPENED) THEN
            CLOSE(UNIT=DTUNIT,IOSTAT=IERR)
          END IF
        END IF
        IF(DIRESTART.EQ.2)THEN
          CALL WRITINT(ATEMP3,IDCOUNT)
          WRITE(6,198) TRIM(ATEMP3)
          NOWRITE=0
          IF(IDCOUNT.EQ.NESPAR) NOWRITE=1
#ifdef INTEL
        IF((IDCOUNT.NE.0).AND.(IDCOUNT.NE.NESPAR))THEN
          WRITE(6,2561)
          WRITE(6,'(a)',ADVANCE='NO') '     '
        END IF
#endif
        END IF
        DRST=0

#ifdef PARALLEL

#ifdef BEO
       if (BEOMASTER) then
         call RUNMASTER(PARREG,OBSREG,NRUN,JFAIL)
         if (JFAIL.ne.0) goto 9891
       else
#endif
#ifdef MPEST
        FLENME=TRIM(CASEFL)//'.jacobian_runs'
        CALL FFOPEN(JFAIL,IRMR,'w',' ',6,CLINE)
        IF(JFAIL.NE.0) GO TO 9891
        WRITE(IRMR,'(I5)') IOPT
        RTEMPLAM=LAMLST
        IF(LAMFLAG.EQ.0) RTEMPLAM=-1.0D0
        WRITE(IRMR,'(1X,1PG14.7)') RTEMPLAM
        WRITE(IRMR,'(1X,I5,2X,I5)') SWITCHFLAG,SBACKFLAG
        CLOSE(UNIT=IRMR)
        IF(SWITCHFLAG.EQ.1)SWITCHFLAG=0
        IF(SBACKFLAG.EQ.1)SBACKFLAG=0
        FLENME=TRIM(CASEFL)//'.lambda_runs'
        CALL DELFILE1(JFAIL,FLENME,CLINE)
        CALL DORUNS_M(JFAIL,NSLAVE,NRUN,
     +  ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,LL,OBSN1,
     +  OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,
     +  NPAR,PRECIS,NOPNT,NTPLFLE,NW,SCALE,
     +  OFFSET,PARDEL,PWORD,TEMPFLE,APAR,
     +  OUTFLE,INSFLE,LDSIN,LDSOU,SINFLE,
     +  SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,MSINFLE,MSOUFLE,
     +  MODFLE,WORKVEC7DIM,WORKVEC7)
        IF(JFAIL.EQ.2) GO TO 9890
        IF(JFAIL.NE.0) GO TO 9891
#else
        IF(SRST.EQ.1)THEN
          PIRESTART=2
        ELSE IF(SRST.EQ.0)THEN
          PIRESTART=1
        END IF
        IF(RSTFLE.EQ.0) PIRESTART=0
        CALL DORUNS(JFAIL,NSLAVE,NRUN,ISTATS,ISTATR,
     +  ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,LL,OBSN1,
     +  OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,ISTRTME,
     +  IRUNTME,JRUN,NPAR,PRECIS,NOPNT,NTPLFLE,NW,SCALE,
     +  OFFSET,PARDEL,PWORD,TEMPFLE,APAR,OREADFLE,PREADFLE,
     +  MANFLE,ASLAVE,NNRUN,OUTFLE,INSFLE,LDSIN,LDSOU,SINFLE,
     +  SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,IDET,SREADFLE,MREADFLE,
     +  SCOM,PIRESTART,PTUNIT,PTFILE,IOPT,REPEATRUN,0,SLAVEGROUP,
     +  WORKVEC7DIM,WORKVEC7)
        IF(JFAIL.NE.0) GO TO 9891
#endif
#ifdef BEO
       endif
#endif
        IF(ISTOP.EQ.2) THEN
          IFIN=10
          GO TO 6000
        ELSE IF(ISTOP.EQ.1) THEN
          IPFAIL=-1
          GO TO 9891
        END IF
        SRST=0
        IES=0
        IRUN=0
        DO 201 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 201
        IF(SCALE(IPP).LT.-1.0D35) GO TO 201
        IES=IES+1
        IF((ICOVOBS.EQ.0).AND.(JSTK(IPP).LT.0)) GO TO 201
        IF(ISENREUSE.NE.0)THEN
          IF(SCREUSE(IPP).GE.0.0D0) GO TO 201
        END IF
        IF(MAXCOMPDIM.LE.1)THEN
          CALL DERCLC(JFAIL,DERFLAG,NSCALL,MMCALL,NSCALLP,MMCALLP,
     +    SOPDIM,I2OR3,IPP,NPAR,NOBS,NPARGP,ASIZE,NINSTR,NTPLFLE,
     +    NINSFLE,NUML,NBLBMX,PRECIS,NOPNT,PVAL,PARLBND,PARUBND,
     +    ITRANS,SCALE,OFFSET,IPARGP,W2,NW,PWORD,APAR,AOBS,DERINC,
     +    IDBND,INCTYP,DERINCLB,FORCEN,DERMTHD,DERINCMUL,REFOBS,
     +    X(1,IES),W1,LCINS,LL,OBSN1,OBSN2,IIOBS,INSFLE,TEMPFLE,INFLE,
     +    OUTFLE,PARDEL,MRKDEL,A,CLINE,BUF,MODFLE,
     +    NRUN,IRUN,2,INCPAR,
     +    MESSFILE,IIRUN,NORMRETURN,WV5DIM,WORKVEC5,SPLITFLAG)
          IF(JFAIL.NE.0) GO TO 9891
        ELSE
          IF((SPLITFLAG.NE.0).AND.(JSPLIT.NE.0))THEN
            IF((SPLITTHRESH(IPARGP(IPP)).NE.0.0D0).AND.
     +         (SPLITACTION(IPARGP(IPP)).EQ.3))THEN
              IF(NCOMPDIM.EQ.0)THEN
                DO I=1,NOBS
                  WORK_C(I)=0.0D0
                END DO
              ELSE
                ISTART_C=1
                CALL GET_VECTOR(NCOMPDIM,NXROW,XC,IXC,WORK_C,IES)
                ISTART_C=IFOUND_C
              END IF
            END IF
          END IF
          CALL DERCLC(JFAIL,DERFLAG,NSCALL,MMCALL,NSCALLP,MMCALLP,
     +    SOPDIM,I2OR3,IPP,NPAR,NOBS,NPARGP,ASIZE,NINSTR,NTPLFLE,
     +    NINSFLE,NUML,NBLBMX,PRECIS,NOPNT,PVAL,PARLBND,PARUBND,
     +    ITRANS,SCALE,OFFSET,IPARGP,W2,NW,PWORD,APAR,AOBS,DERINC,
     +    IDBND,INCTYP,DERINCLB,FORCEN,DERMTHD,DERINCMUL,REFOBS,
     +    WORK_C,W1,LCINS,LL,OBSN1,OBSN2,IIOBS,INSFLE,TEMPFLE,INFLE,
     +    OUTFLE,PARDEL,MRKDEL,A,CLINE,BUF,MODFLE,
     +    NRUN,IRUN,2,INCPAR,
     +    MESSFILE,IIRUN,NORMRETURN,WV5DIM,WORKVEC5,SPLITFLAG)
          IF(JFAIL.NE.0) GO TO 9891
          ISTART_C=1
          DO IOBS=1,NOBS
            CALL STORE_VALUE(IFAIL,NCOMPDIM,XC,IXC,WORK_C(IOBS),
     +      IOBS,IES)
            IF(IFAIL.NE.0) GO TO 9970
            ISTART_C=IFOUND_C
          END DO
        END IF
201     CONTINUE
#else

#ifdef LAHEY
        IF(NOWRITE.EQ.0) WRITE(6,*)
#endif

#endif

#ifndef PARALLEL
        IF(JACFILE.NE.0)THEN
          NXDIM=NXROW
          TEMPJACDIM=MAX(NXROW,NPAR)
          IF(JACFILE.EQ.2)THEN
             DIM_DI_PAR=DI_NPAR
             DIM_DI_OBS=DI_NDEP
             DIM_DI_IES=DI_NPAR
             DIM_DI_PARENT=DI_NPAR
           ELSE
             DIM_DI_PAR=1
             DIM_DI_OBS=1
             DIM_DI_IES=1
             DIM_DI_PARENT=1
           END IF
          IEXTJACMODE=0
          CALL EXTJAC(JFAIL,NXDIM,NOBS,NESPAR,NPAR,
     +    DIM_DI_PAR,DIM_DI_OBS,DIM_DI_IES,DIM_DI_PARENT,
     +    PVAL,ORGVAL,ITRANS,
     +    SCALE,W2,X,COMJAC,EXTDERFLE,CLINE,APAR,AOBS,DERCOM,PRECIS,
     +    NOPNT,NTPLFLE,NW,OFFSET,PARDEL,PWORD,INFLE,TEMPFLE,MESSFILE,
     +    DI_NSKIP,DI_NDEP,DI_NPAR,DI_ORIENTATION,DI_DERFORMAT,DI_PAR,
     +    DI_IES,DI_PARENT,DI_OBS,JACFILE,TEMPJACDIM,NUMTIED,W1,
     +    IEXTJACMODE)
          IF(JFAIL.EQ.1) GO TO 9891
          IF(JFAIL.EQ.2) GO TO 9890
        END IF
c        note, in the above, well have to give the "jacobian model" information
c        on which parameters are tied and fixed.

#endif

! -- Set a flag in the PESTDATA module saying that sensitivities are now available.

2013    CONTINUE

! -- Do we need to cease or pause execution?

        CALL STOPRESS(0)
        IF(ISTOP.EQ.2)THEN
          IFIN=10
          GO TO 6000
        ELSE IF(ISTOP.EQ.1) THEN
          IPFAIL=-1
          GO TO 9891
        END IF

        SENFLAG=1

! -- Save restart information.

        IF(RSTFLE.NE.0)THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.jac'
          CALL FFOPEN(JFAIL,-IRSF,'w',' ',25,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(IRSF) IOPT,IPRED,IREG
          IF(MAXCOMPDIM.LE.1)THEN
            ICOUNT=0
            DO IES=1,NESPAR
              DO IROW=1,NXROW
                IF(X(IROW,IES).NE.0.0D0) ICOUNT=ICOUNT+1
              END DO
            END DO
            WRITE(IRSF) ICOUNT
            IF(ICOUNT.GT.0)THEN
              DO IES=1,NESPAR
                DO IROW=1,NXROW
                  IF(X(IROW,IES).NE.0.0D0) WRITE(IRSF) IROW,IES,
     +               X(IROW,IES)
                END DO
              END DO
            END IF
          ELSE
            WRITE(IRSF) NCOMPDIM
            DO I=1,NCOMPDIM
              WRITE(IRSF) IXC(I),XC(I)
            END DO
          END IF
          WRITE(IRSF) WFSOL
        END IF

C -- The JCO file is written if this is the first optimisation iteration.

        IF((NOPTMAX.EQ.-1).OR.(NOPTMAX.EQ.-2).OR.(IOPT.EQ.1))THEN

C -- Prior information is un-rotated.

          IF(NUMCOV.NE.0)THEN
            IF(NPRIOR.NE.0)THEN
              MATDIM1=NXROW
              NM=MXOBSCOV
              IPSTART=NOBS+1
              DO 6899 JCOV=1,NUMCOV
                IGROUP=COVGP(JCOV)
                DO I=IPSTART,NXROW
                  IF(NOBGNM(I).EQ.IGROUP)THEN
                    CALL ROTATE(-2,NOBSCOV(JCOV),NOBSCOV(JCOV),MATDIM1,
     +              NESPAR,IPSTART,COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,
     +              NXROW,NOBGNM)
                    GO TO 6899
                  END IF
                END DO
6899          CONTINUE
            END IF
          END IF
          IFLAG_X=1
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.jco'
          CALL FFOPEN(JFAIL,-IRSF,'w',' ',25,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          CALL JCOWRITE(NPAR,NESPAR,NXROW,IRSF,X,ITRANS,APAR,
     +    AOBS,REFOBS,OVAL,PVAL,SCALE,OFFSET)
          CLOSE(UNIT=IRSF)
          IF(NOPTMAX.EQ.-2) THEN
            IFIN=11
            GO TO 6000
          END IF
          IF(NUMCOV.NE.0)THEN
            IF(NPRIOR.NE.0)THEN
              MATDIM1=NXROW
              NM=MXOBSCOV
              IPSTART=NOBS+1
              DO 6898 JCOV=1,NUMCOV
                IGROUP=COVGP(JCOV)
                DO I=IPSTART,NXROW
                  IF(NOBGNM(I).EQ.IGROUP)THEN
                    CALL ROTATE(2,NOBSCOV(JCOV),NOBSCOV(JCOV),MATDIM1,
     +              NESPAR,IPSTART,COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,
     +              NXROW,NOBGNM)
                    GO TO 6898
                  END IF
                END DO
6898          CONTINUE
            END IF
          END IF
        END IF

C -- THE SENSITIVITY MATRIX IS NOW ROTATED.
C    HOWEVER THIS IS ONLY DONE IF OBSERVATION COVARIANCE MATRICES ARE PROVIDED.

        IF(NUMCOV.NE.0)THEN
          MATDIM1=NXROW
          NM=MXOBSCOV
          IPSTART=1
          DO 6900 JCOV=1,NUMCOV
            IGROUP=COVGP(JCOV)
            IF(NPRIOR.NE.0)THEN
              DO 6890 I=NOBS+1,NXROW
                IF(NOBGNM(I).EQ.IGROUP) GO TO 6900
6890          CONTINUE
            END IF
            DO 6920 I=IPSTART,NXROW
              IF(NOBGNM(I).EQ.IGROUP)THEN
                CALL ROTATE(2,NOBSCOV(JCOV),NOBSCOV(JCOV),MATDIM1,
     +          NESPAR,IPSTART,COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,NXROW,
     +          NOBGNM)
                GO TO 6900
              END IF
6920        CONTINUE
6900      CONTINUE
        END IF

C -- ALSO, NOW THAT THE SENSITIVITY MATRIX HAS BEEN CALCULATED, WE ROTATE
C    THE REFERENCE OBSERVATIONS BACK AGAIN.

        IF(NUMCOV.GT.0)THEN
          DO 6905 JCOV=1,NUMCOV
            IF(NOBSCOV(JCOV).LE.0) GO TO 6905
            IGROUP=COVGP(JCOV)
            I=0
            DO 6903 J=1,NXROW
              IF(NOBGNM(J).EQ.IGROUP)THEN
                I=I+1
                W1(I)=REFOBS(J)
              END IF
6903        CONTINUE
            CALL ROTATE(1,NOBSCOV(JCOV),NOBSCOV(JCOV),1,1,1,
     +      COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,NXROW,NOBGNM)
            I=0
            DO 6904 J=1,NXROW
              IF(NOBGNM(J).EQ.IGROUP)THEN
                I=I+1
                REFOBS(J)=W1(I)
              END IF
6904        CONTINUE
6905      CONTINUE
        END IF


6907    CONTINUE

C -- If subspace-enhanced Tikhonov is requested, that is done here.

        IF(IREG.NE.0)THEN
          IF(((IREGADJ.EQ.4).OR.(IREGADJ.EQ.5)).AND.
     +       (((IOPT-1)/NOPTREGADJ)*NOPTREGADJ.EQ.(IOPT-1)))THEN
              WRITE(IREC,6916)
              WRITE(6,6916)
6916          FORMAT(/,'    Undertaking Tikhonov regularisation sub',
     +        'space enhancment....')
#ifdef FLUSHFILE
              CALL FLUSH(IREC)
#endif

C -- First all regularisation weights are set to zero.

            TEMPDIM1=0
            DO IROW=1,NXROW
              ITEMP=NOBGNM(IROW)
              IF(IRGP(ITEMP).NE.0) THEN
                OWGHT(IROW)=0
              ELSE
                TEMPDIM1=TEMPDIM1+1
              END IF
            END DO

C -- SVD is undertaken on Q1/2X

            TEMPDIM2=NESPAR
            LWORKTEMP=MAX(3*MIN(TEMPDIM1,NESPAR)+MAX(TEMPDIM1,NESPAR),
     +                5*MIN(TEMPDIM1,NESPAR))
            LWORKTEMP=NINT(FLOAT(LWORKTEMP)*1.3)       ! ARBITRARY
            ALLOCATE(TEMPSVD(TEMPDIM1,TEMPDIM2),
     +      WORKVEC6(LWORKTEMP),STAT=IERR)
            IF(IERR.NE.0)THEN
              WRITE(ERRMSG,6911)
6911          FORMAT('Cannot allocate temporary memory required for ',
     +        'Tikhonov regularisation subspace enhancement.')
              GO TO 9890
            END IF
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
                IRR=0
                DO IROW=1,NXROW
                  ITEMP=NOBGNM(IROW)
                  IF(IRGP(ITEMP).EQ.0) THEN
                    IRR=IRR+1
                    TEMPSVD(IRR,IES)=X(IROW,IESS)*SQRT(OWGHT(IROW))
                  END IF
                END DO
              ELSE
                CALL GET_VECTOR(NCOMPDIM,NXROW,XC,IXC,W1,IESS)
                ISTART_C=IFOUND_C
                IRR=0
                DO IROW=1,NXROW
                  ITEMP=NOBGNM(IROW)
                  IF(IRGP(ITEMP).EQ.0) THEN
                    IRR=IRR+1
                    TEMPSVD(IRR,IES)=W1(IROW)*SQRT(OWGHT(IROW))
                  END IF
                END DO
              END IF
            END DO
            IESTOT=IES
            NRMDIM=1
            NRMLODIM=1
            CALL DGESVD('N','O',TEMPDIM1,IESTOT,TEMPSVD,TEMPDIM1,W1,NRM,
     +      NRMDIM,NRMLO,NRMLODIM,WORKVEC6,LWORKTEMP,INFO)


C -- Before we do anything else we calculate the maximum magnitude of any regularisation row
C    of the Jacobian matrix.

            REGSENMAX=-1.0D200
            DO IROW=1,NXROW
              ITEMP=NOBGNM(IROW)
              IF(IRGP(ITEMP).NE.0)THEN
                IES=0
                IESS=0
                RSUM2=0.0D0
                DO IPP=1,NPAR
                  IF(ITRANS(IPP).LT.0)THEN
                    IF(ITRANS(IPP).LT.-100001)IESS=IESS+1
                    CYCLE
                  ELSE
                    IES=IES+1
                    IESS=IESS+1
                    IF(MAXCOMPDIM.LE.1)THEN
                      RSUM2=RSUM2+X(IROW,IESS)*X(IROW,IESS)
                    ELSE
                      CALL GET_VALUE(NCOMPDIM,XC,IXC,RTEMP,IROW,IESS)
                      RSUM2=RSUM2+RTEMP*RTEMP
                      ISTART_C=IFOUND_C
                    END IF
                  END IF
                END DO
                IF(RSUM2.GT.REGSENMAX)REGSENMAX=RSUM2
              END IF
            END DO

C -- New weights are now calculated.

            WEIGHTMAX=-1.0D200
            WEIGHTMIN=1.0D200
            NNEIG=MIN(IESTOT,TEMPDIM1)
            KREG=0
            DO IROW=1,NXROW
              ITEMP=NOBGNM(IROW)
              IF(IRGP(ITEMP).NE.0)THEN
                KREG=KREG+1
                PROJTOT=0.0D0
                WEIGHTPROJTOT=0.0D0
                RSUM2=0.0D0
                DO I=1,NNEIG
                  RSUM1=0.0D0
                  IES=0
                  IESS=0
                  DO IPP=1,NPAR
                    IF(ITRANS(IPP).LT.0)THEN
                      IF(ITRANS(IPP).LT.-100001)IESS=IESS+1
                      CYCLE
                    ELSE
                      IES=IES+1
                      IESS=IESS+1
                      IF(MAXCOMPDIM.LE.1)THEN
                        RSUM1=RSUM1+TEMPSVD(I,IES)*X(IROW,IESS)
                        IF(I.EQ.1)
     +                  RSUM2=RSUM2+X(IROW,IESS)*X(IROW,IESS)
                      ELSE
                        CALL GET_VALUE(NCOMPDIM,XC,IXC,RTEMP,IROW,IESS)
                        IF(RTEMP.NE.0.0D0) THEN
                          RSUM1=RSUM1+TEMPSVD(I,IES)*RTEMP
                          IF(I.EQ.1)
     +                    RSUM2=RSUM2+RTEMP*RTEMP
                        END IF
                        ISTART_C=IFOUND_C
                      END IF
                    END IF
                  END DO
                  IF(RSUM2.LT.1.0D-30*REGSENMAX)THEN   ! Arbitrary
                    IF(OWGHT(IROW).GT.0.0D0)THEN
                      OWGHT(IROW)=-1.1D300               ! A marker value
                    END IF
                    GO TO 6901
                  END IF
                  RTEMP1=W1(I)/W1(1)
                  IF(ABS(RTEMP1).LT.1.0D-200) GO TO 6909
                  RTEMP1=1/RTEMP1
                  IF(IREGADJ.EQ.4)THEN
                    IF(RTEMP1.GE.REGWEIGHTNUL) GO TO 6909
                  ELSE
                    IF(RTEMP1.GE.REGSINGTHRESH) GO TO 6909
                  END IF
                  RSUM1=RSUM1/SQRT(RSUM2)
                  PROJTOT=PROJTOT+RSUM1*RSUM1
                  RSUM1=RSUM1*RTEMP1
                  WEIGHTPROJTOT=WEIGHTPROJTOT+RSUM1*RSUM1
                END DO
6909            CONTINUE
                IF(IREGADJ.EQ.5)THEN
                  IF(OWGHTKP(KREG).EQ.0.0D0)THEN
                    OWGHT(IROW)=0.0D0
                  ELSE
                    IF(PROJTOT.GE.0.5D0)THEN
                      OWGHT(IROW)=1.0D0
                    ELSE
                      OWGHT(IROW)=REGWEIGHTRAT*REGWEIGHTRAT
                    END IF
                  END IF
                ELSE
                  IF(OWGHTKP(KREG).EQ.0.0D0)THEN
                    OWGHT(IROW)=0.0D0
                  ELSE
                    RTEMP1=1.0D0-PROJTOT
                    IF(RTEMP1.GT.0.0)WEIGHTPROJTOT=WEIGHTPROJTOT+
     +              RTEMP1*RTEMP1*REGWEIGHTNUL*REGWEIGHTNUL
                    OWGHT(IROW)=WEIGHTPROJTOT
                  END IF
                END IF
                IF(OWGHT(IROW).GT.WEIGHTMAX)WEIGHTMAX=OWGHT(IROW)
                IF(OWGHT(IROW).GT.0.0D0)THEN
                  IF(OWGHT(IROW).LT.WEIGHTMIN)WEIGHTMIN=OWGHT(IROW)
                END IF
              END IF
6901          CONTINUE
            END DO

! -- Regularisation observations which have proved to be totally insensitive are given a minimum weight.

            DO IROW=1,NXROW
              IF(OWGHT(IROW).LT.-1.0D300) OWGHT(IROW)=WEIGHTMAX
            END DO

C -- The spread of weights is now reduced in accordance with user settings.

            TOTWEIGHTSUM=0.0D0
            WEIGHTMAX=SQRT(WEIGHTMAX)
            WEIGHTMIN=SQRT(WEIGHTMIN)
            IF(ABS(WEIGHTMAX/WEIGHTMIN-1.0D0).LT.1.0D-3)THEN
              DO IROW=1,NXROW
                ITEMP=NOBGNM(IROW)
                IF(IRGP(ITEMP).GT.0)THEN
                  IF(OWGHT(IROW).GT.0.0D0)THEN
                    OWGHT(IROW)=1.0D0
                    TOTWEIGHTSUM=TOTWEIGHTSUM+OWGHT(IROW)
                  END IF
                END IF
              END DO
            ELSE
              IF(REGWEIGHTRAT.GT.0.0D0)THEN
                WEIGHTRANGE=WEIGHTMAX-WEIGHTMIN
                WEIGHTRATIO=(REGWEIGHTRAT*WEIGHTMIN-WEIGHTMIN)/
     +          WEIGHTRANGE
                DO IROW=1,NXROW
                  ITEMP=NOBGNM(IROW)
                  IF(IRGP(ITEMP).GT.0)THEN
                    IF(OWGHT(IROW).GT.0.0D0)THEN
                      OWGHT(IROW)=
     +                WEIGHTMIN+(SQRT(OWGHT(IROW))-
     +                WEIGHTMIN)*WEIGHTRATIO
                      OWGHT(IROW)=OWGHT(IROW)*OWGHT(IROW)
                      TOTWEIGHTSUM=TOTWEIGHTSUM+OWGHT(IROW)
                    END IF
                  END IF
                END DO
              ELSE
                RTEMP=WEIGHTMIN*ABS(REGWEIGHTRAT)
                WEIGHTMIN=LOG(WEIGHTMIN)
                WEIGHTMAX=LOG(WEIGHTMAX)
                WEIGHTRANGE=WEIGHTMAX-WEIGHTMIN
                WEIGHTRATIO=(LOG(RTEMP)-WEIGHTMIN)/WEIGHTRANGE
                DO IROW=1,NXROW
                  ITEMP=NOBGNM(IROW)
                  IF(IRGP(ITEMP).GT.0)THEN
                    IF(OWGHT(IROW).GT.0.0D0)THEN
                      OWGHT(IROW)=WEIGHTMIN+
     +                (0.5*LOG(OWGHT(IROW))-WEIGHTMIN)*WEIGHTRATIO
                      OWGHT(IROW)=EXP(OWGHT(IROW))
                      OWGHT(IROW)=OWGHT(IROW)*OWGHT(IROW)
                      TOTWEIGHTSUM=TOTWEIGHTSUM+OWGHT(IROW)
                    END IF
                  END IF
                END DO
              END IF
            END IF

C -- Weights are now normalised (so as not to be too high, this resulting in the
C    possible need for overloaded weight factors).

            IF(TOTWEIGHTSUM.NE.0.0D0)THEN
              TOTWEIGHTSUM=1.0D0/TOTWEIGHTSUM*WFLAST
              DO IROW=1,NXROW
                ITEMP=NOBGNM(IROW)
                IF(IRGP(ITEMP).NE.0)THEN
                  OWGHT(IROW)=OWGHT(IROW)*TOTWEIGHTSUM
                END IF
              END DO
            END IF

            DEALLOCATE(TEMPSVD,WORKVEC6,STAT=IERR)
            IF(IERR.NE.0)THEN
              WRITE(ERRMSG,6912)
6912          FORMAT('Cannot deallocate temporary memory required for ',
     +        'Tikhonov subspace enhancement.')
              GO TO 9890
            END IF
            WRITE(IREC,6917)
            WRITE(6,6917)
6917        FORMAT('    Regularisation weights have been ',
     +      're-calculated.')
            WRITE(6,*)
#ifdef FLUSHFILE
            CALL FLUSH(IREC)
#endif
          END IF

          WFLAST=WF
          WF=1.0
          IF(WF.LT.WFMIN)WF=WFMIN
          IF(WF.GT.WFMAX)WF=WFMAX

        END IF

C -- IF WE ARE DOING ADAPTIVE REGULARISATION, PERTINENT OBSERVATION WEIGHTS ARE WORKED
C    OUT BASED ON PERTINENT IW_ PARAMETER VALUES.

        IF(NREGADJPAR.GT.0)THEN
          ICOUNT=0
          DO IPP=1,NPAR
            IF(SCALE(IPP).LT.-1.0D35)THEN
              ITEMP=NINT(OFFSET(IPP))
              DO IOBS=1,NXROW
                IF(NOBGNM(IOBS).EQ.ITEMP)THEN
                  ICOUNT=ICOUNT+1
                  OWGHT(IOBS)=ORIGWGHT(ICOUNT)/PVAL(IPP)/PVAL(IPP)
                END IF
              END DO
            END IF
          END DO
          WF=1.0D0
          IF(WF.LT.WFMIN)WF=WFMIN
          IF(WF.GT.WFMAX)WF=WFMAX

C -- WE CALCULATE THE TOTAL CONTRIBUTION TO THE OBJECTIVE FUNCTION BY ALL
C    REGULARISATION GROUPS

          IRCOUNT=0
          TOTPHIREG=0.0D0
          DO IG=1,NOBSGP
            IF(IRGP(IG).NE.0)THEN
              IRCOUNT=IRCOUNT+1
              TOTPHIREG=TOTPHIREG+PSISUB(IG)
            END IF
          END DO

C -- WE ALSO ALTER THE OBSERVED VALUE OF THE PRIOR INFORMATION EQUATIONS WHICH
C    CITE THE INVERSE WEIGHT PARAMETERS. NOTE THAT THIS ASSUMES LOGARITHMIC
C    TRANSFORMATION OF THE IW PARAMETERS.

          ICOUNT=0
          DO IPP=1,NPAR
            IF(ITRANS(IPP).GE.0)THEN
              IF(SCALE(IPP).LT.-1.0D35)THEN
                ICOUNT=ICOUNT+1
                IOBS=PRIORPAR(ICOUNT)
                IF((IOBS.EQ.0).OR.(IOBS.GT.NXROW))THEN
                  WRITE(ERRMSG,6908)
6908              FORMAT('Incorrect adaptive regularisation ',
     +            'information supplied in PEST control file; ',
     +            'check this file with PESTCHEK.')
                  GO TO 9890
                END IF
                IF(OVAL(IOBS).GE.REFOBS(IOBS))THEN
                  OVAL(IOBS)=NINT(REFOBS(IOBS))
                  IF(OVAL(IOBS).GE.REFOBS(IOBS))
     +            OVAL(IOBS)=OVAL(IOBS)-1.0D0
                ELSE
                  IF(IOPT.GT.1)THEN
                    IF(IWSENS(ICOUNT).GT.GEOMAVSENS*2.0D0)THEN
                      RTEMP1=REFOBS(IOBS)-OVAL(IOBS)
                      RTEMP1=RTEMP1*0.667
                      OVAL(IOBS)=REFOBS(IOBS)-MAX(RTEMP1,0.3D0)           ! 0.3 is arbitrary
                    ELSE IF(IWSENS(ICOUNT).LT.GEOMAVSENS*0.5)THEN
                      IG=NOBGNM(IOBS)
                      IF(PSISUB(IG).LT.TOTPHIREG*0.95)THEN
                        RTEMP1=REFOBS(IOBS)-OVAL(IOBS)
                        RTEMP1=RTEMP1*1.3
                        OVAL(IOBS)=REFOBS(IOBS)-RTEMP1
                      END IF
                    END IF
                  END IF
                END IF
              END IF
            END IF
          END DO

        END IF

C -- IF WE ARE WORKING IN REGULARISATION MODE, HAVE MORE THAN ONE REGULARISATION
C    OBSERVATION GROUP, AND HAVE RELATIVE WEIGHTS ADJUSTMENT ACTIVATED, THE
C    RELATIVE WEIGHTS ARE NOW WORKED OUT.
C -- BUT FIRST WE FIND THE REGULARISATION OBSERVATION GROUP WITH THE HIGHEST SENSITIVITY;
C    THIS WILL BE THE "PIVOT" OBSERVATION GROUP TO WHICH THE WEIGHT FACTOR DIRECTLY APPLIES
C    AND FOR WHICH THE RELATIVE ADJUSTMENT IS 1.0.


        IF((IREG.EQ.1).AND.((IREGADJ.GT.0).AND.(IREGADJ.LT.4)))THEN
          RMAXTOT=0.0D0
          IMAXTOT=0
          DO 5710 I=1,NOBSGP
            IF(IRGP(I).EQ.0)THEN
              SEOGP(I)=-1.0D0
              GO TO 5710
            ELSE
              RSUM2=0.0D0
              RMAX2=0.0D0
              RCOUNT=0.0D0
              ISTART_C=1
              DO 5720 K=1,NXROW
                IF(NOBGNM(K).NE.I) GO TO 5720
                IF(OWGHT(K).LE.0.0D0) GO TO 5720
                RSUM1=0.0D0
                IES=0
                IF(MAXCOMPDIM.LE.1)THEN
                  DO 5730 IES=1,NESPAR
                     RSUM1=RSUM1+X(K,IES)*X(K,IES)
5730              CONTINUE
                ELSE
                  DO IES=1,NESPAR
                    CALL GET_VALUE(NCOMPDIM,XC,IXC,RTEMP,K,IES)
                    IF(RTEMP.NE.0.0D0) RSUM1=RSUM1+RTEMP*RTEMP
                    ISTART_C=IFOUND_C
                  END DO
                END IF
                RSUM2=RSUM2+SQRT(RSUM1*OWGHT(K))
                IF(RSUM1.GT.RMAX2)RMAX2=RSUM1
                RCOUNT=RCOUNT+SQRT(OWGHT(K))
5720          CONTINUE
              IF((IREGADJ.EQ.1).OR.(IREGADJ.EQ.3))THEN
                SEOGP(I)=RSUM2
              ELSE IF(IREGADJ.EQ.2)THEN
                SEOGP(I)=RCOUNT
              END IF
              IF(RMAXTOT.LT.SEOGP(I))THEN
                RMAXTOT=SEOGP(I)
                IMAXTOT=I
              END IF
            END IF
5710      CONTINUE
          IPIVOTREG=IMAXTOT
          IF(IPIVOTREG.EQ.0)IPIVOTREG=1
          SEOGPPIV=SEOGP(IPIVOTREG)
          DO 5740 I=1,NOBSGP
            IF(IRGP(I).NE.0)THEN
              IF(SEOGP(I).GT.0.0D0)THEN
                SEOGP(I)=SEOGP(I)/SEOGPPIV
                IF(SEOGP(I).GT.0.0D0)THEN
                  IF(SEOGP(I).GT.1.0D-10)THEN  !ARBITRARY
                    SEOGP(I)=(1.0/(SEOGP(I)*SEOGP(I)))
                  ELSE
                    SEOGP(I)=0.0D0
                  END IF
                  IF(IOPT.GT.1)THEN
                    IF(SEOGP(I).GT.10.0D0)SEOGP(I)=10.0D0  ! This is supposed to combat massive adjustment - seawat problem.
                  END IF
                END IF
              END IF
            END IF
5740      CONTINUE
          KREG=0
          DO 5750 I=1,NXROW
            IGPNM=NOBGNM(I)
            IF(IRGP(IGPNM).NE.0)THEN
              KREG=KREG+1
              IF(IGPNM.NE.IPIVOTREG)THEN
                IF(SEOGP(IGPNM).GT.0.0D0)
     +          OWGHT(I)=OWGHT(I)*SEOGP(IGPNM)
              END IF
              IF(IREGADJ.EQ.3)THEN
                OWGHT(I)=OWGHT(I)*OWGHTKP(KREG)
              END IF
            END IF
5750      CONTINUE
          WRITE(IREC,5760)
5760      FORMAT(/,'    Regularisation weights adjustment....')
          WRITE(IREC,5762)
5762      FORMAT('    Group Name    Weights adjustment ratio from ',
     +    'previous iteration')
          DO 5765 I=1,NOBSGP
            IF(IRGP(I).NE.0)THEN
              IF(SEOGP(I).GE.0.0)THEN
                WRITE(IREC,5763) OBGNME(I)(1:LEN_TRIM(OBGNME(I))),
     +          SQRT(SEOGP(I))
5763            FORMAT(T6,A,T20,1PG13.6)
              END IF
            END IF
5765      CONTINUE
#ifdef FLUSHFILE
          CALL FLUSH(IREC)
#endif
        END IF

C -- If individual group target objective functions are supplied, weights are adjusted.

        IF(IGTARG.NE.0)THEN
          DO IG=1,NOBSGP
            IF(IRGP(IG).EQ.0)THEN

C -- First we compute contributions to the objective function on the basis of
C    original weights and compute our first cut at weight factor adjustment.

              TPSISUB=PSISUB(IG)/GFAC(IG)
              DTEMP=TPSISUB/GTARG(IG)
              IF(DTEMP.GT.2.0D0)DTEMP=2.0D0
              IF(DTEMP.LT.0.5D0)DTEMP=0.5D0
              GFAC(IG)=GFAC(IG)*DTEMP
              IF(GFAC(IG).GT.128.0) GFAC(IG)=128.0
              IF(GFAC(IG).LT.1.0D0/128.0D0)GFAC(IG)=1.0D0/128.0D0
            END IF
          END DO

C -- Now we adjust the adjustment so that the overall target measurement
C -- objective function is the same.

          TOTGPHIMLIM=0.0D0
          GPHIMLIM=0.0D0
          DO IG=1,NOBSGP
            IF(IRGP(IG).EQ.0)THEN
              TOTGPHIMLIM=TOTGPHIMLIM+GTARG(IG)*GFAC(IG)
              GPHIMLIM=GPHIMLIM+GTARG(IG)
            END IF
          END DO
          FACADJUST=GPHIMLIM/TOTGPHIMLIM
          DO IG=1,NOBSGP
            IF(IRGP(IG).EQ.0)THEN
              GFAC(IG)=GFAC(IG)*FACADJUST
            END IF
          END DO
          PHIMLIM=GPHIMLIM
          PHIMACCEPT=PHIMLIM*1.05
          PHIMLIMKP=PHIMLIM
          PD1RFAC=1.05

C -- Weights are now adjusted.

          DO IG=1,NOBSGP
            IF(IRGP(IG).EQ.0)THEN
              DTEMP=GFAC(IG)/OLDGFAC(IG)
              OLDGFAC(IG)=GFAC(IG)
              DO IROW=1,NXROW
                IF(NOBGNM(IROW).EQ.IG) OWGHT(IROW)=OWGHT(IROW)*DTEMP
              END DO
            END IF
          END DO
          WRITE(6,5766)
          WRITE(IREC,5766)
5766      FORMAT(/,'    Measurement weights adjusted for ',
     +    'fitting individual group phi targets.')
        END IF

C -- REGULARISATION WEIGHT FACTOR IS CALCULATED

        IF(IREG.EQ.1)THEN
          IF((NOPTMAX.NE.-1).AND.(NOPTMAX.NE.-2))THEN
          IF(FRACPHIM.GT.0.0)THEN
            PHIMLIM=PHIMLO*FRACPHIM
            IF(PHIMLIM.LE.PHIMLIMKP)PHIMLIM=PHIMLIMKP
            PHIMACCEPT=PHIMLIM*PD1RFAC
            WRITE(6,7454) PHIMLIM
            WRITE(IREC,7454) PHIMLIM
7454        FORMAT('    FRACPHIM-adjusted target measurement ',
     +      'objective function',T62,': ',1PG12.5)
#ifdef FLUSHFILE
            CALL FLUSH(IREC)
#endif
          END IF
7546      CONTINUE
          WRITE(6,5250)
5250      FORMAT('    Calculating optimal regularisation weight ',
     +    'factor .....')
          NSP4=NESPAR
          NXDIM=NXROW
          IF((MEMSAVE.NE.0).OR.(LSQRMODE.NE.0))THEN
          IF(CONJGRAD.EQ.1)THEN
            LHSD1=1
            LHSD2=1
            LHSVD=LHSVDIM
          ELSE IF(LSQRMODE.NE.0)THEN
            LHSD1=1
            LHSD2=1
            LHSVD=1
          ELSE IF(SVDMODE.EQ.2)THEN
            LHSD1=NXROW
            LHSD2=NESPAR
            LHSVD=1
          ELSE
            LHSD1=NSP4
            LHSD2=NESPAR
            LHSVD=1
          END IF
          IF(LSQRMODE.NE.0)THEN
            IF(PR_INDEX.NE.0)THEN
              IES=0
              IESS=0
              DO IPP=1,NPAR
                IF(ITRANS(IPP).LT.0)THEN
                  IF(ITRANS(IPP).LT.-100001)IESS=IESS+1
                  CYCLE
                END IF
                IES=IES+1
                IESS=IESS+1
                IESTRANS(IESS)=IES
              END DO
            END IF
          END IF
          CALL OPTWT_SL(IFAIL,NXDIM,NSP4,NXROW,NPRIOR,NOBS,NPAR,NESPAR,
     +    W1DIM,RHSDIM,REGITN,WF,WFFAC,PHIMLIM,WFMAX,WFMIN,WFSOL,WFTOL,
     +    ITRANS,REFOBS,OVAL,OWGHT,NOBGNM,W2,TMPOBS,SC,W1,X,RHS,LHS,
     +    INFOCOUNT,IUNSTABLE,LAMBDA,RLAMFAC,ICOVOBS,JSTK,NOBSGP,IRGP,
     +    CONJGRAD,LHSD1,LHSD2,npcgdim,cgr1,cgr2,cgv,cgw,cgy,
     +    cgprecon,cgshift,0,cgitnlim,cgeps,cgrtol,cgistop,cgitn,
     +    cganorm,cgacond,cgrnorm,cgxnorm,lhsvdim,lhsvec,
     +    WORKVEC1,WORKVEC2,WORKVEC3,
     +    SVDMODE,MAXSING,LWORK,EIGTHRESH,WORKVEC4)
          IF(IFAIL.EQ.1) GO TO 9970
          ELSE
          CALL OPTWT(IFAIL,NXDIM,NSP4,NXROW,NPRIOR,NOBS,NPAR,NESPAR,
     +    W1DIM,REGITN,WF,WFFAC,PHIMLIM,WFMAX,WFMIN,WFSOL,WFTOL,ITRANS,
     +    REFOBS,OVAL,OWGHT,NOBGNM,W2,TMPOBS,SC,W1,X,RHS,LHS,INFOCOUNT,
     +    IUNSTABLE,LAMBDA,RLAMFAC,ICOVOBS,JSTK,NORM,NRM,
     +    WORKVEC1,WORKVEC2,WORKVEC3,WORKVEC4,NOBSGP,IRGP,LHSFLAG,
     +    LASTOBSROW,FIRSTREGROW,INOCOV,IOPTCALL,LINREG,WFSTART,
     +    SVDMODE,MAXSING,LWORK,EIGTHRESH,WORKVEC4)
          IF(IFAIL.EQ.1) GO TO 9970
          END IF

          IF(ISTOP.EQ.2)THEN
            IFIN=10
            GO TO 6000
          ELSE IF(ISTOP.EQ.1) THEN
            IPFAIL=-1
            GO TO 9891
          END IF

          WRITE(IREC,*,ERR=9350)
          WRITE(IREC,5220,ERR=9350) WFSOL
          WRITE(6,5220) WFSOL
5220      FORMAT(T5,'Re-calculated regularisation weight factor',T62,
     +    ': ',1PG12.5)
          IF(ABS(WFSOL-WFMAX).LT.1.0D-5*WFMAX)THEN
            WRITE(6,5221)
            WRITE(IREC,5221,ERR=9350)
5221        FORMAT(T5,'***Warning: weight factor at its upper ',
     +      'bound***')
            IF(INFOCOUNT.NE.0)THEN
              WRITE(6,5222)
              WRITE(IREC,5222,ERR=9350)
5222          FORMAT(T5,'***Warning: near singular normal matrix***')
              WRITE(6,5223)
              WRITE(IREC,5223,ERR=9350)
5223          FORMAT(T5,'***Regularisation strategy may need ',
     +        'improvement***')
            ELSE
              IF(FRACPHIM.GT.0.0)THEN
                WRITE(6,5227)
                WRITE(IREC,5227,ERR=9350)
5227            FORMAT(T5,'***Consider lowering FRACPHIM***')
              END IF
            END IF
          ELSE IF(ABS(WFSOL-WFMIN).LT.1.0D-5*WFMIN)THEN
            WRITE(6,5224)
            WRITE(IREC,5224,ERR=9350)
5224        FORMAT(T5,'***Warning: weight factor at its lower ',
     +      'bound***')
          END IF
          IF(IUNSTABLE.EQ.1)THEN
            IF((FRACPHIM.GT.0.0).AND.(PHIMLIM.GT.PHIMLIMKP))THEN
                WRITE(6,7456)
                WRITE(IREC,7456)
7456            FORMAT('    Instability in weight factor calc. - ',
     +          'reducing target meas. obj. fn.')
                PHIMLIM=PHIMLIM/2.0
                IF(PHIMLIM.LT.PHIMLIMKP)PHIMLIM=PHIMLIMKP
                PHIMACCEPT=PHIMLIM*PD1RFAC
                WRITE(6,7454) PHIMLIM
                WRITE(IREC,7454) PHIMLIM
                GO TO 7546
            END IF
          END IF
        END IF
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif
        END IF

        IF(RSTFLE.NE.0)THEN
          INQUIRE(UNIT=IRSF,OPENED=LOPENED)
          IF(LOPENED) THEN
            WRITE(IRSF) WFSOL
            CLOSE(UNIT=IRSF)
          END IF
        END IF
302     CONTINUE
        IF(IAUI.EQ.0)THEN
          IF(JIRST.EQ.1)THEN
            JIRST=0
            IF(IREG.EQ.1)THEN
              IF((NOPTMAX.NE.-1).AND.(NOPTMAX.NE.-2))THEN
                IF(FRACPHIM.GT.0.0)THEN
                  PHIMLIM=PHIMLO*FRACPHIM
                  IF(PHIMLIM.LE.PHIMLIMKP)PHIMLIM=PHIMLIMKP
                  PHIMACCEPT=PHIMLIM*PD1RFAC
                  WRITE(6,7454) PHIMLIM
                  WRITE(IREC,7454) PHIMLIM
                  WRITE(IREC,*,ERR=9350)
                  WRITE(IREC,5220,ERR=9350) WFSOL
                  WRITE(6,5220) WFSOL
#ifdef FLUSHFILE
                  CALL FLUSH(IREC)
#endif
                END IF
              END IF
            END IF
          END IF
        END IF

        IF(HOLDFLAG.NE.0)THEN
          IF(IAUI.EQ.0)THEN
            CALL HLDREAD(JFAIL,NPAR,NPARGP,NESPAR,NLAMBDA,NRELPMX,
     +      NFACPMX,APAR,PARGNME,PHOLD,GHOLD,EHOLD,CLINE,IREG,NVECBND,
     +      IREC,NRELPREDSTP,NABSPREDSTP,NINITSCHFAC,NMULSCHFAC,
     +      NNSEARCH,IPRED)
            IF(JFAIL.NE.0) GO TO 9891
#ifdef FLUSHFILE
            CALL FLUSH(IREC)
#endif
          END IF
          IF(NLAMBDA.GT.0.0D0)LAMBDA=NLAMBDA
          IF(NRELPMX.GT.0.0D0) RELPARMAX=NRELPMX
          IF(NRELPREDSTP.GE.0.0D0) RELPREDSTP=NRELPREDSTP
          IF(NABSPREDSTP.GE.0.0D0) ABSPREDSTP=NABSPREDSTP
          IF(NINITSCHFAC.GT.0.0D0) INITSCHFAC=NINITSCHFAC
          IF(NMULSCHFAC.GT.1.0D0) MULSCHFAC=NMULSCHFAC
          IF(NNSEARCH.GE.1) NSEARCH=NNSEARCH
          IF(NSEARCH.GT.MAXSEARCH)NSEARCH=MAXSEARCH
          IF(NFACPMX.GT.0.0D0)THEN
            DMAX2=NFACPMX
            FACPARMAX=LOG10(NFACPMX)
          END IF
          IF(NVECBND.GE.0)UPVECBEND=NVECBND
        END IF

        IF(IAUI.EQ.0)THEN
          IF(IIIRST.NE.2) THEN
            WRITE(ISNS,128) IOPT
128         FORMAT(/,/,' OPTIMISATION ITERATION NO.',I3,' ----->')
            DO 123 IPP=1,NPAR
              IF(JSTK(IPP).LT.0) GO TO 124
123         CONTINUE
            GO TO 125
124         WRITE(ISNS,327)
327         FORMAT(1X,'Note that sensitivities for parameters glued ',
     +      'to their bounds may be in error.')
125         CONTINUE
            WRITE(ISNS,126)
126         FORMAT(1X,'Parameter name',t20,'Group',t35,
     +      'Current value',t52,'Sensitivity')
#ifdef FLUSHFILE
          CALL FLUSH(ISNS)
#endif
          END IF

          IF(IREG.EQ.1)THEN
            IF((NOPTMAX.NE.-1).AND.(NOPTMAX.NE.-2))THEN
            WFTEMP=WFSOL*WFSOL/WF/WF
            DO 5240 IROW=1,NXROW
              IGPNM=NOBGNM(IROW)
              IF(IRGP(IGPNM).NE.0)OWGHT(IROW)=OWGHT(IROW)*WFTEMP
5240        CONTINUE
            WF=WFSOL
            CALL OBJCLC(1,PSI,NXROW,NPRIOR,NOBS,REFOBS,OVAL,OWGHT,
     +      NOBGNM,NOBSGP,IRGP,SUM1)
            IF(PREDNOISE.GT.0) PSI=PSI+RES_PRED
            PSIL=PSI
            WRITE(IREC,5245,ERR=9350) PSI
            WRITE(6,5245) PSI
5245        FORMAT(T5,'New starting objective function for this itn. ',
     +      '(ie. phi)',T62,': ',1PG12.5)
            DO 5292 I=1,NOBSGP
              CALL OBJCLC(-I,PSISUB(I),NXROW,NPRIOR,NOBS,REFOBS,OVAL,
     +        OWGHT,NOBGNM,NOBSGP,IRGP,SUM1)
              WRITE(IREC,5293,ERR=9350)
     +        OBGNME(I)(1:LEN_TRIM(OBGNME(I))),PSISUB(I)
              WRITE(6,5293) OBGNME(I)(1:LEN_TRIM(OBGNME(I))),PSISUB(I)
5293          FORMAT('    Contribution to phi from observation group "',
     +        A,'"',T62,': ',1PG12.5)
5292        CONTINUE
            IF(NPRIOR.NE.0)THEN
              CALL OBJCLC(-999,PSISUB(NOBSGP+1),NXROW,NPRIOR,NOBS,
     +        REFOBS,OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,SUM1)
              IF(PSISUB(NOBSGP+1).GT.-1.0D299)THEN
                 WRITE(IREC,5294,ERR=9350) PSISUB(NOBSGP+1)
                 WRITE(6,5294) PSISUB(NOBSGP+1)
5294             FORMAT('    Contribution to phi from ungrouped prior ',
     +           'information',T62,': ',1PG12.5)
              END IF
            END IF
            END IF
            IF(IREGADJ.NE.0)THEN
              IF(OWGHTLO(1).LT.-1.0E35)THEN
                DO 5751 I=1,NXROW
                  OWGHTLO(I)=OWGHT(I)
5751            CONTINUE
              END IF
            END IF
          END IF

#ifdef FLUSHFILE
            CALL FLUSH(IREC)
#endif

#ifdef PARALLEL
#ifndef MPEST
          WRITE(IRMR,2360)
2360      FORMAT(/,' Testing parameter upgrades .....')
#endif
#endif
        END IF

        NOWRLM=-1
        ILAMPL=1
        NOPAR=0
215     ITN=0
        NREV=0
        IMM=0
        IPASS1=0
        IF(DOAUI.EQ.'aui')THEN
          IF(LAMBDA.GT.1.0D8)LAMBDA=1.0D8
        END IF
        IF(RLAMFAC_ADJUST.EQ.1)THEN
          IF(LAMBDA.LT.1.0D-8) LAMBDA=1.0D-8
        END IF
        LAMSTR=LAMBDA
        NOWRLM=NOWRLM+1
        JACUPDATEFLAG=1

C -- NEXT THE GRADIENT VECTOR IS CALCULATED

        IES=0
        IESS=0
        ISTART_C=1
        DO 150 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0)THEN
          IF(ITRANS(IPP).LT.-100001) IESS=IESS+1
          GO TO 150
        END IF
        IES=IES+1
        IESS=IESS+1
        RTEMP=0.0D0
        IF(MAXCOMPDIM.LE.1)THEN
        DO 170 IROW=1,NXROW
170     RTEMP=RTEMP+X(IROW,IESS)*OWGHT(IROW)*(OVAL(IROW)-REFOBS(IROW))
        ELSE
          CALL SINGLE_VECTOR_MUL1(IFAIL,NCOMPDIM,1,NXROW,XC,IXC,
     +    IESS,OWGHT,OVAL,REFOBS,RTEMP)
          IF(IFAIL.NE.0) GO TO 9970
          ISTART_C=IFOUND_C
        END IF
        GRAD(IES)=-2.0D0*RTEMP
150     CONTINUE
        NESTMP=IES
        IF(NESTMP.EQ.0) THEN
          WRITE(IREC,305,ERR=9350)
          WRITE(6,305)
305       FORMAT(T8,'all parameters frozen')
          IF(JMM.NE.0)THEN
            NOAUI=1
            GO TO 1200
          END IF
          IF((I2OR3.EQ.2).AND.(ISWTCH.EQ.1)) GO TO 1250
          IFIN=4
          GO TO 6000
        END IF
        CALL OBJCLC1(GRDNRM,NESTMP,GRAD)
        GRDNRM=SQRT(GRDNRM)
        IF(GRDNRM.LE.0.0D0) THEN
          IF(NOWRLM.NE.0)THEN
            WRITE(IREC,310,ERR=9350)
            WRITE(6,310)
310         FORMAT(T8,'phi gradient zero in non-frozen parameter ',
     +      'space')
          ELSE
            WRITE(IREC,311,ERR=9350)
            WRITE(6,311)
311         FORMAT(T5,'phi gradient zero in non-frozen parameter ',
     +      'space')
          END IF
          IF(SENSTO.EQ.1)THEN
            WRITE(ISNS,312)
312         FORMAT(' Phi gradient zero:- ',
     +      '  all parameter sensitivities are zero or near-zero.')
#ifdef FLUSHFILE
          CALL FLUSH(ISNS)
#endif
          END IF
          IF(JMM.NE.0) THEN
            NOAUI=0
            GO TO 1200
          END IF
          IF((I2OR3.EQ.2).AND.(ISWTCH.EQ.1)) GO TO 1250
          IF(PESTMODE.EQ.4) GO TO 12501
          IFIN=5
          GO TO 6000
        END IF

C -- NEXT THE DIAGONAL ELEMENTS OF THE SCALING MATRIX ARE CALCULATED

275     CONTINUE

C -- BUT FIRST WE CALCULATE THE NUMBER OF OBSERVATIONS WITH NON-ZERO WEIGHTS.

        IF(IPRED.EQ.0)THEN
          NN=NOBS
        ELSE
          NN=NOBS-1
        END IF
        NOBSNZERO=0
        DO 276 I=1,NN
          IF(OWGHT(I).GT.0.0D0) NOBSNZERO=NOBSNZERO+1
276    CONTINUE
        IF(NPRIOR.GT.0)THEN
          DO 277 I=NOBS+1,NXROW
            IF(OWGHT(I).GT.0.0D0) NOBSNZERO=NOBSNZERO+1
277       CONTINUE
        END IF
        IF(NOBSNZERO.LE.0)NOBSNZERO=1

        RTEMPG=0.0D0
        ISENSCOUNT=0
        IWCOUNT=0
        IES=0
        IESS=0
        NOEFF=0
        ISTART_C=1
        DO 300 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) THEN
          IF(ITRANS(IPP).LT.-100001) IESS=IESS+1
          GO TO 300
        END IF
        IES=IES+1
        IESS=IESS+1
        IF(MAXCOMPDIM.LE.1)THEN
        CALL OBJCLC(2,SC(IES),NXROW,NPRIOR,NOBS,X(1,IESS),OVAL,OWGHT,
     +  NOBGNM,NOBSGP,IRGP,SUM1)
        ELSE
          CALL SINGLE_VECTOR_MUL2(IFAIL,NCOMPDIM,1,NXROW,XC,IXC,
     +    IESS,OWGHT,SC(IES))
          IF(IFAIL.NE.0) GO TO 9970
          ISTART_C=IFOUND_C
        END IF
        IF(SC(IES).LE.0.0D0) THEN
          SC(IES)=0.0D0
          NOEFF=NOEFF+1
          IF(IAUI.EQ.0)THEN
            IF(DECNOF.EQ.0) THEN
              IF(NOEFF.EQ.1)WRITE(IREC,*,ERR=9350)
              WRITE(IREC,295,ERR=9350) APAR(IPP)(:LEN_TRIM(APAR(IPP)))
              WRITE(6,295) APAR(IPP)(:LEN_TRIM(APAR(IPP)))
295           FORMAT(T5,'Parameter "',A,
     +        '" has no effect on observations.')
             END IF
           END IF
        END IF
        SC(IES)=SQRT(SC(IES))
        IF(HOLDFLAG.NE.0)THEN
          IF(IAUI.EQ.0)SENS(IPP)=SC(IES)
        END IF
        IF(IAUI.EQ.0)THEN
        IF(SENSTO.EQ.1)THEN
        IF(IIIRST.EQ.0)THEN
          IF(DECNOF.EQ.0)THEN
            WRITE(ISNS,299) APAR(IPP),PARGNME(IPARGP(IPP)),PVAL(IPP),
     +      SC(IES)/NOBSNZERO
299         FORMAT(3X,A12,T20,A12,T35,1PG13.6,T52,1PG13.6)
            IF(SCALE(IPP).GT.-1.0D35)THEN
              IF(SC(IES).NE.0)THEN
                RTEMPG=RTEMPG+LOG(SC(IES)/NOBSNZERO)
                ISENSCOUNT=ISENSCOUNT+1
              END IF
            ELSE
              IWCOUNT=IWCOUNT+1
              IWSENS(IWCOUNT)=SC(IES)/NOBSNZERO
            END IF
          END IF
        END IF
        END IF
        END IF
        IF(SC(IES).EQ.0.0D0) GO TO 300
        IF(HOLDFLAG.NE.0)THEN
          IF(ISENS(IPP).GT.0)THEN
            IF(ISENS(IPP).EQ.IAUI)THEN
              IF(DECNOF.EQ.0)THEN
                WRITE(IREC,7560,ERR=9350)
     +          APAR(IPP)(1:LEN_TRIM(APAR(IPP)))
                WRITE(6,7560) APAR(IPP)(1:LEN_TRIM(APAR(IPP)))
7560            FORMAT('    Parameter "',A,'" now held through ',
     +          'automatic user intervention.')
              END IF
            END IF
            SC(IES)=-1.0D0/SC(IES)
            GO TO 300
          END IF
          IF(PHOLD(IPP).NE.0)THEN
            IF(DECNOF.EQ.0)THEN
              IF(IAUI.EQ.0)THEN
                WRITE(IREC,301,ERR=9350)
     +          APAR(IPP)(1:LEN_TRIM(APAR(IPP)))
301             FORMAT('    Parameter "',A,'" held through manual ',
     +          'intervention.')
                WRITE(6,301) APAR(IPP)(1:LEN_TRIM(APAR(IPP)))
              END IF
            END IF
            SC(IES)=-1.0D0/SC(IES)
            IF(ISENS(IPP).EQ.0)ISENS(IPP)=-1
            GO TO 300
          END IF
          IF(SC(IES)/NOBSNZERO.LE.GHOLD(IPARGP(IPP)))THEN
            IF(DECNOF.EQ.0)THEN
              IF(SC(IES).GT.0.0D0)THEN
                IF(IAUI.EQ.0)THEN
                WRITE(IREC,301,ERR=9350)
     +          APAR(IPP)(1:LEN_TRIM(APAR(IPP)))
                WRITE(6,301) APAR(IPP)(1:LEN_TRIM(APAR(IPP)))
              END IF
              END IF
            END IF
            SC(IES)=-1.0D0/SC(IES)
            IF(ISENS(IPP).EQ.0)ISENS(IPP)=-1
          ELSE
            SC(IES)=1.0D0/SC(IES)
          END IF
        ELSE
          SC(IES)=1.0D0/SC(IES)
        END IF
300     CONTINUE
#ifdef FLUSHFILE
            CALL FLUSH(ISNS)
            CALL FLUSH(IREC)
#endif

        SENSTO=0
        IF(ISENSCOUNT.NE.0) GEOMAVSENS=EXP(RTEMPG/ISENSCOUNT)

        IF(HOLDFLAG.NE.0)THEN
          DO 4210 J=1,NPARGP
            IF(GHOLD(J).LT.0.0)THEN
              IITEMP=NINT(-GHOLD(J))
              IF(IITEMP.EQ.0) GO TO 4209
              ICOUNT1=0
4214          IF(ICOUNT1.GE.IITEMP) GO TO 4209
              ICOUNT2=0
              RRRTEMP=0.0
              IES=0
              DO 4213 IPP=1,NPAR
                IF(ITRANS(IPP).GE.0)THEN
                  IES=IES+1
                  IF(IPARGP(IPP).EQ.J)THEN
                    IF(ABS(SC(IES)).GT.RRRTEMP)THEN
                      ICOUNT2=ICOUNT2+1
                      RRRTEMP=ABS(SC(IES))
                      MIES=IES
                      MIPP=IPP
                    END IF
                  END IF
                END IF
4213          CONTINUE
              IF(ICOUNT2.EQ.0) GO TO 4209
              IF(DECNOF.EQ.0)THEN
                IF(SC(MIES).GT.0.0D0)THEN
                  IF(IAUI.EQ.0)THEN
                    WRITE(IREC,301) APAR(MIPP)(:LEN_TRIM(APAR(MIPP)))
                    WRITE(6,301)  APAR(MIPP)(:LEN_TRIM(APAR(MIPP)))
#ifdef FLUSHFILE
                    CALL FLUSH(IREC)
#endif
                  END IF
                END IF
              END IF
              SC(MIES)=0.0D0
              IF(ISENS(MIPP).EQ.0) ISENS(MIPP)=-1
              ICOUNT1=ICOUNT1+1
              GO TO 4214
4209          CONTINUE
            END IF
4210      CONTINUE
4215      CONTINUE
        END IF

        IF(IAUI.EQ.0)THEN
          ISTKFLG=0
          DO 379 IPP=1,NPAR
            IF(JSTK(IPP).LT.0) THEN
              ISTKFLG=1
              GO TO 380
            END IF
379       CONTINUE
380       CONTINUE
        END IF

C -- THE NORMAL MATRIX IS CALCULATED AND STORED

        IF((MEMSAVE.EQ.0).AND.(IFLAG_NOCOV.EQ.0))THEN
        IF(MATSTO.EQ.1)THEN
          IF((ICOVF.NE.0).OR.(ICORF.NE.0).OR.(IEIGF.NE.0))THEN
            LHSFLAG=0
            IES=0
            IESS=0
            ICOUNT=0
            DO 378 IPP=1,NPAR
              IF(ITRANS(IPP).LT.0) THEN
                IF(ITRANS(IPP).LT.-100001) IESS=IESS+1
                GO TO 378
              END IF
              IES=IES+1
              IESS=IESS+1
              ICOUNT=ICOUNT+1
              JES=0
              JESS=0
              ISTART_C=1
              DO 377 I=1,IPP
                IF(ITRANS(I).LT.0) THEN
                  IF(ITRANS(I).LT.-100001) JESS=JESS+1
                  GO TO 377
                END IF
                JES=JES+1
                JESS=JESS+1
                RTEMP=0
                IF(MAXCOMPDIM.LE.1)THEN
                  DO 376 IROW=1,NXROW
376               RTEMP=RTEMP+OWGHT(IROW)*X(IROW,IESS)*X(IROW,JESS)
                ELSE
                  CALL DUAL_VECTOR_MUL(IFAIL,NCOMPDIM,1,NXROW,XC,
     +            IXC,OWGHT,IESS,JESS,RTEMP)
                  IF(IFAIL.NE.0) GO TO 9970
                END IF
                NORM(JES,IES)=RTEMP
377           CONTINUE
378         CONTINUE
            IF(ICOUNT.GT.1)THEN
              DO IES=1,ICOUNT-1
                DO JES=IES+1,ICOUNT
                  NORM(JES,IES)=NORM(IES,JES)
                END DO
              END DO
            END IF

            IF(IAUI.EQ.0)THEN
              IF((ICOVF.NE.0).OR.(ICORF.NE.0).OR.(IEIGF.NE.0))THEN
                DO 395 IES=1,NESPAR
                  DO 394 JES=1,NESPAR
                    NRMLO(JES,IES)=NORM(JES,IES)
394               CONTINUE
395             CONTINUE
              END IF
              CALL WRTMAT(JFAIL,IGNF,NPAR,NOBS,NXROW,NESPAR,NPRIOR,
     +        NPARGP,NWDIM,IOPT,ITRANS,PVAL,W1,W2,OVAL,REFOBS,OWGHT,
     +        NRM,NRMLO,PARGNME,IPARGP,APAR,ICOVF,ICORF,IEIGF,CLINE,
     +        NOBSGP,NOBGNM,IPRED,IREG,APAR1,APAR2,ISTKFLG,IRGP,
     +        MEMSAVE,RES_PRED,NOPTMAX,NDIM4,WORKVEC4)
              IF(JFAIL.NE.0) GO TO 9891
            END IF
          ELSE
            ITEMP=1
            CALL WRTMAT(JFAIL,IGNF,NPAR,NOBS,NXROW,ITEMP,NPRIOR,NPARGP,
     +      NWDIM,IOPT,ITRANS,PVAL,W1,W2,OVAL,REFOBS,OWGHT,NRM,
     +      NRMLO,PARGNME,IPARGP,APAR,ICOVF,ICORF,IEIGF,CLINE,NOBSGP,
     +      NOBGNM,IPRED,IREG,APAR1,APAR2,ISTKFLG,IRGP,MEMSAVE,
     +      RES_PRED,NOPTMAX,NDIM4,WORKVEC4)
            IF(JFAIL.NE.0) GO TO 9891
          END IF
        END IF
        ELSE
          IF(IAUI.EQ.0)THEN
            ITEMP=1
            CALL WRTMAT(JFAIL,IGNF,NPAR,NOBS,NXROW,ITEMP,NPRIOR,NPARGP,
     +      NWDIM,IOPT,ITRANS,PVAL,W1,W2,OVAL,REFOBS,OWGHT,NRM,
     +      NRMLO,PARGNME,IPARGP,APAR,ICOVF,ICORF,IEIGF,CLINE,NOBSGP,
     +      NOBGNM,IPRED,IREG,APAR1,APAR2,ISTKFLG,IRGP,MEMSAVE,
     +      RES_PRED,NOPTMAX,NDIM4,WORKVEC4)
            IF(JFAIL.NE.0) GO TO 9891
          END IF
        END IF

C -- Does a parameter need to be held on account of eigenvalues/eigenvectors?

        IF(HOLDFLAG.NE.0)THEN
          IF(IEIGF.NE.0)THEN
            DO 5600 IES=1,NESPAR
              IF(EHOLD(IES).LE.0) GO TO 5600
              IVEC=NESPAR-IES+1
              DO 5605 I=1,NESPAR
                W1(I)=NRM(I,IVEC)
5605          CONTINUE
              ICOUNT1=0
5610          CONTINUE
              IF(ICOUNT1.GE.EHOLD(IES)) GO TO 5600
              JES=0
              JESS=0
              RTEMPMAX=0.0D0
              MJES=0
              MJESS=0
              MIPP=0
              ICOUNT2=0
              DO 5630 IPP=1,NPAR
                IF(ITRANS(IPP).LT.0)THEN
                 IF(ITRANS(IPP).LT.-100001)THEN
                    JES=JES+1
                  ELSE
                    GO TO 5630
                  END IF
                ELSE
                  JES=JES+1
                  JESS=JESS+1
                END IF
                RTEMP=ABS(W1(JES))
                IF(RTEMP.GT.RTEMPMAX)THEN
                  RTEMPMAX=RTEMP
                  MJES=JES
                  MJESS=JESS
                  MIPP=IPP
                  ICOUNT2=ICOUNT2+1
                END IF
5630          CONTINUE
              IF(ICOUNT2.EQ.0) GO TO 5600
              W1(MJES)=0.0D0
              IF(SC(MJESS).GT.0.0D0)THEN
                IF(DECNOF.EQ.0)THEN
                IF(IAUI.EQ.0)THEN
                WRITE(IREC,301,ERR=9350)
     +          APAR(MIPP)(1:LEN_TRIM(APAR(MIPP)))
                WRITE(6,301) APAR(MIPP)(1:LEN_TRIM(APAR(MIPP)))
#ifdef FLUSHFILE
                CALL FLUSH(IREC)
#endif
                END IF
                END IF
              END IF
              SC(MJESS)=0.0D0
              LHSFLAG=0
              IF(ISENS(MIPP).EQ.0)ISENS(MIPP)=-1
              ICOUNT1=ICOUNT1+1
              GO TO 5610
5600        CONTINUE
          END IF

          DO 4216 IES=1,NESPAR
            IF(SC(IES).LT.0.0D0) SC(IES)=0.0D0
4216      CONTINUE
        END IF

        ICOUNT1=0
        DO 5216 IES=1,NESPAR
          IF(SC(IES).GT.0.0) ICOUNT1=ICOUNT1+1
5216    CONTINUE
        IF(ICOUNT1.EQ.0)THEN
          CALL STPERR(130,0,' ',0,' ',CLINE)
          GO TO 9891
        END IF

        DECNOF=1
        IIIRST=0

C -- NEXT THE RHS OF THE NORMAL EQUATIONS IS CALCULATED

        IF(LSQRMODE.NE.0) GO TO 251
        IES=0
        IESS=0
        ISTART_C=1
        DO 250 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) THEN
          IF(ITRANS(IPP).LT.-100001) IESS=IESS+1
          GO TO 250
        END IF
        IES=IES+1
        IESS=IESS+1
        RTEMP=0.0D0
        IF(MAXCOMPDIM.LE.1)THEN
        DO 270 IROW=1,NXROW
270     RTEMP=RTEMP+X(IROW,IESS)*OWGHT(IROW)*(OVAL(IROW)-REFOBS(IROW))
        ELSE
          CALL SINGLE_VECTOR_MUL1(IFAIL,NCOMPDIM,1,NXROW,XC,IXC,
     +    IESS,OWGHT,OVAL,REFOBS,RTEMP)
          IF(IFAIL.NE.0) GO TO 9970
          ISTART_C=IFOUND_C
        END IF
        IF(SVDMODE.EQ.0)THEN
          RHS(IES)=RTEMP*SC(IES)
        ELSE IF((SVDMODE.EQ.1).OR.(SVDMODE.EQ.2))THEN
          RHS(IES)=RTEMP
        END IF
250     CONTINUE
        IF(SVDMODE.EQ.2) GO TO 251

C -- NEXT THE LHS OF THE NORMAL EQUATIONS IS CALCULATED

        LHSIND=0
        IF((LHSFLAG.EQ.0).OR.
     +     (ICOVF.EQ.1).OR.(ICORF.EQ.1).OR.(IEIGF.EQ.1))THEN
          IES=0
          IESS=0
          DO 400 IPP=1,NPAR
            IF(ITRANS(IPP).LT.0) THEN
              IF(ITRANS(IPP).LT.-100001) IESS=IESS+1
              GO TO 400
            END IF
            IES=IES+1
            IESS=IESS+1
            JES=0
            JESS=0
            ISTART_C=1
            DO 390 I=1,IPP
            IF(ITRANS(I).LT.0) THEN
              IF(ITRANS(I).LT.-100001) JESS=JESS+1
              GO TO 390
            END IF
            JES=JES+1
            JESS=JESS+1
            RTEMP=0.0D0
            IF(MAXCOMPDIM.LE.1)THEN
              DO 360 IROW=1,NXROW
360           RTEMP=RTEMP+OWGHT(IROW)*X(IROW,IESS)*X(IROW,JESS)
            ELSE
              CALL DUAL_VECTOR_MUL(IFAIL,NCOMPDIM,1,NXROW,XC,
     +        IXC,OWGHT,IESS,JESS,RTEMP)
              IF(IFAIL.NE.0) GO TO 9970
            END IF
            IF(CONJGRAD.EQ.0)THEN
              IF(SVDMODE.EQ.0)THEN
                 LHS(JES,IES)=RTEMP*SC(IES)*SC(JES)
              ELSE IF(SVDMODE.EQ.1)THEN
                 LHS(JES,IES)=RTEMP
              END IF
              IF((MATSTO.EQ.1).AND.
     +          ((MEMSAVE.EQ.0).AND.(IFLAG_NOCOV.EQ.0)))
     +          NORM(JES,IES)=RTEMP
            ELSE
              IF(JES.LE.IES)THEN
                LHSIND=LHSIND+1
                LHSVEC(LHSIND)=RTEMP*SC(IES)*SC(JES)
              END IF
            END IF
390         CONTINUE
400       CONTINUE
          IF(CONJGRAD.EQ.0)THEN
            DO IES=1,NESTMP-1
              DO JES=IES+1,NESTMP
                LHS(JES,IES)=LHS(IES,JES)
              END DO
            END DO
            IF((MATSTO.EQ.1).AND.
     +        ((MEMSAVE.EQ.0).AND.(IFLAG_NOCOV.EQ.0))) THEN
              DO IES=1,NESTMP-1
                DO JES=IES+1,NESTMP
                  NORM(JES,IES)=NORM(IES,JES)
                END DO
              END DO
            END IF
          END IF
        END IF
        IF((SVDMODE.EQ.1).AND.(NOLAMBDA.EQ.0))THEN
           DO IES=1,NESTMP
             WORKVEC3(IES)=LHS(IES,IES)
           END DO
        END IF
251     CONTINUE
        LHSFLAG=0
        MATSTO=0

        IF((NOPTMAX.EQ.-1).OR.(NOPTMAX.EQ.-2))THEN
          IFIN=11
          GO TO 6000
        END IF

        JJ=0
        DO 214 IPP=1,NPAR
          IF((JSTK(IPP).LT.0).AND.(ITRANS(IPP).GE.0)) THEN
            ITRANS(IPP)=ITRANS(IPP)-200000
            JJ=JJ+1
          END IF
214     CONTINUE
        IF(JJ.NE.0) GO TO 215

        IF(IPRED.GT.0)THEN
C -- NEXT THE PN VECTOR IS FILLED (PREDICTIVE ANALYSER)

          IES=0
          IESS=0
          ISTART_C=1
          DO 3050 IPP=1,NPAR
            IF(ITRANS(IPP).LT.0)THEN
              IF(ITRANS(IPP).LT.-100001) IESS=IESS+1
              GO TO 3050
            END IF
            IES=IES+1
            IESS=IESS+1
            IF(MAXCOMPDIM.LE.1)THEN
              PN(IES)=X(NOBS,IESS)!PREDICTION MUST BE LAST OBSERVATION
            ELSE
              CALL GET_VALUE(NCOMPDIM,XC,IXC,RTEMP_C,NOBS,IESS)
              ISTART_C=IFOUND_C
              PN(IES)=RTEMP_C
            END IF
3050      CONTINUE
        END IF

C -- THE MARQUARDT LAMBDA IS NEXT CALCULATED AND ADDED

        FRFLAG=0
9020    CONTINUE
#ifdef PARALLEL
        LSCH1=0
        IF((LDOSCH.EQ.1).OR.(LDOSCH.LT.0))THEN
          LSEARCH=1
#ifdef MPEST
          CALL HMPARLAM(IFAIL,MLAMRUN)
          IF(IFAIL.NE.0)MLAMRUN=MSRUN
          LMRUN=MIN(MLAMRUN,NSRUN)        ! NSRUN is dimension of reglam
#else
          CALL HMRUNS(NSLAVE,ISTATS,IRUNTME,IORDER,LMRUN,IDET,MANYLAM)
          LMRUN=MIN(LMRUN,NSRUN)          ! NSRUN is dimension of reglam
#endif
          IF(LDOSCH.LT.0)THEN
            IF(LMRUN.GT.ABS(LDOSCH))LMRUN=ABS(LDOSCH)
          END IF
          IF(MANYLAM.NE.0)THEN
            IF(NUMLAM.LT.NUMLAMORIG)NUMLAM=NUMLAMORIG
            IF(LMRUN.GT.NUMLAM)THEN
              LMRUN=NUMLAM
            ELSE
              NUMLAM=LMRUN
            END IF
          END IF
        ELSE
          LSEARCH=0
        END IF
        IF((NOPTMAX.EQ.-1).OR.(NOPTMAX.EQ.-2)) LSEARCH=0
        IF((IPRED.EQ.1).AND.(NSEARCH.GT.1))LSEARCH=0
        IF(MANYLAM.EQ.0)THEN
          DO 9021 IPP=1,NPAR
            IF(ITRANS(IPP).LT.-100001)THEN
              LSEARCH=0
              GO TO 9022
            END IF
9021      CONTINUE
9022      CONTINUE
        END IF
#endif

        SLAMBDA=LAMBDA
        SIMM=IMM
        SJMM=JMM
        SITN=ITN

9030    LAMBDA=SLAMBDA
        IMM=SIMM
        JMM=SJMM
        ITN=SITN
        LRUN=1

410     ITN=ITN+1

C -- Broyden's rank 1 update is activated if requested.

        IF((JACUPDATE.NE.0).AND.(FAILUREFLAG.EQ.0))THEN
          IF(ITN.GT.1)THEN
            IF(JACUPCOUNT.LT.JACUPDATE)THEN
              IF(JACUPDATEFLAG.NE.0)THEN
                WRITE(IREC,412)
412             FORMAT(/,'        Broyden''s update of ',
     +          'Jacobian matrix undertaken.')
                CALL X_UPDATE(IFAIL,NXROW,NOBS,NPAR,NESPAR,ITRANS,
     +          TMPVAL,PVAL,TMPOBS,REFOBS,X,W1,W2)
                IF(IFAIL.NE.0)THEN
                  WRITE(ERRMSG,411)
411               FORMAT('Error encountered in implementing ',
     +            'Broyden''s Jacobian update. Set JACUPDATE to ',
     +            'zero and re-commence PEST execution.')
                  GO TO 9890
                END IF
                CALL SC_UPDATE(IFAIL,NXROW,NPRIOR,NOBS,NPAR,NESPAR,
     +          NOBSGP,ITRANS, NOBGNM,OVAL,OWGHT,IRGP,SC,X)
                IF(IFAIL.NE.0)THEN
                  WRITE(ERRMSG,411)
                  GO TO 9890
                END IF
                CALL GRAD_UPDATE(IFAIL,NXROW,NOBS,NPRIOR,NPAR,NESPAR,
     +          NOBSGP,ITRANS,X,OWGHT,OVAL,REFOBS,GRAD,NOBGNM,IRGP)
                IF(IFAIL.NE.0)THEN
                  WRITE(ERRMSG,411)
                  GO TO 9890
                END IF
                IF(LSQRMODE.EQ.0)THEN
                  CALL RHS_UPDATE(NXROW,NPAR,NESPAR,SVDMODE,ITRANS,RHS,
     +            X,OVAL,REFOBS,OWGHT,SC)
                END IF
                IF((SVDMODE.NE.2).AND.(LSQRMODE.EQ.0))THEN
                  CALL LHS_UPDATE(NXROW,NPAR,NESPAR,SVDMODE,NOLAMBDA,
     +            ITRANS,X,OWGHT,SC,LHS,WORKVEC3DIM,WORKVEC3)
                END IF
                JACUPDATEFLAG=0
                JACUPCOUNT=JACUPCOUNT+1
              END IF
            END IF
          END IF
        END IF

        ITNJMM=JMM
        IF((CONJGRAD.EQ.0).AND.(SVDMODE.EQ.0).AND.(LSQRMODE.EQ.0))THEN
          DO 420 IES=2,NESTMP
          DO 420 I=1,IES-1
420       LHS(I,IES)=LHS(IES,I)
        END IF
425     CONTINUE
        IF((LSQRMODE.EQ.0).AND.(SVDMODE.NE.2))THEN
          IF(LAMBDA.NE.0.0D0) THEN
            RTEMP=0.0D0
            DO 440 IES=1,NESTMP
            IF(ABS(SC(IES)).GT.RTEMP) RTEMP=ABS(SC(IES))
440         CONTINUE
            RRTEMP=LAMBDA/RTEMP/RTEMP
            ALPHA=RRTEMP
            LHSIND=0
            DO 445 IES=1,NESTMP
            IF(CONJGRAD.EQ.0)THEN
              IF(SVDMODE.EQ.0)THEN
                LHS(IES,IES)=1.0D0+SC(IES)*SC(IES)*RRTEMP
              ELSE IF(SVDMODE.EQ.1)THEN
                IF(NOLAMBDA.EQ.0)THEN
                  LHS(IES,IES)=WORKVEC3(IES)+RRTEMP
                END IF
              END IF
            ELSE
              LHSIND=LHSIND+IES
              LHSVEC(LHSIND)=1.0D0+SC(IES)*SC(IES)*RRTEMP
            END IF
445         CONTINUE
          ELSE
            LHSIND=0
            DO 446 IES=1,NESTMP
              IF(CONJGRAD.EQ.0)THEN
                IF(SVDMODE.EQ.0)THEN
                  LHS(IES,IES)=1.0D0
                END IF
              ELSE
                LHSIND=LHSIND+IES
                LHSVEC(LHSIND)=1.0D0
              END IF
446         CONTINUE
          END IF
        END IF

        NSP4=NESPAR
        NST4=NESTMP
        IF(CONJGRAD.EQ.1)THEN
          INFO=0
          write(cgnout,447) lambda
447       format(/,/,' CURRENT VALUE OF MARQUARDT LAMBDA = ',1pg12.3,
     +    ' ----->')
          call cgsolve(NST4,RHS,cgr1,cgr2,cgv,cgw,W1,cgy,
     +    cgprecon,cgshift,cgnout,cgitnlim,cgeps,cgrtol,cgistop,cgitn,
     +    cganorm,cgacond,cgrnorm,cgxnorm,lhsvdim,lhsvec)
          IF(CGISTOP.GE.4)INFO=1
#ifdef FLUSHFILE
          CALL FLUSH(CGNOUT)
#endif
        ELSE
          IF(SVDMODE.NE.0)THEN
#ifdef PARALLEL
            CALL SKIP_COMP(JFLAG,NSRUN,LDOSCH,MANYLAM,LSEARCH,LSCH1,
     +      LMRUN,LAMBDA,REGLAM)
            IF(JFLAG.NE.0) GO TO 1150
#endif
            IF(LSCH1.EQ.0)THEN
              WRITE(SVDUNIT,447) LAMBDA
            END IF
            IF(SVDMODE.EQ.1)THEN
              DO I=1,NST4
                DO J=1,NST4
                  NRM(J,I)=LHS(J,I)
                END DO
              END DO
              CALL DGESVD('O','N',NST4,NST4,LHS,NSP4,W1,NRM,NSP4,NRMLO,
     +        NSP4,WORKVEC4,LWORK,INFO)
            ELSE IF(SVDMODE.EQ.2)THEN
              IF(IPASS1.EQ.0)THEN
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
                      LHS(IROW,IES)=X(IROW,IESS)*SQRT(OWGHT(IROW))
                    END DO
                  ELSE
                    CALL GET_VECTOR(NCOMPDIM,NXROW,XC,IXC,
     +              LHS(1,IES),IESS)
                    ISTART_C=IFOUND_C
                    DO IROW=1,NXROW
                      LHS(IROW,IES)=LHS(IROW,IES)*SQRT(OWGHT(IROW))
                    END DO
                  END IF
                END DO
                NRMDIM=1
                NRMLODIM=1
                CALL DGESVD('N','O',NXROW,NST4,LHS,NXROW,W1,NRM,NRMDIM,
     +          NRMLO,NRMLODIM,WORKVEC4,LWORK,INFO)
                DO I=1,MIN(NXROW,NST4)
                  W1(I)=W1(I)*W1(I)
                END DO
                DO I=1,MIN(NST4,NXROW)
                  WORKVEC3(I)=W1(I)
                END DO
                IPASS1=IPASS1+1
              END IF
              RTEMP=WORKVEC3(1)
              DO I=1,MIN(NST4,NXROW)
C                W1(I)=WORKVEC3(I)+LAMBDA*RTEMP
                W1(I)=WORKVEC3(I)+LAMBDA
              END DO
            END IF
86002       CONTINUE
            IF(LSCH1.EQ.0)THEN
              WRITE(SVDUNIT,8600)
8600          FORMAT(/,' SINGULAR VALUES:-')
              WRITE(SVDUNIT,8603) (W1(I),I=1,NST4)
8603          FORMAT(7(1X,1PG14.7))
              IF(EIGWRITE.NE.0)THEN
                WRITE(SVDUNIT,*)
                IF(SVDMODE.EQ.1)THEN
                  WRITE(SVDUNIT,8604)
8604              FORMAT(/,' MATRIX OF EIGENVECTORS:-')
                  DO J=1,NST4
                    WRITE(SVDUNIT,8603) (LHS(J,I),I=1,NST4)
                  END DO
                ELSE IF(SVDMODE.EQ.2)THEN
                  ITEMP=MIN(NST4,NXROW)
                  CALL WRITINT(ATEMP1,ITEMP)
                  WRITE(SVDUNIT,86041) TRIM(ATEMP1)
86041             FORMAT(/,' MATRIX OF FIRST ',A,' EIGENVECTORS:-')
                  DO J=1,NST4
                    WRITE(SVDUNIT,8603) (LHS(I,J),I=1,ITEMP)
                  END DO
                END IF
              END IF
            END IF

C -- Low eigenvalues are now removed.

            IICOUNT=1
            EIGMAX=W1(1)
            IF(EIGMAX.LE.0.0D0)THEN
              WRITE(ERRMSG,8605)
8605          FORMAT('Error in carrying out singular value ',
     +        'decomposition; highest eivenvalue is zero.')
              GO TO 9890
            END IF
            IF(SVDMODE.EQ.1)THEN
              ITEMP=NST4
            ELSE IF(SVDMODE.EQ.2)THEN
              ITEMP=MIN(NST4,NXROW)
            END IF
            DO I=2,ITEMP
              IF(ABS(W1(I)/EIGMAX).LT.EIGTHRESH)EXIT
              IF(IICOUNT.GE.MAXSING)EXIT
              IICOUNT=IICOUNT+1
            END DO
            IF(LSCH1.EQ.0)THEN
              CALL WRITINT(ATEMP1,IICOUNT)
              WRITE(SVDUNIT,8631) TRIM(ATEMP1)
8631          FORMAT(/,' Number of singular values used in ',
     +        'solution = ',A)
#ifdef FLUSHFILE
              CALL FLUSH(SVDUNIT)
#endif
            END IF
            NUMEIG=IICOUNT

C -- Next the solution is computed.

            IF(SVDMODE.EQ.1)THEN
              DO I=1,NST4
                RTEMP=0.0D0
                DO J=1,NST4
                  DO K=1,IICOUNT
                    RTEMP=RTEMP+LHS(I,K)*LHS(J,K)/W1(K)*RHS(J)
                  END DO
                END DO
                W2(I)=RTEMP
              END DO
              DO I=1,NST4
                W1(I)=W2(I)
              END DO
              DO I=1,NST4
                DO J=1,NST4
                  LHS(J,I)=NRM(J,I)
                END DO
              END DO
            ELSE IF(SVDMODE.EQ.2)THEN
              DO I=1,IICOUNT
                RTEMP=0.0D0
                DO K=1,NST4
                  RTEMP=RTEMP+LHS(I,K)*RHS(K)
                END DO
                W2(I)=RTEMP/W1(I)
              END DO
              DO I=1,NST4
                RTEMP=0.0D0
                DO K=1,IICOUNT
                  RTEMP=RTEMP+LHS(K,I)*W2(K)
                END DO
                W1(I)=RTEMP
              END DO
            END IF
            GO TO 8700
          ELSE IF(LSQRMODE.EQ.1)THEN
#ifdef PARALLEL
            CALL SKIP_COMP(JFLAG,NSRUN,LDOSCH,MANYLAM,LSEARCH,LSCH1,
     +      LMRUN,LAMBDA,REGLAM)
            IF(JFLAG.NE.0) GO TO 1150
#endif
            IF(LSQR_NOUT.NE.0)THEN
              WRITE(LSQR_NOUT,447) LAMBDA
            END IF
            WANTSE=.FALSE.
            LSQR_LENIW=1
            LSQR_LENRW=NXROW
            DO IROW=1,NXROW
              W2(IROW)=SQRT(OWGHT(IROW))*(OVAL(IROW)-REFOBS(IROW))
            END DO
            IF(PR_INDEX.NE.0)THEN
              IES=0
              IESS=0
              IESTRANS=0                ! An array
              DO IPP=1,NPAR
                IF(ITRANS(IPP).LT.0)THEN
                  IF(ITRANS(IPP).LT.-100001)IESS=IESS+1
                  CYCLE
                END IF
                IES=IES+1
                IESS=IESS+1
                IESTRANS(IESS)=IES
              END DO
            END IF
            CALL LSQR(NXROW,NESTMP,CPROD,LAMBDA,WANTSE,
     +                LSQR_LENIW,LSQR_LENRW,LSQR_IW,OWGHT,
     +                W2,WORKVEC1,WORKVEC2,W1,WORKVEC3,
     +                LSQR_ATOL,LSQR_BTOL,LSQR_CONLIM,LSQR_ITNLIM,
     +                LSQR_NOUT,
     +                LSQR_ISTOP,LSQR_ITN,LSQR_ANORM,LSQR_ACOND,
     +                LSQR_RNORM,LSQR_ARNORM,LSQR_XNORM,0)
                   IF(LSQR_NOUT.NE.0)THEN
#ifdef FLUSHFILE
                      CALL FLUSH(LSQR_NOUT)
#endif
                   END IF
            GO TO 8700
          ELSE
#ifdef PARALLEL
            CALL SKIP_COMP(JFLAG,NSRUN,LDOSCH,MANYLAM,LSEARCH,LSCH1,
     +      LMRUN,LAMBDA,REGLAM)
            IF(JFLAG.EQ.1) GO TO 1150
#endif
            IF(LSCH1.EQ.0)THEN
            WRITE(NCONDUNIT,449) LAMBDA
449         FORMAT(/,/,' CURRENT VALUE OF MARQUARDT LAMBDA = ',1PG12.3)
            END IF
            CALL DPOCO(LHS,NSP4,NST4,RCOND,W2,INFO)
            IF(LSCH1.EQ.0)THEN
            IF(INFO.NE.0)THEN
              WRITE(NCONDUNIT,451)
451           FORMAT('    Matrix is not positive definite.')
            ELSE
              IF(RCOND.LE.0.0) THEN
                RCOND=1.0E30
              ELSE
                RCOND=1.0D0/RCOND
              END IF
              WRITE(NCONDUNIT,448) RCOND
448           FORMAT  ('           MATRIX CONDITION NUMBER = ',1PG12.5)
            END IF
#ifdef FLUSHFILE
            CALL FLUSH(NCONDUNIT)
#endif
            END IF
          END IF
        END IF
        IF(INFO.NE.0) THEN
          NOPAR=1
          IF(MANYLAM.EQ.0)THEN
            IF(LSEARCH.EQ.1)THEN
              LSEARCH=0
              GO TO 9030
            END IF
          END IF
          IF(ISOLMSG.EQ.0)THEN
            WRITE(IREC,296,ERR=9350)
            WRITE(6,296)
296         FORMAT(T9,'normal matrix not positive definite: ',
     +      'cannot solve for update direction')
#ifdef FLUSHFILE
            CALL FLUSH(IREC)
#endif
            ISOLMSG=1
          END IF
          IF(ITN.EQ.1)THEN
            NREV=1
            LAMBDA=LAMBDA*RLAMFAC
            PSI1=1.0D300
            IF(IPRED.GT.0)THEN
              IF(PREDHL.EQ.'l')THEN
                PRED1=1.0D300
              ELSE
                PRED1=-1.0D300
              END IF
            END IF

          ELSE
            IF(NREV.NE.0) THEN
              NREV=NREV+1
              LAMBDA=LAMBDA*RLAMFAC
            ELSE
              IF(ITN.EQ.2) THEN
                NREV=1
                LAMBDA=LAMSTR*RLAMFAC
              ELSE
                GO TO 1200
              END IF
            END IF
          END IF
          TEMPSI=1.0D300

          IF(IPRED.GT.0)THEN
            IF(PREDHL.EQ.'l')THEN
              TEMPRD=1.0D300
            ELSE
              TEMPRD=-1.0D300
            END IF
          END IF
          JACUPDATEFLAG=0
          GO TO 410
        END IF

        ISOLMSG=0

        IF(CONJGRAD.EQ.1) GO TO 513  ! Note - CONJGRAD must not be 1 if doing pred. anal.
        IF(IPRED.GT.0)THEN
C -- NOW WE SOLVE FOR THE PREDICTIVE ANALYZERS LAMBDA

          DO 3045 IES=1,NESTMP
            PNTEMP(IES)=PN(IES)*SC(IES)
3045      CONTINUE
          CALL DPOSL(LHS,NSP4,NST4,PNTEMP)
          PDEN=0.0
          DO 3060 IES=1,NESTMP
            PNTEMP(IES)=PNTEMP(IES)*SC(IES)
            PDEN=PDEN+PN(IES)*PNTEMP(IES)
3060      CONTINUE
          IF(ABS(PDEN).LE.1.0D-300)THEN
            DIRNRM=0.0D0
            GO TO 511
          END IF

          DO 3100 IES=1,NESTMP
            PNTEMP(IES)=RHS(IES)
3100      CONTINUE
          CALL DPOSL(LHS,NSP4,NST4,PNTEMP)
          DO 3120 IES=1,NESTMP
            PNTEMP(IES)=PNTEMP(IES)*SC(IES)
3120      CONTINUE
          RTEMP1=0.0
          DO 3130 IROW=1,NXROW
            RTEMP=0.0
            IES=0
            IESS=0
            ISTART_C=1
            DO 3140 IPP=1,NPAR
              IF(ITRANS(IPP).LT.0) THEN
                IF(ITRANS(IPP).LT.-100001)IESS=IESS+1
                GO TO 3150
              END IF
              IES=IES+1
              IESS=IESS+1
              IF(MAXCOMPDIM.LE.1)THEN
                RTEMP=RTEMP+X(IROW,IESS)*PNTEMP(IES)
              ELSE
                CALL GET_VALUE(NCOMPDIM,XC,IXC,RTEMP_C,IROW,IESS)
                ISTART_C=IFOUND_C
                IF(RTEMP_C.NE.0.0D0)
     +          RTEMP=RTEMP+RTEMP_C*PNTEMP(IES)
              END IF
3150        CONTINUE
3140        CONTINUE
            RTEMP1=RTEMP1+(OVAL(IROW)-REFOBS(IROW))*OWGHT(IROW)
     +      *(RTEMP-OVAL(IROW)+REFOBS(IROW))
3130      CONTINUE
          PNUM=PD0+RTEMP1
          IF(PREDNOISE.EQ.0)THEN
            PLAMDASH=PNUM/PDEN
          ELSE
            PLAMDASH=PNUM/(PDEN+1.0D0/PWTKP/PWTKP)
          END IF
          IF(PLAMDASH.LE.0.0)THEN
            PLAMDASH=0.0
          ELSE
            PLAMDASH=SQRT(PLAMDASH)
          END IF
          IF(PPOSNEG.EQ.'n') PLAMDASH=-PLAMDASH
          IF(PREDNOISE.GT.0)THEN
            EW_PRED=-PLAMDASH
            E_PRED=EW_PRED/PWTKP/PWTKP
            RES_PRED=EW_PRED*E_PRED
          END IF
        END IF

C -- NEXT PEST SOLVES FOR THE UPDATING DIRECTION

490     DO 500 IES=1,NESTMP
        IF(IPRED.GT.0)THEN
          W1(IES)=RHS(IES)-PLAMDASH*PN(IES)*SC(IES)
        ELSE
          W1(IES)=RHS(IES)
        END IF
500     CONTINUE
        CALL DPOSL(LHS,NSP4,NST4,W1)

513     CONTINUE
        DO 510 IES=1,NESTMP
510     W1(IES)=W1(IES)*SC(IES)
8700    CONTINUE

        CALL OBJCLC1(DIRNRM,NESTMP,W1)
        DIRNRM=SQRT(DIRNRM)
511     IF(DIRNRM.LE.0.0D0) THEN
          NOPAR=1
          IF(MANYLAM.EQ.0)THEN
            IF(LSEARCH.EQ.1)THEN
              LSEARCH=0
              GO TO 9030
            END IF
          END IF
          WRITE(IREC,515,ERR=9350)
          WRITE(6,515)
515       FORMAT(T8,'parameter update vector zero')
          IF(JMM.NE.0) THEN
            NOAUI=1
            GO TO 1200
          END IF
          IF((I2OR3.EQ.2).AND.(ISWTCH.EQ.1)) GO TO 1250
          IFIN=6
          GO TO 6000
        END IF

C -- GAMMA, AND THEN BETA, IS NEXT CALCULATED

700     DO 800 IROW=1,NXROW
          RTEMP=0.0D0
          IF(IPRED.GT.0)PRTEMP=0.0D0
          IES=0
          IESS=0
          ISTART_C=1
          DO 750 IPP=1,NPAR
            IF(ITRANS(IPP).LT.0) THEN
              IF(ITRANS(IPP).LT.-100001) IESS=IESS+1
              GO TO 750
            END IF
            IES=IES+1
            IESS=IESS+1
            IF(MAXCOMPDIM.LE.1)THEN
              RTEMP=RTEMP+X(IROW,IESS)*W1(IES)     ! CHECK IESS AND IES HERE
              IF(IPRED.GT.0)PRTEMP=PRTEMP+X(IROW,IESS)*PVAL(IPP)
            ELSE
              CALL GET_VALUE(NCOMPDIM,XC,IXC,RTEMP_C,IROW,IESS)
              ISTART_C=IFOUND_C
              IF(RTEMP_C.NE.0.0D0)RTEMP=RTEMP+RTEMP_C*W1(IES)
              IF(IPRED.GT.0)PRTEMP=PRTEMP+RTEMP_C*PVAL(IPP)
            END IF
750       CONTINUE

          IF(IPRED.GT.0)THEN
            W2(IROW)=RTEMP
            PC0(IROW)=REFOBS(IROW)
          ELSE
            W2(IROW)=RTEMP
          END IF
800     CONTINUE
        IF(IPRED.GT.0)THEN
          PPA=0.0D0
          PPB=0.0D0
          PPC=0.0D0
          DO 805 IROW=1,NXROW
            RTEMP=PC0(IROW)-OVAL(IROW)
            IF(ABS(W2(IROW)).LT.1.0D150)
     +      PPA=PPA+W2(IROW)*OWGHT(IROW)*W2(IROW)
            IF((ABS(W2(IROW)).LT.1.0D150).AND.(ABS(RTEMP).LT.1.0D150))
     +      PPB=PPB+RTEMP*W2(IROW)*OWGHT(IROW)
            IF(ABS(RTEMP).LT.1.0D150)
     +      PPC=PPC+RTEMP*RTEMP*OWGHT(IROW)
805       CONTINUE
          PPB=PPB*2.0
          PPC=PPC-PD0

          RTEMP=PPB*PPB-4.0D0*PPA*PPC
          IF(RTEMP.LT.0.0D0) RTEMP=0.0D0
          RTEMP=SQRT(RTEMP)
          IF(PPA.NE.0.0D0)THEN
            PROOT1=(-PPB+RTEMP)/2.0D0/PPA
            PROOT2=(-PPB-RTEMP)/2.0D0/PPA
          ELSE
            PROOT1=1.0D0
            PROOT2=1.0D0
          END IF
          IF(PROOT1.EQ.0.0D0)PROOT1=1.0
          IF(PROOT2.EQ.0.0D0)PROOT2=1.0
          PRTEMP1=0.0D0
          PRTEMP2=0.0D0
          DO 847 IROW=1,NXROW
            IES=0
            IESS=0
            PRTEMP3=0.0D0
            PRTEMP4=0.0D0
            ISTART_C=1
            DO 845 IPP=1,NPAR
              IF(ITRANS(IPP).LT.0)THEN
                IF(ITRANS(IPP).LT.-100001) IESS=IESS+1
                GO TO 840
              END IF
              IES=IES+1
              IESS=IESS+1
              IF(MAXCOMPDIM.LE.1)THEN
                PRTEMP3=PRTEMP3+X(IROW,IESS)*W1(IES)*PROOT1
                PRTEMP4=PRTEMP4+X(IROW,IESS)*W1(IES)*PROOT2
              ELSE
                CALL GET_VALUE(NCOMPDIM,XC,IXC,RTEMP_C,IROW,IESS)
                ISTART_C=IFOUND_C
                IF(RTEMP_C.NE.0.0D0)THEN
                  PRTEMP3=PRTEMP3+RTEMP_C*W1(IES)*PROOT1
                  PRTEMP4=PRTEMP4+RTEMP_C*W1(IES)*PROOT2
                END IF
              END IF
840         CONTINUE
845         CONTINUE
            PRTEMP3=OVAL(IROW)-PRTEMP3-REFOBS(IROW)
            PRTEMP4=OVAL(IROW)-PRTEMP4-REFOBS(IROW)
            PRTEMP1=PRTEMP1+PRTEMP3*PRTEMP3*OWGHT(IROW)
            PRTEMP2=PRTEMP2+PRTEMP4*PRTEMP4*OWGHT(IROW)
847       CONTINUE
	  BETA=PROOT1
ccccc    IF(ABS(PD0-PRTEMP2).LT.ABS(PD0-PRTEMP1))BETA=PROOT2
c  noticed how we are assuming which root to take. If that works
c  out we should get rid of the testing stuff above
        ELSE
          RTEMP=0.0D0
          RRTEMP=0.0D0
          DO 850 IROW=1,NXROW
            RTEMP=RTEMP+(OVAL(IROW)-REFOBS(IROW))*OWGHT(IROW)*W2(IROW)
850       RRTEMP=RRTEMP+OWGHT(IROW)*W2(IROW)*W2(IROW)
          IF(RRTEMP.EQ.0.0D0) GO TO 811
          BETA=RTEMP/RRTEMP
        END IF

880     DO 810 IES=1,NESTMP
810     W1(IES)=W1(IES)*BETA
811     CONTINUE

C -- IF BOTH THE GRADIENT AND THE CHANGE VECTOR POINT OUT OF BOUNDS
C -- FOR A PARTICULAR PARAMETER, THAT PARAMETER IS FROZEN

        ILOCK=0
        IES=0
        DO 580 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 580
        IES=IES+1
        IF(SC(IES).EQ.0.0D0) GO TO 580
        IF((PVAL(IPP).LE.PARLBND(IPP)).OR.
     +    (ABS(PVAL(IPP)-PARLBND(IPP)).LE.ABS(PARLBND(IPP)*1.0D-8)))
     +    THEN
          PVAL(IPP)=PARLBND(IPP)
          IF((-GRAD(IES).LE.0.0D0).AND.(W1(IES).LE.0.0D0)) GO TO 575
        ELSE IF((PVAL(IPP).GE.PARUBND(IPP)).OR.
     +    (ABS(PVAL(IPP)-PARUBND(IPP)).LE.ABS(PARUBND(IPP)*1.0D-8)))
     +    THEN
          PVAL(IPP)=PARUBND(IPP)
          IF((-GRAD(IES).GE.0.0D0).AND.(W1(IES).GE.0.0D0)) GO TO 575
        END IF
        GO TO 580
575     CONTINUE
        NOPAR=1
        IF(MANYLAM.EQ.0)THEN
          IF(LSEARCH.EQ.1)THEN
            LSEARCH=0
            GO TO 9030
          END IF
        END IF
        ITRANS(IPP)=ITRANS(IPP)-200000
        ILOCK=ILOCK+1
        WRITE(IREC,576,ERR=9350) APAR(IPP)(:LEN_TRIM(APAR(IPP)))
        WRITE(6,576) APAR(IPP)(:LEN_TRIM(APAR(IPP)))
576     FORMAT(T9,'parameter "',A,'" frozen: gradient and update ',
     +  'vectors out of bounds')
#ifdef FLUSHFILE
          CALL FLUSH(IREC)
#endif
        IF(SVDMODE.NE.0) THEN
          WRITE(SVDUNIT,*)
          WRITE(SVDUNIT,5761)APAR(IPP)(:LEN_TRIM(APAR(IPP)))
5761      FORMAT(' Parameter "',A,'" frozen: gradient and update ',
     +    'vectors out of bounds')
#ifdef FLUSHFILE
          CALL FLUSH(SVDUNIT)
#endif
        ELSE IF(LSQRMODE.NE.0) THEN
          IF(LSQR_NOUT.NE.0)THEN
            WRITE(LSQR_NOUT,*)
            WRITE(LSQR_NOUT,5761)APAR(IPP)(:LEN_TRIM(APAR(IPP)))
#ifdef FLUSHFILE
            CALL FLUSH(LSQR_NOUT)
#endif
          END IF
        END IF
580     CONTINUE
        IF(ILOCK.NE.0) GO TO 215

C -- IF THE CHANGE DIRECTION IS OUT OF BOUNDS FOR A PARTICULAR PARAMETER
C -- THAT PARAMETER IS FROZEN

        ILOCK=0
        IES=0
        DO 600 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 600
        IES=IES+1
        IF(SC(IES).EQ.0.0D0) GO TO 600
        IF((PVAL(IPP).LE.PARLBND(IPP)).OR.
     +    (ABS(PVAL(IPP)-PARLBND(IPP)).LE.ABS(PARLBND(IPP)*1.0D-8)))
     +    THEN
          PVAL(IPP)=PARLBND(IPP)
          IF(W1(IES).LE.0.0D0) GO TO 590
        ELSE IF((PVAL(IPP).GE.PARUBND(IPP)).OR.
     +    (ABS(PVAL(IPP)-PARUBND(IPP)).LE.ABS(PARUBND(IPP)*1.0D-8)))
     +    THEN
          PVAL(IPP)=PARUBND(IPP)
          IF(W1(IES).GE.0.0D0) GO TO 590
        END IF
        GO TO 600
590     CONTINUE
        NOPAR=1
        IF(MANYLAM.EQ.0)THEN
          IF(LSEARCH.EQ.1)THEN
            LSEARCH=0
            GO TO 9030
          END IF
        END IF
        ITRANS(IPP)=ITRANS(IPP)-200000
        ILOCK=ILOCK+1
        WRITE(IREC,577,ERR=9350) APAR(IPP)(:LEN_TRIM(APAR(IPP)))
        WRITE(6,577) APAR(IPP)(:LEN_TRIM(APAR(IPP)))
577     FORMAT(T9,'parameter "',A,'" frozen: - update vector out ',
     +  'of bounds')
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif
        IF(SVDMODE.NE.0) THEN
          WRITE(SVDUNIT,*)
          WRITE(SVDUNIT,5771)APAR(IPP)(:LEN_TRIM(APAR(IPP)))
5771      FORMAT(' Parameter "',A,'" frozen: - update vector out ',
     +    'of bounds')
#ifdef FLUSHFILE
          CALL FLUSH(SVDUNIT)
#endif
        ELSE IF(LSQRMODE.NE.0)THEN
          IF(LSQR_NOUT.NE.0)THEN
            WRITE(LSQR_NOUT,*)
            WRITE(LSQR_NOUT,5771)APAR(IPP)(:LEN_TRIM(APAR(IPP)))
#ifdef FLUSHFILE
            CALL FLUSH(LSQR_NOUT)
#endif
          END IF
        END IF
600     CONTINUE
        IF(ILOCK.NE.0) GO TO 215

C -- NEXT ESTABLISH RO
C -- FIRST ADMX

        JPP0=0
        ADMX0=0.0D0
        JPP1=0
        ADMX1=0.0D0
        JPP2=0
        ADMX2=0.0D0
        RO0=1.0
        RO1=1.0
        RO2=1.0
        IES=0
        DO 900 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 900
        IES=IES+1
        IF(W1(IES).EQ.0.0D0) GO TO 900
        IF(ITRANS(IPP).EQ.0) THEN
          IF(PARCHGLIM(IPP).EQ.0)THEN
            IF(ABS(PVAL(IPP)).GT.ABS(ORGVAL(IPP))*FACORIG)THEN
              IF(PVAL(IPP).EQ.0.0D0) GO TO 900
              RTEMP=W1(IES)/ABS(PVAL(IPP))
            ELSE
              IF(ORGVAL(IPP).EQ.0.0D0) GO TO 900
              RTEMP=W1(IES)/ABS(ORGVAL(IPP)*FACORIG)
            END IF
            IF(PVAL(IPP).NE.0.0D0)THEN
              IF(RELPARMAX.LT.1.0D0)THEN
                IF(((W1(IES).GT.0.0D0).AND.(PVAL(IPP).LT.0.0D0)).OR.
     +             ((W1(IES).LT.0.0D0).AND.(PVAL(IPP).GT.0.0D0)))
     +             RTEMP=W1(IES)/ABS(PVAL(IPP))
              END IF
            END IF
            IF(UPVECBEND.EQ.1)THEN
              IF(ABS(RTEMP).GT.RELPARMAX)THEN
                W1(IES)=W1(IES)*RELPARMAX/ABS(RTEMP)
                RTEMP=RELPARMAX*RTEMP/ABS(RTEMP)
              END IF
            END IF
            IF(ABS(RTEMP).GT.ADMX0)THEN
              ADMX0=ABS(RTEMP)
              DMX0=RTEMP
              JPP0=IPP
            END IF
          ELSE
            IF(ABS(PVAL(IPP)).GT.ABS(ORGVAL(IPP)*FACORIG))THEN
              RTEMP=PVAL(IPP)
            ELSE
              RTEMP=ORGVAL(IPP)*FACORIG
            END IF
            IF(((W1(IES).GT.0.0D0).AND.(PVAL(IPP).LT.0.0D0)).OR.
     +         ((W1(IES).LT.0.0D0).AND.(PVAL(IPP).GT.0.0D0)))
     +         RTEMP=PVAL(IPP)
            IF(RTEMP.EQ.0.0D0) GO TO 900
            IF(((W1(IES).GT.0.0D0).AND.(RTEMP.GT.0.0D0)).OR.
     +         ((W1(IES).LT.0.0D0).AND.(RTEMP.LT.0.0D0)))THEN
              RTEMP=RTEMP*(DMAX2-1.0D0)
            ELSE
              RTEMP=RTEMP*(1.0D0-1.0D0/DMAX2)
            END IF
            IF(RTEMP.EQ.0.0D0) GO TO 900
            RTEMP=W1(IES)/RTEMP
            IF(UPVECBEND.EQ.1)THEN
              IF(ABS(RTEMP).GT.1.0D0)THEN
                W1(IES)=W1(IES)*1.0D0/ABS(RTEMP)
                RTEMP=1.0*RTEMP/ABS(RTEMP)
              END IF
            END IF
            IF(ABS(RTEMP).GT.ADMX2)THEN
              DMX2=RTEMP
              ADMX2=ABS(DMX2)
              JPP2=IPP
            END IF
          END IF
        ELSE
          RTEMP=W1(IES)
          IF(UPVECBEND.EQ.1)THEN
            IF(ABS(RTEMP).GT.FACPARMAX)THEN
              W1(IES)=W1(IES)*FACPARMAX/ABS(RTEMP)
              RTEMP=FACPARMAX*RTEMP/ABS(RTEMP)
            END IF
          END IF
          IF(ABS(RTEMP).GT.ADMX1)THEN
            ADMX1=ABS(RTEMP)
            DMX1=RTEMP
            JPP1=IPP
          END IF
        END IF
900     CONTINUE
        IF((JPP0.EQ.0).AND.(JPP1.EQ.0).AND.(JPP2.EQ.0)) GO TO 930

        S0=0.0D0
        S1=0.0D0
        S2=0.0D0
        IF((IOPT.EQ.1).OR.
     +  ((JPP0.NE.JPP0LT).AND.(JPP1.NE.JPP1LT)).AND.
     +   (JPP2.NE.JPP2LT))THEN
          ROSTAR=1
        ELSE
          IF((JPP0.EQ.0).OR.(DMX0LT.EQ.0.0D0).OR.(ROLAST.EQ.0.0D0))
     +    GO TO 905
          IF(JPP0.EQ.JPP0LT) S0=DMX0/DMX0LT/ROLAST
905       IF((JPP1.EQ.0).OR.(DMX1LT.EQ.0.0D0).OR.(ROLAST.EQ.0.0D0))
     +    GO TO 910
          IF(JPP1.EQ.JPP1LT) S1=DMX1/DMX1LT/ROLAST
910       IF((JPP2.EQ.0).OR.(DMX2LT.EQ.0.0D0).OR.(ROLAST.EQ.0.0D0))
     +    GO TO 915
          IF(JPP2.EQ.JPP2LT) S2=DMX2/DMX2LT/ROLAST
915       S=MIN(S0,S1,S2)
          IF(S.GE.-1.0D0) THEN
            ROSTAR=(3.0D0+S)/(3.0D0+ABS(S))
          ELSE
            ROSTAR=1.0D0/2.0D0/ABS(S)
          END IF
        END IF
        IF(JPP0.NE.0) THEN
          IF(ROSTAR*ADMX0.LE.RELPARMAX) THEN
            RO0=ROSTAR
          ELSE
            RO0=RELPARMAX/ADMX0
          END IF
        END IF
        IF(JPP1.NE.0) THEN
          IF(ROSTAR*ADMX1.LE.FACPARMAX) THEN
            RO1=ROSTAR
          ELSE
            RO1=FACPARMAX/ADMX1
          END IF
        END IF
        IF(JPP2.NE.0)THEN
          IF(ROSTAR*ADMX2.LE.1.0D0)THEN
            RO2=ROSTAR
          ELSE
            RO2=1.0D0/ADMX2
          END IF
        END IF
        RO=MIN(RO0,RO1,RO2)
        DO 920 IES=1,NESTMP
920     W1(IES)=W1(IES)*RO
930     CONTINUE

C -- THE PARAMETERS ARE NEXT UPGRADED

C -- BUT FIRST THE LINE-SEARCH VARIABLES ARE INITIATED

        LTFLAG=1
        IF((IPRED.EQ.0).OR.(NSEARCH.LE.1))LTFLAG=0
        IF(LTFLAG.EQ.1)THEN
          IF(PSILO.GT.PD2)LTFLAG=0
        END IF
        IF(LTFLAG.NE.0)THEN
          LITN=0
          LTLAST=0
          LTBIS=0
          LTCOUNT=0
          LT1LAST=-999
          LT2LAST=-999
          LTNEG=0
          LTFLAGX=0
          PD12=(PD1+PD0)*0.5
          LTFAC=INITSCHFAC  !Maybe this should be the best one from last search
          IF(ITN.GT.1)THEN
            IF(LINESAVE.EQ.1)THEN
              IF(MULSCHFAC.GT.1.0)THEN
                LTFAC=BESTFAC/SQRT(MULSCHFAC)
              END IF
            END IF
          END IF
          DO 3440 IES=1,NESTMP
            W1KP(IES)=W1(IES)
3440      CONTINUE
        END IF
        TMPSIL=TEMPSI
        IF(IPRED.GT.0) TMPRL=TEMPRD
        ISUBTEND=0
        ICONVERGE=0

3400    CONTINUE          ! START OF LINE SEARCH
        IF(LTFLAG.NE.0)THEN
          IF(LTLAST.EQ.0) THEN
            LITN=LITN+1
            IF(LITN.GT.NSEARCH) THEN
              LITN=LITN-1
              GO TO 3650
            END IF
          END IF
          DO 3420 IES=1,NESTMP
            W1(IES)=W1KP(IES)*LTFAC
3420      CONTINUE
        END IF

        IES=0
        DO 1000 IPP=1,NPAR
        IF(ITRANS(IPP).LT.0) GO TO 1000
        IES=IES+1
        IF(ITRANS(IPP).EQ.0) THEN
          TMPVAL(IPP)=PVAL(IPP)+W1(IES)
          IF(TMPVAL(IPP).GE.PARUBND(IPP)) TMPVAL(IPP)=PARUBND(IPP)
          IF(TMPVAL(IPP).LE.PARLBND(IPP)) TMPVAL(IPP)=PARLBND(IPP)
        ELSE
          IF(PVAL(IPP).LE.0.0D0) GO TO 970
          RTEMP=LOG10(PVAL(IPP))+W1(IES)
          IF(RTEMP.GT.290.0D0)THEN
            TMPVAL(IPP)=1.0D290
          ELSE IF(RTEMP.LT.-290.0D0) THEN
            TMPVAL(IPP)=1.0D-290
          ELSE
            TMPVAL(IPP)=PVAL(IPP)*(10**W1(IES))
          END IF
          IF(TMPVAL(IPP).GE.PARUBND(IPP)) TMPVAL(IPP)=PARUBND(IPP)
          IF(TMPVAL(IPP).LE.PARLBND(IPP)) TMPVAL(IPP)=PARLBND(IPP)
970       CONTINUE
        END IF
1000    CONTINUE
        DO 1005 IPP=1,NPAR
          IF((ITRANS(IPP).EQ.-200000).OR.(ITRANS(IPP).EQ.-199999))
     +    TMPVAL(IPP)=PVAL(IPP)
1005    CONTINUE
        DO 1010 IPP=1,NPAR
        J=ITRANS(IPP)
        IF(J.LT.0)THEN
          IF(J.GT.-100000)THEN
            TMPVAL(IPP)=TMPVAL(-J)*ORGVAL(IPP)/ORGVAL(-J)
          ELSE IF(J.EQ.-100000) THEN
            TMPVAL(IPP)=ORGVAL(IPP)
          ELSE
            TMPVAL(IPP)=PVAL(IPP)
          END IF
        END IF
1010    CONTINUE
        IF(LTLAST.NE.0) GO TO 3490

        DO 1100 IPP=1,NPAR
        IF(TMPVAL(IPP).NE.PVAL(IPP)) GO TO 1150
1100    CONTINUE
        IF(LDOSCH.EQ.0)GO TO 1155

C -- NEXT THE NEW PARAMETERS ARE TESTED

1150    CONTINUE
        I=-1
        J=0
        INCPAR(1)=0

#ifdef PARALLEL
6906    CONTINUE
        IF(LDOSCH.EQ.0) GO TO 7050
        IF(LSEARCH.EQ.1)THEN
          CALL ZINCTEST(JFAIL,NOCHNG,NPAR,PRECIS,NOPNT,NW,-1,TMPVAL,
     +    PVAL,SCALE,OFFSET,PWORD,APAR,CLINE,J,RTEMP)
          IF(JFAIL.NE.0) GO TO 9891
          NOCHNG=0
          CALL STORE_PARALLEL_REGISTER
     +    (IFAIL,NPAR,PARREG,LRUN,TMPVAL,PARREGFILE)
          IF(IFAIL.NE.0) GO TO 9890
          REGLAM(LRUN)=LAMBDA
          INCPAR(LRUN)=0
          LRUN=LRUN+1
          IF(LRUN.LE.LMRUN)THEN
            IF(ILAMPL.EQ.1)THEN
              IF(LRUN.EQ.2)THEN
                LAMKPH=LAMSTR
                LAMBDA=LAMBDA/RLAMFAC
                IF((LAMBDA.NE.0.0D0).AND.(LAMBDA.LT.1.0D-20))
     +          LAMBDA=1.0D-20
                LAMKPL=LAMBDA
              ELSE IF(LRUN.EQ.3)THEN
                LAMBDA=LAMKPH*RLAMFAC
                LAMKPH=LAMBDA
              ELSE
                IF(MANYLAM.EQ.0)THEN
                  IF(((LRUN-3)/3)*3.EQ.LRUN-3)THEN
                    LAMBDA=LAMKPH*RLAMFAC
                    LAMKPH=LAMBDA
                  ELSE
                    LAMBDA=LAMKPL/RLAMFAC
                    IF((LAMBDA.NE.0.0D0).AND.(LAMBDA.LT.1.0D-20))
     +              LAMBDA=1.0D-20
                    LAMKPL=LAMBDA
                  END IF
                ELSE
                  IF(((LRUN-2)/2)*2.EQ.LRUN-2)THEN
                    LAMBDA=LAMKPL/RLAMFAC
                    IF((LAMBDA.NE.0.0D0).AND.(LAMBDA.LT.1.0D-20))
     +              LAMBDA=1.0D-20
                    LAMKPL=LAMBDA
                  ELSE
                    LAMBDA=LAMKPH*RLAMFAC
                    LAMKPH=LAMBDA
                  END IF
                END IF
              END IF
            ELSE
              IF(NREV.EQ.0)THEN
                LAMBDA=LAMBDA/RLAMFAC
                IF((LAMBDA.NE.0.0D0).AND.(LAMBDA.LT.1.0D-20))
     +          LAMBDA=1.0D-20
              ELSE
                LAMBDA=LAMBDA*RLAMFAC
              END IF
            END IF
            JACUPDATEFLAG=0
            GO TO 410
          ELSE
            CALL WRITINT(ATEMP1,LMRUN)
            WRITE(6,6926) TRIM(ATEMP1)
6926        FORMAT(/,'    Partial parallelisation of lambda search: ',
     +      'running model ',A,' times.....')

#ifdef FLUSHFILE
            CALL FLUSH(IREC)
#endif
#ifdef BEO
            if (BEOMASTER) then
              call RUNMASTER(PARREG,OBSREG,LMRUN,JFAIL)
              if (JFAIL.ne.0) goto 9891
            else
#endif
#ifdef MPEST
            FLENME=TRIM(CASEFL)//'.lambda_runs'
            CALL FFOPEN(JFAIL,IRMR,'w',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(IRMR,'(I5)') LMRUN
            DO LR=1,LMRUN
              WRITE(IRMR,6027) LR,REGLAM(LR)
6027          FORMAT(I5,1X,1PG14.7)
            END DO
            CLOSE(UNIT=IRMR)
            FLENME=TRIM(CASEFL)//'.jacobian_runs'
            CALL DELFILE1(JFAIL,FLENME,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            CALL DORUNS_M(JFAIL,NSLAVE,LMRUN,
     +      ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,
     +      LL,OBSN1,OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,
     +      NPAR,PRECIS,NOPNT,NTPLFLE,NW,
     +      SCALE,OFFSET,PARDEL,PWORD,TEMPFLE,APAR,
     +      OUTFLE,INSFLE,LDSIN,LDSOU,
     +      SINFLE,SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,MSINFLE,
     +      MSOUFLE,MODFLE,WORKVEC7DIM,WORKVEC7)
            IF(JFAIL.EQ.2) GO TO 9890
            IF(JFAIL.NE.0) GO TO 9891
#else
            PIRESTART=0
            CALL DORUNS(JFAIL,NSLAVE,LMRUN,ISTATS,
     +      ISTATR,ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,
     +      LL,OBSN1,OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,
     +      ISTRTME,IRUNTME,JRUN,NPAR,PRECIS,NOPNT,NTPLFLE,NW,
     +      SCALE,OFFSET,PARDEL,PWORD,TEMPFLE,APAR,OREADFLE,
     +      PREADFLE,MANFLE,ASLAVE,NNRUN,OUTFLE,INSFLE,LDSIN,LDSOU,
     +      SINFLE,SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,IDET,SREADFLE,
     +      MREADFLE,SCOM,PIRESTART,PTUNIT,PTFILE,IOPT,REPEATRUN,1,
     +      SLAVEGROUP,WORKVEC7DIM,WORKVEC7)
            IF(JFAIL.NE.0) GO TO 9891
#endif
#ifdef BEO
            endif
#endif
            IF(ISTOP.EQ.2)THEN
              IFIN=10
              GO TO 6000
            ELSE IF(ISTOP.EQ.1) THEN
              IPFAIL=-1
              GO TO 9891
            END IF

            LSEARCH=0
            LSCH1=1
            GO TO 9030
          END IF
        ELSE IF(LSCH1.EQ.1)THEN
          IF(MANYLAM.EQ.0)THEN
            IF(NOPAR.EQ.1) GO TO 7050
          ELSE
            LAMBDA=REGLAM(ITN)
          END IF
          DO 7007 LR=1,LMRUN
            IF(ABS(LAMBDA-REGLAM(LR))/LAMBDA.LE.1.0D-5)THEN
C              IF(NOWRLM.EQ.0) THEN
                IF(LITN.LE.1)THEN
                WRITE(IREC,452,ERR=9350) LAMBDA
                WRITE(6,452) LAMBDA
452             FORMAT(/,T9,'Lambda = ',1PG12.5,' ----->')
#ifdef FLUSHFILE
                CALL FLUSH(IREC)
#endif
                END IF
C              END IF
              NOWRLM=0
              IF((LTFLAG.EQ.0).OR.
     +          ((LTFLAG.EQ.1).AND.(LITN.EQ.1)))THEN
                IMM=IMM+1
                JMM=JMM+1
              END IF
              IF(LTFLAG.NE.0)THEN
                WRITE(6,1145) LTFAC
1145            FORMAT(/,T11,'Line search factor = ',1PG12.5)
                WRITE(IREC,1145,ERR=9350) LTFAC
#ifdef FLUSHFILE
                CALL FLUSH(IREC)
#endif
              END IF
              CALL RETRIEVE_PARALLEL_REGISTER
     +        (IFAIL,NOBS,OBSREG,LR,TMPOBS,OBSREGFILE)
              IF(IFAIL.NE.0) GO TO 9890
C             IF(MANYLAM.NE.0)THEN                   7th march 2009
                CALL RETRIEVE_PARALLEL_REGISTER
     +          (IFAIL,NPAR,PARREG,LR,TMPVAL,PARREGFILE)
                IF(IFAIL.NE.0) GO TO 9890
C             END IF                                 7th march 2009
              GO TO 8010
            END IF
7007      CONTINUE

C -- IF NO PARAMETERS ARE FROZEN WE DO MORE PARALLEL RUNS

          IF(MANYLAM.EQ.0)THEN
            DO 9015 IPP=1,NPAR
              IF(ITRANS(IPP).LT.-100001) GO TO 7050
9015        CONTINUE
          END IF
          ILAMPL=ILAMPL+1
          IF(ILAMPL.GT.ILAMMAX) GO TO 7050  ! FOR IF WE WISH TO SET A LIMIT

#ifdef MPEST
          CALL HMPARLAM(IFAIL,MLAMRUN)
          IF(IFAIL.NE.0)MLAMRUN=MSRUN
          LMRUN=MIN(MLAMRUN,NSRUN)
#else
          CALL HMRUNS(NSLAVE,ISTATS,IRUNTME,IORDER,LMRUN,IDET,MANYLAM)
          LMRUN=MIN(LMRUN,NSRUN)          ! NSRUN is dimension of reglam
#endif
          IF(LDOSCH.LT.0)THEN
            IF(LMRUN.GT.ABS(LDOSCH))LMRUN=ABS(LDOSCH)
          END IF
          IF(MANYLAM.NE.0)THEN
            IF(NUMLAM.LT.NUMLAMORIG)NUMLAM=NUMLAMORIG
            IF(LMRUN.GT.NUMLAM)THEN
              LMRUN=NUMLAM
            ELSE
              NUMLAM=LMRUN
            END IF
          END IF

          IF(LMRUN.LE.0) GO TO 7050
          LRUN=1
          LSCH1=0
          LSEARCH=1

          SITN=ITN-1
          SJMM=ITNJMM
          SLAMBDA=LAMBDA
          SIMM=IMM
          GO TO 6906
        END IF

7050    CONTINUE

C        IF(NOWRLM.EQ.0) THEN
          IF(LITN.LE.1)THEN
          WRITE(IREC,452,ERR=9350) LAMBDA
          WRITE(6,452) LAMBDA
#ifdef FLUSHFILE
          CALL FLUSH(IREC)
#endif
          END IF
C        END IF
        NOWRLM=0
        IF((LTFLAG.EQ.0).OR.
     +    ((LTFLAG.EQ.1).AND.(LITN.EQ.1)))THEN
          IMM=IMM+1
          JMM=JMM+1
        END IF
        IF(LTFLAG.NE.0)THEN
          WRITE(6,1145) LTFAC
          WRITE(IREC,1145,ERR=9350) LTFAC
#ifdef FLUSHFILE
          CALL FLUSH(IREC)
#endif
        END IF
        IF(LTFLAG.NE.0)THEN
          WRITE(6,1149)
1149      FORMAT(T11,'running model .....')
        ELSE
          WRITE(6,1151)
1151      FORMAT(T9,'running model .....')
        END IF

        CALL ZINCTEST(JFAIL,NOCHNG,NPAR,PRECIS,NOPNT,NW,-1,TMPVAL,
     +  PVAL,SCALE,OFFSET,PWORD,APAR,CLINE,J,RTEMP)
        IF(JFAIL.NE.0) GO TO 9891
        CALL STORE_PARALLEL_REGISTER
     +  (IFAIL,NPAR,PARREG,1,TMPVAL,PARREGFILE)
        IF(IFAIL.NE.0) GO TO 9890
        IF(NOCHNG.EQ.1)THEN
          CALL STORE_PARALLEL_REGISTER
     +    (IFAIL,NOBS,OBSREG,1,REFOBS,OBSREGFILE)
          IF(IFAIL.NE.0) GO TO 9890
          NOCHNG=0
        ELSE
#ifdef BEO
          if (BEOMASTER) then
            call RUNMASTER(PARREG,OBSREG,1,JFAIL)
            if (JFAIL.ne.0) goto 9891
          else
#endif
#ifdef MPEST
          FLENME=TRIM(CASEFL)//'.lambda_runs'
          CALL FFOPEN(JFAIL,IRMR,'w',' ',6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          WRITE(IRMR,'(I5)') 1
          WRITE(IRMR,6027) 1,LAMBDA
          CLOSE(UNIT=IRMR)
          FLENME=TRIM(CASEFL)//'.jacobian_runs'
          CALL DELFILE1(JFAIL,FLENME,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          CALL DORUNS_M(JFAIL,NSLAVE,1,
     +    ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,LL,OBSN1,
     +    OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,
     +    NPAR,PRECIS,NOPNT,NTPLFLE,NW,SCALE,
     +    OFFSET,PARDEL,PWORD,TEMPFLE,APAR,
     +    OUTFLE,INSFLE,LDSIN,LDSOU,SINFLE,
     +    SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,MSINFLE,MSOUFLE,
     +    MODFLE,WORKVEC7DIM,WORKVEC7)
          IF(JFAIL.EQ.2) GO TO 9890
          IF(JFAIL.NE.0) GO TO 9891
#else
          PIRESTART=0
          CALL DORUNS(JFAIL,NSLAVE,1,ISTATS,ISTATR,
     +    ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,LL,OBSN1,
     +    OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,ISTRTME,
     +    IRUNTME,JRUN,NPAR,PRECIS,NOPNT,NTPLFLE,NW,SCALE,
     +    OFFSET,PARDEL,PWORD,TEMPFLE,APAR,OREADFLE,PREADFLE,
     +    MANFLE,ASLAVE,NNRUN,OUTFLE,INSFLE,LDSIN,LDSOU,SINFLE,
     +    SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,IDET,SREADFLE,MREADFLE,
     +    SCOM,PIRESTART,PTUNIT,PTFILE,IOPT,REPEATRUN,1,SLAVEGROUP,
     +    WORKVEC7DIM,WORKVEC7)
          IF(JFAIL.NE.0) GO TO 9891
#endif
#ifdef BEO
          endif
#endif
          IF(ISTOP.EQ.2)THEN
            IFIN=10
            GO TO 6000
          ELSE IF(ISTOP.EQ.1) THEN
            IPFAIL=-1
            GO TO 9891
          END IF

        END IF
C        DO 2805 IPP=1,NPAR
C          TMPVAL(IPP)=PARREG(IPP,1)
C2805    CONTINUE

        CALL RETRIEVE_PARALLEL_REGISTER
     +  (IFAIL,NOBS,OBSREG,1,TMPOBS,OBSREGFILE)
        IF(IFAIL.NE.0) GO TO 9890
#else

C        IF(NOWRLM.EQ.0) THEN
          IF(LITN.LE.1)THEN
          WRITE(IREC,452,ERR=9350) LAMBDA
          WRITE(6,452) LAMBDA
452       FORMAT(/,T9,'Lambda = ',1PG12.5,' ----->')
#ifdef FLUSHFILE
          CALL FLUSH(IREC)
#endif
          END IF
C        END IF
        NOWRLM=0
        IF((LTFLAG.EQ.0).OR.
     +    ((LTFLAG.EQ.1).AND.(LITN.EQ.1)))THEN
          IMM=IMM+1
          JMM=JMM+1
        END IF
        IF(LTFLAG.NE.0)THEN
          WRITE(6,1145) LTFAC
1145      FORMAT(/,T11,'Line search factor = ',1PG12.5)
          WRITE(IREC,1145,ERR=9350) LTFAC
#ifdef FLUSHFILE
          CALL FLUSH(IREC)
#endif
        END IF
        IF(LTFLAG.NE.0)THEN
          WRITE(6,1149)
1149      FORMAT(T11,'running model .....')
        ELSE
          WRITE(6,1151)
1151      FORMAT(T9,'running model .....')
        END IF

#ifdef BEO
        REPEATFLAG=0
#else
        REPEATFLAG=1
#endif
        FAILUREFLAG=0
        CALL MODEL(JFAIL,NPAR,NOBS,ASIZE,NINSTR,NTPLFLE,NINSFLE,NUML,
     +  NBLBMX,SCALE,OFFSET,TMPVAL,NW,PWORD,APAR,TMPOBS,AOBS,I,PVAL,
     +  NOCHNG,PRECIS,NOPNT,LCINS,LL,OBSN1,OBSN2,IIOBS,INSFLE,TEMPFLE,
     +  INFLE,OUTFLE,PARDEL,MRKDEL,A,CLINE,BUF,COMLIN(1),J,RTEMP,
     +  MESSFILE,INCPAR(1),ITRANS,REPEATFLAG,0)
        IF(JFAIL.NE.0) THEN
          JFAIL=0
          IF(LAMFORGIVE.EQ.1)THEN
            DO I=1,NOBS
              TMPOBS(I)=-1.11E35
            END DO
            FAILUREFLAG=1
          ELSE
            GO TO 9891
          END IF
        END IF

#endif

8010    CONTINUE

        IF(IPRLL.EQ.1)THEN
          FAILUREFLAG=0
          IF(LAMFORGIVE.EQ.1)THEN
            DO I=1,NOBS
              IF(.NOT.EQUALS(TMPOBS(I),-1.11D35)) GO TO 8011
            END DO
            FAILUREFLAG=1
8011        CONTINUE
          END IF
        END IF

C -- If PEST is running in Pareto mode then normal weights are re-employed
C    for purpose of objective function calculation.

c        IF(PESTMODE.EQ.4)THEN
c          J=0
c          DO I=1,NXROW
c            IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
c              J=J+1
c              OWGHT(I)=OWGHTKP(J)
c            END IF
c          END DO
c        END IF

C -- IF ANY COVARIANCE MATRICES HAVE BEEN SUPPLIED, THEN THE CALCULATED
C    "OBSERVATIONS" ARE ROTATED.

        IF(NUMCOV.GT.0)THEN
          DO 6970 JCOV=1,NUMCOV
            IF(NOBSCOV(JCOV).LE.0) GO TO 6970
            IGROUP=COVGP(JCOV)
            I=0
            DO 6960 J=1,NOBS
              IF(NOBGNM(J).EQ.IGROUP)THEN
                I=I+1
                W1(I)=TMPOBS(J)
              END IF
6960        CONTINUE
            IF(I.GT.0)THEN
              CALL ROTATE(1,NOBSCOV(JCOV),NOBSCOV(JCOV),1,1,1,
     +        COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,NXROW,NOBGNM)
              I=0
              DO 6965 J=1,NOBS
                IF(NOBGNM(J).EQ.IGROUP)THEN
                  I=I+1
                  TMPOBS(J)=W1(I)
                END IF
6965          CONTINUE
            END IF
6970      CONTINUE
        END IF

        IF(IPRED.NE.0)TMPOBS(NOBS)=TMPOBS(NOBS)+E_PRED

        IF(LTFLAG.NE.0)THEN  ! only if being done in a line search
          DO 3480 IOBS=1,NOBS
            LTTMPOBS(IOBS,LITN)=TMPOBS(IOBS)
3480      CONTINUE
        END IF
        GO TO 3500

3490    CONTINUE     ! Line search finished.

        DO 3495 IOBS=1,NOBS
          TMPOBS(IOBS)=LTTMPOBS(IOBS,KITN)
3495    CONTINUE
        TEMPSI=LTPHI(KITN)
        GO TO 1160

3500    CONTINUE

        IF(ISTOP.EQ.2)THEN
          IFIN=10
          GO TO 6000
        ELSE IF(ISTOP.EQ.1) THEN
          IPFAIL=-1
          GO TO 9891
        END IF
        IF(NOCHNG.NE.1) GO TO 1160
1155    DO 1156 IOBS=1,NOBS
1156    TMPOBS(IOBS)=REFOBS(IOBS)
1160    IF(NPRIOR.NE.0)  CALL PRRCLC(JFAIL,NPAR,NOBS,NXROW,NESPAR,
     +    TMPVAL,ITRANS,TMPOBS,X,CLINE,SCALE,NREGADJPAR,PRIORPAR)
          IF(JFAIL.NE.0) GO TO 9891
C        TMPSIL=TEMPSI
C        IF(IPRED.GT.0)TMPRL=TEMPRD
        CALL OBJCLC(1,TEMPSI,NXROW,NPRIOR,NOBS,TMPOBS,OVAL,OWGHT,
     +  NOBGNM,NOBSGP,IRGP,SUM1)
        IF(PREDNOISE.GT.0) TEMPSI=TEMPSI+RES_PRED
        IF(IREG.EQ.1)THEN
          CALL OBJCLC(888888,TEMPHIR,NXROW,NPRIOR,NOBS,TMPOBS,
     +    OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,TEMPHIM)
          TEMPHIR=TEMPHIR/WF/WF
        END IF
        IF(FAILUREFLAG.EQ.1)THEN
          FAILURECOUNT=FAILURECOUNT+1
          IF(ITN.EQ.1)THEN
            TEMPSI=1.0E35
            TEMPHIM=1.0E35
            TEMPHIR=1.0E35
          ELSE IF(ITN.EQ.2)THEN
            TEMPSI=1.1E35
            TEMPHIM=1.1E35
            TEMPHIR=1.1E35
          ELSE
            TEMPSI=0.9E35
            TEMPHIM=0.9E35
            TEMPHIR=0.9E35
          END IF
        ELSE
          SUCCESSCOUNT=SUCCESSCOUNT+1
        END IF

        IF(LTLAST.EQ.1) GO TO 3700
        IF(LTFLAG.EQ.0) GO TO 3700
        WRITE(IREC,3510,ERR=9350) TEMPSI
        WRITE(6,3510) TEMPSI
3510    FORMAT(T26,'Phi = ',1PG12.5)
        WRITE(6,3530) TMPOBS(NOBS)
        WRITE(IREC,3530,ERR=9350) TMPOBS(NOBS)
3530    format(T19,'Prediction = ',1PG12.5)
        IF(PREDNOISE.NE.0)THEN
          WRITE(6,3531) E_PRED
          WRITE(IREC,3531,ERR=9350) E_PRED
3531      FORMAT(T13,'Predictive error = ',1PG12.5)
        END IF
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif

C -- THE NEXT SEGMENT OF CODE IS REQUIRED ONLY FOR LINE SEARCH ALGORITHM

        LTPHI(LITN)=TEMPSI
        LTPRED(LITN)=TMPOBS(NOBS)
        LTFACC(LITN)=LTFAC
        IF(LTBIS.NE.0) GO TO 3300
        IF(LITN.EQ.1)THEN
          LTPHI(0)=PSI
          LTPRED(0)=REFPRED
          LTFACC(0)=0.0
        END IF

C -- TERMINATE LINE SEARCH?

        LTRTEMP1=ABS(LTPRED(LITN)+LTPRED(LITN-1))*0.5
        LTRTEMP2=ABS(LTPRED(LITN)-LTPRED(LITN-1))
        IF(LTRTEMP1.NE.0.0D0) THEN
          IF(ABS(LTRTEMP2/LTRTEMP1).LE.RELPREDSTP*0.1) THEN
            ICONVERGE=1
            GO TO 3650
          END IF
        END IF
        IF(LTRTEMP2.LE.ABSPREDSTP*0.1) THEN
          ICONVERGE=1
          GO TO 3650
        END IF
        IF(LITN.GE.3)THEN
          IF(LTPHI(LITN).LE.PD1)THEN
            IF(LTPHI(LITN).GT.PD12)THEN
              ICONVERGE=1
              GO TO 3650
            END IF
          END IF
        END IF

C -- ASSESS WHERE WE ARE IN RELATION TO OUR TARGET

        IF(LITN.EQ.1)THEN
          IF(LTPHI(0).GT.PD12)THEN
            IF(LTPHI(1).LT.LTPHI(0))THEN
              IF(LTPHI(1).LE.PD12) THEN
                IF(PREDHL.EQ.'l')THEN
                  IF(LTPRED(1).LT.LTPRED(0)) THEN
                    LTFLAGX=1
                    GO TO 3351
                  END IF
                ELSE IF(PREDHL.EQ.'h')THEN
                  IF(LTPRED(1).GT.LTPRED(0)) THEN
                    LTFLAGX=1
                    GO TO 3351
                  END IF
                END IF
                GO TO 3300  !subtended
              END IF
3351          CONTINUE
              LTFAC=LTFAC*MULSCHFAC
              LTREV=0
              GO TO 3400
            ELSE
C              LTFAC=LTFAC/MULSCHFAC
C              LTREV=1
C              GO TO 3400
               IF(LTPHI(1)-LTPHI(0).EQ.0.0D0) GO TO 3650
              LTFAC=(PD12-LTPHI(0))/(LTPHI(1)-LTPHI(0))*LTFAC
              LTNEG=1
              GO TO 3400
            END IF
          ELSE
            IF(LTPHI(1).GT.LTPHI(0))THEN
              IF(LTPHI(1).GE.PD12)GO TO 3300               !subtended
              IF(PREDHL.EQ.'l') THEN
                IF(LTPRED(1).GT.LTPRED(0)) GO TO 3650     !abort
              ELSE IF(PREDHL.EQ.'h')THEN
                IF(LTPRED(1).LT.LTPRED(0)) GO TO 3650
              END IF
              LTFAC=LTFAC*MULSCHFAC
              LTREV=0
              GO TO 3400
            ELSE
              IF(PREDHL.EQ.'l')THEN
                IF(LTPRED(1).GT.LTPRED(0)) GO TO 3650     !abort
              ELSE IF(PREDHL.EQ.'h')THEN
                IF(LTPRED(1).LT.LTPRED(0)) GO TO 3650
              END IF
              LTFAC=LTFAC*MULSCHFAC
              LTREV=0
              GO TO 3400
            END IF
          END IF
        ELSE
          IF(LTNEG.EQ.1)THEN
            IF(LTPHI(LITN).LE.PD12) GO TO 3300            !subtended
            LTFAC=LTFAC+(PD12-LTPHI(LITN))/
     +            (LTPHI(LITN-1)-LTPHI(LITN))*(LTFACC(LITN-1)-LTFAC)
            GO TO 3400
          END IF
          IF(LTPHI(0).GT.PD12)THEN
            IF(LTFLAGX.NE.0)THEN
              IF(LTPHI(LITN).GT.PD12) GO TO 3300             !subtended
              IF(PREDHL.EQ.'h')THEN
                IF(LTPRED(LITN).LE.LTPRED(LITN-1)) GO TO 3300
              ELSE IF(PREDHL.EQ.'l')THEN
                IF(LTPRED(LITN).GE.LTPRED(LITN-1)) GO TO 3300
              END IF
              LTFAC=LTFAC*MULSCHFAC
              GO TO 3400
            END IF
            IF(LTPHI(LITN).LE.PD12) THEN
              IF(PREDHL.EQ.'l') THEN
                IF(LTPRED(LITN).LT.LTPRED(LITN-1)) THEN
                  LTFLAGX=1
                  LTFAC=LTFAC*MULSCHFAC
                  GO TO 3400
                END IF
              ELSE IF(PREDHL.EQ.'h')THEN
                IF(LTPRED(LITN).GT.LTPRED(LITN-1)) THEN
                  LTFLAGX=1
                  LTFAC=LTFAC*MULSCHFAC
                  GO TO 3400
                END IF
              END IF
              GO TO 3300              !subtended
            END IF
            IF(LTPHI(LITN).GT.LTPHI(LITN-1)) GO TO 3650    !abort
            IF(LTREV.EQ.0) THEN
              LTFAC=LTFAC*MULSCHFAC
            ELSE
              IF((LITN.EQ.2).AND.(LTPHI(LITN).GE.LTPHI(0))) GO TO 3650 !arbitrary
              LTFAC=LTFAC/MULSCHFAC
            END IF
            GO TO 3400
          ELSE
            IF(PREDHL.EQ.'h')THEN
              IF(LTPRED(LITN).LT.LTPRED(LITN-1)) GO TO 3650  !abort
            ELSE
              IF(LTPRED(LITN).GT.LTPRED(LITN-1)) GO TO 3650  !abort
            END IF
            IF(LTPHI(LITN).GE.PD12) GO TO 3300   !subtended
            LTFAC=LTFAC*MULSCHFAC
            GO TO 3400
          END IF
        END IF

3300    CONTINUE   !phi is subtended
        ISUBTEND=1
        IF(LTBIS.NE.0)THEN
          LTRTEMP1=ABS(PD12-LTPHI(LITN-1))
          LTRTEMP2=ABS(PD12-LTPHI(LITN))
          IF(LTRTEMP2.GE.LTRTEMP1) THEN
            LTCOUNT=LTCOUNT+1
            IF(LTCOUNT.GE.6) GO TO 3650   !arbitrary
          END IF
        END IF

C -- AS A BASIS FOR THE LINE SEARCH WE FIND THE FACTOR FOR WHICH PHI IS
C    JUST ABOVE PD12. NOTE THAT THERE MAY BE TWO OF THESE. IF SO, WE PICK
C    THE ONE FOR WHICH THE PREDICTION IS HIGHEST/LOWEST.

        LT1=-999
        LT2=-999
        DO 3315 LJTN=0,LITN
          IF(LTPHI(LJTN).LT.PD12)GO TO 3315
          LTFAC1=LTFACC(LJTN)
          LTFACMIN=-1.0D300
          LTFLOTN=-999
          DO 3314 LLTN=0,LITN
            IF(LLTN.EQ.LJTN) GO TO 3314
            IF(LTFACC(LLTN).GT.LTFAC1) GO TO 3314
            IF(LTFACC(LLTN).GT.LTFACMIN) THEN
              LTFACMIN=LTFACC(LLTN)
              LTFLOTN=LLTN
            END IF
3314      CONTINUE
          LTFACMAX=1.0D300
          LTFHITN=-999
          DO 3316 LLTN=0,LITN
            IF(LLTN.EQ.LJTN) GO TO 3316
            IF(LTFACC(LLTN).LE.LTFAC1) GO TO 3316
            IF(LTFACC(LLTN).LT.LTFACMAX)THEN
              LTFACMAX=LTFACC(LLTN)
              LTFHITN=LLTN
            END IF
3316      CONTINUE
          IF(LTFLOTN.NE.-999)THEN
            IF(LTPHI(LTFLOTN).LE.PD12) GO TO 3323
          END IF
          IF(LTFHITN.NE.-999)THEN
            IF(LTPHI(LTFHITN).LE.PD12) GO TO 3323
          END IF
          GO TO 3315
3323      CONTINUE
          IF(LT1.EQ.-999)THEN
            IF(LTFLOTN.NE.-999)THEN
              IF(LTPHI(LTFLOTN).LE.PD12)THEN
                LT1=LTFLOTN
              ELSE
                LT1=LTFHITN
              END IF
            ELSE
              IF(LTFHITN.NE.-999)LT1=LTFHITN
            END IF
            LT2=LJTN
            LTPRED1=LTPRED(LT2)
          ELSE
            IF(PREDHL.EQ.'h')THEN
              IF(LTPRED(LJTN).LT.LTPRED1) GO TO 3315
            ELSE
              IF(LTPRED(LJTN).GT.LTPRED1) GO TO 3315
            END IF
            IF(LTFLOTN.NE.-999)THEN
              IF(LTPHI(LTFLOTN).LE.PD12)THEN
                LT1=LTFLOTN
              ELSE
                LT1=LTFHITN
              END IF
            ELSE
              IF(LTFHITN.NE.-999)LT1=LTFHITN
            END IF
            LT2=LJTN
            LTPRED1=LTPRED(LT2)
          END IF
3315    CONTINUE

C -- THE FOLLOWING IS FOR EMERGENCIES

        IF((LT1.EQ.-999).OR.(LT2.EQ.-999))THEN
          DO 3335 LJTN=0,LITN
            IF(LTPHI(LJTN).GE.PD12)THEN
              LT2=LJTN
              GO TO 3336
            END IF
3335      CONTINUE
3336      CONTINUE
          DO 3339 LJTN=LITN,0,-1
            IF(LJTN.EQ.LT2) GO TO 3339
            IF(LTPHI(LJTN).LE.PD12) THEN
              LT1=LJTN
              GO TO 3345
            END IF
3339      CONTINUE
3345      CONTINUE
        END IF

C -- TERMINATE LINE SEARCH?

        LTRTEMP1=ABS(LTPRED(LITN)+LTPRED(LITN-1))*0.5
        LTRTEMP2=ABS(LTPRED(LITN)-LTPRED(LITN-1))
        IF(LTRTEMP1.NE.0.0D0) THEN
          IF(ABS(LTRTEMP2/LTRTEMP1).LE.RELPREDSTP*0.1) THEN
            ICONVERGE=1
            GO TO 3650
          END IF
        END IF
        IF(LTRTEMP2.LE.ABSPREDSTP*0.1) THEN
          ICONVERGE=1
          GO TO 3650
        END IF
        IF(LITN.GE.3)THEN
          IF(LTPHI(LITN).LE.PD1)THEN
            IF(LTPHI(LITN).GT.PD12)THEN
              ICONVERGE=1
              GO TO 3650
            END IF
          END IF
        END IF

C -- A NEW LINE SEARCH FACTOR IS EVALUATED.

C -- PARABOLIC OPTION

        IF(LITN.LE.1) GO TO 3620
        IF(((LT2LAST.EQ.LT2).AND.(LT1LAST.EQ.LT1)).OR.
     +     ((LT1LAST.EQ.LT2).AND.(LT2LAST.EQ.LT1)))THEN
            GO TO 3650
        ELSE
          LT1LAST=LT1
          LT2LAST=LT2
        END IF
        LTRTEMP2=1.0D300
        LT3=-999
        DO 3550 LJTN=0,LITN
        IF((LJTN.EQ.LT1).OR.(LJTN.EQ.LT2)) GO TO 3550
        LTRTEMP1=MIN(ABS(LTFACC(LJTN)-LTFACC(LT1)),
     +               ABS(LTFACC(LJTN)-LTFACC(LT2)))
        IF(LTRTEMP1.LT.LTRTEMP2)THEN
          LTRTEMP2=LTRTEMP1
          LT3=LJTN
        END IF
3550    CONTINUE
        IF(LT3.EQ.-999) GO TO 3620
        LTF1=LTFACC(LT1)
        LTF2=LTFACC(LT2)
        LTF3=LTFACC(LT3)
        LTP1=LTPHI(LT1)
        LTP2=LTPHI(LT2)
        LTP3=LTPHI(LT3)
        LTF2F1=LTF2-LTF1
        LTF1F3=LTF1-LTF3
        LTF3F2=LTF3-LTF2
        IF((LTF2F1.EQ.0.0D0).OR.(LTF1F3.EQ.0.0D0).OR.
     +     (LTF3F2.EQ.0.0D0)) GO TO 3620
        LTA=-LTP1/LTF2F1/LTF1F3
     +      -LTP2/LTF2F1/LTF3F2
     +      -LTP3/LTF3F2/LTF1F3
        LTB= LTP1*(LTF2+LTF3)/LTF2F1/LTF1F3
     +      +LTP2*(LTF3+LTF1)/LTF2F1/LTF3F2
     +      +LTP3*(LTF1+LTF2)/LTF3F2/LTF1F3
        LTC=-LTP2*LTF1*LTF3/LTF2F1/LTF3F2
     +      -LTP1*LTF2*LTF3/LTF2F1/LTF1F3
     +      -LTP3*LTF1*LTF2/LTF3F2/LTF1F3
        IF(LTA.EQ.0.0D0) GO TO 3620
        LTRTEMP1=LTB*LTB-4.0D0*LTA*(LTC-(PD1+PD0)*0.5D0)  !a little arbitrary
        IF(LTRTEMP1.LT.0.0D0) GO TO 3620
        LTFACHI=(-LTB+SQRT(LTRTEMP1))/2.0D0/LTA
        LTFACLO=(-LTB-SQRT(LTRTEMP1))/2.0D0/LTA
        RMAXFAC=-1.0D300
        RMINFAC=1.0D300
        DO 3619 LKTN=0,LITN
          IF(LTFACC(LKTN).LT.RMINFAC)RMINFAC=LTFACC(LKTN)
          IF(LTFACC(LKTN).GT.RMAXFAC)RMAXFAC=LTFACC(LKTN)
3619    CONTINUE
        ILTFACHI=1
        ILTFACLO=1
        IF((LTFACHI.GT.RMAXFAC).OR.(LTFACHI.LT.RMINFAC))ILTFACHI=0
        IF((LTFACLO.GT.RMAXFAC).OR.(LTFACLO.LT.RMINFAC))ILTFACLO=0
        IF((ILTFACHI.EQ.0).AND.(ILTFACLO.EQ.0))THEN
          RDISTLO=ABS(LTFACLO-LTFACC(LITN))
          RDISTHI=ABS(LTFACHI-LTFACC(LITN))
          IF(RDISTLO.LE.RDISTHI)THEN
            LTFAC=LTFACLO
          ELSE
            LTFAC=LTFACHI
          END IF
        ELSE IF(ILTFACHI.EQ.0)THEN
          LTFAC=LTFACLO
        ELSE IF(ILTFACLO.EQ.0)THEN
          LTFAC=LTFACHI
        ELSE
        IF(PREDHL.EQ.'h')THEN
          IF(LTPRED(LT2).GE.LTPRED(LT1))THEN
            IF(LTFACC(LT2).GE.LTFACC(LT1))THEN
              LTFAC=MAX(LTFACHI,LTFACLO)
            ELSE
              LTFAC=MIN(LTFACHI,LTFACLO)
            END IF
          ELSE
            IF(LTFACC(LT2).GE.LTFACC(LT1))THEN
              LTFAC=MIN(LTFACHI,LTFACLO)
            ELSE
              LTFAC=MAX(LTFACHI,LTFACLO)
            END IF
          END IF
        ELSE
          IF(LTPRED(LT2).GE.LTPRED(LT1))THEN
            IF(LTFACC(LT2).GE.LTFACC(LT1))THEN
              LTFAC=MIN(LTFACHI,LTFACLO)
            ELSE
              LTFAC=MAX(LTFACHI,LTFACLO)
            END IF
          ELSE
            IF(LTFACC(LT2).GE.LTFACC(LT1))THEN
              LTFAC=MAX(LTFACHI,LTFACLO)
            ELSE
              LTFAC=MIN(LTFACHI,LTFACLO)
            END IF
          END IF
        END IF
        END IF
        IF((LTFAC.EQ.LTF1).OR.(LTFAC.EQ.LTF2).OR.(LTFAC.EQ.LTF3))
     +  GO TO 3650
        LTBIS=LTBIS+1
        GO TO 3400

C -- LINEAR OPTION

3620    CONTINUE
        IF(((LT2LAST.EQ.LT2).AND.(LT1LAST.EQ.LT1)).OR.
     +     ((LT1LAST.EQ.LT2).AND.(LT2LAST.EQ.LT1)))THEN
            GO TO 3650
        ELSE
          LT1LAST=LT1
          LT2LAST=LT2
        END IF
        LTPHIHI=LTPHI(LT2)
        LTPHILO=LTPHI(LT1)
        IF(LTPHIHI.NE.LTPHILO)THEN
          LTFACLO=LTFACC(LT1)
          LTFACHI=LTFACC(LT2)
          IF(LTFACC(LT2).GE.LTFACC(LT1))THEN
            LTFAC=LTFACLO+(PD12-LTPHILO)/
     +      (LTPHIHI-LTPHILO)*(LTFACHI-LTFACLO)
          ELSE
            LTFAC=LTFACLO-(PD12-LTPHILO)/
     +      (LTPHIHI-LTPHILO)*(LTFACHI-LTFACLO)
          END IF
        ELSE
          GO TO 3650
        END IF
C        IF(LTFAC.LE.0.0) GO TO 3650
        IF((LTFAC.EQ.LTFACLO).OR.(LTFAC.EQ.LTFACHI)) GO TO 3650
        LTBIS=LTBIS+1
        GO TO 3400

3650    CONTINUE     !no more line search - tidy up

C -- If this is the first Marquardt lambda, we decide whether it is a good idea
C    to save model runs on the next Marquardt lambda.

        LINESAVE=0
        IF(ITN.GE.1)THEN
          IF(NSEARCH.GT.1)THEN
C            IF(ISUBTEND.NE.0)THEN
              IF(ICONVERGE.NE.0)THEN
                IF(LTFAC.GT.0.0D0)THEN
                  LINESAVE=1
                  BESTFAC=LTFAC
                END IF
C              END IF
            END IF
          END IF
        END IF

        IF(PREDHL.EQ.'h')THEN
          LTRTEMP1=-1.0D300
        ELSE
          LTRTEMP1=1.0D300
        END IF
        KITN=-1
        DO 3670 LJTN=1,LITN
          IF(LTPHI(LJTN).LE.PD1) THEN
            IF(PREDHL.EQ.'h')THEN
              IF(LTPRED(LJTN).GT.LTRTEMP1)THEN
                KITN=LJTN
                LTRTEMP1=LTPRED(LJTN)
              END IF
            ELSE
              IF(LTPRED(LJTN).LT.LTRTEMP1)THEN
                KITN=LJTN
                LTRTEMP1=LTPRED(LJTN)
              END IF
            END IF
          END IF
3670    CONTINUE

        IF(KITN.EQ.-1)THEN
          LTRTEMP1=1.0D300
          DO 3680 LJTN=1,LITN
            IF(LTPHI(LJTN).LT.LTRTEMP1)THEN
              KITN=LJTN
              LTRTEMP1=LTPHI(LJTN)
            END IF
3680      CONTINUE
        END IF
        LTLAST=1
        LTFAC=LTFACC(KITN)
        GO TO 3400

3700    CONTINUE    ! End of line search code

        IF(IPRED.GT.0)THEN
          TEMPRD=TMPOBS(NOBS)
          IF(ITN.EQ.1)THEN
            PSI1=TEMPSI
            PRED1=TMPOBS(NOBS)
          END IF
        ELSE
          IF(ITN.EQ.1) PSI1=TEMPSI
        END IF
        FRCPSI=TEMPSI/PSI
        IF(FRCPSI.LT.BESTFRAC)BESTFRAC=FRCPSI
        PDFLAG=0
        IF(IPRED.GT.0)THEN
          IF(LTFLAG.EQ.0)THEN
            CALL PSIWRT(JFAIL,IREC,TEMPSI,-1.0D0,CLINE,FAILUREFLAG)
            IF(JFAIL.NE.0) GO TO 9891
            CALL PSIWRT(JFAIL,6,TEMPSI,-1.0D0,CLINE,FAILUREFLAG)
            IF(JFAIL.NE.0) GO TO 9891
            WRITE(IREC,2202) TMPOBS(NOBS)
            WRITE(6,2202) TMPOBS(NOBS)
2202        FORMAT('    Prediction = ',1PG12.5)
            IF(PREDNOISE.NE.0)THEN
              WRITE(6,2203) E_PRED
              WRITE(IREC,2203,ERR=9350) E_PRED
2203          FORMAT('   Pred. error = ',1PG12.5)
            END IF
#ifdef FLUSHFILE
            CALL FLUSH(IREC)
#endif
          END IF
          IF(NEWPSI.GT.PD1)THEN
            IF(TEMPSI.LT.NEWPSI)PDFLAG=1
          ELSE
            IF(TEMPSI.LE.PD1)THEN
              IF(PREDHL.EQ.'l')THEN
                IF(TEMPRD.LT.NEWPRED) PDFLAG=1
              ELSE
                IF(TEMPRD.GT.NEWPRED) PDFLAG=1
              END IF
            END IF
          END IF
        ELSE
          CALL PSIWRT(JFAIL,IREC,TEMPSI,FRCPSI,CLINE,FAILUREFLAG)
          IF(JFAIL.NE.0) GO TO 9891
          CALL PSIWRT(JFAIL,6,TEMPSI,FRCPSI,CLINE,FAILUREFLAG)
          IF(JFAIL.NE.0) GO TO 9891
          IF(IREG.EQ.1)THEN
            IF(FAILUREFLAG.EQ.0)THEN
              WRITE(IREC,5255) TEMPHIM
              WRITE(6,5255) TEMPHIM
5255          FORMAT('     Meas. fn. = ',1PG12.5)
              WRITE(IREC,5256) TEMPHIR
              WRITE(6,5256) TEMPHIR
5256          FORMAT('    Regul. fn. = ',1PG12.5)
            END IF
          END IF
          IF(KPRED.NE.0)THEN
            WRITE(IREC,2202) TMPOBS(KPRED)
            WRITE(6,2202) TMPOBS(KPRED)
          END IF
          IF(IREG.EQ.0)THEN
            IF(TEMPSI.LT.NEWPSI) PDFLAG=1
          ELSE
            IF(NEWPHIM.GT.PHIMACCEPT)THEN
              IF(TEMPHIM.LT.NEWPHIM)PDFLAG=1
            ELSE
              IF(TEMPHIM.LE.PHIMACCEPT)THEN
                IF(TEMPHIR.LT.NEWPHIR)PDFLAG=1
              END IF
            END IF
          END IF
#ifdef FLUSHFILE
          CALL FLUSH(IREC)
#endif
        END IF
        IF(PDFLAG.EQ.1)THEN
          DO 1170 IPP=1,NPAR
1170      NEWVAL(IPP)=TMPVAL(IPP)
          DO 1180 IROW=1,NXROW
1180      NEWOBS(IROW)=TMPOBS(IROW)
          NEWPSI=TEMPSI
          IF(IREG.EQ.1)THEN
            NEWPHIM=TEMPHIM
            NEWPHIR=TEMPHIR
          END IF
          NROL=RO
          NJPPL0=JPP0
          NJPPL1=JPP1
          NJPPL2=JPP2
          NDMXL0=DMX0
          NDMXL1=DMX1
          NDMXL2=DMX2
          NEWLAM=LAMBDA
          NEWALPHA=ALPHA
          NEWNUMEIG=NUMEIG
          IF(IPRED.GT.0)THEN
            NEWPRED=TEMPRD
            NEWE_PRED=E_PRED
            NEWEW_PRED=EW_PRED
            NEWRES_PRED=RES_PRED
          END IF
        END IF

C -- THE MARQUARDT LAMBDA IS ADJUSTED

        IF(NUMLAM.EQ.1) THEN
          WRITE(IREC,1191,ERR=9350) NUMLAM
          WRITE(6,1191) NUMLAM
1191      FORMAT(/,T5,'No more lambdas: allowed lambdas per iteration',
     +    ' =',I3)
          GO TO 1200
        END IF
        PDFLAG=0
        IF(IPRED.GT.0)THEN
          IF((FRCPSI.LE.PHIRATSUF).AND.(TEMPSI.GT.PD2))PDFLAG=1
        ELSE
          IF(FRCPSI.LE.PHIRATSUF) PDFLAG=1
        END IF
        IF(FRFLAG.EQ.1) PDFLAG=1
        IF(MANYLAM.NE.0) PDFLAG=0
        IF(PDFLAG.EQ.1)THEN
          IF((FRFLAG.EQ.1).AND.(TEMPSI.GE.TMPSIL))THEN
            WRITE(IREC,1193,ERR=9350)
            WRITE(6,1193)
            GO TO 1200
          END IF
#ifdef PARALLEL
          IF(LSCH1.EQ.1)THEN
            IF(NREV.EQ.0)THEN
              DO 6995 LR=1,LMRUN
                IF(REGLAM(LR).LT.LAMBDA)THEN
                   LAMBDA=REGLAM(LR)
                   FRFLAG=1
                   JACUPDATEFLAG=0
                   GO TO 410
                END IF
6995          CONTINUE
            END IF
          END IF
#endif
          WRITE(IREC,1114,ERR=9350) PHIRATSUF
          WRITE(6,1114) PHIRATSUF
1114      FORMAT(/,T5,'No more lambdas: phi is less than ',F6.4,
     +    ' of starting phi')
          GO TO 1200
        END IF
        IF(MANYLAM.EQ.0)THEN
        IF(IREG.NE.0)THEN
          IF(REGCONTINUE.EQ.0)THEN
            IF(NEWPHIM.LE.PHIMLIM)THEN
              WRITE(6,1115)
              WRITE(IREC,1115,ERR=9350)
1115          FORMAT(/,T5,'No more lambdas: measurement obj. fn. less ',
     +        'than PHIMLIM.')
              GO TO 1200
            END IF
          END IF
        END IF
        END IF
        IF(IMM.EQ.NUMLAM) THEN
          WRITE(IREC,1191,ERR=9350) NUMLAM
          WRITE(6,1191) NUMLAM
          GO TO 1200
        END IF
        IF(ITN.EQ.1) THEN
          LAMBDA=LAMBDA/RLAMFAC
          IF((LAMBDA.NE.0.0D0).AND.(LAMBDA.LT.1.0D-20))
     +    LAMBDA=1.0D-20
          JACUPDATEFLAG=1
          IF(LSCH1.EQ.1)JACUPDATEFLAG=0
          GO TO 410
        END IF
        IF((ITN.EQ.3).AND.(NREV.EQ.1).AND.(PSI1.LT.1.0D300))THEN
          RTEMP=PSI1
          IF(IPRED.GT.0) RPTEMP=PRED1
        ELSE
          RTEMP=TMPSIL
          IF(IPRED.GT.0)RPTEMP=TMPRL
        END IF

        IF((IPRED.EQ.0).OR.
     +    ((IPRED.GT.0).AND.(NEWPSI.GT.PD1)))THEN
          IF(TEMPSI.LE.RTEMP) THEN
            IF((IPRED.EQ.0).OR.
     +        ((IPRED.GT.0).AND.(TEMPSI.GT.PD2)))THEN
              IF((RTEMP-TEMPSI)/RTEMP.LT.PHIREDLAM) THEN
                IF(MANYLAM.EQ.0)THEN
                WRITE(IREC,1192,ERR=9350) PHIREDLAM
                WRITE(6,1192) PHIREDLAM
1192            FORMAT(/,T5,'No more lambdas: relative phi reduction ',
     +          'between lambdas less than ',F6.4)
                GO TO 1200
              END IF
              END IF
            ELSE IF(IPRED.GT.0)THEN
              IF(ABS((RTEMP-TEMPSI)/(RTEMP-PD0)).LT.PHIREDLAM)THEN
                IF(MANYLAM.EQ.0)THEN
                WRITE(IREC,1194,ERR=9350) PHIREDLAM
                WRITE(6,1194) PHIREDLAM
1194            FORMAT(/,T5,'No more lambdas: rel. [phi-pd0] ',
     +          'reduction between lambdas less than ',F6.4)
                GO TO 1200
                END IF
              END IF
            END IF
            IF(NREV.EQ.0) THEN
              LAMBDA=LAMBDA/RLAMFAC
              IF((LAMBDA.NE.0.0D0).AND.(LAMBDA.LT.1.0D-20))
     +        LAMBDA=1.0D-20
            ELSE
              NREV=NREV+1
              LAMBDA=LAMBDA*RLAMFAC
            END IF
            JACUPDATEFLAG=1
            IF(LSCH1.EQ.1)JACUPDATEFLAG=0
            GO TO 410
          END IF
          IF(NREV.EQ.1) THEN
            IF(MANYLAM.EQ.0)THEN
            WRITE(IREC,1193,ERR=9350)
            WRITE(6,1193)
1193        FORMAT(/,T5,'No more lambdas: phi rising')
            GO TO 1200
            END IF
          END IF
          IF(ITN.GE.3) THEN
            IF(MANYLAM.EQ.0)THEN
            WRITE(IREC,1193,ERR=9350)
            WRITE(6,1193)
            GO TO 1200
            END IF
          END IF
          LAMBDA=LAMSTR*RLAMFAC
          NREV=1
          JACUPDATEFLAG=1
          IF(LSCH1.EQ.1)JACUPDATEFLAG=0
          GO TO 410
        END IF

        IF(IPRED.GT.0)THEN
          IF(TEMPSI.GT.PD1) THEN
            IF((NREV.EQ.1).OR.(ITN.GE.3)) THEN
              IF(MANYLAM.EQ.0)THEN
              WRITE(IREC,2300,ERR=9350)
              WRITE(6,2300)
2300          FORMAT(/,T5,'No more lambdas: phi has risen above ',
     +        'target')
              GO TO 1200
              END IF
            ELSE
              NREV=1
              LAMBDA=LAMSTR*RLAMFAC
              JACUPDATEFLAG=1
              IF(LSCH1.EQ.1)JACUPDATEFLAG=0
              GO TO 410
            END IF
          END IF
          IF(((PREDHL.EQ.'l').AND.(TEMPRD.LE.RPTEMP)).OR.
     +      ((PREDHL.EQ.'h').AND.(TEMPRD.GE.RPTEMP)))THEN
            IF(RPTEMP.NE.0.0)THEN
              IF(ABS((RPTEMP-TEMPRD)/RPTEMP).LT.RELPREDLAM)THEN
                IF(MANYLAM.EQ.0)THEN
                WRITE(IREC,2310,ERR=9350) RELPREDLAM
                WRITE(6,2310) RELPREDLAM
2310            FORMAT(/,T5,'No more lambdas: relative prediction ',
     +          'change less than ',F8.4)
                GO TO 1200
                END IF
              END IF
              IF(ABS(RPTEMP-TEMPRD).LT.ABSPREDLAM)THEN
                IF(MANYLAM.EQ.0)THEN
                WRITE(IREC,2311,ERR=9350) ABSPREDLAM
                WRITE(6,2311) ABSPREDLAM
2311            FORMAT(/,T5,'No more lambdas: absolute prediction ',
     +          'change less than ',F8.4)
                GO TO 1200
                END IF
              END IF
            ELSE
              IF(ABS(RPTEMP-TEMPRD).LT.ABSPREDLAM)THEN
                IF(MANYLAM.EQ.0)THEN
                WRITE(IREC,2311,ERR=9350) ABSPREDLAM
                WRITE(6,2311) ABSPREDLAM
                GO TO 1200
                END IF
              END IF
              IF(RPTEMP.EQ.TEMPRD)THEN
                IF(MANYLAM.EQ.0)THEN
                WRITE(IREC,2312,ERR=9350)
                WRITE(6,2312)
2312            FORMAT(/,T5,'No more lambdas: no change in prediction')
                GO TO 1200
                END IF
              END IF
            END IF
            IF(NREV.EQ.0)THEN
              LAMBDA=LAMBDA/RLAMFAC
              IF((LAMBDA.NE.0.0D0).AND.(LAMBDA.LT.1.0D-20))
     +        LAMBDA=1.0D-20
            ELSE
              NREV=NREV+1
              LAMBDA=LAMBDA*RLAMFAC
            END IF
            JACUPDATEFLAG=1
            IF(LSCH1.EQ.1)JACUPDATEFLAG=0
            GO TO 410
          END IF
          IF((RTEMP.GT.PD1).AND.(TEMPSI.GT.PD0-(PD1-PD0)))THEN
            IF(NREV.EQ.0)THEN
              LAMBDA=LAMBDA/RLAMFAC
              IF((LAMBDA.NE.0.0D0).AND.(LAMBDA.LT.1.0D-20))
     +        LAMBDA=1.0D-20
            ELSE
              NREV=NREV+1
              LAMBDA=LAMBDA*RLAMFAC
            END IF
            JACUPDATEFLAG=1
            IF(LSCH1.EQ.1)JACUPDATEFLAG=0
            GO TO 410
          END IF
C          IF((NREV.EQ.1).OR.(ITN.EQ.3))THEN
          IF((NREV.EQ.1).OR.(ITN.GE.3))THEN
            IF(MANYLAM.EQ.0)THEN
            IF(PREDHL.EQ.'l')THEN
              WRITE(IREC,2320,ERR=9350)
              WRITE(6,2320)
2320          FORMAT(/,T5,'No more lambdas: prediction rising')
            ELSE
              WRITE(IREC,2330,ERR=9350)
              WRITE(6,2330)
2330          FORMAT(/,T5,'No more lambdas: prediction falling')
            END IF
            GO TO 1200
            END IF
          END IF
          LAMBDA=LAMSTR*RLAMFAC
          NREV=1
          JACUPDATEFLAG=1
          IF(LSCH1.EQ.1)JACUPDATEFLAG=0
          GO TO 410
        END IF

C -- NO MORE LAMBDAS: MOVE TO THE NEXT OPTIMISATION ITERATION

1200    CONTINUE
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif
        MX1CHG=0.0D0
        MX2CHG=1.0D0
        MXCHG=0.0D0
        IPMX1CHG=1
        IPMX2CHG=1
        IPMXCHG=1
        DO 1210 IPP=1,NPAR
        IF((ITRANS(IPP).NE.0).AND.(ITRANS(IPP).NE.1)) GO TO 1210
        IF(PVAL(IPP).NE.0.0D0) THEN
          RTEMP=ABS((PVAL(IPP)-NEWVAL(IPP))/PVAL(IPP))
        ELSE
          IF(NEWVAL(IPP).EQ.0.0D0)THEN
            RTEMP=0.0D0
          ELSE
            RTEMP=1.0D30
          END IF
        END IF
        IF(RTEMP.GT.MXCHG) THEN
          MXCHG=RTEMP
          IPMXCHG=IPP
        END IF
        IF(PARCHGLIM(IPP).EQ.0)THEN
          IF(RTEMP.GT.MX1CHG) THEN
            MX1CHG=RTEMP
            IPMX1CHG=IPP
          END IF
        ELSE
          IF(PVAL(IPP).NE.0.0D0)THEN
            RTEMP=ABS(NEWVAL(IPP)/PVAL(IPP))
          ELSE
            IF(NEWVAL(IPP).EQ.0.0D0)THEN
              RTEMP=1.0D0
            ELSE
              RTEMP=1.0D30
            END IF
          END IF
          IF(RTEMP.LT.1.0D0) THEN
            IF(RTEMP.NE.0.0D0)THEN
              RTEMP=1.0D0/RTEMP
            ELSE
              RTEMP=1.0D30
            END IF
          END IF
          IF(RTEMP.GT.MX2CHG) THEN
            MX2CHG=RTEMP
            IPMX2CHG=IPP
          END IF
        END IF
1210    CONTINUE

C -- THE FOLLOWING CODE HANDLES THE AUTOMATIC USER INTERVENTION
C    FUNCTIONALITY.

        IISPACE=0
        IAUI=IAUI+1
        IF((IREG.NE.0).OR.(IPRED.NE.0)) THEN
          IF(IREG.NE.0)THEN
            IF(DOAUI.NE.'auid')THEN
              GO TO 7500
            END IF
          ELSE
            GO TO 7500
          END IF
        END IF
        IF(MAXAUI.EQ.0) GO TO 7500
        IF(IOPT.LT.AUISTARTOPT) GO TO 7500
        IF(PSI.LE.0.0D0) GO TO 7500
        IF(IAUI.EQ.1)THEN
          IF((AUIRESTITN.GT.1).AND.(FIRSTAUI.NE.0))THEN
            IF(((IOPT-FIRSTAUI+1)/AUIRESTITN)*
     +        AUIRESTITN.EQ.IOPT-FIRSTAUI+1)THEN
              IAFIN=8
              GO TO 7300
            END IF
          END IF
        END IF
        IF(NOAUI.EQ.1) THEN
          IAFIN=1
          GO TO 7300
        END IF
        IF(NEWPSI.GT.1.0D299) GO TO 7300
        IF(IAUI.EQ.1)THEN
          IF((NEWPSI/PSI.LE.NOAUIPHIRAT).OR.(NEWPSI/PSI.LE.PHIRATSUF))
     +      THEN
            IAFIN=9
            GO TO 7400
          END IF
          IF(IREG.EQ.0)THEN
            FIRSTPSI=NEWPSI
          ELSE
            IF(REGCONTINUE.EQ.0)THEN
              FIRSTPSI=NEWPHIM
            ELSE
              FIRSTPSI=NEWPSI
            END IF
          END IF
          AFRACPSI=1.0D0
        ELSE
          IF(IREG.EQ.0)THEN
            AFRACPSI=NEWPSI/FIRSTPSI
          ELSE
            IF(REGCONTINUE.EQ.0)THEN
              AFRACPSI=NEWPHIM/FIRSTPSI
            ELSE
              AFRACPSI=NEWPSI/FIRSTPSI
            END IF
          END IF
          IF(DOAUI.EQ.'aui')THEN
c             IF(AFRACPSI.LE.AUIPHIRATSUF) THEN
c              IAFIN=2
c              GO TO 7400
c            END IF
            RTEMP2=NEWPSI/PSI
            IF(RTEMP2.LE.AUIPHIRATSUF)THEN
              IAFIN=2
              GO TO 7400
            END IF
          ELSE IF(DOAUI.EQ.'auid')THEN
            RTEMP2=NEWPSI/PSI
            IF(RTEMP2.LE.AUIPHIRATSUF)THEN
              IAFIN=2
              GO TO 7400
            END IF
            IF(IAUI.EQ.1)THEN
              OLDARAT=RTEMP2
            ELSE
              IF(RTEMP2.LT.OLDARAT*AUIPHIRATACCEPT)THEN
                JFRAC=0
              ELSE
                JFRAC=JfRAC+1
              END IF
              OLDARAT=RTEMP2
            END IF
          END IF
        END IF

        RTEMP2=AUIPHIRATACCEPT*ALOFRAC
        IF(DOAUI.EQ.'auid') RTEMP2=ALOFRAC
        IF((IAUI.EQ.1).OR.(AFRACPSI.LE.RTEMP2))THEN
          ALOFRAC=AFRACPSI
          DO 7150 IPP=1,NPAR
            NAVAL(IPP)=NEWVAL(IPP)
7150      CONTINUE
          DO 7155 IROW=1,NXROW
            NAOBS(IROW)=NEWOBS(IROW)
7155      CONTINUE

c -- If doing predictive analysis, dont forget newpred.

C -- Note: when automatic user intervention using the "auid" option is employed when
C    PEST is runnning in regularisation mode, it would probably be a good idea not
C    to include regularisation observations in computation of the parameter with
C    maximum composite sensitivity. But for the moment this is done (for ease or programming).

          NAROL=NROL
          NAPSI=NEWPSI
          NAPHIM=NEWPHIM
          NAPHIR=NEWPHIR
          NANJPPL0=NJPPL0
          NANJPPL1=NJPPL1
          NANJPPL2=NJPPL2
          NANDMXL0=NDMXL0
          NANDMXL1=NDMXL1
          NANDMXL2=NDMXL2
          NANLAM=NEWLAM
          NANALPH=NEWALPHA
          NANNUMEIG=NEWNUMEIG
          AMXCHG=MXCHG
          IAPMXCHG=IPMXCHG
          AMX1CHG=MX1CHG
          IAPMX1CHG=IPMX1CHG
          AMX2CHG=MX2CHG
          IAPMX2CHG=IPMX2CHG
          IFRAC=0
          IF(IAUI.GT.1)THEN
            WRITE(6,7157)
            WRITE(IREC,7157,ERR=9350)
7157        FORMAT(/,'    Phi less than AUPHIRATACCEPT of accepted ',
     +      'phi - new parameters accepted.')
            IISPACE=1
          END IF
        ELSE
          IFRAC=IFRAC+1
          WRITE(6,7156)
          WRITE(IREC,7156,ERR=9350)
7156      FORMAT(/,'    Phi greater than AUIPHIRATACCEPT of accepted ',
     +    'phi - new params not accepted.')
          IISPACE=1
          IF(IFRAC.GE.NAUINOACCEPT)THEN
            IAFIN=7
            GO TO 7300
          END IF
        END IF
        IF(DOAUI.EQ.'auid')THEN
          IF(JFRAC.GE.NAUINOACCEPT)THEN
            IAFIN=7
            GO TO 7300
          END IF
          IF(IREG.NE.0)THEN
            IF(REGCONTINUE.EQ.0)THEN
              IF(NAPHIM.LE.PHIMLIM)THEN
                IAFIN=11
                GO TO 7300
              END IF
            END IF
          END IF
        END IF

        IF(IAUI.GT.MAXAUI) THEN
          IAFIN=3
          GO TO 7300
        END IF

C --THE NUMBER OF PARAMETERS THAT ARE NOT FIXED, TIED, FROZEN OR HELD BY
C   THE USER OR AUTOMATIC USER INTERVENTION ARE NOW COUNTED.

        ICC=0
        IMINSENS=0
        IMAXSENS=0
        MAXSENS=0.0D0
        MINSENS=1.0D300
        DO 7180 IPP=1,NPAR
          IF(ITRANS(IPP).GE.0)THEN
            IF(ISENS(IPP).EQ.0)THEN
              IF(SENS(IPP).GT.0.0D0)THEN
                ICC=ICC+1
                IF(SENS(IPP).GT.MAXSENS)THEN
                  MAXSENS=SENS(IPP)
                  IMAXSENS=IPP
                END IF
                IF(SENS(IPP).LT.MINSENS)THEN
                  MINSENS=SENS(IPP)
                  IMINSENS=IPP
                END IF
              END IF
            END IF
          END IF
7180    CONTINUE


        IF(ICC-1.LE.0)THEN
          IAFIN=10
          GO TO 7300
        END IF
        IF(ICC-1.LT.AUINUMFREE)THEN
          IAFIN=4
          GO TO 7300
        END IF

        IF(MAXSENS.EQ.0.0D0)THEN
          IAFIN=5
          GO TO 7300
        END IF
        IF(MAXSENS/MINSENS.LT.AUISENSRAT)THEN
          IAFIN=6
          GO TO 7300
        END IF

C        should we count the parameters which are frozen? these might
C        be insensitive. well i guess that if they become frozen e
C        can hold them next time.

C -- THE DECISION IS NOW MADE ON WHICH PARAMETERS TO HOLD.

        IPHOLD=0
        IF(AUIHOLDMAXCHG.EQ.1)THEN
          IF(ABS(MX2CHG-DMAX2).LT.ABS(1.0D-5*DMAX2))THEN
            IF(ISENS(IPMX2CHG).EQ.0)THEN
              IF(ITRANS(IPMX2CHG).GE.0)THEN
                IF(PARCHGLIM(IPMX2CHG).NE.0)THEN
                  IAHOLD=0
                  IF(SENS(IPMX2CHG).LE.0.0D0)THEN
                    IAHOLD=1
                  ELSE
                    IF(MAXSENS/SENS(IPMX2CHG).GE.AUISENSRAT)THEN
                      IAHOLD=1
                    END IF
                  END IF
                  IF(IAHOLD.EQ.1)THEN
                    IPHOLD=IPMX2CHG
                    ISENS(IPMX2CHG)=IAUI
                    GO TO 7250
                  END IF
                END IF
              END IF
            END IF
          END IF
          IF(ABS(MX1CHG-RELPARMAX).LT.ABS(1.0D-5*RELPARMAX))THEN
            IF(ISENS(IPMX1CHG).EQ.0)THEN
              IF(ITRANS(IPMX1CHG).GE.0)THEN
                IF(PARCHGLIM(IPMX1CHG).EQ.0)THEN
                  IAHOLD=0
                  IF(SENS(IPMX1CHG).LE.0.0D0)THEN
                    IAHOLD=1
                  ELSE
                    IF(MAXSENS/SENS(IPMX1CHG).GE.AUISENSRAT)THEN
                      IAHOLD=1
                    END IF
                  END IF
                  IF(IAHOLD.EQ.1)THEN
                    IPHOLD=IPMX1CHG
                    ISENS(IPMX1CHG)=IAUI
                    GO TO 7250
                  END IF
                END IF
              END IF
            END IF
          END IF
        END IF

        IF(DOAUI.EQ.'aui')THEN
          IPHOLD=IMINSENS
          ISENS(IPHOLD)=IAUI
        ELSE IF(DOAUI.EQ.'auid')THEN
          IPHOLD=IMAXSENS
          ISENS(IPHOLD)=IAUI
        END IF

7250    CONTINUE
        IF(FIRSTAUI.EQ.0) FIRSTAUI=IOPT
        DECNOF=0
        IF(IISPACE.EQ.0)THEN
          WRITE(6,*)
          WRITE(IREC,*,ERR=9350)
        END IF
        WRITE(6,7520)
        WRITE(IREC,7520,ERR=9350)
7520    FORMAT('    Automatic user intervention: ',
     +  'repeating parameter upgrade process...')
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif
        GO TO 7550

C -- NO MORE AUI ITERATIONS

7300    CONTINUE
        IF(IAUI.GT.1)THEN
          DO 7310 IPP=1,NPAR
            NEWVAL(IPP)=NAVAL(IPP)
7310      CONTINUE
          DO 7320 IROW=1,NXROW
            NEWOBS(IROW)=NAOBS(IROW)
7320      CONTINUE
          NROL=NAROL
          NEWPSI=NAPSI
          NEWPHIM=NAPHIM
          NEWPHIR=NAPHIR
          NJPPL0=NANJPPL0
          NJPPL1=NANJPPL1
          NJPPL2=NANJPPL2
          NDMXL0=NANDMXL0
          NDMXL1=NANDMXL1
          NDMXL2=NANDMXL2
          NEWLAM=NANLAM
          NEWALPHA=NANALPH
          NEWNUMEIG=NANNUMEIG
          MXCHG=AMXCHG
          IPMXCHG=IAPMXCHG
          MX1CHG=AMX1CHG
          IPMX1CHG=IAPMX1CHG
          MX2CHG=AMX2CHG
          IPMX2CHG=IAPMX2CHG
        END IF

7400    CONTINUE
        IF(IAFIN.EQ.1)THEN
          AEXPL='Difficulties encountered in calculating new '//
     +    'parameters.'
        ELSE IF(IAFIN.EQ.2)THEN
c          IF(DOAUI.EQ.'aui')THEN
c            AEXPL='Objective function less than AUIPHIRATSUF '//
c     +      'of phi without user intervention.'
c          ELSE IF(DOAUI.EQ.'auid')THEN
            AEXPL='Objective function less than AUIPHIRATSUF '//
     +      'of starting phi.'
c          END IF
        ELSE IF(IAFIN.EQ.3)THEN
          AEXPL='MAXAUI automatic user intervention '//
     +    'iterations carried out.'
        ELSE IF(IAFIN.EQ.4)THEN
          AEXPL='Unheld/unfrozen params with non-zero '//
     +    'sensit. would not exceed AUINUMFREE.'
        ELSE IF(IAFIN.EQ.5)THEN
          AEXPL='Maximum sensitivity of unheld and unfrozen '//
     +    'parameters is zero.'
        ELSE IF(IAFIN.EQ.6)THEN
          AEXPL='Max/min sens. ratio of unheld and unfrozen'//
     +    ' params less than AUISENSRAT.'
        ELSE IF(IAFIN.EQ.7)THEN
          AEXPL='NAUINOACCEPT AUI iterations since '//
     +    'phi improvement of AUIPHIRATACCEPT.'
        ELSE IF(IAFIN.EQ.8)THEN
          AEXPL='Rest iteration for automatic user intervention.'
        ELSE IF(IAFIN.EQ.9)THEN
          AEXPL='New/old objective function ratio less than '//
     +    'NOAUIPHIRAT and/or PHIRATSUF.'
        ELSE IF(IAFIN.EQ.10)THEN
          AEXPL='There are no free parameters left to hold.'
        ELSE IF(IAFIN.EQ.11)THEN
          AEXPL='Measurement objective function less than target.'
        END IF
        IF(IAFIN.NE.0)THEN
          IF(IISPACE.EQ.0)THEN
            WRITE(6,*)
            WRITE(IREC,*,ERR=9350)
          END IF
          IF(IAUI.EQ.1)THEN
            WRITE(6,7410)
            WRITE(IREC,7410,ERR=9350)
7410       FORMAT('    No automatic user intervention attempted:-')
          ELSE
            WRITE(6,7420)
            WRITE(IREC,7420,ERR=9350)
7420        FORMAT('    No more automatic user intervention ',
     +      'iterations:-')
          END IF
          WRITE(6,7430) AEXPL(1:LEN_TRIM(AEXPL))
          WRITE(IREC,7430,ERR=9350) AEXPL(1:LEN_TRIM(AEXPL))
7430      FORMAT(4X,A,/)
        END IF
7500    NOAUI=0
        IAUI=0
        IAFIN=0
        ALOFRAC=1.0D30
        IFRAC=0
        JFRAC=0
        IF(HOLDFLAG.NE.0)THEN
          DO 7450 IPP=1,NPAR
            ISENS(IPP)=0
            SENS(IPP)=0.0D0
7450      CONTINUE
        END IF

C -- END OF AUTOMATIC USER INTERVENTION

        DO 1202 IPP=1,NPAR
          LSTVAL(IPP)=PVAL(IPP)
          PVAL(IPP)=NEWVAL(IPP)
1202    CONTINUE
        IF(MXCHG.LT.RELPARSTP)THEN
          ICHNG=ICHNG+1
        ELSE
          ICHNG=0
        END IF
        DO 1220 IROW=1,NXROW
1220    REFOBS(IROW)=NEWOBS(IROW)
        PSI=NEWPSI
        IF(IREG.EQ.1)THEN
          PHIM=NEWPHIM
          PHIR=NEWPHIR
        END IF
        ROLAST=NROL
        JPP0LT=NJPPL0
        JPP1LT=NJPPL1
        JPP2LT=NJPPL2
        DMX0LT=NDMXL0
        DMX1LT=NDMXL1
        DMX2LT=NDMXL2
        LAMLST=NEWLAM
        LAMFLAG=1
        ALPHLST=NEWALPHA
        NUMEIGLST=NEWNUMEIG
        IF(IPRED.GT.0)THEN
          REFPRED=NEWPRED
          E_PRED=NEWE_PRED
          RES_PRED=NEWRES_PRED
          EW_PRED=NEWEW_PRED
        END IF
        IF(IPRED.GT.0)THEN
          IF(PSI.GT.PD1)THEN
            WRITE(6,1226)PSI
            WRITE(IREC,1226,ERR=9350) PSI
1226        FORMAT(T5,'Best phi this iteration: ',1PG12.5)
          ELSE
            WRITE(6,1228) PSI
            WRITE(IREC,1228,ERR=9350) PSI
1228        FORMAT(T5,'Best  phi  this  iteration    : ',1PG12.5)
            WRITE(6,1227) REFPRED
            WRITE(IREC,1227,ERR=9350) REFPRED
1227        FORMAT(T5,'Best prediction this iteration: ',1PG12.5)
          END IF
        ELSE
          IF(IREG.NE.1)THEN
            IF((PSI.GT.8.999D34).AND.(PSI.LT.9.001d34))THEN
c            IF(EQUALS(PSI,0.9D35))THEN
              WRITE(6,1224)
1224          FORMAT(T5,'Lowest phi this iteration: none calculated.')
            ELSE
              WRITE(6,1225) PSI
              WRITE(IREC,1225,ERR=9350) PSI
1225          FORMAT(T5,'Lowest phi this iteration: ',1PG12.5)
            END IF
          END IF
        END IF
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif

C -- If running in Pareto mode, restore current weights.

C       IF(PESTMODE.EQ.4)THEN
C         J=0
C         DO I=1,NXROW
C          IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
C            J=J+1
C            OWGHT(I)=OWGHTKP(J)*PAR_WTFAC
C          END IF
C        END DO
C      END IF

C -- STORE PARAMETER SET IF LOWEST OBJECTIVE FUNCTION HAS JUST BEEN
C -- ACHIEVED

        PDFLAG=0
        IF(IREG.EQ.0)THEN
          IF(IPRED.GT.0)THEN
            IF(PSILO.GT.PD1)THEN
              IF(PSI.LT.PSILO) PDFLAG=1
            ELSE
              IF(PSI.LE.PD1)THEN
                IF((PREDHL.EQ.'l').AND.(REFPRED.LT.PREDLO))PDFLAG=2
                IF((PREDHL.EQ.'h').AND.(REFPRED.GT.PREDLO))PDFLAG=3
              END IF
            END IF
          ELSE
            IF(PSI.LT.PSILO)PDFLAG=1
          END IF
        ELSE
          IF(PHIM.LT.PHIMLO)PDFLAG=1
C          IF(PHIM.GT.PHIMLIMKP*0.98)THEN
          IF(PHIM.GT.PHIMLIMKP*PD1RFAC)THEN
            CONTINUE
C            IF(PHIM.LT.PHIMLO)PDFLAG=1
          ELSE
            IF(PHIR.LT.PHIRLO)PDFLAG=1
          END IF
        END IF
        IF(PESTMODE.EQ.4) THEN
          PDFLAG=0
          IF(SVDA.EQ.1)THEN
            SVDA_SAVE=1
          END IF
        END IF
        IF(PDFLAG.NE.0)THEN
          DO 1230 IPP=1,NPAR
1230      PVAL_O(IPP)=PVAL(IPP)
          CALL PRMSAV(JFAIL,NPAR,APAR,PVAL_O,CLINE,APRECS,APOINT,SCALE,
     +    OFFSET,-1)
          IF(JFAIL.NE.0) GO TO 9891
          IF(SVDA.EQ.1) SVDA_SAVE=1
        END IF

C -- If working in Pareto mode, the Pareto data is stored and a termination
C    criterion is evaluated.

        IF(PESTMODE.EQ.4)THEN
          J=0
          DO I=1,NXROW
            IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
              J=J+1
              RTEMP=OWGHT(I)
              OWGHT(I)=OWGHTKP(J)
              OWGHTKP(J)=RTEMP
            END IF
          END DO
          DO I=1,NOBSGP
            CALL OBJCLC(-I,PARPSISUB(I),NXROW,NPRIOR,
     +      NOBS,REFOBS,OVAL,OWGHT,NOBGNM,NOBSGP,IRGP,SUM1)
          END DO
          J=0
          DO I=1,NXROW
            IF(NOBGNM(I).EQ.PAR_NOBGNM) THEN
              J=J+1
              RTEMP=OWGHT(I)
              OWGHT(I)=OWGHTKP(J)
              OWGHTKP(J)=RTEMP
            END IF
          END DO
          DO I=1,NOBSGP
            WRITE(ATEMP12,'(1PG12.6)') PARPSISUB(I)
            ATEMP12=ADJUSTL(ATEMP12)
            WRITE(IPOD,'(1X,A12,1X)',ADVANCE='NO') TRIM(ATEMP12)
          END DO
          IF(NOBS_REPORT.NE.0)THEN
            DO I=1,NOBS_REPORT
              WRITE(ATEMP20,'(1PG13.6)') REFOBS(IOBS_REPORT(I))
              ATEMP20=ADJUSTL(ATEMP20)
              WRITE(IPOD,'(1X,A20,1X)',ADVANCE='NO') TRIM(ATEMP20)
            END DO
          END IF
          WRITE(IPOD,'(A)') ' '

          WRITE(IPPD) (PARPSISUB(I),I=1,NOBSGP)
          IF(NOBS_REPORT.NE.0)THEN
            WRITE(IPPD) (REFOBS(IOBS_REPORT(I)),I=1,NOBS_REPORT)
          END IF
          IF(SVDA.EQ.0)THEN
            WRITE(IPPD) (PVAL(I),I=1,NPAR)
            CALL PRMSAV(JFAIL,NPAR,APAR,PVAL,CLINE,APRECS,APOINT,SCALE,
     +      OFFSET,IOPT)
            IF(JFAIL.NE.0) GO TO 9891
          END IF
#ifdef FLUSHFILE
          CALL FLUSH(IPPD)
          CALL FLUSH(IPOD)
#endif
          IF(NOBS_TERM.NE.0)THEN
            RTEMP=REFOBS(NOBS_TERM)
            JFLAG=0
            IF(N_ABOVE_BELOW.EQ.1)THEN
              IF(RTEMP.GE.OBS_THRESH)THEN
                JFLAG=1
              END IF
            ELSE IF(N_ABOVE_BELOW.EQ.-1)THEN
              IF(RTEMP.LE.OBS_THRESH)THEN
                JFLAG=1
              END IF
            END IF
            IF(JFLAG.EQ.0)THEN
              PAR_TERM_COUNT=0
            ELSE
              PAR_TERM_COUNT=PAR_TERM_COUNT+1
              IF(PAR_TERM_COUNT.GE.NUM_ITER_THRESH)THEN
                PARETO_FIN=2
              END IF
            END IF
          END IF
        END IF

C -- THE BEST JACOBIAN CALCULATED SO FAR IS STORED ON FILE.
C    HOWEVER IT IS ROTATED FIRST AND RE-ROTATED AFTERWARDS

        IF((PDFLAG.NE.0).OR.(IJCO.NE.0))THEN
          IF(NUMCOV.NE.0)THEN
            MATDIM1=NXROW
            NM=MXOBSCOV
            IPSTART=1
            DO 6990 JCOV=1,NUMCOV
              IGROUP=COVGP(JCOV)
              DO 6985 I=IPSTART,NXROW
                IF(NOBGNM(I).EQ.IGROUP)THEN
                  CALL ROTATE(-2,NOBSCOV(JCOV),NOBSCOV(JCOV),MATDIM1,
     +            NESPAR,IPSTART,COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,
     +            NXROW,NOBGNM)
                  GO TO 6990
                END IF
6985          CONTINUE
6990        CONTINUE
          END IF

          IF(PDFLAG.NE.0)THEN
            IF(IOPT.NE.1)THEN                    ! because it has already been done
              FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.jco'
              CALL FFOPEN(JFAIL,-IRSF,'w',' ',25,CLINE)
              IF(JFAIL.NE.0) GO TO 9891
              CALL JCOWRITE(NPAR,NESPAR,NXROW,IRSF,X,ITRANS,APAR,
     +        AOBS,REFOBS,OVAL,PVAL,SCALE,OFFSET)
              CLOSE(UNIT=IRSF)
            END IF
          END IF
          IF(IJCO.NE.0)THEN
            CALL WRITINT(AATEMP,IOPT)
            FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.jco'//'.'//TRIM(AATEMP)
            CALL FFOPEN(JFAIL,-IRSF,'w',' ',25,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            CALL JCOWRITE(NPAR,NESPAR,NXROW,IRSF,X,ITRANS,APAR,
     +      AOBS,REFOBS,OVAL,PVAL,SCALE,OFFSET)
            CLOSE(UNIT=IRSF)
          END IF

          IF(NUMCOV.NE.0)THEN
            MATDIM1=NXROW
            NM=MXOBSCOV
            IPSTART=1
            DO 6991 JCOV=1,NUMCOV
              IGROUP=COVGP(JCOV)
              DO 6986 I=IPSTART,NXROW
                IF(NOBGNM(I).EQ.IGROUP)THEN
                  CALL ROTATE(2,NOBSCOV(JCOV),NOBSCOV(JCOV),MATDIM1,
     +            NESPAR,IPSTART,COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,
     +            NXROW,NOBGNM)
                  GO TO 6991
                END IF
6986          CONTINUE
6991        CONTINUE
          END IF
        END IF

C -- CONTINUE WITH ASSIGNMENT OF OPTIMISED VALUES FOR VARIABLES.

        IF(PDFLAG.NE.0)THEN
          DO 1240 IROW=1,NXROW
1240      MODVAL_O(IROW)=REFOBS(IROW)
          IF(((IPRED.GT.0).AND.(PSI.LT.PSILO)).OR.
     +        (IPRED.EQ.0))THEN
            IF(NPHISTP.GE.2) THEN
              DO 1243 I=NPHISTP,2,-1
                PSILOL(I)=PSILOL(I-1)
                IF(IREG.EQ.1)THEN
                  PHIMLOL(I)=PHIMLOL(I-1)
                  PHIRLOL(I)=PHIRLOL(I-1)
                END IF
1243          CONTINUE
            END IF
            PSILOL(1)=PSILO
            PSILO=PSI
            IF(IREG.EQ.1)THEN
              PHIMLOL(1)=PHIMLO
              PHIRLOL(1)=PHIRLO
              PHIMLO=PHIM
              PHIRLO=PHIR
              WFLO=WF
              IF(IREGADJ.NE.0)THEN
                DO 5311 IROW=1,NXROW
                  OWGHTLO(IROW)=OWGHT(IROW)
5311            CONTINUE
              ENDIF
            END IF
          END IF
          LOOPT=IOPT
          IF(IPRED.GT.0)THEN
            IF((PDFLAG.GT.1).OR.
     +        ((PDFLAG.EQ.1).AND.(PSI.LE.PD1)))THEN
              IF(NPREDSTP.GE.2)THEN
                DO 3405 I=NPREDSTP,2,-1
                  PRDLOL(I)=PRDLOL(I-1)
3405            CONTINUE
              END IF
              PRDLOL(1)=PREDLO
              PREDLO=REFPRED
              E_PREDLO=E_PRED
              EW_PREDLO=EW_PRED
              RES_PREDLO=RES_PRED
              PSILO=PSI
            END IF
          END IF
          IF((I2OR3.EQ.2).AND.(ISWTCH.EQ.1)) THEN
            RLO=ROLAST
            JPP0LO=JPP0LT
            JPP1LO=JPP1LT
            JPP2LO=JPP2LT
            DMX0LO=DMX0LT
            DMX1LO=DMX1LT
            DMX2LO=DMX2LT
            LAMLO=LAMLST
            NREVLO=NREV
          END IF
        ELSE
          IF(IREG.EQ.1)THEN
            DO 5300 I=1,NPHISTP
              IF(PHIMLOL(I).GT.PHIMLIMKP*0.98)THEN
                IF(PHIM.LT.PHIMLOL(I)) GO TO 5320
              ELSE
                IF(PHIR.LT.PHIRLOL(I)) GO TO 5320
              END IF
5300        CONTINUE
            GO TO 1249
5320        CONTINUE
            DO 5310 J=NPHISTP,I+1,-1
              PHIMLOL(J)=PHIMLOL(J-1)
              PHIRLOL(J)=PHIRLOL(J-1)
5310        CONTINUE
            PHIMLOL(I)=PHIM
            PHIRLOL(I)=PHIR
          ELSE
          IF((IPRED.EQ.0).OR.
     +      ((IPRED.GT.0).AND.(PSILO.GT.PD1)))THEN
            DO 1262 I=1,NPHISTP
              IF(IPRED.GT.0)THEN
                IF(PSI.LT.PSILOL(I)) GO TO 1264
              ELSE
                IF(PSI.LT.PSILOL(I)) GO TO 1264
              END IF
1262        CONTINUE
            GO TO 1249
1264        DO 1266 J=NPHISTP,I+1,-1
1266        PSILOL(J)=PSILOL(J-1)
            PSILOL(I)=PSI
          ELSE IF(IPRED.GT.0)THEN
            IF(PREDHL.EQ.'l')THEN
              DO 2430 I=1,NPREDSTP
                IF(REFPRED.LT.PRDLOL(I)) GO TO 2450
2430          CONTINUE
            ELSE
              DO 2431 I=1,NPREDSTP
                IF(REFPRED.GT.PRDLOL(I)) GO TO 2450
2431          CONTINUE
            END IF
            GO TO 1249
2450        DO 2460 J=NPREDSTP,I+1,-1
              PRDLOL(J)=PRDLOL(J-1)
2460        CONTINUE
            PRDLOL(I)=REFPRED
          END IF
          END IF
        END IF

C -- IF NECESSARY, SAVE THE RESOLUTION DATA FILE.

        IF(PESTMODE.NE.4)THEN
        IF((PDFLAG.NE.0).OR.(IOPT.EQ.1))THEN
          IF(IRES.NE.0)THEN
            CALL RESDAT(IFAIL,IRSF,NESPAR,NXROW,NOBSGP,IREG,SVDMODE,
     +      SVDA,NUMLAM,IMCOVFLAG,ICOUNTREGOBS,ICOUNTREGNZ,RLAMBDA1,
     +      LAMLST,ALPHLST,NUMEIGLST,PHIM,PHIR,NOBGNM,IRGP,OWGHT,
     +      CLINE,ERRMSG,SVDA_SCALADJ,SVDA_EXTSUPER,LSQRMODE)
            IF(IFAIL.EQ.1) THEN
              GO TO 9891
            ELSE IF(IFAIL.EQ.2)THEN
              GO TO 9890
            END IF
          END IF
        END IF
        END IF

1249    CONTINUE
        IF(NMLODIM2.GE.NESPAR)THEN
          IF((PDFLAG.GT.0).OR.(IOPT.EQ.1))THEN
cxxx         if((psi.lt.psilo).or.(iopt.eq.1))then
            DO 1255 IES=1,NESPAR
            DO 1255 JES=1,NESPAR
1255        NORMLO(JES,IES)=NORM(JES,IES)
          END IF
        END IF
        NEPL23=1

C -- SWITCH TO USE OF CENTRAL DERIVATIVES IF NECESSARY

1250    IF(ISWTCH.EQ.0) GO TO 1300

        PDFLAG=0
        IF(IPRED.GT.0)THEN
          IF(I2OR3.EQ.2)THEN
            IF(PSILO.GT.PD1)THEN
              IF(PSI.GT.PSIL)THEN
                PDFLAG=1
              ELSE
                IF(IOPT.GE.NOPTSWITCH)THEN
                  IF((PSIL-PSI)/PSI.LT.PHIREDSWH)PDFLAG=1
                END IF
              END IF
            ELSE
              IF(PREDHL.EQ.'l') THEN
                IF(PREDL.EQ.0.0)THEN
                  IF(REFPRED.GE.PREDL)PDFLAG=2
                  IF(PREDL-REFPRED.LT.ABSPREDSWH)PDFLAG=3
                ELSE
                  IF((PREDL-REFPRED)/ABS(PREDL).LT.RELPREDSWH)PDFLAG=4
                  IF(PREDL-REFPRED.LT.ABSPREDSWH)PDFLAG=3
                END IF
              ELSE
                IF(PREDL.EQ.0.0)THEN
                  IF(REFPRED.LE.PREDL)PDFLAG=2
                  IF(REFPRED-PREDL.LT.ABSPREDSWH)PDFLAG=3
                ELSE
                  IF((REFPRED-PREDL)/ABS(PREDL).LT.RELPREDSWH) PDFLAG=4
                  IF(REFPRED-PREDL.LT.ABSPREDSWH)PDFLAG=3
                END IF
              END IF
            END IF
          END IF
        ELSE
          IF(IREG.EQ.1)THEN
            IF(I2OR3.EQ.2)THEN
              IF(1.0-BESTFRAC.LT.0.0D0)THEN
                PDFLAG=1
              ELSE
                IF(IOPT.GE.NOPTSWITCH)THEN
                  IF(1.0-BESTFRAC.LT.PHIREDSWH) PDFLAG=1
                END IF
              END IF
            END IF
C              IF(ABS(PHIML-PHIM)*2.0/(PHIML+PHIM).LT.PHIREDSWH)THEN
C                IF(ABS(PHIR-PHIRL)*2.0/(PHIRL+PHIR).LT.PHIREDSWH)THEN
C                  PDFLAG=1
C                END IF
C              END IF
C              IF(PHIM.GT.PHIMACCEPT)THEN
C                IF(PHIML.GE.PHIMACCEPT)THEN
C                  IF(PHIM.GE.PHIML) PDFLAG=1
C                END IF
C              ELSE
C                IF(PHIML.LE.PHIMACCEPT)THEN
C                  IF(PHIR.GE.PHIRL)PDFLAG=1
C                END IF
C              END IF
C            END IF
          ELSE
            IF(I2OR3.EQ.2)THEN
              IF(PSIL-PSI.LT.0.0D0)THEN
                IF(IOPT.GE.NOPTSWITCH)THEN
                  PDFLAG=1
                END IF
              ELSE
                IF(IOPT.GE.NOPTSWITCH)THEN
                  IF((PSIL-PSI)/PSIL.LT.PHIREDSWH) PDFLAG=1
                END IF
              END IF
            END IF
          END IF
        END IF
        IF(PDFLAG.GT.0)THEN
          IF(LOOPT.NE.IOPT) THEN
            IF(SVDA.EQ.0)THEN
              DO 1260 IPP=1,NPAR
1260          PVAL(IPP)=PVAL_O(IPP)
              DO 1270 IROW=1,NXROW
1270          REFOBS(IROW)=MODVAL_O(IROW)
              PSI=PSILO
              IF(IREG.EQ.1)THEN
                PHIM=PHIMLO
                PHIR=PHIRLO
                IF(IREGADJ.GT.0)THEN
                  DO 1271 IROW=1,NXROW
                    OWGHT(IROW)=OWGHTLO(IROW)
1271              CONTINUE
                END IF
              END IF
              ROLAST=RLO
              JPP0LT=JPP0LO
              JPP1LT=JPP1LO
              JPP2LT=JPP2LO
              DMX0LT=DMX0LO
              DMX1LT=DMX1LO
              DMX2LT=DMX2LO
              LAMLST=LAMLO
              NREV=NREVLO
              ICHNG=MAX(ICHNG-(IOPT-LOOPT),0)
              IF(IPRED.GT.0)THEN
                REFPRED=PREDLO
                E_PRED=E_PREDLO
                EW_PRED=EW_PREDLO
                RES_PRED=RES_PREDLO
              END IF
            END IF
          END IF
          I2OR3=3
          IF(NEPL23.EQ.1) THEN
C            IF(IREG.EQ.1)THEN
C              WRITE(IREC,5325,ERR=9350) PHIREDSWH
C              WRITE(6,5325) PHIREDSWH
C5325          FORMAT(T5,'Rel. improvement in measurement and ',
C     +        'regularisation fns. less than ',F6.4)
C            ELSE
            IF(((IPRED.GT.0).AND.(PDFLAG.EQ.1)).OR.
     +          (IPRED.EQ.0))THEN
              WRITE(IREC,1254,ERR=9350) PHIREDSWH
              WRITE(6,1254) PHIREDSWH
1254          FORMAT(T5,'Relative phi reduction between optimisation ',
     +        'iterations less than ',F6.4)
            ELSE
              IF(PDFLAG.EQ.2)THEN
                WRITE(IREC,3461,ERR=9350)
                WRITE(6,3461,ERR=9350)
3461            FORMAT(T5,'No improvement in prediction between ',
     +          'optimisation iterations.')
              ELSE IF(PDFLAG.EQ.3)THEN
                WRITE(IREC,3460,ERR=9350) ABSPREDSWH
                WRITE(6,3460) ABSPREDSWH
3460            FORMAT(T5,'Absolute predictive improvement between ',
     +          'iterations less than ',F6.4)
              ELSE IF(PDFLAG.EQ.4)THEN
                WRITE(IREC,3462,ERR=9350) RELPREDSWH
                WRITE(6,3462) RELPREDSWH
3462            FORMAT(T5,'Relative predictive improvement between ',
     +          'iterations less that ',F6.4)
              END IF
C            END IF
            END IF
            NEPL23=0
          END IF

          WRITE(IREC,1251,ERR=9350)
          WRITE(6,1251)
1251      FORMAT(T5,'Switch to central derivatives calculation')
          SWITCHFLAG=1
          IF(IOPT.NE.LOOPT) THEN
            IF(SVDA.EQ.0)THEN
              SBACKFLAG=1
              WRITE(IREC,1252,ERR=9350) LOOPT
              WRITE(6,1252) LOOPT
1252          FORMAT(T5,'(restart from best parameters so far - ',
     +        'these achieved at iteration ',I4,')',/)
            END IF
          END IF
          OPTSWH=IOPT
          IF(SPLITFLAG.EQ.1)THEN
            IF(JSPLIT.NE.0)THEN
              WRITE(6,1253)
              WRITE(IREC,1253,ERR=9350)
            END IF
          END IF
        END IF

1300    CONTINUE

C -- DO WE NEED TO SWITCH ON SPLIT SLOPE ANALYSIS?

        IF((SPLITFLAG.EQ.0).AND.(JSPLIT.EQ.1).AND.(JA2.EQ.0))THEN
          IF((I2OR3.EQ.3).OR.(I2OR3.EQ.0))THEN
            IF((IOPT.GT.OPTSWH).AND.(IOPT.GT.1))THEN
              IF(IPRED.GT.0)THEN
                IF(PSILO.GT.PD1)THEN
                  IF(PSI.GT.PSIL*SPLITSWH)SPLITFLAG=1
                ELSE
                  SPLITFLAG=1
                END IF
              ELSE
                IF(IREG.EQ.1)THEN
                  IF(BESTFRAC.GT.SPLITSWH) SPLITFLAG=1
                ELSE
                  IF(PSI/PSIL.GT.SPLITSWH) SPLITFLAG=1
                END IF
              END IF
            END IF
          END IF
          IF(SPLITFLAG.EQ.1)THEN
            WRITE(6,1253)
            WRITE(IREC,1253,ERR=9350)
1253        FORMAT(T5,'Split slope derivatives analysis begins during ',
     +      'next iteration')
          END IF
        END IF

C -- LEAVE FLAG TO SAVE REI FILE IF TERMINATING

        REIFLAG=1

C -- RECORD DETAILS OF OPTIMISATION ITERATION

        IF((OPTSWH.EQ.IOPT).AND.(IOPT.NE.LOOPT))THEN
          IF(SVDA.EQ.0)LSTVAL(1)=1.0D300
        END IF
        CALL PRMWRT(JFAIL,IREC,NPAR,APAR,PVAL,LSTVAL,CLINE)
        IF(JFAIL.NE.0) GO TO 9891
        IF((OPTSWH.NE.IOPT).OR.(IOPT.EQ.LOOPT)) THEN
          IF(NUMFAC.NE.0) THEN
            IF(NUMREL.EQ.0)THEN
              WRITE(IREC,5070,ERR=9350) MX2CHG,
     +        APAR(IPMX2CHG)(1:LEN_TRIM(APAR(IPMX2CHG)))
              WRITE(6,5070) MX2CHG,
     +        APAR(IPMX2CHG)(1:LEN_TRIM(APAR(IPMX2CHG)))
5070          FORMAT(T5,'Maximum   factor change: ',1PG10.4,
     +        ' ["',A,'"]')
              WRITE(IREC,5040,ERR=9350) MXCHG,
     +        APAR(IPMXCHG)(1:LEN_TRIM(APAR(IPMXCHG)))
              WRITE(6,5040) MXCHG,
     +        APAR(IPMXCHG)(1:LEN_TRIM(APAR(IPMXCHG)))
            ELSE
              WRITE(IREC,5025,ERR=9350) MX2CHG,
     +        APAR(IPMX2CHG)(1:LEN_TRIM(APAR(IPMX2CHG)))
              WRITE(6,5025) MX2CHG,
     +        APAR(IPMX2CHG)(1:LEN_TRIM(APAR(IPMX2CHG)))
5025          FORMAT(T5,'Maximum   factor change (  factor-',
     +        'limited params): ',1PG10.4,
     +        ' ["',A,'"]')
              WRITE(IREC,5031,ERR=9350) MX1CHG,
     +        APAR(IPMX1CHG)(1:LEN_TRIM(APAR(IPMX1CHG)))
              WRITE(6,5031) MX1CHG,
     +        APAR(IPMX1CHG)(1:LEN_TRIM(APAR(IPMX1CHG)))
5031          FORMAT(T5,'Maximum relative change (relative-',
     +        'limited params): ',1PG10.4,
     +        ' ["',A,'"]')
              WRITE(IREC,5050,ERR=9350) MXCHG,
     +        APAR(IPMXCHG)(1:LEN_TRIM(APAR(IPMXCHG)))
              WRITE(6,5050) MXCHG,
     +        APAR(IPMXCHG)(1:LEN_TRIM(APAR(IPMXCHG)))
5050          FORMAT(T5,'Maximum relative change (      all',
     +        '        params): ',1PG10.4,
     +        ' ["',A,'"]')
            END IF
          ELSE
            WRITE(IREC,5040,ERR=9350) MXCHG,
     +      APAR(IPMXCHG)(1:LEN_TRIM(APAR(IPMXCHG)))
            WRITE(6,5040) MXCHG,
     +      APAR(IPMXCHG)(1:LEN_TRIM(APAR(IPMXCHG)))
5040        FORMAT(T5,'Maximum relative change: ',1PG10.4,
     +      ' ["',A,'"]')
          END IF
        END IF
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif

12501   CONTINUE
        IF(PESTMODE.EQ.4)THEN
          IF(PARETO_FIN.EQ.1)THEN
            IFIN=21
            GO TO 6000
          ELSE IF(PARETO_FIN.EQ.2)THEN
            IFIN=22
            GO TO 6000
          END IF
          GO TO 100
        ELSE
          IF(IREG.EQ.0)THEN
          IF(IPRED.EQ.0)THEN
            IF(PSI.EQ.0.0D0) THEN
              IFIN=2
              GO TO 6000
            END IF
          END IF
          END IF
          IF(IREG.EQ.0)THEN
            IF(PSI.LE.PHISTOPTHRESH)THEN
              IFIN=14
              GO TO 6000
            END IF
            IF(PHIABANDON(IOPT).GT.0.0D0)THEN
              IF(PSI.GE.PHIABANDON(IOPT))THEN
                IFIN=15
                PHIAB=PHIABANDON(IOPT)
                GO TO 6000
              END IF
            END IF
          ELSE
            IF(PHIM.LE.PHISTOPTHRESH)THEN
              IFIN=14
              GO TO 6000
            END IF
            IF(PHIABANDON(IOPT).GT.0.0D0)THEN
              IF(PHIM.GE.PHIABANDON(IOPT))THEN
                IFIN=15
                PHIAB=PHIABANDON(IOPT)
                GO TO 6000
              END IF
            END IF
          END IF
          IF(ICHNG.GE.NRELPAR)THEN
            IFIN=9
            GO TO 6000
          END IF
          IF(IREG.EQ.1)THEN
            IF(PHIMLO.LT.PHIMLIMKP) THEN
              IF(REGCONTINUE.EQ.0)THEN
                IFIN=13
                GO TO 6000
              END IF
            END IF
            DO 5350 I=1,NPHISTP
              IF(ABS(PHIMLOL(I)-PHIMLO)/PHIMLOL(I).GT.PHIREDSTP)
     +        GOTO 1320
              IF(ABS(PHIRLOL(I)-PHIRLO)/PHIRLOL(I).GT.PHIREDSTP)
     +        GOTO 1320
5350        CONTINUE
            IFIN=1
            GO TO 6000
          ELSE
          IF(((IPRED.GT.0).AND.(PSILO.GT.PD1)).OR.
     +        (IPRED.EQ.0))THEN
            DO 1310 I=1,NPHISTP
            IF(ABS((PSILOL(I)-PSILO)/PSILOL(I)).GT.PHIREDSTP)GO TO 1320
1310        CONTINUE
            IFIN=1
            GO TO 6000
          ELSE
            DO 2520 I=1,NPREDSTP
              IF(ABS(PRDLOL(I)-PREDLO).GT.ABS(ABSPREDSTP)) GO TO 2525
2520        CONTINUE
            IFIN=12
            IFIN1=1
            GO TO 6000
2525        CONTINUE
            IF(PREDLO.NE.0.0D0)THEN
              DO 2530 I=1,NPREDSTP
                IF(ABS((PRDLOL(I)-PREDLO)/PREDLO).GT.ABS(RELPREDSTP))
     +          GO TO 2540
2530          CONTINUE
              IFIN=12
              IFIN1=2
              GO TO 6000
            END IF
          END IF
          END IF
2540      CONTINUE

1320      IF(OPTSWH.EQ.IOPT) GO TO 4999
          IF(((IPRED.GT.0).AND.(IOPT-LOOPT.GE.NPREDNORED)).OR.
     +       ((IPRED.EQ.0).AND.(IOPT-LOOPT.GE.NPHINORED)))THEN
            IFIN=8
            GO TO 6000
          END IF
          IF((SUCCESSCOUNT.EQ.0).AND.(FAILURECOUNT.NE.0))THEN
            IFIN=17
            GO TO 6000
          END IF
4999      IF(IOPT.LT.NOPTMAX) GO TO 100
          IFIN=3
        END IF

C -- OPTIMISATION IS COMPLETE: TIDY UP

6000    CONTINUE
        MSG(1)=' '
        MSG(2)=' '
        IF(IFIN.EQ.1)THEN
          IF(IREG.EQ.0)THEN
          WRITE(MSG(1),6035,ERR=9350) NPHISTP+1
6035      FORMAT('the ',I2,' lowest phi''s are within a relative ',
     +    'distance')
          WRITE(MSG(2),6036,ERR=9350) PHIREDSTP
6036      FORMAT(T27,'of eachother of ',1PG9.3)
          ELSE
          WRITE(MSG(1),6037,ERR=9350) NPHISTP+1
6037      FORMAT('the ',I2,' best measurement and regularisation ',
     +    'functions ')
          WRITE(MSG(2),6038,ERR=9350) PHIREDSTP
6038      FORMAT(T27,'are within a rel. distance of ',1PG9.3)
          END IF
        ELSE IF(IFIN.EQ.2) THEN
          MSG(1) = 'phi is zero'
          MSG(2) = ' '
        ELSE IF(IFIN.EQ.3)THEN
          WRITE(MSG(1),6050,ERR=9350) MAX(NOPTMAX,0)
6050      FORMAT('optimisation iteration limit of',I4,' realized.')
          MSG(2) = ' '
        ELSE IF(IFIN.EQ.4)THEN
          MSG(1) = 'all parameters at their limits.'
          MSG(2) = ' '
        ELSE IF(IFIN.EQ.5)THEN
          MSG(1) = 'phi gradient zero'
          MSG(2) = ' '
        ELSE IF(IFIN.EQ.6)THEN
          MSG(1) = 'parameter update vector zero.'
          MSG(2) = ' '
        ELSE IF(IFIN.EQ.8)THEN
          IF(IPRED.GT.0)THEN
            IF(PSILO.GT.PD1)THEN
              WRITE(MSG(1),6090,ERR=9350) NPHINORED
6090           FORMAT(I3,' optimisation iterations have elapsed ',
     +         'since lowest')
               WRITE(MSG(2),6089,ERR=9350)
6089           FORMAT(T27,'phi was achieved.')
            ELSE
              WRITE(MSG(1),6093,ERR=9350) NPHINORED
6093          FORMAT(I3,' optimisation iterations have elapsed ',
     +        'since most ')
              WRITE(MSG(2),6094,ERR=9350)
6094          FORMAT(T27,'extreme prediction was achieved.')
            END IF
          ELSE
            IF(IREG.EQ.0)THEN
            WRITE(MSG(1),6090,ERR=9350) NPHINORED
            WRITE(MSG(2),6089,ERR=9350)
            ELSE
            WRITE(MSG(1),6061,ERR=9350) NPHINORED
6061        FORMAT(I3,' optimisation iterations have elapsed since ',
     +      'best')
            WRITE(MSG(2),6062,ERR=9350)
6062        FORMAT(T27,'result was achieved.')
            END IF
          END IF
        ELSE IF(IFIN.EQ.9) THEN
          WRITE(MSG(1),6100,ERR=9350) RELPARSTP
          WRITE(MSG(2),6101,ERR=9350) NRELPAR
6100      FORMAT('relative parameter change less than ',1PG10.4)
6101      FORMAT(T27,'over',I3,' successive iterations.')
        ELSE IF(IFIN.EQ.10) THEN
          MSG(1) = 'user-initiated termination.'
          MSG(2) = ' '
        ELSE IF(IFIN.EQ.11)THEN
          IF(NOPTMAX.EQ.-1)THEN
            MSG(1) = 'NOPTMAX supplied as -1 indicating that '
     +      //'statistics'
            WRITE(MSG(2),6102,ERR=9350)
6102        FORMAT(T27,'for initial parameter set only are required.')
          ELSE
            MSG(1) = 'NOPTMAX supplied as -2 indicating that '
     +      //'only the'
            WRITE(MSG(2),6103,ERR=9350)
6103        FORMAT(T27,'Jacobian for the initial parameter set is ',
     +      'required.')
          END IF
        ELSE IF(IFIN.EQ.12)THEN
          IF(IFIN1.EQ.2)THEN
            WRITE(MSG(1),6140,ERR=9350) NPREDSTP+1
6140        FORMAT('the ',I2,1X,'most extreme predictions ',
     +      'with phi less ')
            WRITE(MSG(2),6139,ERR=9350) PD1,RELPREDSTP
6139        FORMAT(T27,'than ',1PG9.3,' are within a ',
     +      'rel. distance of ',1PG9.3)
          ELSE
            WRITE(MSG(1),6140,ERR=9350) NPREDSTP+1
            WRITE(MSG(2),6141,ERR=9350) PD1,ABSPREDSTP
6141        FORMAT(T27,'than ',1PG9.3,' are within a ',
     +      'distance of ',1PG9.3)
          END IF
        ELSE IF(IFIN.EQ.13)THEN
          WRITE(MSG(1),6142,ERR=9350)
6142      FORMAT('measurement obj. fn. lower than PHIMLIM.')
        ELSE IF(IFIN.EQ.14)THEN
          IF(IREG.EQ.0)THEN
            WRITE(MSG(1),6143,ERR=9350)
6143        FORMAT('objective function less than PHISTOPTHRESH.')
          ELSE
            WRITE(MSG(1),6144,ERR=9350)
6144        FORMAT('measurement obj fn less than ',
     +      'PHISTOPTHRESH.')
          END IF
        ELSE IF(IFIN.EQ.15)THEN
          WRITE(ANUM,'(1PG13.5)') PHIAB
          ANUM=ADJUSTL(ANUM)
          LASTRUN=0
          IF(IREG.EQ.0)THEN
            WRITE(MSG(1),6133,ERR=9350) TRIM(ANUM)
6133        FORMAT('objective function greater than PHIABANDON (',
     +      A,').')
          ELSE
            WRITE(MSG(1),6134,ERR=9350) TRIM(ANUM)
6134        FORMAT('measurement obj fn greater than ',
     +      'PHIABANDON (',A,').')
          END IF
        ELSE IF(IFIN.EQ.17)THEN
          WRITE(MSG(1),6135,ERR=9350)
6135      FORMAT('too many model run failures.')
        ELSE IF(IFIN.EQ.21)THEN
          WRITE(MSG(1),6136,ERR=9350)
6136      FORMAT('end of user-specified Pareto weights adjustment.')
        ELSE IF(IFIN.EQ.22)THEN
          IF(N_ABOVE_BELOW.EQ.1)THEN
            ATEMP12='above'
          ELSE
            ATEMP12='below'
          END IF
          WRITE(ATEMP20,'(1PG13.6)') OBS_THRESH
          WRITE(MSG(1),6137,ERR=9350) TRIM(AOBS(NOBS_TERM)),
     +    TRIM(ATEMP12),TRIM(ATEMP20)
6137      FORMAT('observation "',A,'" ',A,' threshold of ',A)
          CALL WRITINT(ATEMP12,NUM_ITER_THRESH)
          WRITE(MSG(2),6138) TRIM(ATEMP12)
6138      FORMAT(T27,'on ',a,' successive occasions.')
        END IF
        WRITE(IREC,6030,ERR=9350) MSG(1)(:LEN_TRIM(MSG(1)))
        WRITE(6,6030) MSG(1)(:LEN_TRIM(MSG(1)))
6030    FORMAT(/,T5,'Optimisation complete: ',A)
        IF (LEN_TRIM(MSG(2)).GT.0) THEN
          WRITE(IREC,6031,ERR=9350) MSG(2)(:LEN_TRIM(MSG(2)))
          WRITE(6,6031) MSG(2)(:LEN_TRIM(MSG(2)))
6031      FORMAT(1X,A)
        END IF

        IF(IREG.EQ.1)THEN
        IF((NOPTMAX.NE.-1).AND.(NOPTMAX.NE.-2))THEN
          IF(PHIMLO.GT.PHIMACCEPT)THEN
            WRITE(6,5400)
5400        FORMAT(T5,'Target measurement objective function ',
     +      'not achieved.')
            IF(ABS(WFLO-WFMIN).LT.1.0D-5*WFMIN)THEN
              WRITE(6,5410)
5410          FORMAT(T5,'Consider reducing regularisation weight ',
     +        'factor lower limit.')
            END IF
          END IF
        END IF
        END IF

        WRITE(IREC,6110,ERR=9350) NCALL
        WRITE(6,6110) NCALL
6110    FORMAT(T5,'Total model calls: ',I6)
#ifdef PARALLEL
        IF(NCALL.EQ.0) GO TO 6170
#endif
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
#endif

        IF(NOPTMAX.EQ.-2) THEN
          IPFAIL=9999
          SLAVESTOP=1
          GO TO 9891
        END IF

C -- The REI file is saved.

        IF(REIFLAG.NE.0)THEN
          IF(PESTMODE.NE.4)THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rei'
          CALL FFOPEN(JFAIL,IGNF,'w',' ',6,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          CALL REIWRITE(JFAIL,IGNF,IOPT+1,IREG,NXROW,IPRED,NOBS,
     +    NOBSGP,NUMCOV,OWGHT,NOBGNM,NOBSCOV,AOBS,OVAL,MODVAL_O,OBGNME,
     +    COVGP)
          IF(JFAIL.NE.0) GO TO 9300
          END IF
          IF(IREI.NE.0)THEN
            CALL WRITINT(AATEMP,IOPT)
            FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rei'//'.'//TRIM(AATEMP)
            CALL FFOPEN(JFAIL,IGNF,'w',' ',6,CLINE)
            IF(JFAIL.NE.0) GO TO 9891
            CALL REIWRITE(JFAIL,IGNF,IOPT+1,IREG,NXROW,IPRED,NOBS,
     +      NOBSGP,NUMCOV,OWGHT,NOBGNM,NOBSCOV,AOBS,OVAL,MODVAL_O,
     +      OBGNME,COVGP)
            IF(JFAIL.NE.0) GO TO 9300
          END IF
        END IF

C -- A FINAL MODEL RUN IS CARRIED OUT USING OPTIMISED PARAMETERS.

        IF(NOPTMAX.EQ.0) GO TO 5428
        IF((PESTMODE.EQ.4).AND.(SVDA.NE.1)) GO TO 5428
        IIFLAG=0
        IF(SVDA.EQ.1)THEN
          IF(SVDA_SAVE.EQ.1)THEN
            IF((SVDA_MULBPA.EQ.0).OR.(IOPT.EQ.0))THEN
              CALL BASE_PARDEF(IFAIL,IREC,1,NREGADJPAR,-1,
     +        SVDA_SCALADJ)
            ELSE
              CALL BASE_PARDEF(IFAIL,IREC,1,NREGADJPAR,IOPT,
     +        SVDA_SCALADJ)
            END IF
          END IF
          IF(IFAIL.NE.0) GO TO 9890
          IF(PESTMODE.EQ.4) GO TO 5428
          WRITE(6,5487)
5487      FORMAT(/,1X,78('*'))
          WRITE(6,5486)
5486      FORMAT(/,' Note carefully:-')
          WRITE(6,5481)
5481      FORMAT(/,' When SVD-assist functionality is operative, ',
     +    'the model cannot be run one last',/,' time with ',
     +    'optimised parameters.')
          WRITE(6,5482) TRIM(BASEPARFILE)
5482      FORMAT(/,' Optimised base parameters are recorded in file ',
     +    a,'.')
          WRITE(6,5483)
5483      FORMAT(/,' Use PARREP with this file, together with ',
     +    'base PEST ',
     +    'control file, to build a',/,' new PEST control file with ',
     +    'these optimised parameters.')
          WRITE(6,5484)
5484      FORMAT(/,' Then set NOPTMAX in this file to zero and run ',
     +    'PEST.')
          WRITE(6,5487)
          IIFLAG=1
          GO TO 5428
        END IF
#ifdef PARALLEL
        IF(IFIN.NE.10)THEN
#ifndef MPEST
          WRITE(IRMR,5420)
5420      FORMAT(/,' RUNNING MODEL ONE LAST TIME WITH BEST ',
     +    'PARAMETERS  -----> ')
#endif
          WRITE(6,5421)
5421      FORMAT(/,' Running model one last time with best ',
     +    'parameters.....')
        ELSE
          WRITE(6,5422)
5422      FORMAT(/,' Cannot run model one last time with optimised ',
     +    'parameter values when stopping',
     +    /,' Parallel PEST manually. ',
     +    'Use PARREP utility with PEST to run model once ',/,' with ',
     +    'optimised parameters.')
          IIFLAG=1
        END IF
#else
        WRITE(6,5425)
5425    FORMAT(/,' Running model one last time with best ',
     +  'parameters.....')
#endif

        IF(IIFLAG.EQ.0)THEN
        IF(LASTRUN.EQ.0)THEN
          WRITE(6,5439)
5439      FORMAT(' Last model run not undertaken as LASTRUN ',
     +    'control variable set to 0.')
          GO TO 5428
        END IF
        I=-1000
        J=0
        INCPAR(1)=0
#ifdef PARALLEL
        CALL STORE_PARALLEL_REGISTER
     +  (IFAIL,NPAR,PARREG,1,PVAL_O,PARREGFILE)
        IF(IFAIL.NE.0) GO TO 9890
        PIRESTART=0
#ifdef BEO
        if (BEOMASTER) then
          if (BEOLASTLOC) then
           REPEATFLAG=0
           WRITE(6,'(A)') ' Doing last run on MASTER'
           call MODEL(JFAIL,NPAR,NOBS,ASIZE,NINSTR,NTPLFLE,NINSFLE,NUML,
     +     NBLBMX,SCALE,OFFSET,PVAL_O,NW,PWORD,APAR,REFOBS,AOBS,I,W2,J,
     +     PRECIS,NOPNT,LCINS,LL,OBSN1,OBSN2,IIOBS,INSFLE,TEMPFLE,INFLE,
     +     OUTFLE,PARDEL,MRKDEL,A,CLINE,BUF,COMLIN(1),J,RTEMP,MESSFILE,
     +     INCPAR(1),ITRANS,REPEATFLAG,0)
          else
            call RUNMASTER(PARREG,OBSREG,1,JFAIL)
          end if
          if (JFAIL.ne.0) goto 9891
        else
#endif
#ifdef MPEST
        FLENME=TRIM(CASEFL)//'.lambda_runs'
        CALL FFOPEN(JFAIL,IRMR,'w',' ',6,CLINE)
        IF(JFAIL.NE.0) GO TO 9891
        WRITE(IRMR,'(I5)') 1
        CLOSE(UNIT=IRMR)
        FLENME=TRIM(CASEFL)//'.jacobian_runs'
        CALL DELFILE1(JFAIL,FLENME,CLINE)
        IF(JFAIL.NE.0) GO TO 9891
        CALL DORUNS_M(JFAIL,NSLAVE,1,
     +  ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,LL,OBSN1,
     +  OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,
     +  NPAR,PRECIS,NOPNT,NTPLFLE,NW,SCALE,
     +  OFFSET,PARDEL,PWORD,TEMPFLE,APAR,
     +  OUTFLE,INSFLE,LDSIN,LDSOU,SINFLE,
     +  SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,MSINFLE,MSOUFLE,
     +  MODFLE,WORKVEC7DIM,WORKVEC7)
        IF(JFAIL.EQ.2) GO TO 9890
        IF(JFAIL.NE.0) GO TO 9891
#else
        CALL DORUNS(JFAIL,NSLAVE,1,ISTATS,ISTATR,
     +  ASLDIR,NINSTR,NINSFLE,ASIZE,NUML,NOBS,NBLBMX,LCINS,LL,OBSN1,
     +  OBSN2,IIOBS,AOBS,A,MRKDEL,CLINE,BUF,AFILE,ISTRTME,
     +  IRUNTME,JRUN,NPAR,PRECIS,NOPNT,NTPLFLE,NW,SCALE,
     +  OFFSET,PARDEL,PWORD,TEMPFLE,APAR,OREADFLE,PREADFLE,
     +  MANFLE,ASLAVE,NNRUN,OUTFLE,INSFLE,LDSIN,LDSOU,SINFLE,
     +  SOUFLE,ITRIAL,MESSFILE,INCPAR,ITRANS,IDET,SREADFLE,MREADFLE,
     +  SCOM,PIRESTART,PTUNIT,PTFILE,IOPT,REPEATRUN,0,SLAVEGROUP,
     +  WORKVEC7DIM,WORKVEC7)
        IF(JFAIL.NE.0) GO TO 9891
#endif
#ifdef BEO
        endif
#endif
#else
        REPEATFLAG=1
        CALL MODEL(JFAIL,NPAR,NOBS,ASIZE,NINSTR,NTPLFLE,NINSFLE,NUML,
     +  NBLBMX,SCALE,OFFSET,PVAL_O,NW,PWORD,APAR,REFOBS,AOBS,I,W2,J,
     +  PRECIS,NOPNT,LCINS,LL,OBSN1,OBSN2,IIOBS,INSFLE,TEMPFLE,INFLE,
     +  OUTFLE,PARDEL,MRKDEL,A,CLINE,BUF,COMLIN(1),J,RTEMP,MESSFILE,
     +  INCPAR(1),ITRANS,REPEATFLAG,0)
        IF(JFAIL.NE.0) GO TO 9891
#endif
        IF(IPRED.GT.0)REFOBS(NOBS)=REFOBS(NOBS)+E_PREDLO
        WRITE(IREC,5426)
5426    format(/,'    The model has been run one final time using ',
     +  'best parameters. ')
        WRITE(IREC,5427)
5427    format('    Thus all model input files contain best ',
     +  'parameter values, and model ',/,'    output files contain ',
     +  'model results based on these parameters.')
        END IF
5428    CONTINUE

        IIOPT=IOPT
        FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rec'
        IF(PESTMODE.NE.4)THEN
          WRITE(6,6150)
6150      FORMAT(/,' Recording run statistics .....')
        END IF

C -- BEFORE WRTFIN IS CALLED, THE WEIGHTS ARE GIVEN THEIR OPTIMISED
C    VALUES

        IF(IREG.EQ.1)THEN
          IF(IREGADJ.EQ.0)THEN
            WFTEMP=WFLO*WFLO/WF/WF
            DO 5380 IROW=1,NXROW
              IGPNM=NOBGNM(IROW)
              IF(IRGP(IGPNM).NE.0)OWGHT(IROW)=OWGHT(IROW)*WFTEMP
5380        CONTINUE
          ELSE
            IF(OWGHTLO(1).GT.-1.0E35)THEN
              DO 5383 IROW=1,NXROW
                OWGHT(IROW)=OWGHTLO(IROW)
5383          CONTINUE
            END IF
          END IF

C -- If we do the following our weights will reflect our latest parameter estimates.
C    However our uncertainties will be terrible because we will not have calcualted
C    a weight factor for these new weights.

c          IF(NREGADJPAR.NE.0)THEN
c            ICOUNT=0
c            DO IPP=1,NPAR
c              IF(SCALE(IPP).LT.-1.0D35)THEN
c                ITEMP=NINT(OFFSET(IPP))
c                DO IOBS=1,NXROW
c                  IF(NOBGNM(IOBS).EQ.ITEMP)THEN
c                    ICOUNT=ICOUNT+1
c                    OWGHT(IOBS)=ORIGWGHT(ICOUNT)/PVAL_O(IPP)/PVAL_O(IPP)
c                  END IF
c                END DO
c              END IF
c            END DO
c          END IF
        END IF

C -- A FLAG IS ALSO SET IF ANY PARAMETERS ARE GLUED TO THEIR BOUNDS.

        ISTKFLG=0
        DO 5381 IPP=1,NPAR
          IF(JSTK(IPP).LT.0) THEN
            ISTKFLG=1
            GO TO 5382
          END IF
5381    CONTINUE
5382    CONTINUE


C -- THE JUPITER _PE FILE IS UPDATED.

        IF((JUP_PAR_FILES.NE.0).AND.(J_OUTNAM.NE.' ').AND.
     +     (NOPTMAX.NE.-1).AND.(NOPTMAX.NE.-2)
     +     .AND.(IFIN.NE.10))THEN
          WRITE(J_PE,101,ADVANCE='NO') IOPT
          DO I=1,NPAR
            IF(ITRANS(I).LT.0) CYCLE
            WRITE(J_PE,102,ADVANCE='NO') PVAL(I)
          END DO
          WRITE(J_PE,'(A)') ' '
        END IF

C -- THE JUPITER _SS FILE IS UPDATED.


C -- THE JUPITER _SS FILE IS UPDATED.

        IF((JUP_PAR_FILES.NE.0).AND.(J_OUTNAM.NE.' ')
     +     .AND.(NOPTMAX.NE.-1).AND.(NOPTMAX.NE.-2)
     +     .AND.(IFIN.NE.10))THEN
          IF(IFIN.EQ.3)THEN
            CALL OBJCLC(4,PSIPI,NXROW,NPRIOR,NOBS,REFOBS,OVAL,OWGHT,
     +      NOBGNM,NOBSGP,IRGP,SUM1)
            WRITE(J_SS,105)0,PSI-PSIPI,PSIPI,PSI
          ELSE
            DO I=1,NOBS
              TMPOBS(I)=NEWOBS(I)
            END DO
            IF(NPRIOR.NE.0) CALL PRRCLC(JFAIL,NPAR,NOBS,NXROW,NESPAR,
     +      PVAL,ITRANS,TMPOBS,X,CLINE,SCALE,NREGADJPAR,PRIORPAR)
            IF(JFAIL.NE.0) GO TO 9891
            CALL OBJCLC(1,PSI,NXROW,NPRIOR,NOBS,TMPOBS,OVAL,OWGHT,
     +      NOBGNM,NOBSGP,IRGP,SUM1)
            IF(PREDNOISE.GT.0) PSI=PSI+RES_PRED
            CALL OBJCLC(4,PSIPI,NXROW,NPRIOR,NOBS,TMPOBS,OVAL,OWGHT,
     +      NOBGNM,NOBSGP,IRGP,SUM1)
            WRITE(J_SS,105)IOPT,PSI-PSIPI,PSIPI,PSI
          END IF
        END IF

        CALL WRTFIN(JFAIL,IREC,NPAR,NOBS,NXROW,NESPAR,NPRIOR,NPARGP,
     +  NWDIM,IOPT,ITRANS,PSILO,PVAL_O,IFLAG_X,W1,W2,OVAL,MODVAL_O,
     +  OWGHT,
     +  NORM,NORMLO,APAR,AOBS,ICOV,ICOR,IEIG,CLINE,NOBSGP,NOBGNM,OBGNME,
     +  NEWOBS,JPREDICT,IPRED,PD1,IREG,WFLO,PHIMLO,PHIMACCEPT,WFMIN,
     +  PARGNME,IPARGP,PWTKP,NCOVDIM1,NCOVDIM3,NUMCOV,TOTWT,
     +  NOBSCOV,COVGP,RWORK1,RWORK2,COVAR,APAR1,APAR2,ISTKFLG,IRGP,
     +  NMLODIM1,NMLODIM2,MEMSAVE,WORK_CDIM,WORK_C,JUP_PAR_FILES,
     +  PARLBND,PARUBND,
     +  J_OUTNAM,J_UNIT,J_SS,ISNS,E_PREDLO,RES_PREDLO,JCOFILE,X,
     +  IFLAG_NOCOV,NDIM4,WORKVEC4,PSISUB,GFAC,IGTARG)
        IF(JFAIL.NE.0) THEN
          IF(JFAIL.EQ.2)IPFAIL=1
          GO TO 9891
        END IF
#ifdef FLUSHFILE
        CALL FLUSH(IREC)
        CALL FLUSH(ISNS)
#endif

        IF(IPRED.NE.0)THEN
          IF(PSILO.LE.PD1)THEN
            IF(PREDHL.EQ.'h')THEN
              WRITE(6,6155) MODVAL_O(NOBS)
6155          FORMAT(/,' Maximum prediction value = ',1PG13.6)
            ELSE
              WRITE(6,6156) MODVAL_O(NOBS)
6156          FORMAT(/,' Minimum prediction value = ',1PG13.6)
            END IF
            IF(PREDNOISE.NE.0)THEN
              WRITE(6,6166) E_PREDLO
6166          FORMAT(  ' Predictive error         = ',1PG13.6)
            END IF
          ELSE
            WRITE(6,6158)
6158        FORMAT(/,' Note: objective function did not fall ',
     +      'below PD1.')
          END IF
        END IF
        WRITE(6,6160) FLENME(:LEN_TRIM(FLENME))
6160    FORMAT(/,' See file ',A,' for full run details.')
        FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.sen'
        WRITE(6,6162) FLENME(:LEN_TRIM(FLENME))
6162    FORMAT(' See file ',A,' for parameter sensitivities.')
        IF(PESTMODE.NE.4)THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.seo'
          WRITE(6,6163) FLENME(:LEN_TRIM(FLENME))
6163      FORMAT(' See file ',A,' for observation sensitivities.')
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.res'
          WRITE(6,6161) FLENME(:LEN_TRIM(FLENME))
6161      FORMAT(' See file ',A,' for residuals.')
          IF(TOTWT.GT.0)THEN
            FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rsr'
            WRITE(6,6167) FLENME(:LEN_TRIM(FLENME))
6167        FORMAT(' See file ',A,' for rotated residuals.')
          END IF
        END IF
        IF(SVDMODE.NE.0)THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.svd'
          WRITE(6,6168) FLENME(:LEN_TRIM(FLENME))
6168      FORMAT(' See file ',A,' for history of SVD process.')
        END IF


        IF((RSTFLE.NE.0).AND.(IFIN.NE.10)) THEN
          FLENME=CASEFL(:LEN_TRIM(CASEFL))//'.rst'
          FLENME1=CASEFL(:LEN_TRIM(CASEFL))//'.jst'
          INQUIRE(FILE=FLENME,EXIST=LEXIST)
          IF(LEXIST)THEN
            IF(DONTCOPY.EQ.0)THEN
              CALL DELFILE1(JFAIL,FLENME1,CLINE)
              IF(JFAIL.NE.0) GO TO 9891
#ifdef UNIX
#ifdef SYS_FUNCTION
              NNN=SYSTEM('/bin/cp '//FLENME(1:LEN_TRIM(FLENME))//' '//
     +        FLENME1(1:LEN_TRIM(FLENME1))//' > /dev/null')
#else
              CALL SYSTEM('/bin/cp '//FLENME(1:LEN_TRIM(FLENME))//' '//
     +        FLENME1(1:LEN_TRIM(FLENME1))//' > /dev/null')
#endif
#else
#ifdef LAHEY
#ifdef AT
              CALL COPYFILE(FLENME,FLENME1)
#else
              CALL SYSTEM('copy "'//FLENME(1:LEN_TRIM(FLENME))//'" "'//
     +        FLENME1(1:LEN_TRIM(FLENME1))//'" > nul')
#endif
#else
#ifdef SYS_FUNCTION
              NNN=SYSTEM('copy "'//FLENME(1:LEN_TRIM(FLENME))//'" "'//
     +        FLENME1(1:LEN_TRIM(FLENME1))//'" > nul')
#else
              CALL SYSTEM('copy "'//FLENME(1:LEN_TRIM(FLENME))//'" "'//
     +        FLENME1(1:LEN_TRIM(FLENME1))//'" > nul')
#endif
#endif
#endif
            END IF
          END IF
          CALL FFOPEN(JFAIL,-IRSF,'w',' ',25,CLINE)
          IF(JFAIL.NE.0) GO TO 9891
          IIOPT=-IIOPT
          WRITE(IRSF,ERR=9300) IIOPT,NESPAR,IPRED,IREG,SVDA
          CLOSE(UNIT=IRSF)
        END IF

#ifdef PARALLEL
6170    CONTINUE
        SLAVESTOP=1
#endif

C -- If the program calling PEST is other than PEST, observations, their
C    model equivalents, and the Jacobian matrix are all unrotated.

        IF(CALLER.NE.0)THEN
          IF(NUMCOV.NE.0)THEN
            DO 6260 JCOV=1,NUMCOV
              IF(NOBSCOV(JCOV).LE.0) GO TO 6260
              IGROUP=COVGP(JCOV)
              I=0
              DO 6230 J=1,NXROW
                IF(NOBGNM(J).EQ.IGROUP)THEN
                  I=I+1
                  W1(I)=OVAL(J)
                END IF
6230          CONTINUE
              CALL ROTATE(-1,NOBSCOV(JCOV),NOBSCOV(JCOV),1,1,1,
     +        COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,NXROW,NOBGNM)
              I=0
              DO 6250 J=1,NXROW
                IF(NOBGNM(J).EQ.IGROUP)THEN
                  I=I+1
                  OVAL(J)=W1(I)
                END IF
6250          CONTINUE
6260        CONTINUE

            DO 6280 JCOV=1,NUMCOV
              IF(NOBSCOV(JCOV).LE.0) GO TO 6280
              IGROUP=COVGP(JCOV)
              I=0
              DO 6270 J=1,NXROW
                IF(NOBGNM(J).EQ.IGROUP)THEN
                  I=I+1
                  W1(I)=MODVAL_O(J)
                END IF
6270          CONTINUE
              CALL ROTATE(-1,NOBSCOV(JCOV),NOBSCOV(JCOV),1,1,1,
     +        COVAR(JCOV)%DVAL,W1,X,W2,IGROUP,NXROW,NOBGNM)
              I=0
              DO 6290 J=1,NXROW
                IF(NOBGNM(J).EQ.IGROUP)THEN
                  I=I+1
                  MODVAL_O(J)=W1(I)
                END IF
6290          CONTINUE
6280        CONTINUE

          END IF
        END IF

C -- Now we finish up.

        IPFAIL=9999
        GO TO 9891

9200    CONTINUE
        WRITE(6,9005)
9005    FORMAT(/,' *** Cannot allocate sufficient memory to ',
     +  'continue execution ***',/)
        IF((IREG.NE.0).AND.((ICOVF.NE.0).OR.(ICORF.NE.0).OR.
     +     (IEIGF.NE.0)))THEN
           WRITE(6,9007)
9007       FORMAT(' *** Try setting all of ICOV, ICOR and IEIG to ',
     +     'zero ***',/)
        END IF
        RETURN

9060    CALL STPERR(183,0,' ',0,' ',CLINE)
        GO TO 9891
9300    CALL STPERR(10,1,' ',0,' ',CLINE)
        GO TO 9891
9350    CALL STPERR(26,0,' ',0,' ',CLINE)
        GO TO 9891
9550    CALL STPERR(23,1,' ',0,' ',CLINE)
        GO TO 9891
9600    CALL STPERR(24,1,' ',0,' ',CLINE)
        GO TO 9891
9700    CALL STPERR(128,0,' ',0,' ',CLINE)
        GO TO 9891
9710    CALL STPERR(179,0,' ',0,' ',CLINE)
        GO TO 9891
9750    CALL STPERR(129,1,' ',0,' ',CLINE)
        GO TO 9891
9800    CALL STPERR(134,0,' ',0,' ',CLINE)
        GO TO 9891
9970    CALL STPERR(181,0,' ',0,' ',CLINE)
        GO TO 9891
9980    CALL STPERR(170,0,' ',0,' ',CLINE)
        GO TO 9891
9890    IPFAIL=1
9891    CONTINUE
        IF(IPFAIL.EQ.0)THEN
          IPFAIL=2
        ELSE IF(IPFAIL.EQ.9999)THEN
          IPFAIL=0
        END IF

C -- If necessary, slave execution is stopped.

#ifdef PARALLEL
        IF(SLAVESTOP.NE.0)THEN

#ifndef BEO
          DO 6200 I=1,NSLAVE
          AFILE=ASLDIR(I)(1:LEN_TRIM(ASLDIR(I)))//
     +    SFINFLE(1:LEN_TRIM(SFINFLE))
#ifdef UNIX
          OPEN(UNIT=IPCM,FILE=AFILE(1:LEN_TRIM(AFILE)),
     +    IOSTAT=IERR)
#else
#ifdef LF90
          OPEN(UNIT=IPCM,FILE=AFILE(1:LEN_TRIM(AFILE)),
     +    ACTION='READWRITE,DENYNONE',BLOCKSIZE=1,IOSTAT=IERR)
#else
#ifdef INTEL
          OPEN(UNIT=IPCM,FILE=AFILE(1:LEN_TRIM(AFILE)),
     +    BLOCKSIZE=1,SHARED,IOSTAT=IERR)
#else
          OPEN(UNIT=IPCM,FILE=AFILE(1:LEN_TRIM(AFILE)),
     +    BLOCKSIZE=1,IOSTAT=IERR)
#endif
#endif
#endif
          WRITE(IPCM,'(A)',IOSTAT=IERR) ' F'
          CLOSE(UNIT=IPCM,IOSTAT=IERR)
6200      CONTINUE
#endif
        END IF
#endif

        DEALLOCATE(ORGVAL,TMPVAL,NEWVAL,PSILOL,LSTVAL,PWORD,NW,
     +             PHOLD,EHOLD,APAR1,APAR2,JSTK,STAT=IERR)
        DEALLOCATE(ISENS,NAVAL,SENS,STAT=IERR)
        DEALLOCATE(REFOBS,W1,W2,TMPOBS,NEWOBS,OBSN1,OBSN2,IIOBS,
     +             PSISUB,NAOBS,IRGP,SEOGP,STAT=IERR)
        DEALLOCATE(GHOLD,IDBND,STAT=IERR)
        DEALLOCATE(RHS,GRAD,SC,STAT=IERR)
        DEALLOCATE(PARDEL,MRKDEL,STAT=IERR)
        DEALLOCATE(PN,PNTEMP,PC0,STAT=IERR)
        DEALLOCATE(PHIMLOL,PHIRLOL,STAT=IERR)
        DEALLOCATE(PRIORPAR,IWSENS,STAT=IERR)
        DEALLOCATE(A,LL,LCINS,STAT=IERR)
        DEALLOCATE(PRDLOL,STAT=IERR)
        DEALLOCATE(W1KP,LTTMPOBS,LTPHI,LTPRED,LTFACC,STAT=IERR)
        DEALLOCATE(ORIGWGHT,STAT=IERR)
        DEALLOCATE(OWGHTLO,STAT=IERR)
        DEALLOCATE(OWGHTKP,STAT=IERR)
        DEALLOCATE(NORM,NORMLO,LHS,LHSVEC,STAT=IERR)
        DEALLOCATE(NRM,NRMLO,WORKVEC1,WORKVEC2,WORKVEC3,WORKVEC4,
     +             WORK_C,STAT=IERR)
        DEALLOCATE(WORKVEC5,STAT=IERR)
        DEALLOCATE(cgr1,cgr2,cgv,cgw,cgy,STAT=IERR)
        DEALLOCATE(RWORK1,RWORK2,STAT=IERR)
        DEALLOCATE(GFAC,OLDGFAC,STAT=IERR)

        IF(SVDA.NE.0) THEN
          CALL SVDA_DEALLOCATE()
        ELSE
          DEALLOCATE(BPARGPNME,SENFACTOR,STAT=IERR)
        END IF

        DEALLOCATE(SCREUSE,STAT=IERR)

#ifdef PARALLEL
        CLOSE(UNIT=PARREG,STATUS='DELETE',IOSTAT=IERR)
        CLOSE(UNIT=OBSREG,STATUS='DELETE',IOSTAT=IERR)
#ifndef BEO
        DEALLOCATE(ASLAVE,ASLDIR,IDET,ISTATS,IORDER,
     +             ISTATR,ISTRTME,IRUNTME,JRUN,SINFLE,SOUFLE,SCOM,
     +             ITRIAL,REGLAM,MSINFLE,MSOUFLE,SLAVEGROUP,STAT=IERR)
#endif
#else
        DEALLOCATE(INCPAR,STAT=IERR)
#endif
        DEALLOCATE(WORKVEC7,STAT=IERR)
        IF(PESTMODE.EQ.4)THEN
          DEALLOCATE(PARPSISUB,STAT=IERR)
        END IF

C -- Close some files.

        IF(PESTMODE.EQ.4)THEN
          CLOSE(UNIT=IPPD,IOSTAT=IERR)
          CLOSE(UNIT=IPOD,IOSTAT=IERR)
        END IF

        RETURN
#ifdef PESTMOD
9999    END SUBROUTINE RUN_PEST
#else
9999    END
#endif


#ifdef PARALLEL
        SUBROUTINE SKIP_COMP(JFLAG,NSRUN,LDOSCH,MANYLAM,LSEARCH,LSCH1,
     +  LMRUN,LAMBDA,REGLAM)

        IMPLICIT NONE

        INTEGER   :: JFLAG,LDOSCH,MANYLAM,LSEARCH,LSCH1,LMRUN,NSRUN
        INTEGER   :: LR
        DOUBLE PRECISION :: LAMBDA
        DOUBLE PRECISION :: REGLAM(NSRUN)

        JFLAG=0
        IF(LDOSCH.NE.0)THEN
          IF(MANYLAM.EQ.1)THEN
            IF(LSEARCH.EQ.0)THEN
              IF(LSCH1.EQ.1)THEN
                JFLAG=1
                DO LR=1,LMRUN
                  IF(ABS(LAMBDA-REGLAM(LR))/LAMBDA.LE.1.0D-5)
     +            GO TO 7777
                END DO
                JFLAG=0
7777            CONTINUE
              END IF
            END IF
          END IF
        END IF

        RETURN

#ifdef PESTMOD
        END SUBROUTINE SKIP_COMP
#else
        END
#endif
#endif


C -- Issues
C -- Parallel PEST should be turned off when doing LSQR.
C -- As presently programmed all base parameters are expected to have derivatives in the base
C      parameter derivatives file. Maybe in the future we could relax this so that only
C      some need have these derivatives updates, but that we could still re-define super
C      parameters on the basis of the partly new and partly old derivatives and recompute
C      super parameter derivatives using finite differences for some model components and
C      a derivative-specific model run for other model components which are capable of such
C      a run.
C -- We need to calculate resolution matrix for svdmode equal to 2 (and for LSQR).
C -- We have to fix up calculation of the resolution data left by PEST when it does LSQR and SVDMODE
C      of two.
C -- Subroutine RESDAT in the PARDEF file is sure going to need attention.
C -- We need to put an option in RESPROC of whether or not the marquardt lambda is included in calculation of
C      resolution matrix. This is especially the case where SVDMODE is equal to 2.
C -- When computing resolution matrix stuff, don't forget different role of lambda when, for example, doing
C      SVD with mode of 2.
C -- Some utilities read a PEST control file. Will they tolerate an LSQR section?
C -- Do predvar etc work ok when PEST has an LSQR section? What about REGERR etc?
C -- Check SVD-assist and RESPROC when base parameters hit bounds.
C -- Check what happens with parallel PEST when we set JACFILE nonzero.
C -- VERY IMPORTANT - we need to re-define bb when we do SVDA, LSQR and external derivatives.

