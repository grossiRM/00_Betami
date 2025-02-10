       program genlinpred

C -- Program GENLINPRED acts as a driver for PREDVAR*, PREDUNC* and other PEST
C    utilities in order to facilite a variety of error and uncertainty analyses
C    on a PEST input dataset.

       implicit none

       logical        lexist
       integer, parameter :: MAXSING=2000
       integer, parameter :: NOBSTHRESH=1000
       integer        ibeg,iend,nb_pst,iline,icurrent
       integer        pestmode,npar,nobs,npargp,nprior,nobsgp,
     +                nespar,numnonzero,numsup,nespar1
       integer        ipargp,ipar,iobsgp,iobs,ies,jes,jpar
       integer        predpar
       integer        itemp1,itemp2,itemp3
       integer        ifail,ierr,nnn
       integer        n,itemp,i,ncol,j,nb
       integer        ncount,mcount,icount,istep,ilist,iicount,icountlo
       integer        isinglo,isingstart,isingfin,numsing,singmin,ising
       integer        ndim1,ndim2,ndim
       integer        ndimsol,isuppress
       integer        lw(10),rw(10)

       double precision weightfac,refvar,phi,rtemp,var0,var
       double precision varmin
       double precision rtemp1,rtemp2
       double precision precalunc,precalvar,postcalunc,postcalvar,
     +                  precalerr,postcalerr

       character*1    apg,aog,ab,aa
       character*1    ap,awe,awu,ace,acu,ag,as,arpur,arper,asne,aid,
     +                aun
       character*1    avchoice
       character*4    aext
       character*4    aswitch
       character*12   aversion,aapar
       character*15   aline
       character*20   tonull,atemp,anum,apred
       character*25   aprogram,aprogram_lo
       character*25   heading(4)
       character*40   asection
       character*200  def_uncertfile,scaled_uncertfile,scaled_pestfile,
     +                scaled_jcofile,p2pfile,tp2pfile,parfile,
     +                tparfile,matfile,imatfile,vecbase,varbase,
     +                tinfile,toutfile,tdatfile,tsenfile,tplistfile,
     +                tslistfile,tslistfile0,
     +                tolistfile_sub,tolistfile_add,
     +                scaled_covfile
       character*200  tempfile
       character*200  outfile
       character*200  uncertfile,pestfile,jcofile,senfile
       character*200  afile,afile1,afile2
       character*200  astring
       character*300  cline
       character*500  amessage

       integer                       :: sing(2,MAXSING),ss(MAXSING)
       double precision              :: nulterm(2,MAXSING),nt(MAXSING),
     +                                  solterm(2,MAXSING),st(MAXSING),
     +                                  total(2,MAXSING),tt(MAXSING)

       integer,          allocatable :: icovmat(:)
       integer,          allocatable :: nobgnm(:),npargnm(:),itrans(:)
       character*12,     allocatable :: apar(:),apargp(:),aobsgp(:)
       character*20,     allocatable :: acontrib(:)
       character*20,     allocatable :: aobs(:)
       double precision, allocatable :: lbound(:),ubound(:)
       double precision, allocatable :: ident(:),relerr(:),relunc(:)
       double precision, allocatable :: senscale(:),predvec(:)
       double precision, allocatable :: tempvec(:)
       double precision, allocatable :: contrib(:,:),ocontrib(:)


C -- Initialization

       def_uncertfile        = 'p###.unc'
       scaled_uncertfile     = 's###.unc'
       scaled_covfile        = 's###.cov'
       scaled_pestfile       = 'p###.pst'
       scaled_jcofile        = 'p###.jco'
       p2pfile               = 'p2p###.dat'
       tp2pfile              = 'p2p###.tpl'
       parfile               = 'p###.par'
       tparfile              = 'p###.tpl'
       matfile               = 'm###.mat'
       imatfile              = 'i###.mat'
       vecbase               = 'v###'
       varbase               = 's###'

       tinfile               = 't###.in'
       toutfile              = 't###.out'
       tdatfile              = 't###.dat'
       tsenfile              = 't###.vec'
       tplistfile            = 'p###.lst'
       tslistfile            = 's###.lst'
       tslistfile0           = 's##0.lst'
       tolistfile_sub        = 'o##s.lst'
       tolistfile_add        = 'o##a.lst'
       uncertfile=' '

       apg=' '
       aog=' '
       ab=' '
       aa=' '
       ap=' '
       awe=' '
       awu=' '
       ace=' '
       acu=' '
       ag=' '
       as=' '
       arpur=' '
       arper=' '
       asne=' '
       aid=' '
       aun=' '

#ifdef UNIX
       tonull=' > /dev/null'
#else
       tonull=' > nul'
#endif

#ifndef UNIX
#ifdef LAHEY
       open(unit=*,action='read',carriagecontrol='list')
#endif
#endif

       include 'version.inc'
       write(6,10) trim(aversion)
10     format(/,' GENLINPRED Version ',a,'. Watermark Numerical ',
     + 'Computing.',/)

       write(6,11)
11     format(' Note: if there are too many observations in the PEST ',
     + 'control file to conduct')
       write(6,12)
12     format(' uncertainty (rather than error) analysis, ',
     + 'consider using super observations.')
       write(6,13)
13     format(' See the SUPOBSPREP utility.')
       write(6,*)


C -- Is there an "/s" switch.

#ifdef NO_CMDLINE
       write(*,*)
1      write(*,'(a)') ' ENTER COMMAND LINE ARGUMENTS > '
       read(*,'(a)',ERR=1) CLINE
#else
       call pgetcl(aswitch)
#endif

C       aswitch=adjustl(aswitch)
C       call lowcase(aswitch)
C       if(aswitch.eq.' ')then
C         isuppress=0
C       else if(aswitch(1:3).eq.'/s ')then
C         isuppress=1
C       else
C         write(amessage,5) trim(aswitch)
C5        format(' Unknown command line switch - "',a,'".')
C         go to 9890
C       end if
       isuppress=1

20     write(6,30,advance='no')
30     format(' Enter name of PEST control file: ')
       read(5,'(a)') afile
       if(afile.eq.' ') go to 20
       ibeg=1
       iend=len_trim(afile)
       call getfile(ifail,afile,pestfile,ibeg,iend)
       if(ifail.ne.0) go to 20
#ifndef UNIX
       call lowcase(pestfile)
#endif
       nb_pst=len_trim(pestfile)
       nb_pst=nb_pst-3
       if(nb_pst.lt.1)nb_pst=1
       aext=pestfile(nb_pst:)
       call lowcase(aext)
       if(aext.ne.'.pst')then
         write(6,40)
40       format(/,' PEST control file must have an extension of ',
     +   '".pst" - try again.',/)
         go to 20
       end if
       open(unit=10,file=pestfile,status='old',iostat=ierr)
       if(ierr.ne.0)then
         call addquote(pestfile,afile)
         write(6,41) trim(afile)
41       format(/,' Cannot open file ',a,' - try again.',/)
         go to 20
       end if

C -- A check is made that the corresponding Jacobian matrix file is present.

       jcofile=pestfile(1:nb_pst)//'jco'
       inquire(file=jcofile,exist=lexist)
       if(.not.lexist)then
         call addquote(pestfile,afile1)
         call addquote(jcofile,afile2)
         write(amessage,60) trim(afile2),trim(afile1)
60       format('Cannot find JCO file ',a,' corresponding ',
     +   'to PEST control file ',a,'.')
         go to 9890
       end if

C -- Some data is acquired from the PEST control file.

       write(6,*)
       call addquote(pestfile,afile)
       write(6,100) trim(afile)
100    format(' - reading PEST control file ',a,'....')
       iline=1
       read(10,'(a)',err=9000,end=9050) cline
       call lowcase(cline)
       if(cline(1:3).ne.'pcf')then
         write(amessage,110) trim(afile)
110      format('"pcf" header expected at first line of file ',a,'.')
         go to 9890
       end if
       iline=iline+1
       read(10,*,err=9000,end=9050)
       iline=iline+1
       read(10,'(a)',err=9000,end=9050) cline
       call linspl(ifail,2,lw,rw,cline)
       if(ifail.ne.0) go to 9150
       atemp=cline(lw(2):rw(2))
       call lowcase(atemp)
       if(atemp.eq.'estimation')then
         pestmode=1
       else if(atemp.eq.'prediction')then
         pestmode=2
       else if(atemp(1:5).eq.'regul')then
         pestmode=3
       else
         call writint(aline,iline)
         write(amessage,109) trim(aline),trim(afile)
109      format('Unknown mode at line ',a,' of file ',a,'.')
         go to 9890
       end if
       if(pestmode.eq.3)then
         write(amessage,111) trim(afile)
111      format('PEST control file ',a,' instructs PEST to run ',
     +   'in regularisation mode. GENLINPRED will not allow this. ',
     +   'Copy to new PEST control file, alter to "estimation" ',
     +   'mode, remove regularisation ',
     +   'observations and/or prior information equations, and ',
     +   'then run JCO2JCO to build corresponding new JCO file.')
         go to 9890
       end if
       iline=iline+1
       read(10,*,err=9000,end=9050) npar,nobs,npargp,nprior,nobsgp
       if(nprior.ne.0)then
         write(amessage,112) trim(afile)
112      format('PEST control file ',a,' includes prior information. ',
     +   'GENLINPRED will not allow this. Copy to new PEST control ',
     +   'file, remove prior information, ',
     +   'and then run JCO2JCO to build corresponding new JCO file.')
         go to 9890
       end if
       allocate(apar(npar),apargp(npargp),aobs(nobs),aobsgp(nobsgp),
     + nobgnm(nobs),npargnm(npar),stat=ierr)
       if(ierr.ne.0) go to 9200
       allocate(itrans(npar),lbound(npar),ubound(npar),ident(npar),
     + relerr(npar),relunc(npar),stat=ierr)
       if(ierr.ne.0) go to 9200
       allocate(icovmat(nobsgp),stat=ierr)
       if(ierr.ne.0) go to 9200

       asection='parameter groups'
       do
         iline=iline+1
         read(10,'(a)',err=9000,end=9250) cline
         cline=adjustl(cline)
         if(cline(1:1).eq.'*')then
           if(index(cline,'parameter groups').ne.0) exit
         end if
       end do
       do ipargp=1,npargp
         iline=iline+1
         read(10,'(a)',err=9000,end=9050) cline
         call linspl(ifail,1,lw,rw,cline)
         if(ifail.ne.0) go to 9150
         apargp(ipargp)=cline(lw(1):rw(1))
         call lowcase(apargp(ipargp))
       end do

       asection='parameter data'
       do
         iline=iline+1
         read(10,'(a)',err=9000,end=9250) cline
         cline=adjustl(cline)
         if(cline(1:1).eq.'*')then
           if(index(cline,'parameter data').ne.0) exit
         end if
       end do
       icurrent=1
       nespar=0
       do ipar=1,npar
         iline=iline+1
         read(10,'(a)',err=9000,end=9050) cline
         call linspl(ifail,7,lw,rw,cline)
         if(ifail.ne.0) go to 9150
         apar(ipar)=cline(lw(1):rw(1))
         call lowcase(apar(ipar))
         atemp=cline(lw(2):rw(2))
         call lowcase(atemp)
         if(atemp.eq.'log')then
           itrans(ipar)=1
           nespar=nespar+1
         else if(atemp.eq.'none')then
           itrans(ipar)=0
           nespar=nespar+1
         else if((atemp.eq.'fixed').or.(atemp.eq.'tied'))then
           itrans(ipar)=-1
         else
           call writint(aline,iline)
           write(amessage,115) trim(aline),trim(afile)
115        format('Transformation type must be "none", "fixed", ',
     +     '"tied" or "log" at line ',a,' of file ',a,'.')
           go to 9890
         end if
         call drealread(ifail,cline(lw(5):rw(5)),lbound(ipar))
         if(ifail.ne.0) go to 9000
         call drealread(ifail,cline(lw(6):rw(6)),ubound(ipar))
         if(ifail.ne.0) go to 9000
         if(itrans(ipar).ge.0)then
           if(ubound(ipar)-lbound(ipar).le.0.0)then
             call writint(aline,iline)
             write(amessage,120) trim(aline),trim(afile)
120          format('Parameter upper bound does not exceed lower ',
     +       'bound at line ',a,' of file ',a,'.')
             go to 9890
           end if
         end if
         if(itrans(ipar).eq.1)then
           if(lbound(ipar).le.0.0)then
             call writint(aline,iline)
             write(amessage,121) trim(aline),trim(afile)
121          format('Parameter is log transformed, yet lower bound is ',
     +       'not positive at line ',a,' of file ',a,'.')
             go to 9890
           end if
         end if
         atemp=cline(lw(7):rw(7))
         call lowcase(atemp)
         call which1(ifail,npargp,icurrent,apargp,atemp)
         if(ifail.ne.0)then
           call writint(aline,iline)
           write(amessage,150) trim(atemp),trim(aline),trim(afile)
150        format('Parameter group "',a,'" cited at line ',a,
     +     ' of PEST control file ',a,' is not cited in ',
     +     '"parameter groups" section of this file.')
           go to 9890
         end if
         npargnm(ipar)=icurrent
       end do

       asection='observation groups'
       do
         iline=iline+1
         read(10,'(a)',err=9000,end=9250) cline
         cline=adjustl(cline)
         if(cline(1:1).eq.'*')then
           if(index(cline,'observation group').ne.0) exit
         end if
       end do
       do iobsgp=1,nobsgp
         iline=iline+1
         read(10,'(a)',err=9000,end=9050) cline
         call linspl(ifail,1,lw,rw,cline)
         if(ifail.ne.0) go to 9150
         aobsgp(iobsgp)=cline(lw(1):rw(1))
         call lowcase(aobsgp(iobsgp))
         call linspl(ifail,2,lw,rw,cline)
         if(ifail.ne.0)then
           icovmat(iobsgp)=0
         else
           icovmat(iobsgp)=1
           afile1=cline(lw(2):)
           ibeg=1
           iend=len_trim(afile1)
           call getfile(ifail,afile1,tempfile,ibeg,iend)
           if(ifail.ne.0)then
             call writint(aline,iline)
             write(amessage,151) trim(aline),trim(afile)
151          format('Cannot read name of covariance matrix file at ',
     +       'line ',a,' of file 'a,'.')
             go to 9890
           end if
           inquire(file=tempfile,exist=lexist)
           if(.not.lexist)then
             call writint(aline,iline)
             call addquote(tempfile,afile1)
             write(amessage,152) trim(afile1),trim(aline),trim(afile)
152          format('Cannot find covariance matrix file ',a,
     +       ' cited at line ',a,' of file ',a,'.')
             go to 9890
           end if
         end if
       end do

       asection='observation data'
       numnonzero=0
       do
         iline=iline+1
         read(10,'(a)',err=9000,end=9250) cline
         cline=adjustl(cline)
         if(cline(1:1).eq.'*')then
           if(index(cline,'observation data').ne.0) exit
         end if
       end do
       icurrent=1
       do iobs=1,nobs
         iline=iline+1
         read(10,'(a)',err=9000,end=9050) cline
         call linspl(ifail,4,lw,rw,cline)
         if(ifail.ne.0) go to 9150
         aobs(iobs)=cline(lw(1):rw(1))
         call lowcase(aobs(iobs))
         atemp=cline(lw(4):rw(4))
         call lowcase(atemp)
         call which1(ifail,nobsgp,icurrent,aobsgp,atemp)
         if(ifail.ne.0)then
           call writint(aline,iline)
           write(amessage,160) trim(atemp),trim(aline),trim(afile)
160        format('Observation group "',a,'" cited at line ',a,
     +     ' of PEST control file ',a,' is not cited in ',
     +     '"observation groups" section of this file.')
           go to 9890
         end if
         nobgnm(iobs)=icurrent
         if(icovmat(icurrent).eq.0)then
           call drealread(ifail,cline(lw(3):rw(3)),rtemp)
           if(ifail.ne.0) go to 9000
           if(rtemp.gt.0.0d0) numnonzero=numnonzero+1
         end if
       end do

C -- The number of non-zero-weighted observations is corrected for covariance matrices.

       do iobsgp=1,nobsgp
         if(icovmat(iobsgp).ne.0)then
           do iobs=1,nobs
             if(nobgnm(iobs).eq.iobsgp) numnonzero=numnonzero+1
           end do
         end if
       end do

       close(unit=10)
       write(6,190) trim(afile)
190    format(' - file ',a,' read ok.')

C -- It is now ascertained whether an uncertainty file needs to be written, or whether
C    the user has supplied one.

       write(6,*)
250    write(6,260,advance='no')
260    format(' Use bounds or uncertainty file for parameter ',
     + 'uncertainties  [b/u]: ')
       read(5,'(a)') ab
       if((ab.eq.'B').or.(ab.eq.'b'))then
         ab='b'
       else if((ab.eq.'U').or.(ab.eq.'u'))then
         ab='u'
       else
         go to 250
       end if

C -- The name of the uncertainty file is aquired if appropropriate.

       if(ab.eq.'u')then
280      write(6,290,advance='no')
290      format(' Enter name of parameter uncertainty file: ')
         read(5,'(a)') afile
         if(afile.eq.' ') go to 280
         ibeg=1
         iend=len_trim(afile)
         call getfile(ifail,afile,uncertfile,ibeg,iend)
         if(ifail.ne.0) go to 280
#ifndef UNIX
         call lowcase(uncertfile)
#endif
         inquire(file=uncertfile,exist=lexist)
         if(.not.lexist)then
           call addquote(uncertfile,afile)
           write(6,300) trim(afile)
300        format(/,' File ',a,' does not exist - try again.',/)
           go to 280
         end if
       end if

C -- The relationship between observation weights and measurement uncertainty is now determined.

       write(6,*)
350    write(6,360,advance='no')
360    format(' Are observation weights the inverse of measurement ',
     + 'uncertainty?  [y/n]: ')
       read(5,'(a)') aa
       if((aa.eq.'Y').or.(aa.eq.'y'))then
         weightfac=1.0d0
       else if((aa.eq.'N').or.(aa.eq.'n'))then
380      write(6,390,advance='no')
390      format(' Enter factor for weights to make this so: ')
         read(5,*,err=380) weightfac
         if(weightfac.le.0.0d0) go to 380
       else
         go to 350
       end if
       refvar=1.0d0/weightfac/weightfac

C -- The user is asked for details of parameter identifiability calculations (including whether
C    these calculations are required at all).

       write(6,*)
391    write(6,392,advance='no')
392    format(' Perform global parameter estimability analysis?  ',
     + '[y/n]: ')
       read(5,'(a)') ag
       if((ag.eq.'Y').or.(ag.eq.'y'))then
         ag='y'
       else if((ag.eq.'N').or.(ag.eq.'n'))then
         ag='n'
       else
         go to 391
       end if
       if(ag.eq.'y')then
         write(6,397)
         write(6,*)
372      write(6,373,advance='no')
373      format('     Compute parameter identifiabilities?  [y/n]: ')
         read(5,'(a)') aid
         if((aid.eq.'Y').or.(aid.eq.'y'))then
           aid='y'
         else if((aid.eq.'N').or.(aid.eq.'n'))then
           aid='n'
         else
           go to 372
         end if
376      write(6,377,advance='no')
377      format('     Compute relative parameter error reduction?  ',
     +   '[y/n]: ')
         read(5,'(a)') arper
         if((arper.eq.'Y').or.(arper.eq.'y'))then
           arper='y'
         else if((arper.eq.'N').or.(arper.eq.'n'))then
           arper='n'
         else
           go to 376
         end if
         if((aid.eq.'y').or.(arper.eq.'y'))then
393        write(6,394,advance='no')
394        format('     Use SUPCALC to estimate solution space ',
     +     'dimensionality?  [y/n]: ')
           read(5,'(a)') as
           if((as.eq.'Y').or.(as.eq.'y'))then
             as='y'
           else if((as.eq.'N').or.(as.eq.'n'))then
             as='n'
           else
             go to 393
           end if
           if(as.eq.'n')then
387          write(6,389,advance='no')
389          format('     Enter solution space dimensionality? ')
             read(5,*,err=387) ndimsol
             if(ndimsol.le.0) go to 387
           end if
         end if
378      write(6,379,advance='no')
379      format('     Compute relative parameter uncertainty ',
     +   'reduction?  [y/n]: ')
         read(5,'(a)') arpur
         if((arpur.eq.'Y').or.(arpur.eq.'y'))then
           arpur='y'
         else if((arpur.eq.'N').or.(arpur.eq.'n'))then
           arpur='n'
         else
           go to 378
         end if
         if((arpur.eq.'n').and.(arper.eq.'n').and.(aid.eq.'n'))then
           write(6,*)
           write(6,353)
353        format(' You have asked for global parameter estimability ',
     +     'analysis, yet you have ',/,' selected none of the ',
     +     'options for this type of analysis.')
           write(6,354)
354        format(' Respond to the above prompts again.')
           write(6,*)
           go to 391
         end if
         if(nobs.gt.NOBSTHRESH) then
           if(arpur.eq.'y')then
             if(isuppress.eq.0)then
371            write(6,362,advance='no')
362            format('     This may take a while with so many ',
     +         'observations. Are you sure? [y/n]: ')
               read(5,'(a)') aa
               if((aa.eq.'Y').or.(aa.eq.'y'))then
                 aa='y'
               else if ((aa.eq.'N').or.(aa.eq.'n'))then
                 aa='n'
               else
                 go to 371
               end if
               if(aa.eq.'n')then
                 arpur='n'
                 write(6,361)
361              format('     Relative parameter uncertainty'
     +           ' reduction will not be calculated.')
               end if
             end if
           end if
          end if
       end if

C -- The user is asked for details of predictive/parameter error/uncertainty analysis (including
C    if such analysis is required at all).

       write(6,*)
395    write(6,396,advance='no')
396    format(' Perform comprehensive analysis of a prediction/',
     + 'parameter?  [y/n]: ')
       read(5,'(a)') ap
       if(ap.eq.' ') go to 395
       if((ap.eq.'Y').or.(ap.eq.'y'))then
         ap='y'
       else if((ap.eq.'N').or.(ap.eq.'n'))then
         ap='n'
       else
         go to 395
       end if

       if(ap.eq.'y')then
         write(6,397)
397      format(' Provide pertinent information:-')
         write(6,*)
450      write(6,460,advance='no')
460      format('     Enter name of prediction/parameter to analyze: ')
         read(5,'(a)') apred
         if(apred.eq.' ') go to 450
         call lowcase(apred)
480      write(6,490,advance='no')
490      format('     Enter file to read its sensitivities ',
     +   '["p" if a parameter]: ')
         read(5,'(a)') afile
         afile=adjustl(afile)
         if(afile.eq.' ') go to 480
         ibeg=1
         iend=len_trim(afile)
         call getfile(ifail,afile,senfile,ibeg,iend)
         if(ifail.ne.0) go to 480
#ifndef UNIX
         call lowcase(senfile)
#endif
         if(senfile.eq.'P')senfile='p'
         if(senfile.ne.'p')then
           inquire(file=senfile,exist=lexist)
           if(.not.lexist)then
             call addquote(senfile,afile)
             write(6,300) trim(afile)
             go to 480
           end if
         else
           do ipar=1,npar
             if(apred.eq.apar(ipar)) go to 492
           end do
           call addquote(pestfile,afile)
           write(6,491) trim(apred),trim(afile)
491        format(/,' Parameter "',a,'" is not featured in PEST ',
     +     'control file ',a,'.',/)
           go to 450
492        continue
           if(itrans(ipar).lt.0)then
             call addquote(pestfile,afile)
             write(6,493) trim(apred),trim(afile)
493          format(/,' Parameter "',a,'" is not ',
     +       'adjustable in PEST control file ',a,'.',/)
             go to 450
           end if
           predpar=ipar
         end if
       else
         apred=' '
       end if

       if(apred.ne.' ')then

         write(6,*)
482      write(6,483,advance='no')
483      format('     Compute total predictive error and soln/null ',
     +   'space contribs?  [y/n]: ')
         read(5,'(a)') asne
         if((asne.eq.'Y').or.(asne.eq.'y'))then
           asne='y'
         else if((asne.eq.'N').or.(asne.eq.'n'))then
           asne='n'
         else
           go to 482
         end if
484      write(6,485,advance='no')
485      format('     Compute total predictive uncertainty?  [y/n]: ')
         read(5,'(a)') aun
         if((aun.eq.'Y').or.(aun.eq.'y'))then
           aun='y'
         else if((aun.eq.'N').or.(aun.eq.'n'))then
           aun='n'
         else
           go to 484
         end if

508      write(6,*)
509      write(6,510,advance='no')
510      format('     Compute parameter contributions to ',
     +   'param/prediction error?  [y/n]: ')
         read(5,'(a)') ace
         if((ace.eq.'Y').or.(ace.eq.'y'))then
           ace='y'
         else if((ace.eq.'N').or.(ace.eq.'n'))then
           ace='n'
         else
           go to 509
         end if
511      write(6,512,advance='no')
512      format('     Compute parameter contributions to ',
     +   'param/prediction uncertainty?  [y/n]: ')
         read(5,'(a)') acu
         if((acu.eq.'Y').or.(acu.eq.'y'))then
           acu='y'
         else if((acu.eq.'N').or.(acu.eq.'n'))then
           acu='n'
         else
           go to 511
         end if
         if((ace.eq.'y').or.(acu.eq.'y'))then
530        write(6,540,advance='no')
540        format('     For individual parameters or for parameter ',
     +     'groups?  [i/g]: ')
           read(5,'(a)') apg
           if((apg.eq.'I').or.(apg.eq.'i'))then
             apg='i'
           else if((apg.eq.'G').or.(apg.eq.'g'))then
             apg='g'
           else
             go to 530
           end if
         else
           apg=' '
         end if

         write(6,*)
544      write(6,545,advance='no')
545      format('     Compute observation worth wrt ',
     +   'param/prediction error?  [y/n]: ')
         read(5,'(a)') awe
         if((awe.eq.'Y').or.(awe.eq.'y'))then
           awe='y'
         else if((awe.eq.'N').or.(awe.eq.'n'))then
           awe='n'
         else
           go to 544
         end if
546      write(6,547,advance='no')
547      format('     Compute observation worth wrt ',
     +   'param/prediction uncertainty?  [y/n]: ')
         read(5,'(a)') awu
         if((awu.eq.'Y').or.(awu.eq.'y'))then
           awu='y'
         else if((awu.eq.'N').or.(awu.eq.'n'))then
           awu='n'
         else
           go to 546
         end if
         if((awe.eq.'y').or.(awu.eq.'y'))then
550        write(6,560,advance='no')
560        format('     For individual observations or for ',
     +     'observation groups?  [i/g]: ')
           read(5,'(a)') aog
           if((aog.eq.'I').or.(aog.eq.'i'))then
             aog='i'
           else if((aog.eq.'G').or.(aog.eq.'g'))then
             aog='g'
           else
             go to 550
           end if
         else
           aog=' '
         end if
         if(nobs.gt.NOBSTHRESH)then
           if((aun.eq.'y').or.(awu.eq.'y').or.(acu.eq.'y'))then
             if(isuppress.eq.0)then
               write(6,564)
564            format(/,'     You have requested uncertainty ',
     +         'analysis. This may take a while with so ',/
     +         ,'     many observations cited in PEST control file.')
563            write(6,562,advance='no')
562            format('     Is this ok?  [y/n]: ')
               read(5,'(a)') aa
               if((aa.eq.'y').or.(aa.eq.'Y'))then
                 aa='y'
               else if((aa.eq.'n').or.(aa.eq.'N'))then
                 aa='n'
               else
                 go to 563
               end if
               if(aa.eq.'n') then
                 write(6,569)
569              format('     Then respond to above prompts again.')
                 write(6,*)
                 go to 482
               end if
             end if
           end if
         end if
       end if

       if((ap.eq.'n').and.(ag.eq.'n'))then
         write(amessage,561)
561      format('Either/both of global parameter identifiability ',
     +   'analysis or predictive/parameter uncertainty analysis ',
     +   'must be requested. You have requested neither of these. ',
     +   'Hence there is no reason for GENLINPRED to continue ',
     +   'execution.')
         go to 9890
       end if

C -- The name of the output file is ascertained.

       write(6,*)
630    write(6,640,advance='no')
640    format(' Enter name for GENLINPRED output file: ')
       read(5,'(a)') afile
       if(afile.eq.' ') go to 630
       ibeg=1
       iend=len_trim(afile)
       call getfile(ifail,afile,outfile,ibeg,iend)
       if(ifail.ne.0) go to 630
#ifndef UNIX
       call lowcase(outfile)
#endif

C -- All input data has now been acquired.

C -- The formulation used by PREDUNC-suite programs is now set.

       if(nobs.lt.nespar)then
         avchoice='2'
       else
         avchoice='1'
       end if


C -- Data is recorded to the output file.

       open(unit=20,file=outfile,action='write')
       write(20,*)
       write(20,*)
       write(20,700)
700    format(t20,51('#'))
       write(20,710)
710    format(t20,'#',49(' '),'#')
       write(20,730)
730    format(t20,'#      GENLINPRED UNCERTAINTY/ERROR ANALYSIS',
     + t70,'#')
       write(20,710)
       write(20,700)
       do i=1,3
         write(20,*)
       end do
       write(20,740)
740    format(t20,'ANALYSIS DETAILS')
       write(20,750)
750    format(t20,'----------------')

       write(20,*)
       call addquote(pestfile,afile)
       write(20,760) trim(afile)
760    format(t5,'Name of PEST control file',t65,': ',a)
       write(atemp,'(1pg12.5)') weightfac
       atemp=adjustl(atemp)
       write(20,765) trim(atemp)
765    format(t5,'Factor by which to multiply observation ',
     + 'weights',t65,': ',a)
       if(ab.eq.'b')then
         astring='parameter bounds'
       else
         astring='uncertainty file'
       end if
       write(20,770) trim(astring)
770    format(t5,'Source of parameter uncertainty data',t65,': ',a)
       if(ab.eq.'u')then
         call addquote(uncertfile,afile)
         write(20,780) trim(afile)
780      format(t5,'Name of uncertainty file',t65,': ',a)
       end if

       write(20,*)
       if(ag.eq.'n')then
         atemp='no'
       else
         atemp='yes'
       end if
       write(20,789) trim(atemp)
789    format(t5,'Perform global parameter estimability analysis',
     + t65,': ',a)
       if(ag.eq.'y')then
         if(aid.eq.'n')then
           atemp='no'
         else
           atemp='yes'
         end if
         write(20,788) trim(atemp)
788      format(t10,'Parameter identifiability',t65,': ',a)
         if(arper.eq.'n')then
           atemp='no'
         else
           atemp='yes'
         end if
         write(20,787) trim(atemp)
787      format(t10,'Relative parameter error reduction',t65,': ',a)
         if(arpur.eq.'n')then
           atemp='no'
         else
           atemp='yes'
         end if
         write(20,786) trim(atemp)
786      format(t10,'Relative parameter uncertainty reduction',
     +   t65,': ',a)
       end if

       write(20,*)
       if(ap.eq.'y')then
         atemp='yes'
       else
         atemp='no'
       end if
       write(20,790) trim(atemp)
790    format(t5,'Analyse error/uncertainty of an entity',t65,': ',a)
       if(ap.eq.'y')then
         write(20,800) trim(apred)
800      format(t10,'Name of entity',t65,': ',a)
         if(senfile.eq.'p')then
           atemp='parameter'
         else
           atemp='prediction'
         end if
         write(20,810) trim(atemp)
810      format(t10,'Type of entity',t65,': ',a)
         if(senfile.ne.'p')then
           call addquote(senfile,afile)
           write(20,820) trim(afile)
820        format(t10,'File containing prediction sensitivities',t65,
     +     ': ',a)
         end if

         write(20,*)
         if(asne.eq.'y')then
           atemp='yes'
         else
           atemp='no'
         end if
         write(20,821) trim(atemp)
821      format(t10,'Total error variance',t65,': ',a)
         if(aun.eq.'y')then
           atemp='yes'
         else
           atemp='no'
         end if
         write(20,819) trim(atemp)
819      format(t10,'Total uncertainty',t65,': ',a)

         if(ace.eq.'y')then
           atemp='yes'
         else
           atemp='no'
         end if
         write(20,*)
         write(20,822) trim(atemp)
822      format(t10,'Parameter contributions to error ',
     +   'variance',t65,': ',a)
         if(acu.eq.'y')then
           atemp='yes'
         else
           atemp='no'
         end if
         write(20,823) trim(atemp)
823      format(t10,'Parameter contributions to uncertainty',
     +   t65,': ',a)
         if((ace.eq.'y').or.(acu.eq.'y'))then
           if(apg.eq.'g')then
             atemp='by group'
           else
             atemp='individual'
           end if
           write(20,830) trim(atemp)
830        format(t10,'Parameter contributions',t65,': ',a)
         end if

         write(20,*)
         if(awe.eq.'y')then
           atemp='yes'
         else
           atemp='no'
         end if
         write(20,832) trim(atemp)
832      format(t10,'Observation worth in reducing error variance',
     +   t65,': ',a)
         if(awu.eq.'y')then
           atemp='yes'
         else
           atemp='no'
         end if
         write(20,833) trim(atemp)
833      format(t10,'Observation worth in reducing uncertainty ',
     +   'variance',t65,': ',a)
         if((awe.eq.'y').or.(awu.eq.'y'))then
           if(aog.eq.'g')then
             atemp='by group'
           else
             atemp='individual'
           end if
           write(20,840) trim(atemp)
840        format(t10,'Observation worth',t65,': ',a)
         end if
       end if

#ifdef FLUSHFILE
       call flush(20)
#endif

C -- More memory is allocated.

       allocate(senscale(nespar),predvec(nespar),stat=ierr)
       if(ierr.ne.0) go to 9200

       if(apg.eq.'g')then
         ndim1=npargp+1
       else
         ndim1=nespar+1
       end if
       allocate(contrib(2,ndim1),stat=ierr)
       if(ierr.ne.0) go to 9200

       if(aog.eq.'g')then
         ndim2=nobsgp+1
       else
         ndim2=1
       end if
       ndim=max(ndim1,ndim2)
       allocate(acontrib(ndim),stat=ierr)
       if(ierr.ne.0) go to 9200

       if(aog.eq.'g')then
         ndim2=nobsgp+1
       else if(aog.eq.'i')then
         ndim2=nobs+1
       else
         ndim2=1
       end if
       allocate(ocontrib(ndim2),stat=ierr)
       if(ierr.ne.0) go to 9200

C -- Now the work begins.

C -- First a parameter uncertainty file is written if necessary.

       if(ab.eq.'b')then
         write(6,*)
         write(6,845)
845      format(' - writing temporary parameter uncertainty file...')
         uncertfile=def_uncertfile
         open(unit=30,file=def_uncertfile,action='write')
         write(30,846)
846      format(1x,'START STANDARD_DEVIATION')
         do ipar=1,npar
           if(itrans(ipar).ge.0)then
             if(itrans(ipar).eq.1)then
               rtemp=log10(ubound(ipar))-log10(lbound(ipar))
             else
               rtemp=ubound(ipar)-lbound(ipar)
             end if
             rtemp=rtemp*0.25
             write(30,850) trim(apar(ipar)),rtemp
850          format(1x,a,t16,1pg14.7)
           end if
         end do
         write(30,847)
847      format(1x,'END STANDARD_DEVIATION')

         close(unit=30)
         write(6,855)
855      format(' - temporary parameter uncertainty file written ok.')
       end if

C -- SCALEPAR is now run in order to compute a scaled parameter value set.

       aprogram='SCALEPAR'
       aprogram_lo='scalepar'
       write(6,*)
       write(6,900)
900    format(' - running SCALEPAR to build scaled PEST input ',
     + 'dataset...')
       call delfile(scaled_jcofile)
       call delfile(scaled_uncertfile)
       call delfile(scaled_covfile)
       call delfile(matfile)
       open(unit=30,file=tinfile)
       call addquote(pestfile,afile)
       write(30,'(a)') trim(afile)
       call addquote(uncertfile,afile)
       write(30,'(a)') trim(afile)
       write(30,'(a)') trim(scaled_pestfile)//'~^'      ! Special code instructs SCALEPAR not to write new model batch file
       write(30,'(a)') trim(p2pfile)
       write(30,'(a)') trim(tp2pfile)
       write(30,'(a)') trim(parfile)
       write(30,'(a)') trim(tparfile)
       write(30,'(a)') trim(matfile)
       write(30,'(a)') trim(imatfile)
       write(30,'(a)') 's'
       write(30,'(a)') 'y'
       write(30,'(a)') 'y'
       write(30,'(a)') trim(scaled_uncertfile)
       write(30,'(a)') trim(scaled_covfile)
       close(unit=30)
#ifdef SYS_FUNCTION
       nnn=system('scalepar < '//trim(tinfile)//trim(tonull))
#else
       call system('scalepar < '//trim(tinfile)//trim(tonull))
#endif
       inquire(file=scaled_jcofile,exist=lexist)
       if(.not.lexist) go to 9300
       inquire(file=scaled_uncertfile,exist=lexist)
       if(.not.lexist) go to 9300
       call delfile(p2pfile)
       call delfile(tp2pfile)
       call delfile(parfile)
       call delfile(tparfile)
       call delfile(imatfile)
       write(6,910) trim(aprogram)

C -- Scaling factors are now read if this is warranted.

       if(apred.ne.' ')then
         open(unit=11,file=matfile,status='old',iostat=ierr)
         if(ierr.ne.0)then
           write(amessage,875) trim(matfile)
875        format('Cannot open file ',a,' written by SCALEPAR to ',
     +     'determine parameter scaling factors.')
           go to 9890
         end if
         read(11,*,err=9350)itemp1,itemp2,itemp3
         if((itemp1.ne.nespar).or.(itemp2.ne.nespar).or.
     +      (itemp3.ne.-1))then
           write(amessage,880)
880        format('Internal error type 1 - contact programmer.')
           go to 9890
         end if
         ies=0
         do ipar=1,npar
           if(itrans(ipar).lt.0) cycle
           ies=ies+1
           read(11,*,err=9350,end=9350) senscale(ies)
         end do
       end if
       call delfile(matfile)

C -- Next SUPCALC is run in order to compute number of super parameters.

       if(ag.eq.'n') go to 1279

       write(20,920)
920    format(/,/,t20,'ANALYSIS OF PARAMETERS')
       write(20,921)
921    format(    t20,'----------------------',/)


       if((aid.eq.'y').or.(arper.eq.'y'))then
         if(as.eq.'y')then
           aprogram='SUPCALC'
           aprogram_lo='supcalc'
           write(6,*)
           write(6,905)
905        format(' - running SUPCALC to compute dimensions of ',
     +     'solution space...')
           open(unit=30,file=tinfile)
           write(30,'(a)') trim(scaled_pestfile)
           phi=numnonzero/weightfac/weightfac
           write(30,*) phi
           write(30,'(a)') '1'
           write(30,'(a)') 'u'
           write(30,'(a)') trim(scaled_uncertfile)
           write(30,'(a)') trim(tdatfile)
           close(unit=30)
           call delfile(toutfile)
#ifdef SYS_FUNCTION
           nnn=system
     +     ('supcalc < '//trim(tinfile)//' > '//trim(toutfile))
#else
           call system
     +     ('supcalc < '//trim(tinfile)//' > '//trim(toutfile))
#endif
           open(unit=11,file=toutfile,status='old',iostat=ierr)
           if(ierr.ne.0) go to 9300
           do
             read(11,'(a)',end=9300) cline
             if(index(cline,'Minimum number of super').ne.0) exit
           end do
           n=index(cline,'=')
           atemp=cline(n+1:)
           atemp=adjustl(atemp)
           call intread(ifail,atemp,numsup)
           if(ifail.ne.0) go to 9300
           close(unit=11,status='delete')
           call delfile(tdatfile)
           write(6,910) trim(aprogram)
910        format(' - program ',a,' run ok.')

           write(20,930) trim(aprogram)
930        format(/,' Outcomes of running program ',a,' ----->')
           write(20,*)
           call writint(anum,numsup)
           write(20,940) trim(anum)
940        format(t5,'Number of dimensions of calibration solution ',
     +     'space = ',a)
           itemp=nespar-numsup
           call writint(anum,itemp)
           write(20,950) trim(anum)
950        format(t5,'Number of dimensions of calibration null ',
     +     'space     = ',a)

#ifdef FLUSHFILE
           call flush(20)
#endif

C -- The user is notified of SUPCALC's conclusions and asked if he/she wishes to over-ride this.

           write(6,*)
           call writint(anum,numsup)
960        write(6,965) trim(anum)
965        format(' SUPCALC has recommended the use of ',a,
     +     ' solution ',
     +     'space dimensions ',/,' for computation of parameter ',
     +     'identifiability and relative ',/,
     +     ' error reduction.')
           write(6,*)
970        write(6,980,advance='no')
980        format(' Do you wish to over-ride this? [y/n]: ')
           read(5,'(a)') aa
           if((aa.eq.'Y').or.(aa.eq.'y'))then
             aa='y'
           else if((aa.eq.'N').or.(aa.eq.'n'))then
             aa='n'
           else
             go to 970
           end if
           if(aa.eq.'n')then
             if(numsup.eq.0)then
               write(6,985)
985            format(/,' A value of zero MUST be overridden - ',
     +         'try again.',/)
               go to 970
             end if
           else
             call writint(anum,nespar)
988          write(6,990,advance='no') trim(anum)
990          format(' Enter new solution space dimensions ',
     +       '[1 - ',a,']: ')
             read(5,*,err=988) numsup
             if((numsup.lt.1).or.(numsup.gt.nespar)) go to 988
             write(20,*)
             write(20,991)
991          format(' User over-ride of SUPCALC recommendations ----->')
             write(20,*)
             call writint(anum,numsup)
             write(20,940) trim(anum)
             itemp=nespar-numsup
             call writint(anum,itemp)
             write(20,950) trim(anum)
           end if
         else
           numsup=ndimsol
           if(numsup.gt.nespar)numsup=nespar
           write(20,992)
992        format(t5,'Dimensions of calibration solution space were ',
     +     'provided by user.')
           if(ndimsol.gt.nespar)then
             write(20,993)
993          format(t5,'(These were reduced to comply with number of ',
     +       'adjustable parameters.)')
           end if
           write(20,*)
           call writint(anum,numsup)
           write(20,940) trim(anum)
           itemp=nespar-numsup
           call writint(anum,itemp)
           write(20,950) trim(anum)
         end if
       end if

C -- IDENTPAR is now run in order to compute parameter identifiabilities.

       if(aid.eq.'y')then
         aprogram='IDENTPAR'
         aprogram_lo='identpar'
         write(6,*)
         write(6,1000)
1000     format(' - running IDENTPAR to compute parameter ',
     +   'identifiabilities...')
         call writint(anum,numsup)
         call delfile(toutfile)
#ifdef SYS_FUNCTION
         nnn=system('identpar '//trim(scaled_pestfile)//' '//
     +   trim(anum)//
     +   ' null null '//trim(toutfile)//trim(tonull))
#else
         call system('identpar '//trim(scaled_pestfile)//' '
     +   //trim(anum)//
     +   ' null null '//trim(toutfile)//trim(tonull))
#endif

         open(unit=11,file=toutfile,status='old',iostat=ierr)
         if(ierr.ne.0) go to 9300
         read(11,*,err=9300,end=9300)
         do ipar=1,npar
           if(itrans(ipar).lt.0) cycle
           read(11,*,err=9300,end=9300) aapar,(rtemp,i=1,numsup),
     +     ident(ipar)
         end do
         close(unit=11,status='delete')
         write(6,910) trim(aprogram)
       end if

C -- The following sets of files are needed by both PREDVAR and PREDUNC in computing
C    relative error reduction.

       if((arper.eq.'y').or.(arpur.eq.'y')) then
         ies=0
         do ipar=1,npar
           if(itrans(ipar).lt.0) cycle
           ies=ies+1
           call writint(anum,ies)
           afile=trim(vecbase)//trim(anum)//'.vec'
           open(unit=30,file=afile)
           write(30,1120) nespar,1,2
1120       format(3i8)
           do jes=1,nespar
             if(jes.eq.ies) then
               rtemp=1.0
             else
               rtemp=0.0
             end if
             write(30,1130) rtemp
1130         format(f10.2)
           end do
           write(30,1140)
1140       format('* row names')
           jes=0
           do jpar=1,npar
             if(itrans(jpar).lt.0) cycle
             write(30,'(a)') trim(adjustl(apar(jpar)))
           end do
           write(30,1160)
1160       format('* column names')
           write(30,'(a)') trim(adjustl(apar(ipar)))
           close(unit=30)
         end do
       end if

C -- Parameter error variance reduction is now calculated. Note that we do not need to scale parameter
C    sensitivities here.

       if(arper.eq.'y')then
         aprogram='PREDVAR1A'
         aprogram_lo='predvar1a'
         write(6,*)
         write(6,1100)
1100     format(' - running PREDVAR1A to compute rel param ',
     +   'error variance reductions...')
         open(unit=30,file=tdatfile)
         do ies=1,nespar
           call writint(anum,ies)
           afile1=trim(vecbase)//trim(anum)//'.vec'
           afile2=trim(varbase)//trim(anum)//'.var'
           write(30,1110) trim(afile1),trim(afile2)
1110       format(a,1x,a)
           call delfile(afile2)
         end do
         close(unit=30)

         open(unit=30,file=tinfile)
         write(30,'(a)') trim(scaled_pestfile)
         write(30,*) refvar
         write(30,'(a)') trim(scaled_uncertfile)
         write(30,'(a)') trim(tdatfile)
         write(30,1180) 0
1180     format(i5)
         write(30,1180) numsup
         write(30,'(a)') ' '
         close(unit=30)

#ifdef SYS_FUNCTION
         nnn=system('predvar1a < '//trim(tinfile)//trim(tonull))
#else
         call system('predvar1a < '//trim(tinfile)//trim(tonull))
#endif

         ies=0
         do ipar=1,npar
           if(itrans(ipar).lt.0) cycle
           ies=ies+1
           call writint(anum,ies)
           afile=trim(varbase)//trim(anum)//'.var'
           open(unit=11,file=afile,status='old',err=9300)
           do i=1,3
             read(11,*,err=9300,end=9300)
           end do
           read(11,*,err=9300,end=9300) itemp,rtemp,rtemp,var0
           read(11,*,err=9300,end=9300) itemp,rtemp,rtemp,var
           close(unit=11,status='delete')
           if(var0.ge.0.0d0)then
             relerr(ipar)=1.0d0-var/var0
           else
             relerr(ipar)=0.0d0
           end if
         end do

         call delfile(tdatfile)
         call delfile(tinfile)
         write(6,910) trim(aprogram)

       end if

C -- Parameter uncertainty variance reduction is now calculated. Note that we do not need to scale parameter
C    sensitivities here.

       if(arpur.eq.'y')then
         aprogram='PREDUNC1'
         aprogram_lo='predunc1'
         write(6,*)
         write(6,1200)
1200     format(' - running PREDUNC1 to compute rel param ',
     +   'uncert variance reductions...')

         ies=0
         do ipar=1,npar
           if(itrans(ipar).lt.0) cycle
           call delfile(toutfile)
           ies=ies+1
           open(unit=30,file=tinfile)
           write(30,'(a)') scaled_pestfile
           write(30,*) refvar
           write(30,'(a)') trim(scaled_uncertfile)
           call writint(anum,ies)
           afile=trim(vecbase)//trim(anum)//'.vec'
           write(30,'(a)') trim(afile)
           write(30,'(a)') trim(avchoice)
           close(unit=30)

#ifdef SYS_FUNCTION
           nnn=system('predunc1 < '//trim(tinfile)//' > '//
     +     trim(toutfile))
#else
           call system('predunc1 < '//trim(tinfile)//' > '//
     +     trim(toutfile))
#endif

           open(unit=11,file=toutfile,status='old',err=9300)
           do
             read(11,'(a)',err=9300,end=9300) cline
             if(index(cline,'Pre-cal pred').ne.0) exit
           end do
           n=index(cline,'=')
           if(n.eq.0) go to 9300
           atemp=cline(n+1:)
           call drealread(ifail,atemp,var0)
           if(ifail.ne.0) go to 9300
           read(11,'(a)',err=9300,end=9300) cline
           n=index(cline,'=')
           if(n.eq.0) go to 9300
           atemp=cline(n+1:)
           call drealread(ifail,atemp,var)
           if(var0.ge.0.0d0)then
             relunc(ipar)=1.0d0-var*var/var0/var0
           else
             relunc(ipar)=0.0d0
           end if
           close(unit=11,status='delete')

         end do

         call delfile(tinfile)
         write(6,910) trim(aprogram)

       end if

C -- Files needed by both the above PREDVAR1A and PREDVAR runs are deleted.

       ies=0
       do ipar=1,npar
         if(itrans(ipar).lt.0) cycle
         ies=ies+1
         call writint(anum,ies)
         afile=trim(vecbase)//trim(anum)//'.vec'
         call delfile(afile)
       end do

C -- Results are written to the GENLINPRED output file.

       write(20,1250)
1250   format(/,' Outcomes of parameter identifiability',
     + ' and error/uncertainty variance reduction analysis ----->')
       write(20,*)
       heading(1)='Parameter'
       ncol=1
       if(aid.eq.'y')then
         ncol=ncol+1
         heading(ncol)='Identifiability'
       end if
       if(arper.eq.'y')then
         ncol=ncol+1
         heading(ncol)='Rel_err_var_redn'
       end if
       if(arpur.eq.'y')then
         ncol=ncol+1
         heading(ncol)='Rel_uncert_var_redn'
       end if
       write(20,1260) (trim(heading(i)),i=1,ncol)
1260   format(t5,a,t28,a,t51,a,t74,a)
       do i=1,ncol
         do j=1,len_trim(heading(i))
           heading(i)(j:j)='-'
         end do
       end do
       write(20,1260) (trim(heading(i)),i=1,ncol)
       ies=0
       do ipar=1,npar
         if(itrans(ipar).lt.0) cycle
         heading(1)=trim(apar(ipar))
         ncol=1
         if(aid.eq.'y')then
           ncol=ncol+1
           write(heading(ncol),1170) ident(ipar)
         end if
         if(arper.eq.'y')then
           ncol=ncol+1
           write(heading(ncol),1170) relerr(ipar)
1170       format(1pg14.7)
         end if
         if(arpur.eq.'y')then
           ncol=ncol+1
           write(heading(ncol),1170) relunc(ipar)
         end if
         write(20,1260) (trim(heading(i)),i=1,ncol)
        end do
        if(aid.eq.'y')then
          write(20,*)
          write(20,1280)
1280      format(t5,'Note: run IDENTPAR yourself to see identiability ',
     +    'contributions by different solution ',/,t5,'space ',
     +    'eigencomponents.')
        end if

#ifdef FLUSHFILE
        call flush(20)
#endif

C -- If predictive uncertainty is not to be examined, the program is terminated.

        if(apred.eq.' ') go to 9895

1279    continue
        write(20,1281)
1281    format(/,/,t20,'ANALYSIS OF PREDICTION')
        write(20,1282)
1282    format(    t20,'----------------------',/)

C -- The predictive sensitivity is formulated or read.

        if(senfile.eq.'p')then
          ies=0
          do ipar=1,npar
            if(itrans(ipar).lt.0) cycle
            ies=ies+1
            if(ipar.eq.predpar)then
              predvec(ies)=1.0d0
            else
              predvec(ies)=0.0d0
            end if
          end do
        else
          nb=len_trim(senfile)
          i=nb-3
          if(i.lt.1)i=1
          aext=senfile(i:nb)
          call lowcase(aext)
          if(aext.eq.'.jco')then
            write(6,1310)
1310        format(/,' - running program JROW2VEC to extract ',
     +      'predictive sensitivities...')
            aprogram='JROW2VEC'
            aprogram_lo='jrow2vec'
            call delfile(toutfile)
            call addquote(senfile,afile1)
#ifdef SYS_FUNCTION
            nnn=system('jrow2vec '//trim(afile1)//' '//
     +      trim(apred)//' '//trim(toutfile)//trim(tonull))
#else
            call system('jrow2vec '//trim(afile1)//' '//
     +      trim(apred)//' '//trim(toutfile)//trim(tonull))
#endif
            open(unit=11,file=toutfile,status='old',iostat=ierr)
            if(ierr.ne.0) go to 9300
            read(11,*,err=9300,end=9300) nespar1
            if(nespar1.lt.nespar)then
              call addquote(senfile,afile1)
              call addquote(pestfile,afile2)
              write(amessage,1320) trim(afile1),trim(afile2)
1320          format('JCO file ',a,' cites less adjustable parameters ',
     +        'than PEST control file ',a,'.')
              go to 9890
            end if
            afile=toutfile
          else
            call addquote(senfile,afile)
            write(6,1375) trim(afile)
1375        format(/,' - reading predictive sensitivities from ',
     +      'file ',a,'...')
            open(unit=11,file=senfile,status='old',iostat=ierr)
            if(ierr.ne.0)then
              call addquote(senfile,afile)
              write(amessage,1380) trim(afile)
1380          format('Cannot open file ',a,' to read ',
     +        'predictive sensitivities.')
              go to 9890
            end if
            read(11,*,err=9400,end=9400) nespar1
            if(nespar1.lt.nespar)then
              call addquote(senfile,afile1)
              call addquote(pestfile,afile2)
              write(amessage,1390) trim(afile1),trim(afile2)
1390          format('File ',a,' cites less adjustable parameters ',
     +        'than PEST control file ',a,'.')
              go to 9890
            end if
            call addquote(senfile,afile)
          end if
          allocate(tempvec(nespar1),stat=ierr)
          if(ierr.ne.0) go to 9200
          do i=1,nespar1
            read(11,*,err=9400,end=9400) tempvec(i)
          end do
          read(11,'(a)',err=9400,end=9400) cline
          call lowcase(cline)
          if(index(cline,'name').eq.0)then
            call writint(aline,nespar+2)
            write(amessage,1410) trim(aline),trim(afile)
1410        format('"* row names" header expected to follow ',
     +      'predictive sensitivities and precede parameter names ',
     +      'at line ',a,' of file ',a,'.')
            go to 9890
          end if
          predvec=-1.1d35          ! an array
          do i=1,nespar1
            read(11,'(a)') aapar
            call lowcase(aapar)
            ies=0
            do ipar=1,npar
              if(itrans(ipar).lt.0) cycle
              ies=ies+1
              if(aapar.eq.apar(ipar))then
                predvec(ies)=tempvec(i)
                go to 1350
              end if
            end do
1350        continue
          end do
          close(unit=11)
          deallocate(tempvec)
          ies=0
          do ipar=1,npar
            if(itrans(ipar).lt.0) cycle
            ies=ies+1
            if(predvec(ies).lt.-1.0d35)then
              call addquote(senfile,afile1)
              call addquote(pestfile,afile2)
              if(aext.eq.'.jco')then
                atemp='JCO'
              else
                atemp='sensitivity'
              end if
              write(amessage,1360) trim(apar(ipar)),trim(afile2),
     +        trim(atemp),trim(afile1)
1360          format('Parameter "',a,'" cited in PEST control file ',
     +        a,' is not cited in ',a,' file ',a,'.')
              go to 9890
            end if
          end do
          if(aext.eq.'.jco')then
            write(6,1315)
1315        format(' - program JROW2VEC run ok.')
          else
            call addquote(senfile,afile)
            write(6,1361) trim(afile)
1361        format(' - file ',a,' read ok.')
          end if

        end if

C -- The sensitivities of the prediction must now be scaled.

        do ies=1,nespar
          predvec(ies)=predvec(ies)/senscale(ies)   ! check that non divide by zero
        end do

C -- The scaled predictive sensitivity vector is now written to a sensitivity file.

        open(unit=30,file=tsenfile,action='write')
        write(30,1120) nespar,1,2
        do ies=1,nespar
          write(30,1430) predvec(ies)
1430      format(1x,1pg14.7)
        end do
        write(30,1140)
        do ipar=1,npar
          if(itrans(ipar).lt.0) cycle
          write(30,'(a)') trim(adjustl(apar(ipar)))
        end do
        write(30,1440)
1440    format('* column names')
        write(30,'(1x,a)') trim(apred)
        close(unit=30)

C -- PREDVAR1A is now run in order to obtain plots of error variance vs singular number.

        if(asne.eq.'y')then
          aprogram='PREDVAR1A'
          aprogram_lo='predvar1a'
          write(6,1445)
1445      format(/,' - running PREDVAR1A to obtain pred. error ',
     +    'variance sing. val. dependence...')
          open(unit=30,file=tdatfile,action='write')
          write(30,1450) trim(tsenfile),trim(toutfile)
1450      format(1x,a,2x,a)
          close(unit=30)

C -- It is run a first time to put us in the ballpark

          if(nespar.gt.50) then
            call delfile(toutfile)
            open(unit=30,file=tinfile,action='write')
            write(30,1455) trim(scaled_pestfile)
1455        format(1x,a)
            write(30,1460) refvar
1460        format(1x,1pg14.7)
            write(30,1455) trim(scaled_uncertfile)
            write(30,1455) trim(tdatfile)
            istep=nespar/20
            ncount=0
            do i=0,nespar,istep
              ncount=ncount+1
              if(ncount.gt.MAXSING)then
                write(amessage,1475)
                go to 9890
              end if
              write(30,1470) i
1470          format(1x,i5)
              sing(1,ncount)=i
            end do
            if(i.ne.nespar) then
              write(30,1470) nespar
              ncount=ncount+1
              if(ncount.gt.MAXSING)then
                write(amessage,1475)
1475            format('Increase MAXSING and re-compile program.')
                go to 9890
              end if
              sing(1,ncount)=nespar
            end if
            write(30,'(a)') ' '
            close(unit=30)
#ifdef SYS_FUNCTION
            nnn=system('predvar1a < '//trim(tinfile)//' '//
     +      trim(tonull))
#else
            call system('predvar1a < '//trim(tinfile)//' '//
     +      trim(tonull))
#endif
            open(unit=11,file=toutfile,status='old',err=9300)
            read(11,*,err=9300,end=9300)
            read(11,*,err=9300,end=9300)
            read(11,*,err=9300,end=9300)
            varmin=1.0d300
            do icount=1,ncount
              read(11,*,err=9300,end=9300) itemp,
     +        nulterm(1,icount),solterm(1,icount),total(1,icount)
              if(total(1,icount).lt.varmin)then
                varmin=total(1,icount)
                icountlo=icount
              end if
            end do
            close(unit=11,status='delete')
            if(icountlo.eq.1)then
              isingstart=sing(1,1)
              isingfin=sing(1,4)
            else if(icountlo.eq.ncount)then
              isingstart=sing(1,ncount-3)
              isingfin=sing(1,ncount)
            else
              i=icountlo-2
              j=icountlo+2
              if(i.lt.1)then
                i=1
                j=j+1
              else if(j.gt.ncount)then
                j=ncount
                i=i-1
              end if
              if(i.lt.1)i=1
              if(j.gt.ncount)j=ncount
              isingstart=sing(1,i)
              isingfin=sing(1,j)
            end if
          else
            isingstart=0
            isingfin=nespar
          end if

C -- Now we run PREDVAR1A again.

          call delfile(toutfile)
          open(unit=30,file=tinfile,action='write')
          write(30,1455) trim(scaled_pestfile)
          write(30,1460) refvar
          write(30,1455) trim(scaled_uncertfile)
          write(30,1455) trim(tdatfile)
          mcount=0
          do i=isingstart,isingfin
            mcount=mcount+1
            write(30,1470) i
            sing(2,mcount)=i
          end do
          write(30,'(a)') ' '
          close(unit=30)
#ifdef SYS_FUNCTION
          nnn=system('predvar1a < '//trim(tinfile)//' '//trim(tonull))
#else
          call system('predvar1a < '//trim(tinfile)//' '//trim(tonull))
#endif
          open(unit=11,file=toutfile,status='old',err=9300)
          read(11,*,err=9300,end=9300)
          read(11,*,err=9300,end=9300)
          read(11,*,err=9300,end=9300)
          do icount=1,mcount
            read(11,*,err=9300,end=9300) itemp,
     +      nulterm(2,icount),solterm(2,icount),total(2,icount)
          end do
          close(unit=11)

C -- The results from the two runs are combined.

          if(ncount.eq.0)then
            do icount=1,mcount
              ss(icount)=sing(2,icount)
              nt(icount)=nulterm(2,icount)
              st(icount)=solterm(2,icount)
              tt(icount)=total(2,icount)
            end do
            numsing=mcount
          else
            numsing=0
            do
              ilist=0
              singmin=999999
              do icount=1,ncount
                if(sing(1,icount).lt.singmin)then
                  singmin=sing(1,icount)
                  ilist=1
                  iicount=icount
                end if
              end do
              do icount=1,mcount
                if(sing(2,icount).lt.singmin)then
                  singmin=sing(2,icount)
                  ilist=2
                  iicount=icount
                end if
              end do
              if(ilist.ne.0)then
                if(numsing.ne.0)then
                  do j=1,numsing
                    if(singmin.eq.ss(j)) then
                      sing(ilist,iicount)=9999999
                      go to 1499
                    end if
                  end do
                end if
                numsing=numsing+1
                if(numsing.gt.MAXSING)then
                  write(amessage,1475)
                  go to 9890
                end if
                ss(numsing)=singmin
                nt(numsing)=nulterm(ilist,iicount)
                st(numsing)=solterm(ilist,iicount)
                tt(numsing)=total(ilist,iicount)
                sing(ilist,iicount)=9999999
              else
                go to 1500
              end if
1499          continue
            end do
1500        continue
          end if

C -- The results are now written to the GENLINPRED output file.

          write(20,1520) trim(apred)
1520      format(' PREDVAR1 analysis for prediction "',a,'" ----->',/)
          write(20,1540)
1540      format(t5,'Sing_val',t20,'Null-space',t37,'Soln-space',t52,
     +    'Total',  t70,'Total')
          write(20,1550)
1550      format(t5,'index',   t20,'term',      t37,'term',      t52,
     +    'variance',t70,'std_dev')
          write(20,1551)
1551      format(t5,'--------',t20,'----------',t37,'----------',t52,
     +    '--------',t70,'-------')
          do ising=1,numsing
            write(20,1560) ss(ising),nt(ising),st(ising),
     +      tt(ising),sqrt(tt(ising))
1560        format(i8,t20,1pg14.7,t37,1pg14.7,t52,
     +      1pg14.7,t70,1pg14.7)
          end do
          varmin=1.0d300
          do ising=1,numsing
            if(tt(ising).lt.varmin)then
              varmin=tt(ising)
              isinglo=ising
            end if
          end do
          write(20,*)
          write(20,1570)
1570      format(t5,'Pre-calibration:- ')
          write(20,1571) tt(1)
1571      format(t5,'Total error variance                 = ',1pg14.7)
          write(20,1572) sqrt(tt(1))
1572      format(t5,'Total error standard deviation       = ',1pg14.7)
          write(20,*)
          write(20,1580)
1580      format(t5,'Post-calibration:-')
          write(20,1571) varmin
          write(20,1572) sqrt(varmin)
          call writint(anum,ss(isinglo))
          write(20,1590) trim(anum)
1590      format(t5,'Minimum error variance occurs at singular value ',
     +    'number ',a,'.')
          precalerr=tt(1)
          postcalerr=varmin

          write(6,910) trim(aprogram)

        end if

C -- PREDUNC1 is now run in order to obtain pre- and post-calibration predictive uncertainty.

        if(aun.eq.'y')then
          aprogram='PREDUNC1'
          aprogram_lo='predunc1'
          write(6,1600)
1600      format(/,' - running PREDUNC1 to compute predictive ',
     +    'uncertainty...')
          call delfile(toutfile)
          open(unit=30,file=tinfile,action='write')
          write(30,1610) trim(scaled_pestfile)
1610      format(1x,a)
          write(30,1620) refvar
1620      format(1x,1pg14.7)
          write(30,1610) trim(scaled_uncertfile)
          write(30,1610) trim(tsenfile)
          write(30,'(a)') trim(avchoice)
          close(unit=30)
#ifdef SYS_FUNCTION
          nnn=system('predunc1 < '//trim(tinfile)//' > '
     +    //trim(toutfile))
#else
          call system('predunc1 < '//trim(tinfile)//' > '
     +    //trim(toutfile))
#endif
          open(unit=11,file=toutfile,status='old',err=9300)
          do
            read(11,'(a)',end=9300) cline
            if(index(cline,'Pre-cal').ne.0) exit
          end do
          n=index(cline,'=')
          atemp=cline(n+1:)
          call drealread(ifail,atemp,precalunc)
          if(ifail.ne.0) go to 9300
          read(11,'(a)',end=9300) cline
          n=index(cline,'=')
          atemp=cline(n+1:)
          call drealread(ifail,atemp,postcalunc)
          if(ifail.ne.0) go to 9300
          close(unit=11)
          precalvar=precalunc*precalunc
          postcalvar=postcalunc*postcalunc

          if(asne.eq.'y')then
            write(20,*)
            write(20,*)
          end if
          write(20,1650) trim(apred)
1650      format(' PREDUNC1 analysis for prediction "',a,'" ----->',/)
          write(20,1570)
          write(20,1651) precalvar
1651      format(t5,'Total uncertainty variance           = ',1pg14.7)
          write(20,1652) precalunc
1652      format(t5,'Total uncertainty standard deviation = ',1pg14.7)
          write(20,*)
          write(20,1580)
          write(20,1651) postcalvar
          write(20,1652) postcalunc
          write(6,910) trim(aprogram)

        end if

#ifdef FLUSHFILE
        call flush(20)
#endif

C -- Parameter contributions to predictive error variance and uncertainty are now computed.

        if((ace.eq.'n').and.(acu.eq.'n')) go to 1866

C -- First a parameter list file is made.

        open(unit=30,file=tplistfile,action='write')
        write(30,1700)
1700    format('* parameter type none')
        mcount=1
        acontrib(mcount)='none'
        if(apg.eq.'i')then
          do ipar=1,npar
            if(itrans(ipar).lt.0) cycle
            write(30,1710) trim(apar(ipar))
1710        format('* parameter type ',a)
            mcount=mcount+1
            acontrib(mcount)=apar(ipar)
            write(30,1712) trim(apar(ipar))
1712        format(1x,a)
          end do
        else
          do ipargp=1,npargp
            icount=0
            do ipar=1,npar
              if(itrans(ipar).ge.0)then
                if(npargnm(ipar).eq.ipargp) then
                  if(icount.eq.0)then
                    mcount=mcount+1
                    acontrib(mcount)=apargp(ipargp)
                    write(30,1710) trim(apargp(ipargp))
                  end if
                  icount=icount+1
                  write(30,1712) trim(apar(ipar))
                end if
              end if
            end do
1715        continue
          end do
        end if
        close(unit=30)

C -- PREDVAR4 is run.

        if(ace.eq.'y')then
          aprogram='PREDVAR4'
          aprogram_lo='predvar4'
          write(6,1720)
1720      format(/,' - running PREDVAR4 to compute param contribs ',
     +    'to pred error variance...')
          open(unit=30,file=tslistfile,action='write')
          do ies=0,nespar
            write(30,1730) ies
1730        format(1x,i5)
          end do
          close(unit=30)
          open(unit=30,file=tslistfile0,action='write')
          write(30,1730) 0
          close(unit=30)
          if(mcount.eq.2) go to 1759
          call delfile(toutfile)
          open(unit=30,file=tinfile,action='write')
          write(30,1740) trim(scaled_pestfile)
1740      format(1x,a)
          write(30,1750) refvar
1750      format(1x,1pg14.7)
          write(30,1740) trim(scaled_uncertfile)
          write(30,1740) trim(tsenfile)
          write(30,1740) trim(tslistfile)
          write(30,1740) trim(tplistfile)
          write(30,1740) trim(toutfile)
          close(unit=30)
#ifdef SYS_FUNCTION
          nnn=system('predvar4 < '//trim(tinfile)//trim(tonull))
#else
          call system('predvar4 < '//trim(tinfile)//trim(tonull))
#endif
          open(unit=11,file=toutfile,status='old',err=9300)
          do i=1,3
            read(11,*,err=9300,end=9300)
          end do
          do i=1,mcount
            read(11,*,err=9300,end=9300) atemp,rtemp,rtemp,contrib(2,i)
          end do
          close(unit=11)

          call delfile(toutfile)
          open(unit=30,file=tinfile,action='write')
          write(30,1740) trim(scaled_pestfile)
          write(30,1750) refvar
          write(30,1740) trim(scaled_uncertfile)
          write(30,1740) trim(tsenfile)
          write(30,1740) trim(tslistfile0)
          write(30,1740) trim(tplistfile)
          write(30,1740) trim(toutfile)
          close(unit=30)
#ifdef SYS_FUNCTION
          nnn=system('predvar4 < '//trim(tinfile)//trim(tonull))
#else
          call system('predvar4 < '//trim(tinfile)//trim(tonull))
#endif
          open(unit=11,file=toutfile,status='old',err=9300)
          do i=1,3
            read(11,*,err=9300,end=9300)
          end do
          do i=1,mcount
            read(11,*,err=9300,end=9300) atemp,rtemp,rtemp,contrib(1,i)
          end do
          close(unit=11)

1759      continue
          if((asne.eq.'y').or.(aun.eq.'y'))then
            write(20,*)
            write(20,*)
          end if
          write(20,1760) trim(apred)
1760      format(' PREDVAR4 analysis for prediction "',a,'" ----->',/)
          write(20,1765)
1765      format(t5,'Contributions to predictive error variance',/)
          write(20,1780)
1780      format(t5,'Parameter',t20,'Pre-cal',t40,'Post-cal')
          if(apg.eq.'g')then
            atemp='group'
          else
            atemp=' '
          end if
          write(20,1790) trim(atemp)
1790      format(t5,a,t20,'contribution',t40,'contribution')
          write(20,1800)
1800      format(t5,'---------',t20,'------------',t40,'------------')
          if(mcount.eq.2)then
            write(20,1810) trim(acontrib(2)),precalerr,postcalerr
          else
            do i=2,mcount
              rtemp1=contrib(1,1)-contrib(1,i)
              rtemp2=contrib(2,1)-contrib(2,i)
              write(20,1810) trim(acontrib(i)),rtemp1,rtemp2
1810          format(t5,a,t20,1pg14.7,t40,1pg14.7)
            end do
          end if

          call delfile(tslistfile0)
          write(6,910) trim(aprogram)
        end if

#ifdef FLUSHFILE
        call flush(20)
#endif

C -- PREDUNC4 is run.

        if(acu.eq.'y')then
          aprogram='PREDUNC4'
          aprogram_lo='predunc4'
          write(6,1820)
1820      format(/,' - running PREDUNC4 to compute param contribs ',
     +    'to pred uncert variance...')
          if(mcount.eq.2) go to 1859

          call delfile(toutfile)
          open(unit=30,file=tinfile,action='write')
          write(30,1740) trim(scaled_pestfile)
          write(30,1750) refvar
          write(30,1740) trim(scaled_uncertfile)
          write(30,1740) trim(tsenfile)
          write(30,1740) trim(tplistfile)
          write(30,1740) trim(toutfile)
          write(30,'(a)') trim(avchoice)
          close(unit=30)
#ifdef SYS_FUNCTION
          nnn=system('predunc4 < '//trim(tinfile)//trim(tonull))
#else
          call system('predunc4 < '//trim(tinfile)//trim(tonull))
#endif
          open(unit=11,file=toutfile,status='old',err=9300)
          do i=1,3
            read(11,*,err=9300,end=9300)
          end do
          do i=1,mcount
            read(11,*,err=9300,end=9300) atemp,contrib(1,i),contrib(2,i)
          end do
          close(unit=11)

1859      continue
          if((asne.eq.'y').or.(aun.eq.'y').or.
     +       (ace.eq.'y'))then
             write(20,*)
             write(20,*)
          end if
          write(20,1860) trim(apred)
1860      format(' PREDUNC4 analysis for prediction "',a,'" ----->',/)
          write(20,1865)
1865      format(t5,'Contributions to predictive uncertainty ',
     +    'variance',/)
          write(20,1780)
          if(apg.eq.'g')then
            atemp='group'
          else
            atemp=' '
          end if
          write(20,1790) trim(atemp)
          write(20,1800)
          if(mcount.eq.2)then
            write(20,1810) trim(acontrib(2)),precalvar,postcalvar
          else
            do i=2,mcount
              rtemp1=contrib(1,1)-contrib(1,i)
              rtemp2=contrib(2,1)-contrib(2,i)
              write(20,1810) trim(acontrib(i)),rtemp1,rtemp2
            end do
          end if

          call delfile(tplistfile)
          write(6,910) trim(aprogram)
        end if

#ifdef FLUSHFILE
        call flush(20)
#endif

C -- Observation worth is now computed. First through observation subtraction.

1866    continue
        if((awe.eq.'n').and.(awu.eq.'n')) go to 2400

        open(unit=30,file=tolistfile_sub,action='write')
        open(unit=31,file=tolistfile_add,action='write')
        write(30,1900)
1900    format('* observation type none')
        write(31,1901)
1901    format('* observation type base')
        mcount=1
        acontrib(mcount)='first'
        if(aog.eq.'i')then
          do iobs=1,nobs
            write(30,1910) trim(aobs(iobs))
            write(31,1910) trim(aobs(iobs))
1910        format('* observation type ',a)
            write(30,1912) trim(aobs(iobs))
            write(31,1912) trim(aobs(iobs))
1912        format(1x,a)
            mcount=mcount+1
          end do
        else
          do iobsgp=1,nobsgp
            icount=0
            do iobs=1,nobs
              if(nobgnm(iobs).eq.iobsgp) then
                if(icount.eq.0)then
                  mcount=mcount+1
                  acontrib(mcount)=aobsgp(iobsgp)
                  write(30,1910) trim(aobsgp(iobsgp))
                  write(31,1910) trim(aobsgp(iobsgp))
                  acontrib(mcount)=aobsgp(iobsgp)
                end if
                icount=icount+1
                write(30,1912) trim(aobs(iobs))
                write(31,1912) trim(aobs(iobs))
              end if
            end do
          end do
        end if
        close(unit=30)
        close(unit=31)

C -- PREDVAR5 is run.

        if(awe.eq.'y')then
          aprogram='PREDVAR5'
          aprogram_lo='predvar5'
          write(6,1920)
1920      format(/,' - running PREDVAR5 to compute observ worth ',
     +    'through subtraction...')
          open(unit=30,file=tslistfile,action='write')
          do ies=0,nespar
            write(30,1730) ies
          end do
          close(unit=30)
          if(mcount.eq.2) go to 1959

          call delfile(toutfile)
          open(unit=30,file=tinfile,action='write')
          write(30,1740) trim(scaled_pestfile)
          write(30,1750) refvar
          write(30,1740) trim(scaled_uncertfile)
          write(30,1740) trim(tsenfile)
          write(30,1740) trim(tslistfile)
          write(30,1740) trim(tolistfile_sub)
          write(30,'(a)') 's'
          write(30,1740) trim(toutfile)
          close(unit=30)
#ifdef SYS_FUNCTION
          nnn=system('predvar5 < '//trim(tinfile)//trim(tonull))
#else
          call system('predvar5 < '//trim(tinfile)//trim(tonull))
#endif
          open(unit=11,file=toutfile,status='old',err=9300)
          do i=1,3
            read(11,*,err=9300,end=9300)
          end do
          do i=1,mcount
            read(11,*,err=9300,end=9300) atemp,rtemp,rtemp,ocontrib(i)
          end do
          close(unit=11)

1959      continue
          if((asne.eq.'y').or.(aun.eq.'y').or.
     +       (ace.eq.'y').or.(acu.eq.'y'))then
             write(20,*)
             write(20,*)
          end if
          write(20,1960) trim(apred)
1960      format(' PREDVAR5 observation subtraction analysis for ',
     +    'prediction "',a,'" ----->',/)
          write(20,1965)
1965      format(t5,'Increases in predictive error variance incurred ',
     +    'through loss of observations',/)
          if(aog.eq.'g')then
            write(20,1980)
1980        format(t5,'Observation_group',t30,'Variance_increase')
          else
            write(20,1981)
1981        format(t5,'Observation',t30,'Variance_increase')
          end if
          if(aog.eq.'g')then
            write(20,1990)
1990        format(t5,'-----------------',t30,'-----------------')
          else
            write(20,1991)
1991        format(t5,'-----------',t30,'-----------------')
          end if
          if(mcount.eq.2)then
            if(aog.eq.'i')then
              write(20,2010) trim(aobs(1)),postcalerr
            else
              write(20,2010) trim(acontrib(2)),postcalerr
            end if
          else
            do i=2,mcount
              rtemp1=ocontrib(i)-ocontrib(1)
              if(aog.eq.'i')then
                write(20,2010) trim(aobs(i-1)),rtemp1
              else
                write(20,2010) trim(acontrib(i)),rtemp1
2010            format(t5,a,t30,1pg14.7)
              end if
            end do
          end if
          write(6,910) trim(aprogram)
        end if

#ifdef FLUSHFILE
        call flush(20)
#endif

C -- PREDUNC5 is run.

        if(awu.eq.'y')then
          aprogram='PREDUNC5'
          aprogram_lo='predunc5'
          write(6,2020)
2020      format(/,' - running PREDUNC5 to compute observ worth ',
     +    'through subtraction...')
          if(mcount.eq.2) go to 2059

          call delfile(toutfile)
          open(unit=30,file=tinfile,action='write')
          write(30,1740) trim(scaled_pestfile)
          write(30,1750) refvar
          write(30,1740) trim(scaled_uncertfile)
          write(30,1740) trim(tsenfile)
          write(30,1740) trim(tolistfile_sub)
          write(30,'(a)') 's'
          write(30,1740) trim(toutfile)
          write(30,'(a)') trim(avchoice)
          close(unit=30)
#ifdef SYS_FUNCTION
          nnn=system('predunc5 < '//trim(tinfile)//trim(tonull))
#else
          call system('predunc5 < '//trim(tinfile)//trim(tonull))
#endif
          open(unit=11,file=toutfile,status='old',err=9300)
          do i=1,3
            read(11,*,err=9300,end=9300)
          end do
          do i=1,mcount
            read(11,*,err=9300,end=9300) atemp,rtemp,ocontrib(i)
          end do
          close(unit=11)

2059      continue
          if((asne.eq.'y').or.(aun.eq.'y').or.
     +       (ace.eq.'y').or.(acu.eq.'y').or.
     +       (awe.eq.'y'))then
             write(20,*)
             write(20,*)
          end if
          write(20,2060) trim(apred)
2060      format(' PREDUNC5 observation subtraction analysis for ',
     +    'prediction "',a,'" ----->',/)
          write(20,2065)
2065      format(t5,'Increases in predictive uncertainty variance ',
     +    'incurred through loss of observations',/)
          if(aog.eq.'g')then
            write(20,1980)
          else
            write(20,1981)
          end if
          if(aog.eq.'g')then
            write(20,1990)
          else
            write(20,1991)
          end if
          if(mcount.eq.2)then
            if(aog.eq.'i')then
              write(20,2010) trim(aobs(1)),postcalvar
            else
              write(20,2010) trim(acontrib(2)),postcalvar
            end if
          else
            do i=2,mcount
              rtemp1=ocontrib(i)-ocontrib(1)
              if(aog.eq.'i')then
                write(20,2010) trim(aobs(i-1)),rtemp1
              else
                write(20,2010) trim(acontrib(i)),rtemp1
              end if
            end do
          end if
          write(6,910) trim(aprogram)
        end if

        call delfile(tolistfile_sub)

#ifdef FLUSHFILE
        call flush(20)
#endif

C -- Observation worth is now computed. First through observation addition.

C -- PREDVAR5 is run.

        if(awe.eq.'y')then
          aprogram='PREDVAR5'
          aprogram_lo='predvar5'
          write(6,2120)
2120      format(/,' - running PREDVAR5 to compute observ worth ',
     +    'through addition...')
          if(mcount.eq.2) go to 2159

          call delfile(toutfile)
          open(unit=30,file=tinfile,action='write')
          write(30,1740) trim(scaled_pestfile)
          write(30,1750) refvar
          write(30,1740) trim(scaled_uncertfile)
          write(30,1740) trim(tsenfile)
          write(30,1740) trim(tslistfile)
          write(30,1740) trim(tolistfile_add)
          write(30,'(a)') 'a'
          write(30,1740) trim(toutfile)
          close(unit=30)
#ifdef SYS_FUNCTION
          nnn=system('predvar5 < '//trim(tinfile)//trim(tonull))
#else
          call system('predvar5 < '//trim(tinfile)//trim(tonull))
#endif
          open(unit=11,file=toutfile,status='old',err=9300)
          do i=1,3
            read(11,*,err=9300,end=9300)
          end do
          do i=1,mcount
            read(11,*,err=9300,end=9300) atemp,rtemp,rtemp,ocontrib(i)
          end do
          close(unit=11)

2159      continue
          write(20,*)
          write(20,*)
          write(20,2160) trim(apred)
2160      format(' PREDVAR5 observation addition analysis for ',
     +    'prediction "',a,'" ----->',/)
          write(20,2165)
2165      format(t5,'Decreases in pre-calibration predictive error ',
     +    'variance incurred through addition of observations',/)
          if(aog.eq.'g')then
            write(20,2180)
2180        format(t5,'Observation_group',t30,'Variance_decrease')
          else
            write(20,2181)
2181        format(t5,'Observation',t30,'Variance_decrease')
          end if
          if(aog.eq.'g')then
            write(20,2190)
2190        format(t5,'-----------------',t30,'-----------------')
          else
            write(20,2191)
2191        format(t5,'-----------',t30,'-----------------')
          end if
          if(mcount.eq.2)then
            if(aog.eq.'i')then
              write(20,2010) trim(aobs(1)),postcalerr
            else
              write(20,2010) trim(acontrib(2)),postcalerr
            end if
          else
            do i=2,mcount
              rtemp1=ocontrib(1)-ocontrib(i)
              if(aog.eq.'i')then
                write(20,2010) trim(aobs(i-1)),rtemp1
              else
                write(20,2010) trim(acontrib(i)),rtemp1
              end if
            end do
          end if
          write(6,910) trim(aprogram)
          call delfile(tslistfile)
        end if

#ifdef FLUSHFILE
        call flush(20)
#endif

C -- PREDUNC5 is run.

        if(awu.eq.'y')then
          aprogram='PREDUNC5'
          aprogram_lo='predunc5'
          write(6,2220)
2220      format(/,' - running PREDUNC5 to compute observ worth ',
     +    'through addition...')
          if(mcount.eq.2) go to 2259

          call delfile(toutfile)
          open(unit=30,file=tinfile,action='write')
          write(30,1740) trim(scaled_pestfile)
          write(30,1750) refvar
          write(30,1740) trim(scaled_uncertfile)
          write(30,1740) trim(tsenfile)
          write(30,1740) trim(tolistfile_add)
          write(30,'(a)') 'a'
          write(30,1740) trim(toutfile)
          write(30,'(a)')trim(avchoice)
          close(unit=30)
#ifdef SYS_FUNCTION
          nnn=system('predunc5 < '//trim(tinfile)//trim(tonull))
#else
          call system('predunc5 < '//trim(tinfile)//trim(tonull))
#endif
          open(unit=11,file=toutfile,status='old',err=9300)
          do i=1,3
            read(11,*,err=9300,end=9300)
          end do
          do i=1,mcount
            read(11,*,err=9300,end=9300) atemp,rtemp,ocontrib(i)
          end do
          close(unit=11)

2259      continue
          write(20,*)
          write(20,*)
          write(20,2260) trim(apred)
2260      format(' PREDUNC5 observation addition analysis for ',
     +    'prediction "',a,'" ----->',/)
          write(20,2265)
2265      format(t5,'Decreases in pre-calibration predictive ',
     +    'uncertainty variance incurred through addition of ',
     +    'observations',/)
          if(aog.eq.'g')then
            write(20,2180)
          else
            write(20,2181)
          end if
          if(aog.eq.'g')then
            write(20,2190)
          else
            write(20,2191)
          end if
          if(mcount.eq.2)then
            if(aog.eq.'i')then
              write(20,2010) trim(aobs(1)),postcalvar
            else
              write(20,2010) trim(acontrib(2)),postcalvar
            end if
          else
            do i=2,mcount
              rtemp1=ocontrib(1)-ocontrib(i)
              if(aog.eq.'i')then
                write(20,2010) trim(aobs(i-1)),rtemp1
              else
                write(20,2010) trim(acontrib(i)),rtemp1
              end if
            end do
          end if
          write(6,910) trim(aprogram)
        end if

        call delfile(tolistfile_sub)

        close(unit=20)
        call addquote(outfile,afile)
        write(6,2300) trim(afile)
2300    format(/,' - see file ',a,' for complete GENLINPRED output.')

2400    continue
        go to 9895


9000   call writint(aline,iline)
       write(amessage,9010) trim(aline),trim(afile)
9010   format('Error encountered when reading line ',a,' of PEST ',
     + 'control file ',a,'.')
       go to 9890
9050   write(amessage,9060) trim(afile)
9060   format('Premature end encountered to PEST control file ',a,'.')
       go to 9890
9150   call writint(aline,iline)
       write(amessage,9160) trim(aline),trim(afile)
9160   format('Insufficient entries on line ',a,' of file ',a,'.')
       go to 9890
9200   write(amessage,9210)
9210   format('Cannot allocate sufficient memory to continue ',
     + 'execution.')
       go to 9890
9250   write(amessage,9260) trim(asection),trim(afile)
9260   format('Cannot find "',a,'" section of PEST control file ',a,'.')
       go to 9890
9300   call program_error(aprogram,aprogram_lo,tinfile,numsup,toutfile,
     + scaled_pestfile,senfile,apred)
       go to 9900
9350   write(amessage,9360) trim(matfile)
9360   format('Error reading data from matrix file ',a,' written ',
     + 'by SCALEPAR.')
       go to 9890
9400   write(amessage,9410) trim(afile)
9410   format('Error reading predictive sensitivities from file ',a,'.')
       go to 9890

9890   continue
       amessage=' '//trim(amessage)
       call writmess(6,amessage)

C -- Tidying up.

9895   continue
       call delfile(def_uncertfile)
       call delfile(scaled_uncertfile)
       call delfile(scaled_covfile)
       call delfile(scaled_pestfile)
       call delfile(scaled_jcofile)
       call delfile(p2pfile)
       call delfile(tp2pfile)
       call delfile(parfile)
       call delfile(tparfile)
       call delfile(matfile)
       call delfile(imatfile)
       call delfile(tinfile)
       call delfile(toutfile)
       call delfile(tdatfile)
       call delfile(tsenfile)
       call delfile(tplistfile)
       call delfile(tslistfile)
       call delfile(tslistfile0)
       call delfile(tolistfile_sub)
       call delfile(tolistfile_add)


9900   deallocate(icovmat,nobgnm,npargnm,itrans,apar,
     + apargp,aobsgp,aobs,lbound,ubound,ident,
     + relerr,relunc,senscale,predvec,stat=ierr)
       deallocate(acontrib,contrib,ocontrib,stat=ierr)

       end



       subroutine delfile(afile)

       implicit none
       character*(*), intent(in) :: afile
       integer                   :: ierr

       open(unit=19,file=afile,status='old',iostat=ierr)
       if(ierr.eq.0)then
         close(unit=19,status='delete',iostat=ierr)
       end if

       end



       subroutine program_error(aprogram,aprogram_lo,tinfile,numsup,
     + toutfile,scaled_pestfile,senfile,apred)

C -- Subroutine PROGRAM_ERROR reports an error message arising from failure
C    of a program to run.

       implicit none

       character*(*), intent(in) :: aprogram,aprogram_lo,tinfile,
     +                              scaled_pestfile,senfile,apred

       integer,       intent(in) :: numsup
       character*(*), intent(in) :: toutfile

       character*10              :: anum
       character*200             :: afile

       write(6,*)
       write(6,5)
5      format(78('*'))
       write(6,*)
       write(6,10) trim(aprogram)
10     format(' An error was detected in running program ',a,'.')
       write(6,*)
       write(6,20)
20     format(' Run this program yourself using the command:-',/)
       if(aprogram_lo.eq.'identpar')then
         call writint(anum,numsup)
         write(6,25) trim(scaled_pestfile),trim(anum),trim(toutfile)
25       format('      identpar ',a,' ',a,' null null ',a)
       else if(aprogram_lo.eq.'jrow2vec')then
         call addquote(senfile,afile)
         write(6,26) trim(afile),trim(apred),trim(toutfile)
26       format('      jrow2vec ',a,' ',a,' ',a)
       else
         write(6,30) trim(aprogram_lo),trim(tinfile)
30       format('     ',a,' < ',a)
       end if
       write(6,*)
       write(6,40)
40     format(' and monitor screen output to see the problem.')
       write(6,*)
       write(6,5)
       return

       end


        subroutine getfile(ifail,cline,filename,ibeg,iend)

C-- Subroutine getfile extracts a filename from a string.

C -- Arguments are as follows:-
C       ifail: returned as zero if filename successfully read
C       cline: a character string containing the file name
C       filename: the name of the file read from the string
C       ibeg: character position at which to begin search for filename
C       iend: on input  - character position at which to end search for filename
C             on output - character postion at which filename ends


        integer, intent(out)               :: ifail
        integer, intent(in)                :: ibeg
        integer, intent(inout)             :: iend
        character (len=*), intent(in)      :: cline
        character (len=*), intent(out)     :: filename

        integer                            :: i,j,k
        character (len=1)                  :: aa

        ifail=0
        do i=ibeg,iend
          aa=cline(i:i)
          if((aa.ne.' ').and.(aa.ne.',').and.(aa.ne.char(9)))go to 50
        end do
        ifail=1
        return

50      if((aa.eq.'"').or.(aa.eq.''''))then
          do j=i+1,iend
            if(cline(j:j).eq.aa) go to 60
          end do
          ifail=1
          return
60        iend=j
          if(i+1.gt.j-1)then
            ifail=1
            return
          else
            filename=cline(i+1:j-1)
          end if
        else
          do j=i+1,iend
            if((cline(j:j).eq.' ').or.(cline(j:j).eq.',').or.
     +         (cline(j:j).eq.char(9)))then
              k=j-1
              go to 100
            end if
          end do
          k=iend
100       filename=cline(i:k)
          if(cline(k:k).eq.'"')then
            ifail=1
            return
          else if(cline(k:k).eq.'''')then
            ifail=1
            return
          end if

          iend=k
        end if
        filename=adjustl(filename)
        return

        end


        subroutine lowcase(astrng)

C -- Subroutine LOWCASE converts a string to lower case.

        integer i,j
        character*(*) astrng

        do 10 i=1,len_trim(astrng)
        j=ichar(astrng(i:i))
        if((j.ge.65).and.(j.le.90)) astrng(i:i)=char(j+32)
10      continue
        return
        end


        subroutine addquote(afile,aqfile)

C -- Subroutine ADDQUOTE adds quotes to a filename if it has a space in it.

        implicit none

        character*(*) afile
        character*(*) aqfile
        integer nbb

        if(index(trim(afile),' ').eq.0)then
          aqfile=afile
        else
          aqfile(1:1)='"'
          aqfile(2:)=trim(afile)
          nbb=len_trim(aqfile)+1
          aqfile(nbb:nbb)='"'
        end if

        return
      end



	subroutine writint(atemp,ival)

c	Subroutine WRITINT writes an integer to a character variable.

	integer*4 ival
	character*6 afmt
	character*(*) atemp

	afmt='(i   )'
	write(afmt(3:5),'(i3)') len(atemp)
	write(atemp,afmt)ival
	atemp=adjustl(atemp)
	return
	end



        subroutine writmess(iunit,amessage)

        implicit none

	integer iunit,jend,i,nblc,junit,leadblank,itake,j
        character*(*) amessage
	character (len=20) ablank

	ablank=' '
	itake=0
	j=0
	junit=iunit

        if(amessage.eq.' ')then
          write(junit,*)
          return
        end if
        write(junit,*)
	do i=1,min(20,len(amessage))
	  if(amessage(i:i).ne.' ')go to 21
20      end do
21	leadblank=i-1
	nblc=len_trim(amessage)
5       jend=j+78-itake
	if(jend.ge.nblc) go to 100
	do i=jend,j+1,-1
	if(amessage(i:i).eq.' ') then
	  if(itake.eq.0) then
	     write(junit,'(a)') amessage(j+1:i)
	     itake=2+leadblank
	  else
	     write(junit,'(a)') ablank(1:leadblank+2)//
     +       amessage(j+1:i)
	  end if
	  j=i
	  go to 5
	end if
	end do
	if(itake.eq.0)then
	  write(junit,'(a)') amessage(j+1:jend)
	  itake=2+leadblank
	else
	  write(junit,'(a)') ablank(1:leadblank+2)//
     +    amessage(j+1:jend)
	end if
	j=jend
	go to 5
100     jend=nblc
	if(itake.eq.0)then
	  write(junit,'(a)') amessage(j+1:jend)
	else
	  write(junit,'(a)') ablank(1:leadblank+2)//
     +    amessage(j+1:jend)
	end if
	return

        end




        subroutine linspl(ifail,num,lw,rw,cline)

c -- Subroutine LINSPL splits a line into whitespace-separated substrings.

        integer ifail,nw,nblc,j,i
        integer num
        integer lw(num),rw(num)
        character*(*) cline

        ifail=0
        nw=0
        nblc=len_trim(cline)
        if(nblc.eq.0) then
          ifail=1
          return
        end if
        j=0
5       if(nw.eq.num) return
        do 10 i=j+1,nblc
        if((cline(i:i).ne.' ').and.(cline(i:i).ne.',')
     +  .and.(ichar(cline(i:i)).ne.9)) go to 20
10      continue
        ifail=1
        return
20      nw=nw+1
        lw(nw)=i
        do 30 i=lw(nw)+1,nblc
        if((cline(i:i).eq.' ').or.(cline(i:i).eq.',')
     +  .or.(ichar(cline(i:i)).eq.9)) go to 40
30      continue
        rw(nw)=nblc
        if(nw.lt.num) ifail=1
        return
40      rw(nw)=i-1
        j=rw(nw)
        go to 5

        end



        SUBROUTINE WHICH1(IFAIL,NPAR,IPAR,APAR,TPAR)

C -- SUBROUTINE WHICH1 LOCATES A STRING IN AN ARRAY

        INTEGER NPAR,IPAR,I
        INTEGER IFAIL
        CHARACTER*(*) TPAR
        CHARACTER*(*) APAR(NPAR)

        IFAIL=0
        IF((IPAR.LT.1).OR.(IPAR.GT.NPAR)) IPAR=1
        CALL LOWCASE(TPAR)
        IF(TPAR.EQ.APAR(IPAR)) RETURN
        IF(IPAR.NE.NPAR)THEN
          DO 20 I=IPAR+1,NPAR
          IF(TPAR.EQ.APAR(I))THEN
            IPAR=I
            RETURN
          END IF
20        CONTINUE
        END IF
        IF(IPAR.NE.1)THEN
          DO 40 I=IPAR-1,1,-1
          IF(TPAR.EQ.APAR(I)) THEN
            IPAR=I
            RETURN
          END IF
40        CONTINUE
        END IF
        IFAIL=1
        RETURN

        END



        SUBROUTINE INTREAD(IFAIL,CLINE,iTEMP)

C -- Subroutine INTREAD reads a real number from a string.

        INTEGER IFAIL
        INTEGER ITEMP
        CHARACTER*6 AFMT
        CHARACTER*(*) CLINE

        IFAIL=0
        AFMT='(i   )'
        WRITE(AFMT(3:5),'(I3)') LEN(CLINE)
        READ(CLINE,AFMT,ERR=100) ITEMP
        RETURN

100     IFAIL=1
        RETURN
        END



        SUBROUTINE DREALREAD(IFAIL,CLINE,RTEMP)

C -- Subroutine DREALREAD reads a real number from a string.

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



C -- Notes

C -- In manual point out that if we select a parameter for which to determine
C    sensitivities then it is really the log of the parameter.
C -- Also covariance matrix must pertain to parameter logs of course.

