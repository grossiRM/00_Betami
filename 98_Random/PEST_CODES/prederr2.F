        include 'matman.f'

        program prederr2

C -- Program PREDERR2 reads resolution and G matrices and parameter and measurement
C    uncertainty data. It also reads a Jacobian matrix. It then calculates
C    error variances of those predictions featured in that matrix which are cited in
C    an external prediction file.

        use matrix_manipulate
        implicit none

        integer       :: ifail,ierr,i,j,ipar,iobs,npar,nobs,iunit,
     +                   junit,itype,vardim,covdim,jpartype,jobstype,
     +                   nb,itemp1,itemp2,nespar,nxrow,icount,ies,irow,
     +                   tempvecdim,iipred
        integer, allocatable :: parorder(:)

        double precision :: dtemp,predterm1,predterm2
        double precision, allocatable :: res(:,:),g(:,:),parcov(:,:),
     +                                   parvar(:),obscov(:,:),
     +                                   obsvar(:),x(:,:),tempvec(:),
     +                                   predrow(:)
        character*4     :: aext
        character*12    :: aversion
        character*15    :: atype
        character*20    :: apred
        character*200   :: rmatfile,gmatfile,obscovfile,afile,
     +  jcofile,bfile,parcovfile,predlistfile,outfile

        character*500   :: cline
        character*600   :: amessage

        character*12, allocatable :: apar(:),japar(:)
        character*20, allocatable :: aobs(:),japred(:)

        type(dmatrix) :: rmat,gmat

#ifndef UNIX
#ifdef LAHEY
        open(unit=*,action='read',carriagecontrol='list')
#endif
#endif

        include 'version.inc'
        write(6,5) trim(aversion)
5       format(/,' PREDERR2 Version ',a,'. Watermark Numerical ',
     +  'Computing.',/)

10      write(6,11,advance='no')
11      format(' Enter name of resolution matrix file: ')
        read(5,'(a)') rmatfile
        if(rmatfile.eq.' ') go to 10
        call remchar(rmatfile,'''')
        call remchar(rmatfile,'"')
        rmatfile=adjustl(rmatfile)
#ifndef UNIX
        call lowcas(rmatfile)
#endif
        call mat_read(ifail,10,rmat,rmatfile,amessage,cline)
        if(ifail.ne.0) go to 9800
        call addquote(rmatfile,afile)
        if(rmat%nrow.ne.rmat%ncol)then
          write(amessage,12) trim(afile)
12        format(' Matrix in file ',a,' cannot be a resolution ',
     +    'matrix because it is not square.')
          go to 9800
        end if
        if(rmat%icode.eq.-1)then
          call mat_diag_expand(ifail,rmat,amessage)
          if(ifail.ne.0) go to 9800
        end if
        if(rmat%icode.eq.2)then
          do irow=1,rmat%nrow
            if(rmat%arow(irow).ne.rmat%acol(irow))then
              write(amessage,13) trim(afile)
13            format(' Matrix in file ',a,' cannot be a resolution ',
     +        'matrix because row names do not equal column names.')
              go to 9800
            end if
          end do
        end if
        write(6,15) trim(afile)
15      format(' - file ',a,' read ok.')


17      write(6,18,advance='no')
18      format(' Enter name of g matrix file: ')
        read(5,'(a)') gmatfile
        if(gmatfile.eq.' ') go to 17
        call remchar(gmatfile,'''')
        call remchar(gmatfile,'"')
        gmatfile=adjustl(gmatfile)
#ifndef UNIX
        call lowcas(gmatfile)
#endif
        call mat_read(ifail,10,gmat,gmatfile,amessage,cline)
        if(ifail.ne.0) go to 9800
        call addquote(gmatfile,afile)
        if(gmat%nrow.ne.rmat%nrow)then
          call addquote(rmatfile,bfile)
          write(amessage,22) trim(afile),trim(bfile)
22        format(' Number of rows in G matrix file ',a,' does not ',
     +    'match number of rows in R matrix file ',a,'.')
          go to 9800
        end if
        do irow=1,gmat%nrow
          if(gmat%arow(irow).ne.rmat%arow(irow))then
            call addquote(rmatfile,bfile)
            write(amessage,21) trim(afile),trim(bfile)
21          format(' Row names in G matrix file ',a,' do not match ',
     +      'those in R matrix file ',a,'.')
            go to 9800
          end if
        end do
        call mat_icode_expand(ifail,gmat,amessage)
        if(ifail.ne.0) go to 9800
        write(6,29) trim(afile)
29      format(' - file ',a,' read ok.')


        write(6,*)
31      write(6,32,advance='no')
32      format(' Enter name of parameter uncertainty file: ')
        read(5,'(a)')parcovfile
        if(parcovfile.eq.' ') go to 31
        call remchar(parcovfile,'''')
        call remchar(parcovfile,'"')
        parcovfile=adjustl(parcovfile)
#ifndef UNIX
        call lowcas(parcovfile)
#endif
        call addquote(parcovfile,afile)
        open(unit=10,file=parcovfile,status='old',iostat=ierr)
        if(ierr.ne.0)then
          write(6,34) trim(afile)
34        format(/,' Cannot open file ',a,' - try again.',/)
          go to 31
        end if

41      write(6,42,advance='no')
42      format(' Enter name of observation uncertainty file: ')
        read(5,'(a)')obscovfile
        if(obscovfile.eq.' ') go to 41
        call remchar(obscovfile,'''')
        call remchar(obscovfile,'"')
        obscovfile=adjustl(obscovfile)
#ifndef UNIX
        call lowcas(obscovfile)
#endif
        call addquote(obscovfile,afile)
        open(unit=11,file=obscovfile,status='old',iostat=ierr)
        if(ierr.ne.0)then
          write(6,34) trim(afile)
          go to 41
        end if

        write(6,*)
310     write(6,320,advance='no')
320     format(' Enter name of Jacobian matrix file: ')
        read(5,'(a)')jcofile
        if(jcofile.eq.' ') go to 310
        call remchar(jcofile,'''')
        call remchar(jcofile,'"')
        jcofile=adjustl(jcofile)
#ifndef UNIX
        call lowcas(jcofile)
#endif
        nb=len_trim(jcofile)
        i=nb-3
        if(i.lt.1)i=1
        aext=jcofile(i:nb)
        call lowcas(aext)
        if(aext.ne.'.jco')then
          call addquote(jcofile,afile)
          write(6,315)
315       format(/,' Expected extension for Jacobian matrix file ',
     +    'is ".jco". Try again.',/)
          go to 310
        end if
        call open_unformatted_file(ifail,16,'read',jcofile,amessage)
        if(ifail.ne.0)then
          call addquote(jcofile,afile)
          write(6,330) trim(afile)
330       format(/,' Cannot open unformatted file ',a,'. Try again.',/)
          go to 310
        end if

350     write(6,360,advance='no')
360     format(' Enter name of prediction list file: ')
        read(5,*) predlistfile
        call remchar(predlistfile,'''')
        call remchar(predlistfile,'"')
        predlistfile=adjustl(predlistfile)
#ifndef UNIX
        call lowcas(predlistfile)
#endif
        call addquote(predlistfile,afile)
        open(unit=18,file=predlistfile,status='old',iostat=ierr)
        if(ierr.ne.0)then
          write(6,34) trim(afile)
          go to 350
        end if

        write(6,*)
520     write(6,530,advance='no')
530     format(' Enter name for prediction error variance output ',
     +  'file: ')
        read(5,'(a)') outfile
        if(outfile.eq.' ') go to 520
#ifdef LF90
        open(unit=30,file=outfile,action='readwrite,denynone',
     +  blocksize=1,iostat=ierr)
#else
        open(unit=30,file=outfile,action='write',iostat=ierr)
#endif
        if(ierr.ne.0)then
          call addquote(outfile,afile)
          write(6,231) trim(afile)
231       format(/,' Cannot open file ',a,' for output - try again.',/)
          go to 520
        end if
        write(30,540)
540     format(' Prediction',t25,'Variance_1',t40,'Variance_2',t55,
     +  'Total',t70,'Standard_Dev')

C -- Matrix data is put into arrays and vectors used by RESPROC
C    so that RESPROC code can be used from now on.

        npar=rmat%nrow
        nobs=gmat%ncol
        tempvecdim=max(npar,nobs)
        allocate(res(npar,npar),g(npar,nobs),apar(npar),aobs(nobs),
     +  tempvec(tempvecdim),predrow(npar),parorder(npar),stat=ierr)
        if(ierr.ne.0) go to 9100
        do i=1,npar
          do j=1,npar
            res(j,i)=rmat%array(j,i)
          end do
        end do
        do i=1,nobs
          do j=1,npar
            g(j,i)=gmat%array(j,i)
          end do
        end do
        do ipar=1,npar
          apar(ipar)=rmat%arow(ipar)
        end do
        do iobs=1,nobs
          aobs(iobs)=gmat%acol(iobs)
        end do

        call mat_deallocate(ifail,rmat)
        if(ifail.ne.0) go to 9800
        call mat_deallocate(ifail,gmat)
        if(ifail.ne.0) go to 9800

C -- More memory is allocated.

        allocate(parcov(npar,npar),parvar(1),stat=ierr)
        if(ierr.ne.0) go to 9100
        allocate(obscov(1,1),obsvar(nobs),stat=ierr)
        if(ierr.ne.0) go to 9100

C -- The Jacobian matrix file is read.

        call addquote(jcofile,afile)
        write(6,362) trim(afile)
362     format(/,' - reading Jacobian matrix file ',a,'...')
        read(16,err=9350,end=9350)itemp1,itemp2
        if(itemp1.ge.0)then
          write(amessage,361) trim(afile)
361       format(' Jacobian matrix file ',a,' uses old format; ',
     +    'use JCOTRANS utility to translate it to new format.')
          go to 9800
        end if
        nespar=abs(itemp1)
        nxrow=abs(itemp2)
        if((nespar.le.0).or.(nxrow.le.0))then
          write(amessage,365) trim(afile)
365       format(' NROW or NCOL not positive in file ',a,
     +    '; is this a valid Jacobian matrix file?')
          go to 9800
        end if
        allocate(x(nxrow,nespar),japar(nespar),japred(nxrow),stat=ierr)
        if(ierr.ne.0) go to 9100
        do i=1,nespar
          do j=1,nxrow
            x(j,i)=0.0d0
          end do
        end do
        read(16,err=9350,end=9350)icount
        do i=1,icount
          read(16,err=9350,end=9350) j,dtemp
          ies=(j-1)/nxrow+1
          irow=j-(ies-1)*nxrow
          x(irow,ies)=dtemp
        end do
        do ies=1,nespar
          read(16,err=9350,end=9350) japar(ies)
          call lowcas(japar(ies))
        end do
        do irow=1,nxrow
          read(16,err=9350,end=9350) japred(irow)
          call lowcas(japred(irow))
        end do
380     continue
        ies=0
        do ipar=1,npar
          call whichone(ifail,npar,ies,japar,apar(ipar))
          if(ifail.ne.0)then
            call addquote(rmatfile,bfile)
            write(amessage,385) trim(apar(ipar)),trim(bfile),trim(afile)
385         format(' Parameter "',a,'" featured in resolution ',
     +      'matrix file ',a,' does not feature in Jacobian ',
     +      'matrix file ',a,'.')
            go to 9800
          end if
          parorder(ipar)=ies
        end do
        deallocate(japar,stat=ierr)
        close(unit=16)
        write(6,370) trim(afile)
370     format(' - Jacobian matrix file ',a,' read ok.')

C -- The parameter uncertainty file is now read.

        write(6,50)
50      format(/,' - reading parameter uncertainty data...')
        iunit=10
        junit=13
        itype=2
        vardim=1
        covdim=npar
        atype='parameter'
        call read_uncert_data_file(ifail,iunit,junit,itype,jpartype,
     +  npar,vardim,covdim,parvar,parcov,atype,parcovfile,amessage,
     +  cline,apar)
        if(ifail.ne.0) go to 9800
        write(6,60)
60      format(' - parameter uncertainty data read ok.')

C -- The observation uncertainty file is now read.

        write(6,70)
70      format(/,' - reading observation uncertainty data...')
        iunit=11
        junit=13
        itype=1
        vardim=nobs
        covdim=1
        atype='observation'
        call read_uncert_data_file(ifail,iunit,junit,itype,jobstype,
     +  nobs,vardim,covdim,obsvar,obscov,atype,obscovfile,amessage,
     +  cline,aobs)
        if(ifail.ne.0) go to 9800
        write(6,75)
75      format(' - observation uncertainty data read ok.')
        write(6,*)

        do i=1,npar
          res(i,i)=res(i,i)-1.0d0
        end do

C -- The prediction list file is now read and calculations are performed.

        iipred=1

515     continue
        read(18,*,end=9900) apred
        call lowcas(apred)
        apred=adjustl(apred)
        call whichone(ifail,nxrow,iipred,japred,apred)
        if(ifail.ne.0)then
          call addquote(predlistfile,afile)
          call addquote(jcofile,bfile)
          write(amessage,521) trim(apred),trim(bfile),trim(afile)
521       format(' Prediction "',a,'" cited in prediction list ',
     +    'file ',a,' does not feature in Jacobian ',
     +    'matrix file ',a,'.')
          call writmess(6,amessage)
          write(30,522) trim(apred),trim(bfile)
          write(6,*)
522       format(1x,a,t25,' prediction not cited in JCO file ',a,'.')
          go to 515
        end if
        do ipar=1,npar
          predrow(ipar)=x(iipred,parorder(ipar))
        end do
        write(6,100) trim(apred)
100     format(' Calculating variance terms for prediction ',a,'...')

C -- I-R term of predictive variance is now calculated.

        do i=1,npar
          dtemp=0.0d0
          do j=1,npar
            dtemp=dtemp+res(j,i)*predrow(j)
          end do
          tempvec(i)=dtemp
        end do

        predterm1=0.0d0
        if(jpartype.eq.2)then
          do i=1,npar
            do j=1,npar
              predterm1=predterm1+tempvec(i)*tempvec(j)*parcov(i,j)
            end do
          end do
        else
          do i=1,npar
            predterm1=predterm1+tempvec(i)*tempvec(i)*parcov(i,i)
          end do
        end if

C -- The C(e) term of predictive error variance is now calculated.

        do i=1,nobs
          dtemp=0.0d0
          do j=1,npar
            dtemp=dtemp+g(j,i)*predrow(j)
          end do
          tempvec(i)=dtemp
        end do

        predterm2=0.0d0
        do i=1,nobs
          predterm2=predterm2+tempvec(i)*tempvec(i)*obsvar(i)
        end do

        write(30,570) trim(apred),predterm1,predterm2,
     +  predterm1+predterm2,sqrt(predterm1+predterm2)
570     format(1x,a,t25,1pg14.7,t40,1pg14.7,t55,1pg14.7,t70,1pg14.7)
#ifdef FLUSHFILE
        call flush(30)
#endif

        go to 515

9100    write(amessage,9110)
9110    format(' Cannot allocate sufficient memory to continue ',
     +  'execution.')
        go to 9800
9350    write(amessage,9360) trim(afile)
9360    format(' Error reading Jacobian matrix from file ',a,'.')
        go to 9800

9800    continue
        call writmess(6,amessage)

9900    deallocate(res,g,parcov,parvar,obscov,obsvar,tempvec,
     +  predrow,japred,x,parorder,stat=ierr)

9999    continue

        end



