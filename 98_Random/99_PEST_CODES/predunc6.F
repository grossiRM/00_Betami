       program predunc6

C -- Program PREDUNC6 calculates conditional predictive uncertainty.

       use pestdata, only  : maxcompdim,pestmode,npar,nobsgp,nxrow,
     +                       errmsg,apar,obgnme,aobs,itrans,numcov,
     +                       covind,owght,x,pval_o,modval_o,
     +                       parchglim,pval,parlbnd,parubnd,
     +                       ipargp,scale,offset,dercom,covgp,nespar,
     +                       nobs,nobgnm,covar,nobscov

       implicit none

       logical lexist
       integer ifail,ibeg,iend,nb_pst,ierr,irestart,itemp1,itemp2,bn,
     + i,j,icount,itemp,iobsgp,jcount,info,jtemp,i_weight_warn
       integer irow,ies,ipar,icurrent,iunit,junit,
     + itype,vardim,covdim,jpartype,jrow,jes,icov
       integer vchoice,nc,maxnc,ig,ir,ic
       integer nespar1,nxrow1,ixrow
       double precision refvar,dtemp,sum,precalvar
       double precision refvarinv,postcalvar
       character*5 aext
       character*10 atype
       character*12 aversion,aapar
       character*20 apred,aaobs
       character*200 afile,bfile
       character*200 pestfile,jcofile,uncertfile,outfile,jcofile1
       character*300  dline

       double precision parvar(1),det(2)

       double precision, allocatable :: y1(:),y(:),xcpxt(:,:),z(:),
     +                                  xcpy(:),u(:),cp(:,:),xcp(:,:)
       double precision, allocatable :: xx(:,:),xtqx(:,:),xtq(:,:)
       double precision, allocatable :: cp_copy(:,:),x1(:,:)
       character*12, allocatable     :: apar1(:),adjapar(:)
       character*20, allocatable     :: aobs1(:)

C -- Initialization

#ifndef UNIX
#ifdef LAHEY
       open(unit=*,action='read',carriagecontrol='list')
#endif
#endif

       include 'version.inc'
       write(6,10) trim(aversion)
10     format(/,' PREDUNC6 Version ',a,'. Watermark Numerical ',
     + 'Computing.',/)


C -- Initialization.

       i_weight_warn=0

C -- Input data is acquired from the user.

20     write(6,30,advance='no')
30     format(' Enter name of PEST control file: ')
       read(5,'(a)') afile
       if(afile.eq.' ') go to 20
       ibeg=1
       iend=len_trim(afile)
       call getfile(ifail,afile,pestfile,ibeg,iend)
#ifndef UNIX
       call lowcas(pestfile)
#endif
       if(ifail.ne.0) go to 20
       nb_pst=len_trim(pestfile)
       nb_pst=nb_pst-3
       if(nb_pst.lt.1)nb_pst=1
       aext=pestfile(nb_pst:)
       call lowcas(aext)
       if(aext.ne.'.pst')then
         write(6,40)
40       format(/,' PEST control file must have an extension of ',
     +   '".pst" - try again.',/)
         go to 20
       end if
       inquire(file=pestfile,exist=lexist)
       if(.not.lexist)then
         call addquote(pestfile,afile)
         write(6,50) trim(afile)
50       format(/,' Cannot find file ',a,' - try again.',/)
         go to 20
       end if

C -- A check is made that the corresponding Jacobian matrix file is present.

       jcofile=pestfile(1:nb_pst)//'jco'
       call open_unformatted_file(ifail,11,'read',jcofile,errmsg)
       if(ifail.ne.0)then
         call addquote(jcofile,afile)
         call addquote(pestfile,bfile)
         write(errmsg,60) trim(afile),trim(bfile)
60       format('Cannot open Jacobian matrix file ',a,
     +   ' corresponding to PEST control file ',a,'.')
         go to 9890
       end if

51     write(6,52,advance='no')
52     format(' Enter observation reference variance: ')
       read(5,*,err=51)refvar
       if(refvar.le.0.0d0) go to 51

C -- More input is sought.

       write(6,*)
80     write(6,90,advance='no')
90     format(' Enter name of parameter uncertainty file: ')
       read(5,'(a)') afile
       if(afile.eq.' ') go to 80
       ibeg=1
       iend=len_trim(afile)
       call getfile(ifail,afile,uncertfile,ibeg,iend)
       if(ifail.ne.0) go to 80
#ifndef UNIX
       call lowcas(uncertfile)
#endif
       open(unit=12,file=uncertfile,status='old',iostat=ierr)
       if(ierr.ne.0)then
         call addquote(uncertfile,afile)
         write(6,95) trim(afile)
95       format(/,' Cannot open file ',a,' - try again.',/)
         go to 80
       end if

100    write(6,110,advance='no')
110    format(' Enter name of predictive sensitivity JCO file: ')
       read(5,'(a)') afile
       if(afile.eq.' ') go to 100
       ibeg=1
       iend=len_trim(afile)
       call getfile(ifail,afile,jcofile1,ibeg,iend)
       if(ifail.ne.0) go to 100
#ifndef UNIX
       call lowcas(jcofile1)
#endif
       call open_unformatted_file(ifail,15,'read',jcofile1,errmsg)
       if(ifail.ne.0)then
         call addquote(jcofile1,afile)
         write(errmsg,111) trim(afile)
111      format(' Cannot open unformatted Jacobian file ',a,
     +   '. Try again.')
         call writmess(6,errmsg)
         write(6,*)
         go to 100
       end if

       write(6,*)
800    write(6,810,advance='no')
810    format(' Enter name for output uncertainty table file: ')
       read(5,'(a)') outfile
       if(outfile.eq.' ') go to 800
       open(unit=20,file=outfile)

       write(6,*)
112    write(6,113)
113    format(' Use which version of linear predictive ',
     + 'uncertainty equation:-')
       write(6,114)
114    format('     if version optimized for small number of ',
     + 'parameters   - enter 1')
       write(6,115)
115    format('     if version optimized for small number of ',
     + 'observations - enter 2')
116    write(6,117,advance='no')
117    format(' Enter your choice: ')
       read(5,*,err=116) vchoice
       if((vchoice.ne.1).and.(vchoice.ne.2)) go to 116

C -- All input data has now been acquired. The complete PEST input
C    dataset is now read.

       write(6,*)
       call addquote(pestfile,afile)
       write(6,235) trim(afile)
235    format(' - reading PEST control file ',a,'....')
       irestart=0
       call read_pest_data(ifail,irestart,pestfile)
       if(ifail.ne.0) go to 9890
       if(maxcompdim.gt.1)then
         write(errmsg,240) trim(afile)
240      format('PREDUNC1 requires that MAXCOMPDIM be set to 1 ',
     +   'or less in PEST control file ',a,'.')
         go to 9890
       end if
       if(pestmode.eq.2)then
         write(errmsg,250) trim(afile)
250      format('PREDUNC1 requires that PEST not be run in predictive ',
     +   'analysis mode in PEST control file ',a,'.')
         go to 9890
       end if
       if(pestmode.eq.3)then
260      write(errmsg,270) trim(afile)
270      format(' File ',a,' tells PEST to run in ',
     +   'regularisation mode. Alter this file, removing ',
     +   'all regularisation observations and ',
     +   'regularisation prior information, set PESTMODE ',
     +   'to "estimation" and re-run PREDUNC1.')
         go to 9890
       end if
       if(nespar.eq.0)then
         write(errmsg,272) trim(afile)
272      format('There are no adjustable parameters cited in PEST ',
     +   'control file ',a,'.')
         go to 9890
       end if
       if(nxrow.eq.0)then
         write(errmsg,273) trim(afile)
273      format('No observations or prior information items are ',
     +   'cited in PEST control file ',a,'.')
         go to 9890
       end if
       write(6,275) trim(afile)
275    format(' - file ',a,' read ok.')

C -- We can delete some unused PEST memory.

       deallocate(pval_o,modval_o,parchglim,pval,parlbnd,parubnd,
     + ipargp,scale,offset,dercom,stat=ierr)

C -- The Jacobian matrix file is now read.

       call addquote(jcofile,afile)
       write(6,276) trim(afile)
276    format(/,' - reading Jacobian matrix file ',a,'....')
       read(11,err=9000,end=9000)itemp1,itemp2
       if(abs(itemp1).ne.nespar) go to 9050
       if(itemp1.lt.0)then
         continue
       else
         write(errmsg,290) trim(afile)
290      format('Jacobian matrix file ',a,' was not produced by ',
     +   'version 8 or later of PEST; use program JCOTRANS to ',
     +   'convert this to new format.')
         go to 9890
       end if
       itemp2=abs(itemp2)
       bn=itemp2
       if(itemp2.ne.nxrow) go to 9050
       do i=1,nespar
         do j=1,nxrow
           x(j,i)=0.0d0
         end do
       end do
       read(11,err=9000,end=9000)icount
       do i=1,icount
         read(11,err=9000,end=9000) j,dtemp
         ies=(j-1)/bn+1
         irow=j-(ies-1)*bn
         x(irow,ies)=dtemp
       end do
       do i=1,npar
         if(itrans(i).lt.0) cycle
         read(11,err=9000,end=9000) aapar
         if(aapar.ne.apar(i)) go to 9050
       end do
       do i=1,bn
         read(11,err=9000,end=9000) aaobs
         call lowcas(aaobs)
         if(aaobs.ne.aobs(i)) go to 9050
       end do
       close(unit=11)
       write(6,275) trim(afile)

C -- The predictive sensitivity JCO file is now read. Note that it is not necessarily
C    assumed that the same parameters are cited in the same order.

       call addquote(jcofile1,afile)
       write(6,505) trim(afile)
505    format(/' - reading predictive sensitivity JCO file ',a,
     + '....')

       read(15,err=9000,end=9000)itemp1,itemp2
       if(itemp1.lt.0)then
         continue
       else
         write(errmsg,290) trim(afile)
         go to 9890
       end if
       nespar1=abs(itemp1)
       nxrow1=abs(itemp2)
       allocate(x1(nxrow1,nespar1),stat=ierr)
       if(ierr.ne.0) go to 9200
       allocate(apar1(nespar1),aobs1(nxrow1),stat=ierr)
       if(ierr.ne.0) go to 9200
       do i=1,nespar1
         do j=1,nxrow1
           x1(j,i)=0.0d0
         end do
       end do
       read(15,err=9000,end=9000)icount
       do i=1,icount
         read(15,err=9000,end=9000) j,dtemp
         ies=(j-1)/nxrow1+1
         irow=j-(ies-1)*nxrow1
         x1(irow,ies)=dtemp
       end do
       do i=1,nespar1
         read(15,err=9000,end=9000) apar1(i)
       end do
       do i=1,nxrow1
         read(15,err=9000,end=9000) aobs1(i)
         call lowcas(aobs1(i))
       end do
       close(unit=15)

C -- We check that every adjustable parameter cited in the original PST file
C    is also cited in the JCO file.

       do i=1,npar
         if(itrans(i).lt.0) cycle
         aapar=apar(i)
         do j=1,nespar1
           if(aapar.eq.apar1(j)) go to 820
         end do
         write(errmsg,815) trim(aapar),trim(pestfile),trim(jcofile1)
815      format('Parameter "',a,'" cited as an adjustable parameter ',
     +   'in PEST control file ',a,' is not featured in JCO ',
     +   'file ',a,'.')
         go to 9890
820      continue
       end do
       write(6,275) trim(afile)
       allocate(y1(nespar1),y(nespar),adjapar(nespar),
     + cp(nespar,nespar),stat=ierr)
       if(ierr.ne.0) go to 9200

C -- A new parameter name array is built comprised only of adjustable parameters.

       ies=0
       do ipar=1,npar
         if(itrans(ipar).ge.0) then
           ies=ies+1
           adjapar(ies)=apar(ipar)
         end if
       end do

C -- The parameter uncertainty file is read.

       call addquote(uncertfile,afile)
       write(6,570) trim(afile)
570    format(/,' - reading parameter uncertainty file ',a,'....')
       iunit=12
       junit=15
       itype=2
       vardim=1
       covdim=nespar
       atype='parameter'
       call read_uncert_data_file(ifail,iunit,junit,itype,jpartype,
     + nespar,vardim,covdim,parvar,cp,atype,uncertfile,errmsg,
     + dline,adjapar)
       if(ifail.ne.0) then
         errmsg=adjustl(errmsg)
         go to 9890
       end if
       write(6,580) trim(afile)
580    format(' - parameter uncertainty file ',a,' read ok.')

       ixrow=0
2000   continue
       ixrow=ixrow+1
       if(ixrow.gt.nxrow1) go to 2500
       if(ixrow.eq.2)then
         write(6,2001)
2001     format(' - calculating predictive uncertainties....')
       end if
       do ies=1,nespar1
         y1(ies)=x1(ixrow,ies)
       end do
       apred=aobs1(ixrow)

C -- The predictive sensitivity vector is now re-arranged (and abridged if necessary).

       ies=0
       icurrent=1
       do ies=1,nespar
         call which1(ifail,nespar1,icurrent,apar1,adjapar(ies))
         y(ies)=y1(icurrent)
       end do

       if(vchoice.eq.1) go to 1000

C -- The XC(p)X' matrix is now formed.

       if(ixrow.gt.1) go to  2100

       write(6,581)
581    format(/,' - forming XC(p)X'' matrix....')
       allocate(xcp(nxrow,nespar),xcpxt(nxrow,nxrow),stat=ierr)
       if(ierr.ne.0) go to 9200
       if(jpartype.eq.1)then
         do irow=1,nxrow
           do ies=1,nespar
             xcp(irow,ies)=x(irow,ies)*cp(ies,ies)
           end do
         end do
       else
         do irow=1,nxrow
           do ies=1,nespar
             sum=0.0d0
             do jes=1,nespar
               sum=sum+x(irow,jes)*cp(jes,ies)
             end do
             xcp(irow,ies)=sum
           end do
         end do
       end if
       do irow=1,nxrow
         do jrow=1,irow
           sum=0.0d0
           do ies=1,nespar
             sum=sum+xcp(irow,ies)*x(jrow,ies)
           end do
           xcpxt(jrow,irow)=sum
         end do
       end do
       do irow=1,nxrow-1
         do jrow=irow+1,nxrow
           xcpxt(jrow,irow)=xcpxt(irow,jrow)
         end do
       end do

C -- The C(e) term is now added.

       write(6,582)
582    format(' - adding C(e) term....')
       if(numcov.eq.0)then
         do irow=1,nxrow
           dtemp=owght(irow)
           if(dtemp.le.0.0d0)then
             dtemp=1.0d35
             i_weight_warn=1
           else
             dtemp=1.0d0/dtemp
           end if
           xcpxt(irow,irow)=xcpxt(irow,irow)+dtemp*refvar
         end do
       else
         do irow=1,nxrow
           itemp=nobgnm(irow)
           if(covind(itemp))then
             continue
           else
             dtemp=owght(irow)
             if(dtemp.le.0.0d0)then
               dtemp=1.0d35
               i_weight_warn=1
             else
               dtemp=1.0d0/dtemp
             end if
             xcpxt(irow,irow)=xcpxt(irow,irow)+dtemp*refvar
           end if
         end do
         do icov=1,numcov
           icount=0
           iobsgp=covgp(icov)
           do irow=1,nxrow
             itemp=nobgnm(irow)
             if(itemp.eq.iobsgp)then
               icount=icount+1
               jcount=0
               do jrow=1,nxrow
                 jtemp=nobgnm(jrow)
                 if(jtemp.eq.iobsgp)then
                   jcount=jcount+1
                   xcpxt(jrow,irow)=xcpxt(jrow,irow)+
     +             covar(icov)%dval(jcount,icount)*refvar
                 end if
               end do
             end if
           end do
         end do
       end if

C -- The matrix is now inverted.

       write(6,583)
583    format(' - inverting [XC(p)X''+C(e)] matrix....')
       call dpofa(xcpxt,nxrow,nxrow,info)
       if(info.ne.0)then
         write(errmsg,600)
600      format(' Cannot invert [XC(p)X''+C(e)] matrix.')
         go to 9890
       end if
       info=1
       call dpodi(xcpxt,nxrow,nxrow,det,info)
       do irow=1,nxrow-1
         do jrow=irow+1,nxrow
           xcpxt(jrow,irow)=xcpxt(irow,jrow)
         end do
       end do


C -- y'C(p) is computed.

2100   continue
C       write(6,584)
C584    format(' - computing predictive uncertainty....')

       if(ixrow.eq.1)then
         allocate(z(nespar),stat=ierr)
         if(ierr.ne.0) go to 9200
       end if

       do ies=1,nespar
         sum=0.0d0
         do jes=1,nespar
           sum=sum+y(jes)*cp(jes,ies)
         end do
         z(ies)=sum
       end do

C -- y'C(p)y (pre-calibration predictive uncertainty) is computed.

       sum=0.0d0
       do ies=1,nespar
         sum=sum+y(ies)*z(ies)
       end do
       precalvar=sum

C -- XC(p)y is formed.

       if(ixrow.eq.1)then
         allocate(xcpy(nxrow),u(nxrow),stat=ierr)
         if(ierr.ne.0) go to 9200
       end if
       do irow=1,nxrow
         sum=0.0d0
         do ies=1,nespar
           sum=sum+x(irow,ies)*z(ies)
         end do
         xcpy(irow)=sum
       end do

C -- The second term is formed.

c       sum=0.0d0
c       do ies=1,nespar
c         zies=z(ies)
c         do jes=1,nespar
c           zjes=z(jes)
c           do irow=1,nxrow
c             do jrow=1,nxrow
c               sum=sum+zies*zjes*xcpxt(jrow,irow)
c             end do
c           end do
c         end do
c       end do

       do irow=1,nxrow
         sum=0.0d0
         do jrow=1,nxrow
           sum=sum+xcpxt(irow,jrow)*xcpy(jrow)
         end do
         u(irow)=sum
       end do
       sum=0.0d0
       do irow=1,nxrow
         sum=sum+xcpy(irow)*u(irow)
       end do

       go to 614

C -- The following code applies to the small parameter number option.

1000   continue

C       write(6,1001)
C1001   format(/,' - computing pre-calibration predictive ',
C     + 'uncertainty....')
       if(ixrow.eq.1)then
         allocate(xtqx(nespar,nespar),z(nespar),stat=ierr)
         if(ierr.ne.0) go to 9200
         allocate(cp_copy(nespar,nespar),stat=ierr)
         if(ierr.ne.0) go to 9200
       end if
       cp_copy=cp           ! arrays
       do ies=1,nespar
         sum=0.0d0
         do jes=1,nespar
           sum=sum+y(jes)*cp_copy(jes,ies)
         end do
         z(ies)=sum
       end do
       sum=0.0d0
       do ies=1,nespar
         sum=sum+y(ies)*z(ies)
       end do
       precalvar=sum

C -- If any observation covariance matrices are present they are inverted.

       if(ixrow.gt.1) go to 2200

       refvarinv=1.0d0/refvar
       if(numcov.ne.0)then
         write(6,1010)
1010     format(' - inverting observation covariance matrices....')
         do i=1,numcov
           if(nobscov(i).gt.0)then
             nc=nobscov(i)
             call dpofa(covar(i)%dval,nc,nc,info)
             if(info.ne.0)then
               write(errmsg,1020) trim(obgnme(covgp(i))),trim(pestfile)
1020           format(' Cannot invert covariance matrix supplied for ',
     +         'observation group "',a,'" cited in PEST control ',
     +         'file ',a,'.')
               go to 9890
             end if
             info=1
             call dpodi(covar(i)%dval,nc,nc,det,info)
             do irow=1,nc-1
               do jrow=irow+1,nc
                 covar(i)%dval(jrow,irow)=covar(i)%dval(irow,jrow)
               end do
             end do
             covar(i)%dval=covar(i)%dval*refvarinv    ! arrays
           end if
           ig=covgp(i)
           do irow=1,nxrow
             if(nobgnm(irow).eq.ig)owght(irow)=0.0d0
           end do
         end do
       end if

       write(6,1030)
1030   format(' - forming XtC-1(e)X matrix....')
       owght=owght*refvarinv       ! an array
       do ies=1,nespar
         do jes=1,ies
           sum=0.0d0
           do irow=1,nxrow
             if(owght(irow).le.0.0d0)i_weight_warn=1
             sum=sum+x(irow,jes)*owght(irow)*x(irow,ies)
           end do
           xtqx(jes,ies)=sum
           xtqx(ies,jes)=sum
         end do
       end do
       if(numcov.ne.0)then
         maxnc=0
         do i=1,numcov
           if(maxnc.le.nobscov(i))maxnc=nobscov(i)
         end do
         if(maxnc.gt.0)then
           allocate(xx(maxnc,nespar),xtq(nespar,maxnc),stat=ierr)
           if(ierr.ne.0) go to 9200
           do i=1,numcov
             ig=covgp(i)
             nc=nobscov(i)
             ir=0
             do irow=1,nxrow
               if(nobgnm(irow).eq.ig)then
                 ir=ir+1
                 do ies=1,nespar
                   xx(ir,ies)=x(irow,ies)
                 end do
                 if(ir.eq.nc) go to 1040
               end if
             end do
1040         continue
             if(ir.eq.0) go to 1041
             do ic=1,nc
               do ies=1,nespar
                 sum=0.0d0
                 do ir=1,nc
                   sum=sum+xx(ir,ies)*covar(i)%dval(ir,ic)
                 end do
                 xtq(ies,ic)=sum
               end do
             end do
             do jes=1,nespar
               do ies=1,nespar
                 sum=0.0d0
                 do ir=1,nc
                   sum=sum+xtq(ies,ir)*xx(ir,jes)
                 end do
                 xtqx(ies,jes)=xtqx(ies,jes)+sum
               end do
             end do
1041         continue
           end do
         end if
       end if

       write(6,1060)
1060   format(' - inverting C(p) matrix....')
       if(jpartype.eq.1)then
         do ies=1,nespar
           dtemp=cp(ies,ies)
           if(abs(dtemp).lt.1.0d-35)dtemp=1.0e-35
           xtqx(ies,ies)=xtqx(ies,ies)+1.0d0/dtemp
         end do
       else
         call dpofa(cp,nespar,nespar,info)
         if(info.ne.0)then
           write(errmsg,1070) trim(uncertfile)
1070       format(' Cannot invert covariance matrix supplied in ',
     +     'parameter uncertainty file ',a,'.')
           go to 9890
         end if
         info=1
         call dpodi(cp,nespar,nespar,det,info)
         do ies=1,nespar-1
           do jes=ies+1,nespar
             cp(jes,ies)=cp(ies,jes)
           end do
         end do
         do ies=1,nespar
           do jes=1,nespar
             xtqx(jes,ies)=xtqx(jes,ies)+cp(jes,ies)
           end do
         end do
       end if

       write(6,1080)
1080   format(' - inverting [XtC-1(e)X + C-1(p)] matrix....')
       call dpofa(xtqx,nespar,nespar,info)
       if(info.ne.0)then
         write(errmsg,1090)
1090     format(' Cannot invert matrix.')
         go to 9890
       end if
       info=1
       call dpodi(xtqx,nespar,nespar,det,info)
       do ies=1,nespar-1
         do jes=ies+1,nespar
           xtqx(jes,ies)=xtqx(ies,jes)
         end do
       end do

2200   continue
C       write(6,1100)
C1100   format(' - calculating post-calibration predictive ',
C     + 'uncertainty....')
       do ies=1,nespar
         sum=0.0d0
         do jes=1,nespar
           sum=sum+y(jes)*xtqx(jes,ies)
         end do
         z(ies)=sum
       end do
       sum=0.0d0
       do ies=1,nespar
         sum=sum+y(ies)*z(ies)
       end do
       postcalvar=sum

C -- Output to the screen is given.

614    continue

       if(vchoice.eq.2)then
         postcalvar=precalvar-sum
       end if
       if(ixrow.eq.1)then
         write(20,2302)
2302     format(' Prediction',t20,'Precal_var',
     +   t40,'Postcal_var',t60,'Precal_uncert',
     +   t80,'Postcal_uncert')
       end if
       if(precalvar.lt.0.0d0)precalvar=0.0d0
       if(postcalvar.lt.0.0d0)postcalvar=0.0d0
       write(20,2301) trim(apred),precalvar,postcalvar,
     + sqrt(precalvar),sqrt(postcalvar)
2301   format(1x,a,t20,1pg14.7,t40,1pg14.7,t60,1pg14.7,t80,1pg14.7)
       go to 2000

2500   continue
       write(6,2010) trim(outfile)
2010   format(' - file ',a,' written ok.')

       if(i_weight_warn.eq.1)then
         write(errmsg,811) trim(pestfile)
811      format('Warning: at least one measurement weight in file ',
     +   a,' is zero. This was assigned an artificial weight of ',
     +   '1.0e-35 by PREDUNC1. ',
     +   'Under some circumstances this may lead to numerical error. ',
     +   'It is suggested that this observation be removed from the ',
     +   'PEST control file. Then use JCO2JCO to generate a new ',
     +   'Jacobian matrix for that file.')
         go to 9890
       end if

       go to 9990

9000   write(errmsg,9010) trim(afile)
9010   format('Error encountered in reading Jacobian matrix ',
     + 'file ',a,'.')
       go to 9890
9050   call addquote(pestfile,bfile)
       write(errmsg,9060) trim(afile),trim(bfile)
9060   format('Jacobian matrix file ',a,' is not compatible ',
     + 'with PEST control file ',a,'; at least one observation ',
     + 'or adjustable parameter is different between them.')
       go to 9890
9200   write(errmsg,9210)
9210   format('Cannot allocate sufficient memory to continue ',
     + 'execution.')
       go to 9890

9890   continue
       errmsg=' '//trim(errmsg)
       call writmess(6,errmsg)

9990   deallocate(y1,y,xcpxt,z,xcpy,u,apar1,adjapar,xcp,stat=ierr)
       deallocate(xtqx,xtq,xx,stat=ierr)
       deallocate(x1,aobs1,cp_copy,stat=ierr)


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



        SUBROUTINE NUMRD(IFAIL,ITYPE,LW,RW,IVAL,RVAL,CLINE)

C -- SUBROUTINE NUMRD READS A NUMBER FROM PART OF A STRING

        INTEGER IFAIL,RW,LW,LN,IVAL
        INTEGER ITYPE
        DOUBLE PRECISION RVAL
        CHARACTER*(*) CLINE
        CHARACTER*10 FMT

        IFAIL=0
        LN=RW-LW+1
        IF(ITYPE.EQ.0) THEN
          WRITE(FMT,10) LN
10        FORMAT('(I',I2,')')
          READ(CLINE(LW:RW),FMT,ERR=50) IVAL
        ELSE IF(ITYPE.EQ.1) THEN
          WRITE(FMT,20) LN
20        FORMAT('(F',I2,'.0)')
          READ(CLINE(LW:RW),FMT,ERR=50) RVAL
        END IF
        RETURN

50      IFAIL=1
        RETURN

        END


        SUBROUTINE WHICH1(IFAIL,NPAR,IPAR,APAR,TPAR)

C -- SUBROUTINE WHICH1 LOCATES A STRING IN AN ARRAY

        INTEGER NPAR,IPAR,I
        INTEGER IFAIL
        CHARACTER*(*) TPAR
        CHARACTER*(*) APAR(NPAR)

        IFAIL=0
        IF((IPAR.LT.1).OR.(IPAR.GT.NPAR)) IPAR=1
        CALL LOWCAS(TPAR)
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


        SUBROUTINE UPCAS(ASTRNG)

C -- SUBROUTINE UPCAS CONVERTS A STRING TO UPPER CASE

        INTEGER I,J
        CHARACTER*(*) ASTRNG

        DO 10 I=1,LEN_TRIM(ASTRNG)
        J=ICHAR(ASTRNG(I:I))
        IF((J.GE.97).AND.(J.LE.122)) ASTRNG(I:I)=CHAR(J-32)
10      CONTINUE
        RETURN

        END


        SUBROUTINE GETINT(IFAIL,BUF,N1,N2,NBLB,MRKDEL)

C -- SUBROUTINE GETINT GETS THE NEXT STORED INSTRUCTION FOR PROCESSING

        INTEGER N1,N2,NBLB,I,II
        INTEGER IFAIL
        CHARACTER MRKDEL
        CHARACTER*(*) BUF

        IFAIL=0
        IF(N2.GE.NBLB) THEN
          N1=0
          RETURN
        END IF
        DO 10 I=N2+1,NBLB
        IF((BUF(I:I).NE.' ').AND.(ICHAR(BUF(I:I)).NE.9)) GO TO 50
10      CONTINUE
        N1=0
        RETURN
50      N1=I
        IF(BUF(N1:N1).NE.MRKDEL)THEN
          I=INDEX(BUF(N1:NBLB),' ')
          II=INDEX(BUF(N1:NBLB),CHAR(9))
          IF((I.EQ.0).AND.(II.EQ.0))THEN
            I=0
          ELSE IF(I.EQ.0)THEN
            I=II
          ELSE IF(II.EQ.0) THEN
            I=I
          ELSE
            I=MIN(I,II)
          END IF
          IF(I.NE.0) THEN
            N2=N1+I-2
          ELSE
            N2=NBLB
          END IF
        ELSE
          IF(N1.EQ.NBLB)THEN
            IFAIL=1
            RETURN
          END IF
          I=INDEX(BUF(N1+1:NBLB),MRKDEL)
          IF(I.EQ.0) THEN
            IFAIL=1
            RETURN
          END IF
          N2=N1+I
        END IF

        RETURN

        END


