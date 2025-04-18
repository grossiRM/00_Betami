       program randpar

C -- Program RANDPAR generates random realisations of PEST parameters.

       use message_utility
       use pdf
       implicit none

       integer, parameter :: MAXTRIAL=1000
       integer, parameter :: MAXORDER=8000
       integer            :: ipar1(MAXORDER),ipar2(MAXORDER),
     +                       igl(MAXORDER)

       integer          :: nb,ierr,ntied,itied,j,k,ibeg,iend,iunit,
     +                     junit,itype,vardim,covdim,jpartype,numfile,
     +                     ndist,ifile,ifail,ies,idist,jes,iseed
       integer          :: pestmode,npar,nobs,npargp,nprior,nobsgp,
     +                     nespar,ipar
       integer          :: iorder,norder,iline,itrial
       real             :: rtemp
       real             :: ranvec(2),dummy(1)
       double precision :: parvar(1)
       character*1      :: ad,am,ap,at
       character*4      :: aext
       character*10     :: aseed
       character*12     :: apar1,apar2
       character*12     :: aapar1,aapar2,aline,atrial
       character*12     :: aversion
       character*15     :: atemp1,atemp2,atrans,achange,agroup,anum,
     +                     atype
       character*200    :: pestinfile,apestinfile,afile,uncertfile,
     +                     auncertfile,basename,outfile
       character*200    :: orderfile,aorderfile
       character*300    :: cline

       integer, allocatable          :: itrans(:)
       real, allocatable             :: mean(:),values(:),ppval(:)
       real, allocatable             :: covar(:,:)
       double precision, allocatable :: pval(:),lbound(:),ubound(:),
     +                                  scale(:),offset(:)
       double precision, allocatable :: cp(:,:)
       character*12, allocatable     :: apar(:),aapar(:)


C -- Initialization

#ifndef UNIX
#ifdef LAHEY
       open(unit=*,action='read',carriagecontrol='list')
#endif
#endif

       include 'version.inc'
       write(6,50) trim(aversion)
50     format(/,' RANDPAR Version ',a,'. Watermark Numerical ',
     + 'Computing.',/)

100    write(6,110,advance='no')
110    format(' Enter name of existing PEST control file: ')
       read(5,*) pestinfile
#ifndef UNIX
       call lowcas(pestinfile)
#endif
       nb=len_trim(pestinfile)
       nb=nb-3
       if(nb.lt.1)nb=1
       aext=pestinfile(nb:)
       call lowcas(aext)
       if(aext.ne.'.pst')then
         write(6,120)
120      format(' *** file extension of ".pst" expected - ',
     +   'try again ***')
         go to 100
       end if
       call addquote(pestinfile,apestinfile)
       open(unit=10,file=pestinfile,status='old',err=100)

C -- The existing PEST control file is partly read.

       read(10,*,err=9000,end=9000) atemp1
       call lowcas(atemp1)
       if(atemp1(1:4).ne.'pcf ') go to 9000
       read(10,*,err=9000,end=9000)
       read(10,*,err=9000,end=9000) atemp1,atemp2
       call lowcas(atemp2)
       if(atemp2.eq.'estimation')then
         pestmode=1
       else if(atemp2.eq.'prediction')then
         pestmode=2
       else if (atemp2(1:5).eq.'regul')then
         pestmode=3
       else
         write(amessage,165) trim(apestinfile)
165      format(' Unknown value for PESTMODE variable on line 3 of ',
     +   'PEST control file ',a,'.')
         go to 9890
       end if
       read(10,*,err=9000,end=9000) npar,nobs,npargp,nprior,nobsgp
       if(npar.eq.0)then
         write(amessage,168) trim(apestinfile)
168      format(' Number of parameters NPAR supplied as zero in ',
     +   'PEST control file ',a,'.')
         go to 9890
       end if

C -- Parameter data is now read.

       allocate(apar(npar),itrans(npar),pval(npar),lbound(npar),
     + ubound(npar),scale(npar),offset(npar),ppval(npar),stat=ierr)
       if(ierr.ne.0) go to 9200
       do
         read(10,'(a)',end=9000) cline
         cline=adjustl(cline)
         if(cline(1:1).eq.'*')then
           call lowcas(cline)
           if(index(cline,'parameter da').ne.0) go to 170
         end if
       end do
170    continue
       nespar=0
       do ipar=1,npar
         read(10,*,err=9000,end=9000)apar(ipar),atrans,achange,
     +   pval(ipar),lbound(ipar),ubound(ipar),agroup,scale(ipar),
     +   offset(ipar)
         call lowcas(apar(ipar))
         call lowcas(atrans)
         if(atrans.eq.'log')then
           itrans(ipar)=1
         else if(atrans.eq.'none')then
           itrans(ipar)=0
         else if(atrans.eq.'fixed')then
           itrans(ipar)=-100000
         else if(atrans.eq.'tied')then
           itrans(ipar)=-1
         else
           go to 9000
         end if
         if(itrans(ipar).eq.1)then
           if(pval(ipar).le.0.0d0)then
             write(amessage,195) trim(apar(ipar)),trim(apestinfile)
195          format(' Parameter "',a,'" is log transformed, yet it ',
     +       'has a negative initial value in PEST control file ',
     +       a,'.')
             go to 9890
           end if
         end if
         if(itrans(ipar).gt.0)then
           if((pval(ipar).lt.lbound(ipar)).or.
     +        (pval(ipar).gt.ubound(ipar))) go to 9000
           if(lbound(ipar).ge.ubound(ipar)) go to 9000
         end if
         if(itrans(ipar).gt.0)then
           if(lbound(ipar).le.0.0) go to 9000
         end if
         if(itrans(ipar).ge.0) nespar=nespar+1
       end do
       if(nespar.eq.0) then
         write(amessage,230) trim(apestinfile)
230      format(' There are no adjustable parameters in PEST ',
     +   'control file ',a,'.')
         go to 9890
       end if

C -- We see if there are any tied parameters.

       ntied=0
       do ipar=1,npar
         if(itrans(ipar).eq.-1) ntied=ntied+1
       end do
       if(ntied.ne.0)then
         do itied=1,ntied
           read(10,*) apar1,apar2
           call lowcas(apar1)
           call lowcas(apar2)
           do j=1,npar
             if(apar1.eq.apar(j)) go to 325
           end do
           go to 9000
325        if(itrans(j).ne.-1) go to 9000
           do k=1,npar
             if(apar2.eq.apar(k)) go to 326
           end do
           go to 9000
326        if(itrans(k).lt.0) go to 9000
           itrans(j)=-k
         end do
       end if
       call writint(anum,npar)
       write(6,210) trim(anum),trim(apestinfile)
210    format(' - ',a,' parameters read from file ',a,'.')
       call writint(anum,nespar)
       write(6,220) trim(anum)
220    format(' - ',a,' of these are adjustable.')

       close(unit=10)

C -- Some information about generating parameters is now sought.

       write(6,*)
260    write(6,270,advance='no')
270    format(' Use (log)normal or (log)uniform ',
     + 'distrib for param generation? [n/u]: ')
       read(5,'(a)') ad
       if((ad.eq.'N').or.(ad.eq.'n'))then
         ad='n'
       else if((ad.eq.'U').or.(ad.eq.'u')) then
         ad='u'
       else
         go to 260
       end if
       if(ad.eq.'n')then
280      write(6,290,advance='no')
290      format(' Compute means as existing param values or range ',
     +   'midpoints? [e/m]: ')
         read(5,'(a)') am
         if((am.eq.'E').or.(am.eq.'e'))then
           am='e'
         else if((am.eq.'M').or.(am.eq.'m'))then
           am='m'
         else
           go to 280
         end if
300      continue
         if(ntied.ne.0)then
           write(6,310,advance='no')
310        format(' Respect parameter ranges (parent parameters)? ',
     +     '[y/n]: ')
         else
           write(6,311,advance='no')
311        format(' Respect parameter ranges? [y/n]: ')
         end if
         read(5,'(a)') ap
         if((ap.eq.'Y').or.(ap.eq.'y'))then
           ap='y'
         else if((ap.eq.'N').or.(ap.eq.'n'))then
           ap='n'
         else
           go to 300
         end if
       end if
       if(ntied.ne.0)then
320      write(6,330,advance='no')
330      format(' Respect parameter ranges (tied parameters)? ',
     +   '[y/n]: ')
         read(5,'(a)') at
         if((at.eq.'Y').or.(at.eq.'y'))then
           at='y'
         else if((at.eq.'N').or.(at.eq.'n'))then
           at='n'
         else
           go to 320
         end if
       end if

C -- We now prompt for the name of the parameter uncertainty file.

       if(ad.eq.'n')then
         write(6,*)
340      write(6,350,advance='no')
350      format(' Enter name of parameter uncertainty file: ')
         read(5,'(a)') afile
         if(afile.eq.' ') go to 340
         ibeg=1
         iend=len_trim(afile)
         call getfile(ifail,afile,uncertfile,ibeg,iend)
         if(ifail.ne.0) go to 340
#ifndef UNIX
         call lowcas(uncertfile)
#endif
         call addquote(uncertfile,auncertfile)
         open(unit=12,file=uncertfile,status='old',iostat=ierr)
         if(ierr.ne.0) go to 340
         allocate(cp(nespar,nespar),stat=ierr)
         if(ierr.ne.0) go to 9200
       end if

C -- A copy of the parameter name array is made in which only adjustable parameters
C    are cited.

       allocate(aapar(nespar),values(nespar),stat=ierr)
       if(ierr.ne.0) go to 9200
       ies=0
       do ipar=1,npar
         if(itrans(ipar).ge.0)then
           ies=ies+1
           aapar(ies)=apar(ipar)
         end if
       end do

C -- The uncertainty file is now read.

       if(ad.eq.'n')then
         iunit=12
         junit=15
         itype=2
         vardim=1
         covdim=nespar
         atype='parameter'
         call read_uncert_data_file(ifail,iunit,junit,itype,jpartype,
     +   nespar,vardim,covdim,parvar,cp,atype,uncertfile,amessage,
     +   cline,aapar)
         if(ifail.ne.0) go to 9890
         if(jpartype.ne.1)then
           if(ad.ne.'n')then
             write(amessage,360) trim(auncertfile)
360          format(' File ',a,' indicates that a covariance ',
     +       'matrix file has been supplied for at least some ',
     +       'parameters. This is not ',
     +       'allowed if parameters belong to a uniform ',
     +       'distribution. They must all be normally distributed.')
             go to 9890
           end if
         end if
         write(6,380) trim(auncertfile)
380      format(' - parameter uncertainty file ',a,' read ok.')

C -- The covariance matrix is transferred to a single precision array.

         allocate(covar(nespar,nespar),stat=ierr)
         if(ierr.ne.0) go to 9890
         do ies=1,nespar
           do jes=1,nespar
             covar(jes,ies)=cp(jes,ies)
           end do
         end do
         deallocate(cp,stat=ierr)
       else
         jpartype=1
       end if

C -- A parameter ordering file is read if present.

       write(6,*)
1000   write(6,1010,advance='no')
1010   format(' Enter name of parameter ordering file ',
     + '(<Enter> if none): ')
       read(5,'(a)') orderfile
       if(orderfile.eq.' ')then
         norder=0
       else
         open(unit=12,file=orderfile,status='old',err=1000)
         call addquote(orderfile,aorderfile)
         iline=0
         do
           ipar=1
           iline=iline+1
           call writint(aline,iline)
           read(12,'(a)',end=1200) cline
           if(cline.eq.' ') cycle
           cline=adjustl(cline)
           if(cline(1:1).eq.'#') cycle
           call lowcas(cline)
           call tabrem(cline)
           nb=index(cline,'<')
           if(nb.eq.0)then
             nb=index(cline,'>')
             if(nb.eq.0)then
               write(amessage,1020) trim(aline),trim(aorderfile)
1020           format(' Line ',a,' of parameter order file ',a,
     +         ' contains neither a "<" nor ">" character.')
               go to 9890
             end if
           end if
           if(nb.eq.1)then
             write(amessage,1030) trim(aline),trim(aorderfile)
1030         format(' Improper format at line ',a,' of parameter ',
     +       'order file ',a,'.')
             go to 9890
           end if
           aapar1=cline(1:nb-1)
           call whichone(ifail,npar,ipar,apar,aapar1)
           if(ifail.ne.0)then
             write(amessage,1040) trim(aapar1),trim(aline),
     +       trim(aorderfile)
1040         format(' Parameter "',a,'" cited at line ',a,' of ',
     +       'parameter order file ',a,' does not appear in PEST ',
     +       'control file.')
             go to 9890
           end if
           iorder=iorder+1
           if(iorder.gt.MAXORDER)then
             write(amessage,1050) trim(aorderfile)
1050         format(' Too many parameter relationships supplied ',
     +       'in file ',a,'. Increase MAXORDER and re-compile ',
     +       'program.')
             go to 9890
           end if
           ipar1(iorder)=ipar
           aapar2=trim(adjustl(cline(nb+1:)))
           aapar2=adjustl(aapar2)
           ipar=1
           call whichone(ifail,npar,ipar,apar,aapar2)
           if(ifail.ne.0)then
             write(amessage,1040) trim(aapar2),trim(aline),
     +       trim(aorderfile)
             go to 9890
           end if
           ipar2(iorder)=ipar
           if(aapar1.eq.aapar2)then
             write(amessage,1060) trim(aline),trim(aorderfile)
1060         format(' Parameter names must not be the same at line ',
     +       a,' of file ',a,'.')
             go to 9890
           end if
           if(cline(nb:nb).eq.'>')then
             igl(iorder)=1
           else
             igl(iorder)=-1
           end if
         end do
1200     continue
         close(unit=12)
         write(6,1210) trim(aorderfile)
1210     format(' - file ',a,' read ok.')
         norder=iorder
       end if

       write(6,*)
       write(6,400,advance='no')
400    format(' Enter filename base for parameter value files: ')
       read(5,'(a)') basename
       nb=len_trim(basename)
       if(basename(nb:nb).eq.'.') basename(nb:nb)=' '
#ifndef UNIX
       call lowcas(basename)
#endif
420    write(6,430,advance='no')
430    format(' How many of these files do you wish to generate? ')
       read(5,*,err=420) numfile
       if(numfile.le.0) go to 420

C -- The random number generator is now initialised.

       write(6,*)
431    continue
       write(6,432,advance='no')
432    format(' Enter integer random number seed ',
     + '(<Enter> if default): ')
       read(5,'(a)') aseed
       if(aseed.ne.' ')then
         call intread(ifail,aseed,iseed)
         if(ifail.ne.0) go to 431
         if(iseed.le.0)then
           write(6,433)
433        format(' *** Must be positive - try again ***')
           go to 431
         end if
       end if
       if(jpartype.eq.1)then
         ndist=nespar
       else
         ndist=1
       end if
       if(aseed.eq.' ')then
         call initialise_density_set(ifail,ndist)
       else
         call initialise_density_set(ifail,ndist,iseed)
       end if
       if(ifail.ne.0) go to 9890

       if(jpartype.eq.1)then
         if(ad.eq.'u')then
           atype='uniform'
         else
           atype='normal'
         end if
         ies=0
         do ipar=1,npar
           if(itrans(ipar).lt.0) cycle
           ies=ies+1
           if(itrans(ipar).eq.0)then
             if(ad.eq.'u')then
               ranvec(1)=lbound(ipar)
               ranvec(2)=ubound(ipar)
             else
               if(am.eq.'e')then
                 ranvec(1)=pval(ipar)
               else
                 ranvec(1)=0.5*(lbound(ipar)+ubound(ipar))
               end if
             end if
           else
             if(ad.eq.'u')then
               ranvec(1)=log10(lbound(ipar))
               ranvec(2)=log10(ubound(ipar))
             else
               if(am.eq.'e')then
                 ranvec(1)=log10(pval(ipar))
               else
                 ranvec(1)=0.5*(log10(lbound(ipar))+
     +                          log10(ubound(ipar)))
               end if
             end if
           end if
           if(ad.eq.'n')then
             ranvec(2)=sqrt(covar(ies,ies))
           end if
           idist=0
           call set_pdf_parameters(ifail,atype,ranvec,idist)
           if(ifail.ne.0)then
             write(amessage,440)
440          format(' Failure condition encountered in initializing ',
     +       'random number generator.')
             go to 9890
           end if
           if(idist.ne.ies)then
             write(amessage,441)
441          format(' Error in call to random number generator - ',
     +       'contact programmer.')
             go to 9890
           end if
         end do
       else
         atype='multinormal'
         allocate(mean(nespar),stat=ierr)
         if(ierr.ne.0) go to 9200
         ndist=1
         ies=0
         do ipar=1,npar
           if(itrans(ipar).lt.0) cycle
           ies=ies+1
           if(itrans(ipar).eq.0)then
             if(am.eq.'e')then
               mean(ies)=pval(ipar)
             else
               mean(ies)=0.5*(lbound(ipar)+ubound(ipar))
             end if
           else
             if(am.eq.'e')then
               mean(ies)=log10(pval(ipar))
             else
               mean(ies)=0.5*(log10(lbound(ipar))+
     +                        log10(ubound(ipar)))
             end if
           end if
         end do
         idist=0
         call set_pdf_parameters(ifail,atype,ranvec,idist,
     +   nespar,nespar,mean,covar)
         if(ifail.ne.0)then
           write(amessage,440)
           go to 9890
         end if
         if(idist.ne.1)then
           write(amessage,441)
           go to 9890
         end if
       end if

C -- Parameter value files are now written.

       do ifile=1,numfile
         itrial=0
1300     continue
         itrial=itrial+1
         if(itrial.gt.MAXTRIAL)then
           call writint(atrial,MAXTRIAL)
           call writint(anum,ifile)
           write(amessage,1310) trim(atrial),trim(anum)
1310       format(' Over ',a,' trials have been spent in trying to ',
     +     'respect user-supplied ordering relationships for ',
     +     'parameter set realisation number ',a,'. Edit parameter ',
     +     'ordering file, ',
     +     'make these relationships less restrictive, and ',
     +     're-run RANDPAR.')
           go to 9890
         end if
         if(jpartype.eq.1)then
           do idist=1,nespar
             call get_random_number(ifail,idist,1,values(idist),dummy)
           end do
         else
           idist=1
           call get_random_number(ifail,idist,1,rtemp,values)
         end if
         ies=0
         do ipar=1,npar
           if(itrans(ipar).eq.1)then
             ies=ies+1
             ppval(ipar)=10**(values(ies))
           else if(itrans(ipar).eq.0)then
             ies=ies+1
             ppval(ipar)=values(ies)
           else if(itrans(ipar).eq.-100000)then
             ppval(ipar)=pval(ipar)
           end if
           if(ap.eq.'y')then
             if(ppval(ipar).gt.ubound(ipar))ppval(ipar)=ubound(ipar)
             if(ppval(ipar).lt.lbound(ipar))ppval(ipar)=lbound(ipar)
           end if
         end do
         if(ntied.ne.0)then
           do ipar=1,npar
             if(itrans(ipar).eq.-100000) cycle
             if(itrans(ipar).lt.0)then
               k=-itrans(ipar)
               ppval(ipar)=ppval(k)*pval(ipar)/pval(k)
               if(at.eq.'y')then
                 if(ppval(ipar).gt.ubound(ipar))
     +              ppval(ipar)=ubound(ipar)
                 if(ppval(ipar).lt.lbound(ipar))
     +              ppval(ipar)=lbound(ipar)
               end if
             end if
           end do
         end if

C -- If necessary parameter ordering relationships are tested.

         if(norder.gt.0)then
           do iorder=1,norder
             if(igl(iorder).eq.1)then
               if(ppval(ipar1(iorder)).le.ppval(ipar2(iorder)))
     +         go to 1300
             else
               if(ppval(ipar1(iorder)).ge.ppval(ipar2(iorder)))
     +         go to 1300
             end if
           end do
         end if

         call writint(anum,ifile)
         anum=adjustl(anum)
         outfile=trim(basename)//trim(anum)//'.par'
         open(unit=20,file=outfile)
         write(20,460)
460      format(' single point')
         do ipar=1,npar
           write(20,470) trim(apar(ipar)),ppval(ipar),
     +     scale(ipar),offset(ipar)
470        format(1x,a,t20,2x,1pg14.7,2x,1pg14.7,2x,1pg14.7)
         end do
         close(unit=20)
         write(6,480) trim(outfile)
480      format(' - file ',a,' written ok.')
       end do

       go to 9900

9000   write(amessage,9010) trim(apestinfile)
9010   format(' Error encountered when reading PEST control file ',
     + a,' - check this file with PESTCHEK.')
       go to 9890
9200   write(amessage,9210)
9210   format(' Cannot allocate sufficient memory to continue ',
     + 'execution.')
       go to 9890

9890   continue
       call writmess(6,amessage)

9900   continue
       if(allocated(itrans)) deallocate(itrans,stat=ierr)
       if(allocated(mean)) deallocate(mean,stat=ierr)
       if(allocated(values)) deallocate(values,stat=ierr)
       if(allocated(ppval)) deallocate(ppval,stat=ierr)
       if(allocated(pval)) deallocate(pval,stat=ierr)
       if(allocated(lbound)) deallocate(lbound,stat=ierr)
       if(allocated(ubound)) deallocate(ubound,stat=ierr)
       if(allocated(scale)) deallocate(scale,stat=ierr)
       if(allocated(offset)) deallocate(offset,stat=ierr)
       if(allocated(cp)) deallocate(cp,stat=ierr)
       if(allocated(apar)) deallocate(apar,stat=ierr)
       if(allocated(aapar)) deallocate(aapar,stat=ierr)
       if(allocated(covar)) deallocate(covar,stat=ierr)

       do idist=1,ndist
         call deallocate_pdf(idist)
       end do
       call deallocate_pdf(-1)

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


        SUBROUTINE TABREM(CLINE)

C -- Subroutine TABREM removes tabs from a string.

        INTEGER I
        CHARACTER*(*) CLINE

        DO 10 I=1,LEN(CLINE)
10      IF(ICHAR(CLINE(I:I)).EQ.9) CLINE(I:I)=' '

        RETURN
        END

