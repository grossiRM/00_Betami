       program addreg1

C -- Program ADDREG1 adds "preferred value regularisation" to the a PEST control file.

       implicit none

       integer   :: ifail,nb,ierr,pestmode,i,ipar,j,iwarn,k,newgroup
       integer   :: npar,nobs,npargp,nprior,nobsgp,ntplfle,ninsfle
       integer   :: lw(10),rw(10)
       double precision :: lbound,ubound,phimlim,phimaccept,fracphim

       character*4    :: aext
       character*12   :: agroup,anew,aversion
       character*15   :: atemp1,atemp2,atrans,achange
       character*200  :: pestinfile,pestoutfile,apestinfile,
     + apestoutfile,casename
       character*500  :: amessage
       character*1000 :: cline

       integer, allocatable          :: itrans(:),npg(:),iadd(:)
       double precision, allocatable :: pval(:)
       character*12, allocatable     :: apar(:),apargp(:),aobsgp(:)

#ifndef UNIX
#ifdef LAHEY
       open(unit=*,action='read',carriagecontrol='list')
#endif
#endif

       include 'version.inc'
       write(6,10) trim(aversion)
10     format(/,' ADDREG1 version ',a,'. Watermark Numerical ',
     + 'Computing.')
#ifdef NO_CMDLINE
       write(*,*)
1      write(*,'(a)') ' ENTER COMMAND LINE ARGUMENTS > '
       read(*,'(a)',ERR=1) CLINE
#else
       call pgetcl(cline)
#endif

C -- The command line is parsed.

       if(cline.eq.' ') go to 8900
       call spacesub(cline)
       call linspl(ifail,3,lw,rw,cline)
       if(ifail.eq.0) go to 8900
       call linspl(ifail,2,lw,rw,cline)
       if(ifail.ne.0) go to 8900
       pestinfile=cline(lw(1):rw(1))
       call remchar(pestinfile,char(211))
       pestoutfile=cline(lw(2):rw(2))
       call remchar(pestoutfile,char(211))
#ifndef UNIX
       call lowcas(pestinfile)
       call lowcas(pestoutfile)
#endif

C -- Further analysis is done on the input and output PEST filenames.

       nb=len_trim(pestinfile)
       nb=nb-3
       if(nb.lt.1)nb=1
       aext=pestinfile(nb:)
       call lowcas(aext)
       if(aext.eq.'.pst')then
         if(nb.eq.1) go to 8900
         casename=pestinfile(1:nb-1)
       else
         casename=pestinfile
         pestinfile=trim(casename)//'.pst'
       end if

       nb=len_trim(pestoutfile)
       nb=nb-3
       if(nb.lt.1)nb=1
       aext=pestoutfile(nb:)
       call lowcas(aext)
       if(aext.ne.'.pst')then
         pestoutfile=trim(pestoutfile)//'.pst'
       end if

C -- Quotes are added to filenames if necessary.

       call addquote(pestinfile,apestinfile)
       call addquote(pestoutfile,apestoutfile)

C -- The old and new PEST control files are opened.

       open(unit=10,file=pestinfile,status='old',iostat=ierr)
       if(ierr.ne.0)then
         write(amessage,30) trim(apestinfile)
30       format(' Cannot open PEST control file ',a,'.')
         go to 9890
       end if

       open(unit=20,file=pestoutfile,action='write',iostat=ierr)
       if(ierr.ne.0)then
         write(amessage,40) trim(apestoutfile)
40       format(' Cannot open file ',a,' for output.')
         go to 9890
       end if

C -- The existing PEST control file is opened and partly read.

       read(10,*,err=9000,end=9000) atemp1
       call lowcas(atemp1)
       if(atemp1.ne.'pcf') go to 9000
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
       if(nprior.ne.0)then
         write(amessage,167) trim(apestinfile)
167      format(' PEST control file ',a,' already contains prior ',
     +   'information. This is not allowed by ADDREG1.')
         go to 9890
       end if
       if(npar.eq.0)then
         write(amessage,168) trim(apestinfile)
168      format(' Number of parameters NPAR supplied as zero in ',
     +   'PEST control file ',a,'.')
         go to 9890
       end if
       read(10,*,err=9000,end=9000) ntplfle,ninsfle

C -- Parameter group names are now read.
       allocate(apargp(npargp),iadd(npargp),stat=ierr)
       if(ierr.ne.0) go to 9200
       do
         read(10,'(a)',err=9000,end=9000) cline
         cline=adjustl(cline)
         if(cline(1:1).eq.'*')then
           call lowcas(cline)
           if(index(cline,'parameter gr').ne.0) go to 180
           if(index(cline,'group defin').ne.0) go to 180
         end if
       end do
180    do i=1,npargp
         read(10,*,err=9000,end=9000) apargp(i)
         call lowcas(apargp(i))
       end do

C -- Parameter data is now read.

       allocate(apar(npar),itrans(npar),pval(npar),npg(npar),stat=ierr)
       if(ierr.ne.0) go to 9200
       read(10,'(a)') cline
       cline=adjustl(cline)
       call lowcas(cline)
       if((cline(1:1).ne.'*').or.
     +    (index(cline,'parameter da').eq.0)) go to 9000
       do i=1,npar
         read(10,*,err=9000,end=9000)apar(i),atrans,achange,pval(i),
     +   lbound,ubound,agroup
         call lowcas(apar(i))
         call lowcas(agroup)
         do j=1,npargp
           if(agroup.eq.apargp(j)) then
             npg(i)=j
             go to 190
           end if
         end do
         go to 9000
190      continue
         if(atrans.eq.'log')then
           itrans(i)=1
         else if(atrans.eq.'none')then
           itrans(i)=0
         else if(atrans.eq.'fixed')then
           itrans(i)=-100000
         else if(atrans.eq.'tied')then
           itrans(i)=-1
         else
           go to 9000
         end if
         if(itrans(i).eq.1)then
           if(pval(i).le.0.0d0)then
             write(amessage,195) trim(apar(i)),trim(apestinfile)
195          format(' Parameter "',a,'" is log transformed, yet it ',
     +       'has a negative initial value in PEST control file ',
     +       a,'.')
             go to 9890
           end if
         end if
       end do

C -- The names of observation groups are now read.

       allocate(aobsgp(nobsgp),stat=ierr)
       if(ierr.ne.0) go to 9200
       do
         read(10,'(a)',err=9000,end=9000) cline
         cline=adjustl(cline)
         if(cline(1:1).eq.'*')then
           call lowcas(cline)
           if(index(cline,'observation gr').ne.0) go to 210
         end if
       end do
210    continue
       do i=1,nobsgp
         read(10,*,err=9000,end=9000) aobsgp(i)
         call lowcas(aobsgp(i))
       end do

! -- We now see whether we need to add new observation group names to the
!    PEST control file and how many.

       iwarn=0
       iadd=0           ! an array
       do ipar=1,npar
         if(itrans(ipar).ge.0)then
           j=npg(ipar)
           anew='regul_'//apargp(j)(1:6)
           do k=1,nobsgp
             if(anew.eq.aobsgp(k)) then
               iwarn=1
               go to 220
             end if
           end do
           iadd(j)=1
220        continue
         end if
       end do
       newgroup=0
       do i=1,npargp
         if(iadd(i).eq.1) newgroup=newgroup+1
       end do

C -- The number of parameters for which we require prior information is now
C    established.

       nprior=0
       do ipar=1,npar
         if(itrans(ipar).ge.0) nprior=nprior+1
       end do
       if(nprior.eq.0)then
         write(amessage,230) trim(apestinfile)
230      format(' No adjustable parameters are cited in PEST ',
     +   'control file ',a,'.')
         go to 9890
       end if

C -- The original PEST control file is re-wound and information transferred to the
C    new PEST control file.

       rewind(unit=10)
       read(10,*)
       write(20,250)
250    format('pcf')
       read(10,*)
       write(20,255)
255    format('* control data')
       read(10,*)
       write(20,260)
260    format('restart regularisation')
       read(10,*)
       write(20,270) npar,nobs,npargp,nprior,nobsgp+newgroup
270    format(5i8)
       do
         read(10,'(a)') cline
         write(20,'(a)') trim(cline)
         cline=adjustl(cline)
         if(cline(1:1).eq.'*')then
           call lowcas(cline)
           if(index(cline,'observation gr').ne.0) go to 300
         end if
       end do
300    continue
       do i=1,nobsgp
         read(10,'(a)') cline
         write(20,'(a)') trim(cline)
       end do
       if(newgroup.ne.0)then
         do i=1,npargp
           if(iadd(i).ne.0)then
             anew='regul_'//apargp(i)(1:6)
             write(20,'(a)') trim(anew)
           end if
         end do
       end if

! -- The remainder of the PEST control file is now transferred.

       do
         read(10,'(a)',err=9000,end=9000) cline
         write(20,'(a)') trim(cline)
         cline=adjustl(cline)
         if(cline(1:1).eq.'*') then
           if(index(cline,'model in').ne.0) go to 400
         end if
       end do
400    continue
       do i=1,ntplfle
         read(10,'(a)',err=9000,end=9000) cline
         write(20,'(a)') trim(cline)
       end do
       do i=1,ninsfle
         read(10,'(a)',err=9000,end=9000) cline
         write(20,'(a)') trim(cline)
       end do

C -- The new prior information equations are now added.

       write(20,401)
401    format('* prior information')
       do ipar=1,npar
         if(itrans(ipar).lt.0) cycle
         j=npg(ipar)
         anew='regul_'//apargp(j)(1:6)
         if(itrans(ipar).eq.0)then
           write(20,430) trim(apar(ipar)),trim(apar(ipar)),pval(ipar),
     +     trim(anew)
430        format(a,t15,'1.0 * ',a,' = ',1pg14.7,'  1.0 ',a)
         else
           write(20,440) trim(apar(ipar)),trim(apar(ipar)),
     +     log10(pval(ipar)),trim(anew)
440        format(a,t15,'1.0 * log(',a,') = ',1pg14.7,'  1.0 ',a)
         end if
       end do

C --  We see if any regularisation section is in the PEST control file.

       phimlim=1.0
       phimaccept=1.05
       fracphim=0.1
       do
         read(10,'(a)',end=600) cline
         cline=adjustl(cline)
         if(cline(1:1).eq.'*') then
           if(index(cline,'regula').ne.0) go to 500
         end if
       end do
500    read(10,*,iostat=ierr)phimlim,phimaccept
       if(ierr.ne.0)then
         phimlim=1.0
         phimaccept=1.05
         go to 600
       end if
       go to 600

600    continue
       write(20,620)
620    format('* regularisation')
       write(20,630) phimlim,phimaccept,fracphim
630    format(3(1x,1pg14.7))
       write(20,640)
640    format(' 1.0   1.0e-10    1.0e10')
       write(20,650)
650    format(' 1.3   1.0e-2     1')

C -- Finishing up

       close(unit=10)
       close(unit=20)
       write(6,660) trim(apestinfile)
660    format(' - file ',a,' read ok.')
       write(6,670) trim(apestoutfile)
670    format(' - file ',a,' written ok.')

       if(iwarn.ne.0)then
         write(amessage,680) trim(apestoutfile), trim(apestinfile)
680      format(' WARNING: at least one observation ',
     +   'group name employed in regularisation prior ',
     +   'information added to file ',a,' was already cited in ',
     +   'the "observation groups" section of file ',a,'.')
         call writmess(6,amessage)
       end if

       go to 9900


8900   continue
       write(6,8910)
8910   format(/,' ADDREG1 is run using the command:',/)
       write(6,8920)
8920   format('     addreg1 pestinfile pestoutfile',/)
       write(6,8930)
8930   format(' where',/)
       write(6,8940)
8940   format('     pestinfile  is an existing ',
     + 'PEST control file, and')
       write(6,8950)
8950   format('     pestoutfile is a new PEST control file.')

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

9900   deallocate(itrans,npg,iadd,pval,apar,apargp,aobsgp,stat=ierr)

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

        subroutine lowcas(astrng)

C -- Subroutine LOWCAS converts a string to lower case.

        integer i,j
        character*(*) astrng

        do 10 i=1,len_trim(astrng)
        j=ichar(astrng(i:i))
        if((j.ge.65).and.(j.le.90)) astrng(i:i)=char(j+32)
10      continue
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




