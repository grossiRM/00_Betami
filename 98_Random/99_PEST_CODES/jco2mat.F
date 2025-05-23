       program jco2mat

C -- Program JCO2MAT rewrites the Jacobian matrix, as read from a JCO file, in
C    matrix file format.

       implicit none

       integer          :: ierr,ifail,itemp1,itemp2,nespar,nxrow,
     +                     icount,irow,ies,nb,i,j
       integer          :: lw(5),rw(5)
       double precision :: dtemp
       character*4      :: aext
       character*12     :: aversion
       character*200    :: jcofile,matfile,afile
       character*500    :: cline,amessage

       double precision, allocatable  :: x(:,:)
       character*12, allocatable      :: apar(:)
       character*20, allocatable      :: aobs(:)

#ifndef UNIX
#ifdef LAHEY
       open(unit=*,action='read',carriagecontrol='list')
#endif
#endif

       include 'version.inc'
       write(6,10) trim(aversion)
10     format(/,' JCO2MAT version ',a,'. Watermark Numerical ',
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
       call linspl(ifail,3,lw,lw,cline)
       if(ifail.eq.0) go to 8900
       call linspl(ifail,2,lw,rw,cline)
       if(ifail.ne.0) go to 8900
       jcofile=cline(lw(1):rw(1))
       call remchar(jcofile,char(211))
       matfile=cline(lw(2):rw(2))
       call remchar(matfile,char(211))
#ifndef UNIX
       call lowcas(jcofile)
       call lowcas(matfile)
#endif

       nb=len_trim(jcofile)
       i=nb-3
       if(i.lt.1)i=1
       aext=jcofile(i:nb)
       call lowcas(aext)
       if(aext.ne.'.jco')then
         call addquote(jcofile,afile)
         write(amessage,15) trim(afile)
15       format(' Expected extension for Jacobian matrix file ',a,
     +   ' is ".jco".')
         go to 9800
       end if

C -- The Jacobian matrix is read.

       call addquote(jcofile,afile)
       write(6,30) trim(afile)
30     format(' - reading Jacobian matrix file ',a,'...')

       call open_unformatted_file(ifail,10,'read',jcofile,amessage)
       if(ifail.ne.0) go to 9800

       read(10,err=9350,end=9350)itemp1,itemp2
       if(itemp1.ge.0)then
         write(amessage,60) trim(afile)
60       format(' Jacobian matrix file ',a,' uses old format; ',
     +   'use JCOTRANS utility to translate it to new format.')
         go to 9800
       end if
       nespar=abs(itemp1)
       nxrow=abs(itemp2)
       if((nespar.le.0).or.(nxrow.le.0))then
         write(amessage,65) trim(afile)
65       format(' NROW or NCOL not positive in file ',a,
     +   '; is this a valid Jacobian matrix file?')
         go to 9800
       end if
       allocate(x(nxrow,nespar),apar(nespar),aobs(nxrow),stat=ierr)
       if(ierr.ne.0) go to 9200
       do i=1,nespar
         do j=1,nxrow
           x(j,i)=0.0d0
         end do
       end do
       read(10,err=9350,end=9350)icount
       do i=1,icount
         read(10,err=9350,end=9350) j,dtemp
         ies=(j-1)/nxrow+1
         irow=j-(ies-1)*nxrow
         x(irow,ies)=dtemp
       end do
       do ies=1,nespar
         read(10,err=9350,end=9350) apar(ies)
       end do
       do irow=1,nxrow
         read(10,err=9350,end=9350) aobs(irow)
       end do
       close(unit=10)
       write(6,70) trim(afile)
70     format(' - Jacobian matrix file ',a,' read ok.')

C -- The matrix file is now written.

        call addquote(matfile,afile)
        open(unit=20,file=matfile,err=9100)
        write(20,280) nxrow,nespar,2
280     format(3i6)
        do irow=1,nxrow
          write(20,285,err=9100) (x(irow,ies),ies=1,nespar)
285       format(8(1x,1pg14.7))
        end do
        write(20,290,err=9100)
290     format('* row names')
        do irow=1,nxrow
          write(20,300,err=9100) trim(aobs(irow))
300       format(A)
        end do
        write(20,310,err=9100)
310     format('* column names')
        do ies=1,nespar
          write(20,300,err=9100) trim(apar(ies))
        end do
        close(unit=20)
        write(6,320) trim(afile)
320     format(' - file ',a,' written ok.')

       go to 9999

8900   continue
       write(6,8910)
8910   format(/,' JCO2MAT is run using the command:',/)
       write(6,8920)
8920   format('     jco2mat jcofile matfile',/)
       write(6,8930)
8930   format(' where',/)
       write(6,8940)
8940   format('     jcofile is the name of a PEST Jacobian matrix ',
     + 'file, and')
       write(6,8960)
8960   format('     matfile is a matrix file to which the ',
     + 'Jacobian matrix will be written.')

       go to 9999

9100   write(amessage,9110) trim(afile)
9110   format(' Cannot write to matrix file ',a,'.')
       go to 9800
9200   write(amessage,9210)
9210   format(' Cannot allocate sufficient memory to continue ',
     + 'execution.')
       go to 9800
9350   write(amessage,9360) trim(afile)
9360   format(' Error reading Jacobian matrix from file ',a,'.')
       go to 9800

9800   continue
       call writmess(6,amessage)

9999   deallocate(x,apar,aobs,stat=ierr)

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

