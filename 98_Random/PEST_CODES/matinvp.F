       include 'matman.f'

       program matinvp

C -- Program MATINVP finds the inverse of a positve definite matrix.

       use matrix_manipulate
       implicit none

       logical          :: lopened
       integer          :: ifail,ncol,nrow,ierr,icol,irow,info
       integer          :: lw(5),rw(5)
       double precision :: det
       character*12     :: aversion
       character*20     :: atemp
       character*200    :: matfile1,matfile2
       character*200    :: afile
       character*500    :: cline
       character*600    :: amessage
       type(dmatrix)    :: mat1

#ifndef UNIX
#ifdef LAHEY
       open(unit=*,action='read',carriagecontrol='list')
#endif
#endif

       include 'version.inc'
       write(6,10) trim(aversion)
10     format(/,' MATINVP version ',a,'. Watermark Numerical ',
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
       matfile1=cline(lw(1):rw(1))
       call remchar(matfile1,char(211))
       matfile2=cline(lw(2):rw(2))
       call remchar(matfile2,char(211))
#ifndef UNIX
       call lowcas(matfile1)
       call lowcas(matfile2)
#endif

C -- The matrix is read.

       call mat_read(ifail,10,mat1,matfile1,amessage,cline)
       if(ifail.ne.0) go to 9800
       call addquote(matfile1,afile)
       write(6,100) trim(afile)
100    format(' - file ',a,' read ok.')

C -- The matrix is checked to ensure that the diagonal can be extracted.

       if(mat1%nrow.ne.mat1%ncol)then
         write(amessage,110) trim(afile)
110      format(' Cannot invert matrix in file ',a,
     +   ' because it is not square.')
         go to 9800
       end if
       ncol=mat1%ncol
       nrow=mat1%nrow

C -- First we take care of the case where the matrix is diagonal.

       if(mat1%icode.eq.-1)then
         do icol=1,mat1%ncol
           if(mat1%vector(icol).le.0.0d0)then
             write(amessage,120) trim(afile)
120          format(' Cannot invert matrix in file ',a,' because ',
     +       'it is not positive definite.')
             go to 9800
           end if
         end do
         do icol=1,ncol
           mat1%vector(icol)=1.0d0/mat1%vector(icol)
         end do
       else
         call dpofa(mat1%array,ncol,ncol,info)
         if(info.ne.0)then
           write(amessage,120) trim(afile)
           go to 9800
         end if
         info=1
         call dpodi(mat1%array,ncol,ncol,det,info)
         do icol=1,ncol-1
           do irow=icol+1,ncol
             mat1%array(irow,icol)=mat1%array(icol,irow)
           end do
         end do
         if(mat1%icode.eq.2)then
           do icol=1,ncol
             atemp=mat1%acol(icol)
             mat1%acol(icol)=mat1%arow(icol)
             mat1%arow(icol)=atemp
           end do
         end if
       end if

C -- The new matrix file is written.

       call mat_write(ifail,20,mat1,matfile2,amessage,cline)
       if(ifail.ne.0) go to 9800
       call addquote(matfile2,afile)
       write(6,250) trim(afile)
250    format(' - file ',a,' written ok.')

       go to 9900

8900   continue
       write(6,8910)
8910   format(/,' MATINVP is run using the command:',/)
       write(6,8920)
8920   format('     matinvp matfile1 matfile2',/)
       write(6,8930)
8930   format(' where',/)
       write(6,8940)
8940   format('     matfile1 contains a square positive definite ',
     + 'matrix, and')
       write(6,8960)
8960   format('     matfile2 contains the inverse of this matrix.')

       go to 9999

9800   continue
       call writmess(6,amessage)

9900   continue
       call mat_deallocate(ifail,mat1)

       inquire(unit=10,opened=lopened)
       if(lopened) close(unit=10,iostat=ierr)
       inquire(unit=20,opened=lopened)
       if(lopened) close(unit=20,iostat=ierr)


9999   continue

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



        subroutine intread(ifail,cline,itemp)

C -- Subroutine INTREAD reads a real number from a string.

        integer ifail
        integer itemp
        character*6 afmt
        character*(*) cline

        ifail=0
        afmt='(i   )'
        write(afmt(3:5),'(i3)') len(cline)
        read(cline,afmt,err=100) itemp
        return

100     ifail=1
        return
        end



