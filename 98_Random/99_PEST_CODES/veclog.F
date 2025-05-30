       include 'matman.f'


       program veclog

C -- Program veclog log-transforms the elements of a vector.

       use matrix_manipulate
       implicit none

       logical          :: lopened
       integer          :: ifail,iwarn,irow,nrow,ierr,itemp
       integer          :: lw(4),rw(4)
       double precision :: dtemp
       character*12     :: aversion
       character*200    :: matfile1,matfile2,matfile3
       character*200    :: amatfile1,amatfile2,amatfile3
       character*500    :: cline
       character*600    :: amessage
       type(dmatrix)    :: mat1,mat2

#ifndef UNIX
#ifdef LAHEY
       open(unit=*,action='read',carriagecontrol='list')
#endif
#endif

       include 'version.inc'
       write(6,10) trim(aversion)
10     format(/,' VECLOG version ',a,'. Watermark Numerical ',
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
       call linspl(ifail,4,lw,lw,cline)
       if(ifail.eq.0) go to 8900
       call linspl(ifail,3,lw,rw,cline)
       if(ifail.ne.0) go to 8900
       matfile1=cline(lw(1):rw(1))
       call remchar(matfile1,char(211))
       matfile2=cline(lw(2):rw(2))
       call remchar(matfile2,char(211))
       matfile3=cline(lw(3):rw(3))
       call remchar(matfile3,char(211))
#ifndef UNIX
       call lowcas(matfile1)
       call lowcas(matfile2)
       call lowcas(matfile3)
#endif
       call addquote(matfile1,amatfile1)
       call addquote(matfile2,amatfile2)
       call addquote(matfile3,amatfile3)

C -- The vector is read.

       call mat_read(ifail,10,mat1,matfile1,amessage,cline)
       if(ifail.ne.0) go to 9800
       write(6,100) trim(amatfile1)
100    format(' - file ',a,' read ok.')

       call mat_read(ifail,10,mat2,matfile2,amessage,cline)
       if(ifail.ne.0) go to 9800
       write(6,100) trim(amatfile2)

C -- The two matrices are checked for compatibility.

       if(mat1%ncol.ne.1)then
         write(amessage,110) trim(amatfile1)
110      format(' Matrix contained in file ',a,' must contain a ',
     +   'single column.')
         go to 9800
       end if
       if(mat2%ncol.ne.1)then
         write(amessage,110) trim(amatfile2)
         go to 9800
       end if
       if(mat1%nrow.ne.mat2%nrow)then
         write(amessage,115) trim(amatfile1),trim(amatfile2)
115      format(' The matrices contained in files ',a,' and ',a,
     +   ' do not have the same number of rows.')
         go to 9800
       end if
       nrow=mat1%nrow

       if(mat2%icode.eq.-1)then
         dtemp=mat2%vector(1)
         if((abs(dtemp-0.0d0).gt.1.0d-3).and.
     +      (abs(dtemp-1.0d0).gt.1.0d-3).and.
     +      (abs(dtemp+1.0d0).gt.1.0d-3))then
           write(amessage,120) trim(amatfile2)
120        format(' Elements of the vector contained in matrix file ',
     +     a,' must be either 1, 0 or -1.')
           go to 9800
         end if
       else
         do irow=1,nrow
           dtemp=mat2%array(irow,1)
           if((abs(dtemp-0.0d0).gt.1.0d-3).and.
     +        (abs(dtemp-1.0d0).gt.1.0d-3).and.
     +        (abs(dtemp+1.0d0).gt.1.0d-3))then
             write(amessage,120) trim(amatfile2)
             go to 9800
           end if
         end do
       end if

! -- The transformation is now carried out.

       if(nrow.eq.1)then
         if(mat2%icode.eq.-1)then
           itemp=nint(mat2%vector(1))
         else
           itemp=nint(mat2%array(1,1))
         end if
         if(mat1%icode.eq.-1)then
           dtemp=mat1%vector(1)
         else
           dtemp=mat1%array(1,1)
         end if
         if(itemp.eq.1)then
           if(dtemp.le.0.0d0)then
             write(amessage,130) trim(amatfile1),trim(amatfile2)
130          format(' At least one element in the matrix housed ',
     +       'in file ',a,' is zero or negative; however, according ',
     +       'to the matrix in file ',a,' this element is to be log-',
     +       'transformed. This is not possible.')
             go to 9800
           end if
           if(mat1%icode.eq.-1)then
             mat1%vector(1)=log10(dtemp)
           else
             mat1%array(1,1)=log10(dtemp)
           end if
         else if(itemp.eq.-1)then
           if(mat1%icode.eq.-1)then
             mat1%vector(1)=10.0d0**min(36.0d0,dtemp)
           else
             mat1%array(1,1)=10.0d0**min(36.0d0,dtemp)
           end if
         end if
       else
         do irow=1,nrow
           itemp=nint(mat2%array(irow,1))
           if(itemp.eq.1)then
             if(mat1%array(irow,1).le.0.0d0)then
               write(amessage,130) trim(amatfile1),trim(amatfile2)
               go to 9800
             end if
             mat1%array(irow,1)=log10(mat1%array(irow,1))
           else if(itemp.eq.-1)then
             mat1%array(irow,1)=10.0d0**min(36.0d0,mat1%array(irow,1))
           end if
         end do
       end if

C -- The new matrix file is written.

       call mat_write(ifail,20,mat1,matfile3,amessage,cline)
       if(ifail.ne.0) go to 9800
       write(6,200) trim(amatfile3)
200    format(' - file ',a,' written ok.')

C -- The warning is written if necessary.

       iwarn=0
       do irow=1,nrow
         if(mat1%arow(irow).ne.mat2%arow(irow)) then
           iwarn=1
           go to 210
         end if
       end do
210    continue
       if(iwarn.ne.0)then
         write(amessage,250) trim(amatfile1),trim(amatfile2)
250      format(' Warning: row names are not ',
     +   'identical (and/or not in the same order) for the matrices ',
     +   'in files ',a,' and ',a,'.')
         go to 9800
       end if

       go to 9900

8900   continue
       write(6,8910)
8910   format(/,' VECLOG is run using the command:',/)
       write(6,8920)
8920   format('     veclog     vecfile transfile vecoutfile',/)
       write(6,8930)
8930   format(' where',/)
       write(6,8940)
8940   format('     vecfile    contains a column matrix,')
       write(6,8950)
8950   format('     transfile  contains a column "transformation ',
     + 'matrix", and')
       write(6,8960)
8960   format('     vecoutfile is the transformed vector output',
     + ' file.')

       go to 9999

9800   continue
       call writmess(6,amessage)

9900   continue
       call mat_deallocate(ifail,mat1)
       call mat_deallocate(ifail,mat2)
       inquire(unit=10,opened=lopened)
       if(lopened) close(unit=10,iostat=ierr)
       inquire(unit=20,opened=lopened)
       if(lopened) close(unit=20,iostat=ierr)


9999   continue

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

