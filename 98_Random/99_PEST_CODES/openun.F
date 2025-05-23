       subroutine open_unformatted_file(ifail,iunit,how,aname,amessage)

C -- Subroutine open_unformatted_file opens an unformatted file for reading
C    or writing.

       implicit none

       integer ifail,iunit
       character*(*) how,aname,amessage

       integer ierr
       character*10 astatus
       character*200 afile

       ifail=0
       astatus='unknown'
       call lowcas(how)
       if(how.eq.'read') astatus='old'

#ifdef LAHEY
#ifdef UNIX
       open(unit=iunit,file=aname,form='unformatted',
     + status=astatus,iostat=ierr)
#else
#ifdef LF90
       open(unit=iunit,file=aname,form='unformatted',
     + access='transparent',status=astatus,iostat=ierr)
C       open(unit=iunit,file=aname,form='binary',
C     + status=astatus,iostat=ierr)
#else
       open(unit=iunit,file=aname,form='binary',
     + status=astatus,iostat=ierr)
#endif
#endif
#else
       open(unit=iunit,file=aname,form='unformatted',
     + status=astatus,iostat=ierr)
#endif
       if(ierr.ne.0)then
         call addquote(aname,afile)
         if(how.eq.'read')then
           write(amessage,370) trim(afile)
370        format(' Cannot open unformatted file ',a,'.')
         else
           write(amessage,380) trim(afile)
380        format(' Cannot open unformatted file ',a,' for writing.')
         end if
         ifail=1
         return
       end if

       return

       end


