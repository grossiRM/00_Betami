       subroutine store_value(ifail,ndim,vector,ivector,rval,irow,icol)

C -- Program stor_value places a number into a compressed matrix. Note that it also
C    maintains sorting order. It can stor a number into a location that is already
C    defined, or, add the location to the vector in correct order.

C    Arguments are as follows:-
C      ifail       : return as 1 if no more room left in vector
C      ndim        : number of used entries of the vector
C      vector      : real vector which stores matrix in compressed form
C      ivector     : integer vector containing element number of matrix
C      rval        : number to be stored
C      irow,icol   : row and column of matrix

       use pestdata, only: maxcompdim,ifound_c,nxrow,derzerolim
       implicit none

       integer icol,irow,ifail,inum,i,ielem,ndim
       integer ivector(maxcompdim)
       double precision rval
       double precision vector(maxcompdim)

C -- The matrix index number is evaluated from its row and column.

       ifail=0
       inum=(icol-1)*nxrow+irow
       if(ndim.eq.0)then
         ndim=1
         if(ndim+1.gt.maxcompdim) go to 9000
         vector(1)=rval
         ivector(1)=inum
         ifound_c=1
         return
       end if
       call search_vector(ielem,ndim,ivector,inum)
       if(ielem.gt.0)then
         vector(ielem)=rval
         ifound_c=ielem
       else
         if(abs(rval).gt.derzerolim) then
           ielem=-ielem
           ndim=ndim+1
           if(ndim+1.gt.maxcompdim) go to 9000
           if(ielem.eq.ndim-1)then
             vector(ndim)=rval
             ivector(ndim)=inum
             ifound_c=ndim
           else
             do i=ndim,ielem+2,-1
               vector(i)=vector(i-1)
               ivector(i)=ivector(i-1)
             end do
             vector(ielem+1)=rval
             ivector(ielem+1)=inum
             ifound_c=ielem+1
           end if
         end if
       end if
       return

9000   ifail=1
       return
#ifdef PESTMOD
       end subroutine store_value
#else
       end
#endif


       subroutine search_vector(ielem,ndim,ivector,inum)

C -- Program search_vector searches a vector for a number, assuming that entries
C    are in increasing order.

       use pestdata, only: istart_c
       implicit none

       integer inum,ielem,itemp,lbound,ubound,irove,ndim,inum1,inum2
       integer ivector(ndim)

       if(istart_c.gt.ndim)istart_c=ndim
       if(istart_c.ne.0)then
         inum1=ivector(istart_c)
         if(inum.eq.inum1)then
           ielem=istart_c
           return
         else if(istart_c.lt.ndim)then
           itemp=istart_c+1
           inum2=ivector(itemp)
           if(inum.eq.inum2)then
             ielem=itemp
             return
           else
             if((inum.gt.inum1).and.(inum.lt.inum2))then
               ielem=-istart_c
               return
             end if
           end if
         end if
       end if
       itemp=ivector(ndim)
       if(inum.eq.itemp)then
         ielem=ndim
         return
       else if(inum.gt.itemp)then
         ielem=-ndim
         return
       end if
       itemp=ivector(1)
       if(inum.eq.itemp)then
         ielem=1
         return
       else if(inum.lt.itemp)then
         ielem=0
         return
       end if

C -- We've tried all the easy options; now lets do the search.

       irove=istart_c
       if(irove.lt.2)irove=2
       if(irove.gt.ndim) irove=ndim
       lbound=1
       ubound=ndim
100    continue
       itemp=ivector(irove)
       if(inum.eq.itemp)then
         ielem=irove
         return
       else if(inum.lt.itemp)then
         ubound=irove
       else if(inum.gt.itemp)then
         lbound=irove
       end if
       if(ubound.eq.lbound+1)then
         ielem=-lbound
         return
       end if
       irove=(lbound+ubound)/2
       go to 100

#ifdef PESTMOD
       end subroutine search_vector
#else
       end
#endif


       subroutine get_value(ndim,vector,ivector,rval,irow,icol)

C -- Program get_value retreives a number from a compressed matrix.

C    Arguments are as follows:-
C      ndim        : number of elements of vector used so far
C      vector      : real vector which stores matrix in compressed form
C      ivector     : integer vector containing element number of matrix
C      rval        : number retreived
C      irow,icol   : row and column of matrix

       use pestdata, only: ifound_c,nxrow
       implicit none

       integer icol,irow,ifail,inum,ielem,ndim
       integer ivector(ndim)
       double precision rval
       double precision vector(ndim)

C -- The matrix index number is evaluated from its row and column.

       inum=(icol-1)*nxrow+irow
       call search_vector(ielem,ndim,ivector,inum)
       if(ielem.gt.0)then
         rval=vector(ielem)
         ifound_c=ielem
       else
         rval=0.0d0
         ifound_c=-ielem
       end if

       return
#ifdef PESTMOD
       end subroutine get_value
#else
       end
#endif



       subroutine get_vector(ndim,nrow,vector,ivector,rvec,icol)

C -- Subroutine get_vector retrieves a column of the true matrix from the
C    compressed matrix.

       use pestdata, only: ifound_c,nxrow
       implicit none

       integer icol,inum,ielem,ndim,inum1,inum2,i,ielemnext,
     + inumnext,in,nrow,j
       integer ivector(ndim)
       double precision vector(ndim),rvec(nrow)

       inum1=(icol-1)*nxrow+1
       call search_vector(ielem,ndim,ivector,inum1)
       inum2=(icol-1)*nxrow+nrow
       if(ielem.gt.0)then
         rvec(1)=vector(ielem)
       else
         rvec(1)=0.0d0
       end if
       inum=inum1+1
       ielem=abs(ielem)
       ielemnext=ielem+1
       i=1
10     continue
       if(ielemnext.gt.ndim)then
         do j=1+1,nrow
           rvec(j)=0.0d0
         end do
         ifound_c=ndim
         go to 500
       end if
       inumnext=ivector(ielemnext)
       if(inumnext.eq.inum)then
         i=i+1
         rvec(i)=vector(ielemnext)
         inum=inum+1
         if(inum.gt.inum2)then
           ifound_c=ielemnext
           go to 500
         end if
         ielem=ielemnext
         ielemnext=ielemnext+1
         go to 10
       end if
       do in=inum,min(inum2,inumnext-1)
         i=i+1
         rvec(i)=0.0d0
       end do
       if(inum2.le.inumnext-1)then
         ifound_c=ielem
         go to 500
       else
         i=i+1
         rvec(i)=vector(ielemnext)
         if(i.eq.nrow)then
           ifound_c=ielemnext
           go to 500
         else
           ielem=ielemnext
           ielemnext=ielemnext+1
           inum=inumnext+1
           go to 10
         end if
       end if

500    return
#ifdef PESTMOD
       end subroutine get_vector
#else
       end
#endif



       subroutine dual_vector_mul(ifail,ndim,startrow,nrow,vector,
     + ivector,rvec,icol1,icol2,rval)

C -- Subroutine dual_vector_mul multiplies one column by another and by an
C    external vector.

       use pestdata, only: maxcompdim,nxrow
       implicit none

       integer ifail,icol1,icol2,ielem1,ielem2,ndim,in1,in2,is1,is2,
     + iflag1,iflag2,irow,nrow,startrow
       integer ivector(ndim+1)
       double precision rval
       double precision vector(ndim+1),rvec(nrow)

       ifail=0
       if(ndim.eq.maxcompdim)then
         ifail=1
         return
       end if
       ivector(ndim+1)=0

       in1=(icol1-1)*nxrow+startrow
       call search_vector(ielem1,ndim,ivector,in1)
       in2=(icol2-1)*nxrow+startrow
       call search_vector(ielem2,ndim,ivector,in2)
       if(ielem1.le.0)ielem1=abs(ielem1)+1
       is1=ivector(ielem1)
       if(ielem2.le.0)ielem2=abs(ielem2)+1
       is2=ivector(ielem2)

       in1=in1-1
       in2=in2-1
       rval=0.0d0
       do irow=startrow,nrow
         in1=in1+1
         in2=in2+1
         if(is1.eq.in1)then
           iflag1=1
         else
           iflag1=0
         end if
         if(is2.eq.in2)then
           iflag2=1
         else
           iflag2=0
         end if
         if((iflag1.eq.1).and.(iflag2.eq.1))
     +     rval=rval+vector(ielem1)*vector(ielem2)*rvec(irow)
         if(iflag1.eq.1)then
           ielem1=ielem1+1   ! note that we assume there is one zero element above ncompdim
           is1=ivector(ielem1)
         end if
         if(iflag2.eq.1)then
           ielem2=ielem2+1
           is2=ivector(ielem2)
         end if
       end do

       return

#ifdef PESTMOD
       end subroutine dual_vector_mul
#else
       end
#endif


       subroutine single_vector_mul1(ifail,ndim,startrow,nrow,vector,
     + ivector,icol,wt,vec1,vec2,rval)

C -- Subroutine single_vector_mul1 multiplies one column of the jacobian
C    by wt(i)*(vec1(i)-vec2(i)) where wt(i) may take on zero values.

       use pestdata, only: maxcompdim,nxrow
       implicit none

       integer ifail,ielem,ndim,in,is,iflag,irow,nrow,startrow,
     + icol
       integer ivector(ndim+1)
       double precision rval
       double precision vector(ndim+1),wt(nrow),vec1(nrow),vec2(nrow)

       ifail=0
       if(ndim.eq.maxcompdim)then
         ifail=1
         return
       end if
       ivector(ndim+1)=0

       in=(icol-1)*nxrow+startrow
       call search_vector(ielem,ndim,ivector,in)
       if(ielem.le.0)ielem=abs(ielem)+1
       is=ivector(ielem)

       in=in-1
       rval=0.0d0
       do irow=startrow,nrow
         in=in+1
         if(is.eq.in)then
           iflag=1
         else
           iflag=0
         end if
         if(iflag.eq.1)
     +     rval=rval+vector(ielem)*wt(irow)*(vec1(irow)-vec2(irow))
         if(iflag.eq.1)then
           ielem=ielem+1   ! note that we assume there is one zero element above ncompdim
           is=ivector(ielem)
         end if
       end do

       return

#ifdef PESTMOD
       end subroutine single_vector_mul1
#else
       end
#endif

       subroutine single_vector_mul2(ifail,ndim,startrow,nrow,vector,
     + ivector,icol,wt,rval)

C -- Subroutine single_vector_mul2 multiplies one column of the jacobian
C    matrix squared by wt(i) (which can have many zero values).

       use pestdata, only: maxcompdim,nxrow
       implicit none

       integer ifail,ielem,ndim,in,is,iflag,irow,nrow,startrow,
     + icol
       integer ivector(ndim+1)
       double precision rval
       double precision vector(ndim+1),wt(nrow)

       ifail=0
       if(ndim.eq.maxcompdim)then
         ifail=1
         return
       end if
       ivector(ndim+1)=0

       in=(icol-1)*nxrow+startrow
       call search_vector(ielem,ndim,ivector,in)
       if(ielem.le.0)ielem=abs(ielem)+1
       is=ivector(ielem)

       in=in-1
       rval=0.0d0
       do irow=startrow,nrow
         in=in+1
         if(is.eq.in)then
           iflag=1
         else
           iflag=0
         end if
         if(iflag.eq.1)
     +     rval=rval+vector(ielem)*vector(ielem)*wt(irow)
         if(iflag.eq.1)then
           ielem=ielem+1   ! note that we assume there is one zero element above ncompdim
           is=ivector(ielem)
         end if
       end do

       return

#ifdef PESTMOD
       end subroutine single_vector_mul2
#else
       end
#endif


       subroutine single_vector_mul3(ifail,ndim,startrow,nrow,vector,
     + ivector,icol,wt,vec,rval)

C -- Subroutine single_vector_mul3 multiplies one column of the jacobian
C    by vec(i)*sqrt(wt(i))where wt(i) may take on zero values.

       use pestdata, only: maxcompdim,nxrow
       implicit none

       integer ifail,ielem,ndim,in,is,iflag,irow,nrow,startrow,
     + icol
       integer ivector(ndim+1)
       double precision rval
       double precision vector(ndim+1),wt(nrow),vec(nrow)

       ifail=0
       if(ndim.eq.maxcompdim)then
         ifail=1
         return
       end if
       ivector(ndim+1)=0

       in=(icol-1)*nxrow+startrow
       call search_vector(ielem,ndim,ivector,in)
       if(ielem.le.0)ielem=abs(ielem)+1
       is=ivector(ielem)

       in=in-1
       rval=0.0d0
       do irow=startrow,nrow
         in=in+1
         if(is.eq.in)then
           iflag=1
         else
           iflag=0
         end if
         if(iflag.eq.1)
     +     rval=rval+vector(ielem)*vec(irow)*sqrt(wt(irow))
         if(iflag.eq.1)then
           ielem=ielem+1   ! note that we assume there is one zero element above ncompdim
           is=ivector(ielem)
         end if
       end do

       return

#ifdef PESTMOD
       end subroutine single_vector_mul3
#else
       end
#endif


       subroutine zero_matrix(ifail,ndim,vector,ivector,endrow)

C -- Subroutine zero_matrix zeroes all elements of compressed matrix below
C    a certain row.

       use pestdata, only: maxcompdim,nxrow
       implicit none

       integer ifail,ndim,endrow,i,j,ies,irow
       integer ivector(ndim+1)
       double precision vector(ndim+1)

       ifail=0
       do i=1,ndim
          j=ivector(i)
          ies=(j-1)/nxrow+1
          irow=j-(ies-1)*nxrow
          if(irow.le.endrow)vector(j)=0.0d0
       end do

       return

#ifdef PESTMOD
       end subroutine zero_matrix
#else
       end
#endif

