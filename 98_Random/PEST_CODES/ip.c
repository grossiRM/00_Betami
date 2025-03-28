/*
 *  TCP/IP library
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <string.h>
#include <fcntl.h>
#ifdef UNIX
#include <netdb.h>
#include <sys/utsname.h>
#else
#include <ws2tcpip.h>
#endif

#define LEN 65536
#define SERVER_QUEUE_SIZE 16
static const int verbose=0;

typedef struct addrinfo addr;

/* ------------------------------------------------------------------ */
/* --------------------  Error Messages and Logging  ---------------- */
/* ------------------------------------------------------------------ */
/*
 *  Print fatal error on stderr and exit
 */
void fatal(const char* format , ...)
{
   va_list args;
   va_start(args,format);
   vfprintf(stderr,format,args);
   va_end(args);
   exit(1);
}

/*
 *  Print info on stderr
 */
void info(const char* format , ...)
{
   va_list args;
   va_start(args,format);
   vfprintf(stderr,format,args);
   va_end(args);
}

/* ------------------------------------------------------------------ */
/* --------------------------  IP Utilities  ------------------------ */
/* ------------------------------------------------------------------ */
/*
 *  Parse host:port
 *    str  is the string host:port or :port
 *    sa   is the returned addrinfo structure
 *  RETURN
 *    1 no host name (port only)
 *    0 host and port
 *   -1 if no name
 *   -2 if no port
 *   -3 if host name resolution fails
 */
static int hostport(char* str,addr** sa)
{
   if (!str || !*str)
   {
      return -1;
   }
   else
   {
      char  buf[LEN];
      char* host;
      char* port;
      addr  hint;
      //  Erase hint structure
      memset(&hint,0,sizeof(hint));
      //  Copy name to buf but no longer than LEN
      strncpy(buf,str,LEN);
      buf[LEN-1] = 0;
      //  IPv6
      if (buf[0] == '[')
      {
         //  Change ]: to 0
         char* ch = strstr(buf,"]:");
         if (!ch) return -2;
         *ch++=0;
         host = buf[1] ? buf+1 : NULL;
         port = ch+1;
         if (!port[0]) return -2;
         hint.ai_family = AF_INET6;
      }
      //  
      else
      {
         //  Change : to 0
         char* ch = strchr(buf,':');
         if (!ch) return -2;
         *ch++ = 0;
         host = buf[0] ? buf : NULL;
         port = ch;
         if (!port[0]) return -2;
         hint.ai_family = AF_INET;
      }
      //  Use any interface
      if (!host) hint.ai_flags = AI_PASSIVE;

      //  Get address
      hint.ai_socktype = SOCK_STREAM;
      if (getaddrinfo(host,port,&hint,sa)) return -3;
      return host ? 0 : 1;
   }
}

/*
 *  Set socket to non-blocking
 */
static int setnonblock(int fd)
{
#ifdef UNIX
   long flag = fcntl(fd,F_GETFL) | O_NONBLOCK;
   return fcntl(fd,F_SETFL,flag);
#else
   return 0;
#endif
}

/*
 *  Set socket to keepalive
 */
static int setkeepalive(int fd)
{
#ifdef UNIX
   int flag=1;
   return setsockopt(fd,SOL_SOCKET,SO_KEEPALIVE,&flag,sizeof(flag));
#else
   return 0;
#endif
}

/* ------------------------------------------------------------------ */
/* ------------------------  TCP/IP Utilities  ---------------------- */
/* ------------------------------------------------------------------ */
/*
 *  Open TCP socket for listening
 *     sa is a pointer to an addrinfo structure (from hostport)
 *  RETURN
 *     socket id on success
 *     throws fatal error on failure
 */
static int open_tcp(addr* sa)
{
   int sock;                /*  TCP socket        */

   //  Open TCP socket
   sock = socket(sa->ai_family,sa->ai_socktype,sa->ai_protocol);
   if (sock<0) fatal("Cannot open socket\n");
   if (verbose>0) info("TCP socket %d\n",sock);

   //  Set non-blocking
   if (setnonblock(sock)<0) fatal("Cannot make socket non-blocking\n");

   //  Bind to port
   if (bind(sock,sa->ai_addr,sizeof(struct sockaddr))<0) fatal("Cannot bind socket to TCP port\n");
   if (verbose>0) info("TCP socket %d bound to port\n",sock);

   // Start accepting connections
   if (listen(sock,SERVER_QUEUE_SIZE)<0)
      fatal("Cannot listen to TCP %d port",sock);

   return sock;
}

/*
 *  Accept TCP connection
 *    sock0 is the listening port (from open_tcp)
 *  RETURN
 *    socket id for new port
 *    throws fatal error on failure
 */
static int accept_tcp(int sock0)
{
   struct sockaddr_in sin;             //  Socket Address
   socklen_t          l = sizeof(sin); //  Address length

   //  Accept connection
   int sock = accept(sock0,(struct sockaddr*)&sin,&l);
   //  Check result
   if (sock<0)
      fatal("Cannot accept TCP %d socket\n",sock0);
   else if (verbose>1)
      info("Accept TCP %d socket %d\n",sock0,sock);

   return sock;
}

/*
 *  Connect to TCP socket
 *     sa is a pointer to an addrinfo structure (from hostport)
 *  RETURN
 *     socket id >=0 on success
 *     -1 on failure
 */
static int connect_tcp(addr* sa)
{
   int k;
   int sock;

   // Create socket
   sock = socket(sa->ai_family,sa->ai_socktype,sa->ai_protocol);
   if (sock<0) fatal("Cannot open socket for connect\n");
   if (verbose>1) info("TCP connect socket %d\n",sock);

   //  Set TCP keepalive
   if (setkeepalive(sock)<0) fatal("Cannot make socket TCP keepalive\n");

   //  Initialize random number generator with time and PID for retries
   srandom(time(NULL)^getpid());

   // Try up to 16 times to establish connection
   for (k=0;k<16;k++)
   {
      //  If connect succeeds, return the socket number
      if (!connect(sock,sa->ai_addr,sizeof(struct sockaddr)))
      {
         if (verbose>1) info("TCP connect socket %d\n",sock);
         return sock;
      }
      //  Sleep a random number of milliseconds from 0-2047 (average 1s)
      int ms = random()&0x7FF;
      msleep_(&ms);
   }

   //  Connect failed
   if (verbose>0) info("Cannot connect to socket\n");
   close(sock);
   return -1;

}

/*
 *  Read len bytes from TCP socket
 *    sock is the TCP socket to read from
 *    buf  is the buffer into which to read
 *    len  is the number of bytes to read
 *  RETURN
 *    0  if read successful
 *    -1 on error (premature close)
 */
static int recv_tcp(int sock,void* buf,int len)
{
   char* ch=buf;  //  Pointer to next byte to receive
   if (verbose>2) info("TCP %d recv %d\n",sock,len);
   //  Loop until required number of bytes read
   while (len>0)
   {
      //  Read up to remaining number of bytes
      int n = read(sock,ch,len);
      //  Read error (closed?)
      if (n<0)
      {
         if (verbose>2) info("TCP %d recv fatal\n",sock);
         return -1;
      }
      //  NULL read (EOF?)
      else if (!n)
      {
         if (verbose>2) info("TCP %d recved EOF\n",sock);
         return -1;
      }
      //  Increment pointer and decrement remaining byte count
      if (verbose>2) info("TCP %d recved %d %d\n",sock,n,len);
      ch  += n;
      len -= n;
   }
   return 0;
}

/*
 *  Send len bytes to TCP socket
 *     sock is the socket to write to
 *     buf  is the message data
 *     len  is the length of the message
 *  RETURN
 *     0 on success
 *    -1 on failure
 */
static int send_tcp(int sock,void* buf,int len)
{
   if (verbose>2) info("TCP %d send %d\n",sock,len);
   //  Really verbose dumps message
   if (verbose>4)
   {
      int i;
      char* ch=buf;
      for (i=0;i<len;i++,ch++)
         if (*ch<33)
            info("<%d>",*ch);
         else
            info("%c",*ch);
      info("\n");
   }
   //  Write to socket
   return (write(sock,buf,len)<len) ? -1 : 0;
}

/*
 *  Reverse n bytes
 *     buf is the buffer reversed in place
 *     n is the total length of the buffer
 */
static void reverse(char* buf,int n)
{
   int i,j;
   for (i=0,j=n-1;i<n/2;i++,j--)
   {
      char t = buf[i];
      buf[i] = buf[j];
      buf[j] = t;
   }
}

/* ------------------------------------------------------------------ */
/* ------------------------  BEO TCP Interface  --------------------- */
/* ------------------------------------------------------------------ */
//  Killer fact:  LENTEXT must match dimension of NAME in BEOPRIV
#define LENTEXT  256
#define MAXSIZE 1024
#define DIMSIZE  12
#define MAXTAG    2
//  Node specific data
typedef struct
{
   int   sock;        //  Socket used to communicate with node
   char* buf[MAXTAG]; //  Buffer for POST receives
   char* ptr[MAXTAG]; //  Pointer for POST receives
   int   max[MAXTAG]; //  Size of POST receive buffer
   int   len[MAXTAG]; //  Length of data remaining in POST receive buffer
} node_t;
static node_t* node;      //  Node array (single entry on slaves)
static int num;           //  Number of active slaves
static int max;           //  Maximum number of slaves possible
static int rev=0;         //  True if reverse byte gender (set only on slave)
static char dim[DIMSIZE]; //  Array used to send Npar, Nobs and CSUM to slaves
static char text[LENTEXT];//  Array used to send slave description to master
static char dir[LENTEXT]; //  Array used for slave working directory

/*
 *  Open TCP/IP socket
 *    n is the maximum number of slaves the master can connect to
 *    name is the host:port string  (:port identifies the master)
 *    nn is the node number (master=0, slave>0)
 *  RETURN
 *    throws a fatal error if connection fails
 */
void tcpopen_(int* n,char* name,int* nn)
{
   //  Translate address
   addr* sa;
   int master = hostport(name,&sa);
   if (master==-3)
     fatal("Cannot resolve %s\n",name);
   else if (master<0)
     fatal("Invalid host:port %s\n",name);

   //  Master:  Open port to accept connections
   if (master)
   {
      int k;
      //  Allocate memory for maximum number of nodes
      max = *n;
      node = (node_t*)malloc(max*sizeof(node_t));
      //  Initialize all sockets to -1 (no connection)
      for (k=0;k<max;k++)
         node[k].sock = -1;
      //  Open one socket to listen for connections on selected port
      num = 1;
      node[0].sock = open_tcp(sa);
      //  Master is node number 0
      *nn = 0;
      //  Set idenitification string
      snprintf(text,LENTEXT,"MASTER");
   }
   //  Slave:  Connect to master
   else
   {
      short mi;
      int   runtime=0;
#ifdef UNIX
      struct utsname un;
#endif
      //  Connect to master node
      int sock = connect_tcp(sa);
      if (sock<0) fatal("Cannot connect to master\n");
      //  Allocate memory for one connection (to master)
      num = max = 1;
      node = (node_t*)malloc(sizeof(node_t));
      node[0].sock = sock;
      //  Determine byte gender from first thing sent from master
      if (recv_tcp(sock,&mi,2)) fatal("TCPRECV error\n");
      rev  = (mi!=1);
      //  Read node number for master for master
      if (recv_tcp(sock,nn,4))
         fatal("TCPRECV error\n");
      else if (rev)
         reverse((char*)nn,4);
      //  Get working directory
      if (!getcwd(dir,LENTEXT)) dir[0] = 0;
      //  Send node info to master
#ifdef UNIX
      if (uname(&un)<0) fatal("UNAME error\n");
      snprintf(text,LENTEXT,"%-32s %-8s %s %s",un.nodename,un.machine,un.sysname,dir);
#else
      snprintf(text,LENTEXT,"%s",dir);
#endif
      if (rev) reverse((char*)&runtime,4);
      if (send_tcp(sock,&runtime,4)) fatal("TCPSEND error\n");
      if (send_tcp(sock,text,LENTEXT)) fatal("TCPSEND error\n");
   }
}
/*
 *  Send n items of size l to id
 *     id  is the node number
 *     buf is the array
 *     l   is the size of the items being sent (in bytes)
 *     n   is the number of items being sent
 *  RETURN
 *     throws a fatal error if the paramaters are invalid
 *     fails silently on send error
 *  Remarks
 *     items are reversed when l>1 and rev is set (slave only)
 *     when reversing item size l must be less than MAXSIZE
 *     it is OK to send n items of l and receive as one n*l
 */
void tcpsend_(int* id,char* buf,int* l,int* n)
{
   //  Check parameters
   if (*id<0 || *id>=num)
      fatal("TCPSEND invalid node %d\n",*id);
   else if (node[*id].sock<0)
      fatal("TCPSEND closed socket %d\n",*id);
   else if (*l<0 || *l>MAXSIZE)
      fatal("TCPSEND invalid size %d\n",*l);
   else if (*n<0)
      fatal("TCPSEND invalid count %d\n",*n);
   //  Reverse and send items one at a time
   else if (rev && *l>1)
   {
      int i,j;
      for (i=0;i<*n;i++)
      {
         char rev[MAXSIZE];
         for (j=*l-1;j>=0;j--)
            rev[j] = *buf++;
         if (send_tcp(node[*id].sock,rev,*l)) info("TCPSEND failed\n");
      }
   }
   //  Send entire buffer
   else if (send_tcp(node[*id].sock,buf,(*l)*(*n)))
      info("TCPSEND failed\n");
}
/*
 *  Receive n items of size l to id
 *     id  is the node number
 *     buf is the array
 *     l   is the size of the items being received (in bytes)
 *     n   is the number of items being received
 *  RETURN
 *     throws a fatal error if receive fails
 *  Remarks
 *     items are reversed when l>1 and rev is set (slave only)
 */
void tcprecv_(int* id,char* buf,int* l,int* n)
{
   //  Check parameters
   if (*id<0 || *id>=num)
      fatal("TCPRECV invalid node %d\n",*id);
   else if (node[*id].sock<0)
      fatal("TCPRECV closed socket %d\n",*id);
   else if (*l<0)
      fatal("TCPRECV invalid size %d\n",*l);
   else if (*n<0)
      fatal("TCPRECV invalid count %d\n",*l);
   //  Receive buffer
   else if (recv_tcp(node[*id].sock,buf,*l * *n))
      fatal("TCPRECV error\n");
   //  Reverse items if necessary
   else if (rev && *l>1)
   {
      int i;
      for (i=0;i<*n;i++)
         reverse(buf+(*l)*i,*l);
   }
}
/*
 *  Post receive of n items of size l to id
 *     id  is the node number
 *     l   is the size of the items being received (in bytes)
 *     n   is the number of items being received
 *  RETURN
 *     throws a fatal error if receive fails
 *  Remarks
 *     Receive buffer is resized as necessary
 *     Reversing is not implemented as this is used on master only
 */
void tcppost_(int* id,int* tag,int* l,int* n)
{
   //  Check parameters
   if (*id<0 || *id>=num)
      fatal("TCPPOST invalid node %d\n",*id);
   else if (node[*id].sock<0)
      fatal("TCPPOST closed socket %d\n",*id);
   else if (*tag<0 || *tag>=MAXTAG)
      fatal("TCPPOST invalid tag %d\n",*tag);
   else if (*l<0)
      fatal("TCPPOST invalid size %d\n",*l);
   else if (*n<0)
      fatal("TCPPOST invalid count %d\n",*n);
   //  Set structure for receive
   else
   {
      int N = (*l)*(*n);
      //  Grow buffer if necesary
      if (N>node[*id].max[*tag])
      {
         node[*id].max[*tag] = N;
         node[*id].buf[*tag] = realloc(node[*id].buf[*tag],N);
         if (!node[*id].buf[*tag]) fatal("TCPPOST realloc error tag %d size %d\n",*tag,N);
      }
      node[*id].ptr[*tag] = node[*id].buf[*tag];
      node[*id].len[*tag] = N;
   }
}

/*
 *  Post receive of n items of size l to id
 *     id  is the node number
 *     l   is the size of the items being received (in bytes)
 *     n   is the number of items being received
 *  RETURN
 *     throws a fatal error if receive fails
 *  Remarks
 *     Receive buffer is resized as necessary
 *     Reversing is not implemented as this is used on master only
 */
void tcpload_(int* id,int* tag,char* buf,int* l,int* n)
{
   int N = (*l)*(*n);
   //  Check parameters
   if (*id<0 || *id>=num)
      fatal("TCPLOAD invalid node %d\n",*id);
   else if (node[*id].sock<0)
      fatal("TCPLOAD closed socket %d\n",*id);
   else if (*tag<0 || *tag>=MAXTAG)
      fatal("TCPLOAD invalid tag %d\n",*tag);
   else if (*l<0)
      fatal("TCPLOAD invalid size %d\n",*l);
   else if (*n<0 || N>node[*id].max[*tag])
      fatal("TCPLOAD invalid count %d\n",*n);
   //  Copy data
   else
      memcpy(buf,node[*id].buf[*tag],N);
}

/*
 *  Check of posted receive has completed
 *     id is the node number
 *     flag is the returned value
 *       1 => receive completed
 *       0 => not yet complete
 *      -1 => error (closed connection)
 */
void tcptest_(int* id,int* flag)
{
   //  Check parameters
   if (*id<0 || *id>=num)
      fatal("TCPTEST invalid node %d\n",*id);
   //  Get outstanding receives 
   else
   {
      int k;
      for (k=0;k<MAXTAG;k++)
      {
         int n;
         fd_set set;
         struct timeval timer = {0,0};
         //  Socket is closed
         if (node[*id].sock<0) break;
         //  Nothing more - skip to next
         if (node[*id].len[k]<=0) continue;
         //  Check if socket has data
         FD_ZERO(&set);
         FD_SET(node[*id].sock ,&set);
         //  If n<0 it will be caught below
         n = select(node[*id].sock+1 , &set , NULL , NULL , &timer);
         //  Timeout (break to guarantee in order reception)
         if (n==0)
           break;
         //  Attempt read
         else if (n>0)
            n = read(node[*id].sock,node[*id].ptr[k],node[*id].len[k]);
         //  Read Error (0 is an error because select said there is something)
         if (n<=0)
         {
            int j;
            close(node[*id].sock);
            node[*id].sock = -1;
            //  Free all buffers
            for (j=0;j<MAXTAG;j++)
            {
               free(node[*id].buf[j]);
               node[*id].buf[j] = NULL;
               node[*id].max[j] = 0;
            }
            break;
         }
         //  Unpack data
         else
         {
            node[*id].ptr[k] += n;
            node[*id].len[k] -= n;
            //  Break here if data is left since reads must be in sequence
            if (node[*id].len[k]>0) break;
         }
      }
      //  Set return flag
      if (node[*id].sock<0)
         *flag = -1;
      else
      {
         *flag = 1;
         for (k=0;k<MAXTAG && *flag;k++)
            if (node[*id].len[k]>0) *flag = 0;
      }
   }
}

/*
 *  Check for new connections
 *     live  is an integer array used to show node status
 *     wall  is an double array used to measure execution wall time
 *     n     is the highest active node 
 *  RETURN
 *     live=0 means offline
 *     live<0 means idle
 *  Remarks
 *     This should be called from the master only
 */
void tcpnode_(int live[],double wall[],char name[],int* n)
{
   int k=1;
   //  Loop while pending connections
   while (k)
   {
      //  Check for a new connection with immediate timeout
      int sock0 = node[0].sock;
      struct timeval timer = {0,0};
      fd_set set;
      FD_ZERO(&set);
      FD_SET(sock0,&set);
      if (verbose>1) info("Select\n");
      if (select(sock0+1 , &set , NULL , NULL , &timer) && FD_ISSET(sock0,&set))
      {
         short  mi=1;
         int    i,j,k,runtime;
         //  Accept connection
         int sock = accept_tcp(sock0);
         //  Find free socket to assign connection to
         for (i=1,j=0;i<num && !j;i++)
            if (node[i].sock<0) j = i;
         if (!j) j = num++;
         if (num>=max) fatal("Too many connections\n");
         //  Send node data && get node info
         if (send_tcp(sock,&mi,2) || send_tcp(sock,&j,4) || send_tcp(sock,dim,DIMSIZE) ||
             recv_tcp(sock,&runtime,4) || recv_tcp(sock,text,LENTEXT))
            close(sock); //  Failed so reset
         //  Mark node as idle
         else
         {
            //  Store socket
            node[j].sock = sock;
            //  Initialize receive buffers
            for (i=0;i<MAXTAG;i++)
            {
               node[j].buf[i] = NULL;
               node[j].max[i] = 0;
               node[j].len[i] = 0;
            }
            //  Set fast idle
            live[j-1] = -1;
            wall[j-1] = runtime;
            //  Announce
            for (i=k=0;i<LENTEXT;i++)
               name[LENTEXT*(j-1)+i] = text[k] ? text[k++] : ' ';
            fprintf(stderr,"Node %4d runtime=%8d %s\n",j,runtime,text);
         }
      }
      //  No pending connection so end loop
      else
         k = 0;
   }
   //  Return number of potential live connections
   *n = num-1;
}
/*
 *  Close all connections
 */
void tcpclose_(void)
{
   int k;
   for (k=0;k<num;k++)
      if (node[k].sock>=0)
         close(node[k].sock);
   free(node);
   num = max = 0;
}
/*
 *  Set dimensions
 */
void tcpcast_(int* np,int* nt,int* csum)
{
   memcpy(dim+0,(char*)np,4);
   memcpy(dim+4,(char*)nt,4);
   memcpy(dim+8,(char*)csum,4);
}
/*
 *  Fatal error with node text
 */
void fatal_(char* msg)
{
   fatal("%s: %s\n",text,msg);
}
/*
 *  Change working directory
 */
void setdir_(const char* dir)
{
   //  Create directory if we can't chdir to it
#ifdef UNIX
   if (chdir(dir) && mkdir(dir,0777)) fatal("Cannot create directory %s\n",dir);
#else
   if (chdir(dir) && mkdir(dir)) fatal("Cannot create directory %s\n",dir);
#endif
   //  Cannot chdir
   if (chdir(dir)) fatal("Cannot change directory to %s\n",dir);
}
