README.TXT


             Modpath Output Examiner - Version: 1.0.00

The Modpath Output Examiner is a Microsoft Windows application for 
viewing and analyzing particle-tracking Output from MODPATH Version 6. 

NOTE: Any use of trade, product or firm names is for descriptive purposes 
      only and does not imply endorsement by the U.S. Government.

This distribution contains a compiled version of the Modpath Output Examiner
application that can be executed on personal computers using the Microsoft 
Windows XP or Windows 7 operating systems.  In addition to the executable  
files, source code also is provided. The source code is packaged as a set of
Microsoft Visual Studio solutions. The application can be rebuilt using
Microsoft Visual Studio 8.

IMPORTANT: Users should review the file TM6A41_Modpath6.pdf, which contains 
the user guide for MODPATH version 6 as well as documentation for the 
Modpath Output Examiner application. Users also should review the file 
release.txt, which describes changes that have been introduced into 
MODPATH with each official release; these changes may substantially affect 
users.

Instructions for installation, execution, and testing of MODPATH are
provided below.


                            TABLE OF CONTENTS

                         A. DISTRIBUTION FILE
                         B. INSTALLING
                         C. EXECUTING MODPATH OUTPUT EXAMINER
                         D. TESTING
                         E. COMPILING

A. DISTRIBUTION FILE

The files for this distribution are provided in a ZIP archive file named:

  ModpathOutputExaminer.1_0_00.zip
  
To extract the files, select a directory and extract the zip file to the 
selected directory.The following directory structure will be created:

  |-- ModpathOutputExaminer.1_0
      |--bin               ; executable files for Modpath Output Examiner
      |--doc               ; User guide for MODPATH version 6
      |--example-out       ; Example simulations with input and output
      |--src               ; Source code for Modpath Output Examiner.
                             The directory src contains a number of
                             subdirectories that are not listed here.
                             Those subdirectories contain all of the 
                             source code components needed to build
                             the executable application with Microsoft
                             Visual Studio.
    
It is recommended that no user files are kept in the ModpathOutputExaminer.1_0
directory structure. If you do plan to put your own files in the 
ModpathOutputExaminer.1_0 directory structure, do so only by creating additional 
subdirectories.

The documentation for MODPATH version 6 and the Modpath Output Examiner is a 
Portable Document Format (PDF) file. PDF files are readable and printable on 
various computer platforms using Acrobat Reader from Adobe. The Acrobat Reader 
is freely available from the following World Wide Web site:

      http://www.adobe.com/


B. INSTALLING

The Modpath Output Examiner application consists of a primary application 
executable file (ModpathOutputExaminer.exe) and several DLL library files.
Those files are located together in the directory, 
ModpathOutputExaminer.1_0\bin. For the ModpathOutputExaminer appliction to
work properly, all of those files must be located in the same directory.
However, that directory may be copied to a new location and also may be 
renamed, if desired. The only requirement is that all the files remain 
together in the same directory. 

The application is built using the Microsoft .NET 3.5 framework. As of 2012, 
most computers running Microsoft Windows already have the .NET 3.5 framework 
installed. However, if necessary, it can be downloaded for free from 
Microsoft and installed. If a computer has the .NET 3.5 framework 
installed, all that is necessary to install the Modpath Output Examiner 
application is to copy the files to a disk drive on the computer.

C. EXECUTING MODPATH OUTPUT EXAMINER

The Modpath Output Examiner is a Microsoft Windows application.
To execute the application, double-click the icon for the file 
ModpathOutputExaminer.exe in Windows Explorer. As an alternative, 
s shortcut to the executable file may be created and placed in a convenient 
location, such as the user desktop. The application may then be executed by 
double-clicking the icon for the shortcut.                     

D. TESTING

A complete set of examples with output is provided in the directory, 
ModpathOutputExaminer.1_0\example-out. To run a specific example
simulations, start the Modpath Output Examiner application and open the
simulation file for the specific simulation. If you have a shortcut on
the desktop for ModpathOutputExaminer.exe, you may also drag the MODPATH
simulation file from Windows Explorer and drop it on the shortcut icon
to start the application and open the simulation file.

E. COMPILING

The source code for the Modpath Output Examiner is provided in the
form of three Microsoft Visual Stuio 8 solutions:

  1. PumaFramework
  2. ModflowTrainingTools
  3. ModpathOutputExaminer
  
The PumaFramework solution is composed of several projects that produce a
number of DLL libraries. The ModflowTraining Tools solution
produces a DLL that contains common components used by applications
that are (or will be) part of the ModflowTrainingTools suite. 
The ModpathOutputExaminer solution produces an end-user Windows
executable file (exe). The combination of the Windows executable
file and the DLL libraries constitute the Modpath Output Examiner
application.

Because the Windows applications are linked together by sequential 
dependencies, it is important to compile and build them in the
order listed below:

  Step 1. Compile and build PumaFramework
  Step 2. Compile and build ModflowTrainingTools
  Step 3. Compile and build ModpathOutputExaminer

Any change to the PumaFramework code requires a complete
recompilation of the solutions in the order listed above (steps 1, 2, and 3). 
A change only to the ModflowTrainingTools code requires steps 2 and 3 only. 
A change only to the ModpathOutputExaminer code requires step 3 only.

The separate projects that make up the PumaFramework have complex 
interdependencies. Consequently, the PumaFramework solution should 
always be used to compile and build those projects as a group to assure
that all of the DLL libraries are consistent with one another. The 
PumaFramework projects should not be compiled and built separately.


