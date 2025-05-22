This directory contains the files needed to install PHAST for Windows 
and Model Viewer, and contains an example and tutorial for the 
use of Phast for Windows (P4W).

To install P4W: Double click on phast4windows-0.14.4-5803.msi

To install PHAST and Model Viewer: Double click on phast-2.2.0-5314.msi
Model Viewer is installed with the batch version of PHAST, but is 
the only part of the batch installation that will be used. 

Phast for Windows can be executed as follows:
Start->All Programs->Phast for Windows 0.14.4->Phast for Windows 0.14.4

Model Viewer can be executed as follows:
Start->All Programs->Phast 2.2.5314->Model Viewer 1.3->Model Viewer

An example based on work at Cape Cod is included for your information.
The example is divided into two parts: (1) a flow simulation, and 
(2) a reactive transport simulation of phosphorus (conservative transport
in this example), and nitrogen (reaction of NH4+ with O2 to form NO3-.

Flow modeling:

Flow simulations can be run or developed in several ways. If you just
want to view the interface and run the flow simulation, you can open the file
flow.p4w.

If you want to develop the entire simulation from scratch, you can follow
the directions in HowToFlow.ppt.

Another alternative is to import the flow.trans.dat file.


Transport modeling:

Again there are several ways to proceed. The simplest is to open the
transport.p4w file.

Directions for extending the flow simulation to reactive transport are given in
HowToTransport.ppt.

Again it is possible to import the transport.trans.dat file and proceed from 
there. 

The transport.chem.dat is a PHREEQC file has the chemistry definitions used in the
reactive-transport simulation.


Additional information:

Manuals for PHAST version 2 and PHREEQC are installed from the .msi files and can be found
through the start menu->All Programs.

The PHASToverview.ppt file has some basic information about the finite difference
grid of PHAST and the way that data are processed.

The NumericalDispersion.ppt file contains a simplified description of numerical 
dispersion and how to estimate it for a given simulation.

A site map for Cape Cod is included in sitemap.jpg and its registration information in 
sitemap.jgw. The HowToFlow.ppt file describes how to use this jpg file as 
a base map in P4W. 