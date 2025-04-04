@echo off
REM This part of batch file added by SVDAPREP
REM
REM Delete model input files.
del A.snn_1 > nul
REM
REM Run PARCALC to compute base parameters from super parameters.
"e:\models\pest\parcalc" > nul
REM
REM Run PICALC to compute base parameter prior information.
"e:\models\pest\picalc" > nul
REM
REM The following is copied directly from file A_bat1.bat
REM
e:\models\mf2k\mf2k_h5.exe < A.in_1
exit
