@echo off
REM This part of batch file added by SVDAPREP
REM
REM Delete model input files.
del A.snn_1 > nul
REM
REM Run PARCALC to compute base parameters from super parameters.
"c:\program files\gms 10.0\models\pest\parcalc" > nul
REM
REM Run PICALC to compute base parameter prior information.
"c:\program files\gms 10.0\models\pest\picalc" > nul
REM
REM The following is copied directly from file A_bat1.bat
REM
"c:\program files\gms 10.0\models\mf2k\64\mf2k_h5_64.exe" < A.in_1
exit
