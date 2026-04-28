@echo off
rem Double-click shim for start_web.ps1.
rem Runs PowerShell with -ExecutionPolicy Bypass at PROCESS scope only -
rem this does not modify machine policy and does not require admin.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_web.ps1" %*
if errorlevel 1 pause
