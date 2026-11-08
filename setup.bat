@echo off
rem One command from a fresh clone to a built, tested k3.exe on Windows.
rem Runs scripts\setup-windows.ps1; see that file for what it downloads and checks.
rem Usage: setup.bat [-SkipTests]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-windows.ps1" %*
exit /b %ERRORLEVEL%
