@echo off
title Neev Production Server (24/7 with 4GB RAM Guard)
echo Starting Neev 24/7 Server with Never-Sleep and 4GB RAM Guardrail...
powershell -ExecutionPolicy Bypass -File "%~dp0run_local_server.ps1"
pause

