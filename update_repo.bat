@echo off
title Update Crypto-Bot Repository
cd /d "%~dp0"

echo Updating main branch...
git fetch origin
if %errorlevel% neq 0 (
    echo Failed to fetch from origin.
    goto end
)
git reset --hard origin/main
if %errorlevel% neq 0 (
    echo Failed to reset local branch to origin/main.
    goto end
)

echo Update of main branch complete.
:end
pause
