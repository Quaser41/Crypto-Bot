@echo off
title Update Crypto-Bot Repository
cd /d "%~dp0"

for /f "delims=" %%i in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%i"
echo Updating branch %BRANCH%...
git fetch origin
if %errorlevel% neq 0 (
    echo Failed to fetch from origin.
    goto end
)
git reset --hard origin/%BRANCH%
if %errorlevel% neq 0 (
    echo Failed to reset local branch. Check that origin/%BRANCH% exists.
    goto end
)

echo Update complete.
:end
pause
