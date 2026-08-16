@echo off
setlocal
cd /d "%~dp0"

echo ==========================================
echo   SwitchBoard - push to GitHub
echo ==========================================
echo.

where git >nul 2>&1
if errorlevel 1 (
  echo [X] git is not installed, or not on your PATH.
  echo     Install it from https://git-scm.com/download/win
  echo     then close this window, reopen it, and run this again.
  echo.
  pause
  exit /b 1
)

echo [1/4] Current repo state:
git status --short
echo.

echo [2/4] Committing the corrected docs...
git add README.md INTERVIEW_GUIDE.md
git commit -m "Correct test-count claim: 46 tests, not 99" >nul 2>&1
if errorlevel 1 echo       Nothing new to commit - continuing.
echo.

echo [3/4] Commits ready to push:
git log --oneline -3
echo.

echo [4/4] Pushing to origin/main...
echo.
git push -u origin main
if errorlevel 1 goto failed

echo.
echo ==========================================
echo   DONE. Your code is live at:
echo   https://github.com/Manav-Jain-2000/SwitchBoard
echo ==========================================
echo.
pause
exit /b 0

:failed
echo.
echo ==========================================
echo   PUSH FAILED - read the git error above,
echo   then match it to one of these:
echo ==========================================
echo.
echo  "Authentication failed", or it asked for a password
echo     FIX:  gh auth login
echo           Pick GitHub.com, then HTTPS, then browser login.
echo           Then run this script again.
echo.
echo  "Updates were rejected", or "fetch first"
echo     The remote has a starter commit that yours does not.
echo     Your history is the real one, so overwrite it:
echo     FIX:  git push -u origin main --force
echo.
echo  "Repository not found"
echo     Either you are signed in as the wrong account, or the
echo     repo does not exist yet. Check which account you are:
echo     FIX:  gh api user --jq .login
echo           That must print  Manav-Jain-2000
echo     If the repo is missing, create it empty:
echo     FIX:  gh repo create Manav-Jain-2000/SwitchBoard --private
echo.
pause
exit /b 1
