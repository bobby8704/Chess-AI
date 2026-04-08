@echo off
cd /d "C:\Users\bobvu\Desktop\Dup_Chess_project"
echo ======================================
echo Chess AI Overnight Training
echo ======================================
echo.
echo Starting 8-hour training session...
echo Press Ctrl+C to stop at any time.
echo.
".venv\Scripts\python.exe" -u training\train_reliable.py --hours 8
echo.
echo Training complete!
pause
