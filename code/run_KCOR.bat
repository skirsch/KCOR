@echo off
REM Batch file to run KCOR analysis

echo Running KCOR analysis...
python KCORv4.py ../../Czech/data/KCOR_output.xlsx KCOR_processed_REAL.xlsx
echo KCOR analysis complete!

pause
