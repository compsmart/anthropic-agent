@echo off
echo Running AGI Agent Tool Tests...
echo.

REM Run automated tests
echo === AUTOMATED TESTS ===
C:/Users/Brad/.pyenv/pyenv-win/versions/3.11.7/python.exe test_tools_interactive.py

echo.
echo === INTERACTIVE MODE ===
echo Press Ctrl+C to exit interactive mode
C:/Users/Brad/.pyenv/pyenv-win/versions/3.11.7/python.exe test_tools_interactive.py --interactive

pause
