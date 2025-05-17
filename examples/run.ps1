mypy --strict --follow-untyped-imports .\simple.py ..\src

if ($?) {
    python simple.py
}
