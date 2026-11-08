#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p build
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build SSU_Data_Dash_5min.tex
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build SSU_Data_Dash_5min.tex
cp build/SSU_Data_Dash_5min.pdf SSU_Data_Dash_5min.pdf
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build -jobname=SSU_Data_Dash_with_Backup '\def\WithAppendix{1}\input{SSU_Data_Dash_5min.tex}'
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=build -jobname=SSU_Data_Dash_with_Backup '\def\WithAppendix{1}\input{SSU_Data_Dash_5min.tex}'
cp build/SSU_Data_Dash_with_Backup.pdf SSU_Data_Dash_with_Backup.pdf
