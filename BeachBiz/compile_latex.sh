#!/bin/bash

# Name of your LaTeX file (without the .tex extension)
filename="BeachBiz"

# Compiling the LaTeX file into a PDF
pdflatex $filename.tex
pdflatex $filename.tex  # Running twice to resolve references (including table of contents ones) , if any

makeindex $filename

pdflatex $filename.tex  # Running again for updates to index

# Cleaning up auxiliary files (optional)
rm $filename.aux $filename.log $filename.out

pandoc BeachBiz.tex -o BeachBiz.md
pandoc BeachBiz.md -o BeachBiz.html

