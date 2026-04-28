# Makefile for IIT BHU M.Tech Thesis
# Usage:
#   make        — compile PDF (pdflatex + bibtex + pdflatex x2)
#   make clean  — remove intermediate files
#   make open   — open the PDF

MAIN = thesis
LATEX = pdflatex
BIBTEX = bibtex
FLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all clean open

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(FLAGS) $(MAIN).tex
	$(LATEX) $(FLAGS) $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.lof *.lot *.toc *.out *.fls *.fdb_latexmk

open: $(MAIN).pdf
	open $(MAIN).pdf
