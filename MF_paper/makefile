TEXFILE=PT_Networks_Mem

all:
	latex ${TEXFILE}.tex
	bibtex ${TEXFILE}
	latex ${TEXFILE}.tex
	latex ${TEXFILE}.tex
	dvips ${TEXFILE}.dvi

clean:
	rm ${TEXFILE}.ps
	rm ${TEXFILE}.dvi
	rm ${TEXFILE}.aux
	rm ${TEXFILE}.bbl
	rm ${TEXFILE}.blg
	rm ${TEXFILE}.log
	rm ${TEXFILE}Notes.bib
