main.pdf: *.tex
	xelatex -synctex=1 -pdf main.tex && \
	rm -rf *.aux *.log *.out *.bbl *.blg *.fdb_latexmk *.fls;

minimal:
	xelatex main.tex

clean:
	$(RM)  *.log *.aux \
	*.cfg *.glo *.idx *.toc \
	*.ilg *.ind *.out *.lof \
	*.lot *.bbl *.blg *.gls *.cut *.hd \
	*.dvi *.ps *.thm *.tgz *.zip *.rpi \
	*.d *.fls *.*.make *.fdb_latexmk *.run.xml \
	*.synctex.gz *.bcf
	$(RM) main.pdf
