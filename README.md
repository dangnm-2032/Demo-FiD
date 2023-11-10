# DemoFiD

Features:
- English language
- Read PDF, TXT (in text format not image)
- Read CSV, JSON (file must have 3 field: id, title, content)
- Speech input/output

How to use:
- wget --no-check-certificate https://dl.xpdfreader.com/xpdf-tools-linux-4.04.tar.gz &&\
	tar -xvf xpdf-tools-linux-4.04.tar.gz && sudo cp xpdf-tools-linux-4.04/bin64/pdftotext /usr/local/bin &&\
	pip install --upgrade pip &&\
	pip install -r requirements.txt
- run main.py
