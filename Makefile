NAME	:=	bathroom_wall


train:
	python .\train.py -s ..\data\$(NAME)_orig\ -m ..\data\$(NAME)_res\tmp_20000 --iterations 20000


render:
	python myrender.py -m ..\data\$(NAME)_res\30000\ --skip_test


view:
	..\viewers\bin\SIBR_gaussianViewer_app.exe -m ..\data\$(NAME)_res\tmp