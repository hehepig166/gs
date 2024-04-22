name	:=	bathroom_wall

train:
	python .\train.py -s ..\data\$(name)_orig\ -m ..\data\$(name)_res\2000 --iterations 2000


render:
	python myrender.py -m ..\data\$(name)_res\30000\ --skip_test


