NAME	:=	bathroom_wall


# Check the operating system
ifeq ($(OS),Windows_NT)
# Windows commands
	TRAIN_CMD := python .\train.py -s ..\data\$(NAME)_orig\ -m ..\data\$(NAME)_res\tmp_20000 --iterations 20000
	RENDER_CMD := python myrender.py -m ..\data\$(NAME)_res\30000\ --skip_test
	VIEW_CMD := ..\viewers\bin\SIBR_gaussianViewer_app.exe -m ..\data\$(NAME)_res\tmp
else
# Linux commands
	TRAIN_CMD := python ./train.py -s ../data/$(NAME)_orig/ -m ../data/$(NAME)_res/tmp_20000 --iterations 20000
	RENDER_CMD := python myrender.py -m ../data/$(NAME)_res/30000/ --skip_test
	VIEW_CMD := ../viewers/bin/SIBR_gaussianViewer_app.exe -m ../data/$(NAME)_res/tmp
endif

# Define the targets
train:
	$(TRAIN_CMD)

render:
	$(RENDER_CMD)

view:
	$(VIEW_CMD)