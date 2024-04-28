NAME		:=	bathroom_wall
SAVE_DIR	:=	tmp
ITER_NUM	:=	9000


# Check the operating system
ifeq ($(OS),Windows_NT)
# Windows commands
	TRAIN_CMD := python .\train.py -s ..\data\$(NAME)_orig\ -m ..\data\$(NAME)_res\$(SAVE_DIR) --iterations $(ITER_NUM)
	RENDER_CMD := python myrender.py -m ..\data\$(NAME)_res\$(SAVE_DIR) --skip_test
	VIEW_CMD := ..\viewers\bin\SIBR_gaussianViewer_app.exe -m ..\data\$(NAME)_res\$(SAVE_DIR)
	COMPILE_CMD := cd .\submodules\diff-gaussian-rasterization\ &  python setup.py install
else
# Linux commands
	TRAIN_CMD := python ./train.py -s ../data/$(NAME)_orig/ -m ../data/$(NAME)_res/$(SAVE_DIR) --iterations $(ITER_NUM)
	RENDER_CMD := python myrender.py -m ../data/$(NAME)_res/$(SAVE_DIR) --skip_test
	VIEW_CMD := ../viewers/bin/SIBR_gaussianViewer_app.exe -m ../data/$(NAME)_res/$(SAVE_DIR)
	COMPILE_CMD := cd ./submodules/diff-gaussian-rasterization/ ; python setup.py install
endif

# Define the targets
train:
	$(TRAIN_CMD)

render:
	$(RENDER_CMD)

view:
	$(VIEW_CMD)

compile:
	$(COMPILE_CMD)