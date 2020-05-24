NAME := open_draw

build:
	docker build . -t ${NAME}

shell:
	docker run --device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1  \
	-v /tmp/.X11-unix:/tmp/.X11-unix  \
	-v $(shell pwd):/app \
	-e DISPLAY=${DISPLAY} \
	-e  QT_X11_NO_MITSHM=1 \
	-it open_draw /bin/bash

calibrate:
	docker run --device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1  \
	-v /tmp/.X11-unix:/tmp/.X11-unix  \
	-v $(shell pwd):/app \
	-e DISPLAY=${DISPLAY} \
	-e  QT_X11_NO_MITSHM=1 \
	-it open_draw python calibrator/calibrate.py

sure:
	docker run --device=/dev/video0:/dev/video0 --device=/dev/video1:/dev/video1  \
	-v /tmp/.X11-unix:/tmp/.X11-unix  \
	-v $(shell pwd):/app \
	-e DISPLAY=${DISPLAY} \
	-e  QT_X11_NO_MITSHM=1 \
	-it open_draw python calibrator/clean.py

tests:
	docker run -v $(shell pwd):/app  ${NAME} python -m unittest
