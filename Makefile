NAME := open_draw

build:
	docker build . -t ${NAME}

shell:
	docker run -v $(shell pwd):/tax_to_json -it --entrypoint /bin/bash ${NAME}

up:
	docker run --device=/dev/video0:/dev/video0 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=${DISPLAY} \
	--env="QT_X11_NO_MITSHM=1" -p 5000:5000 -p 8888:8888 -it open_draw /bin/bash

tests:
	docker run -v $(shell pwd):/app  ${NAME} python -m unittest
