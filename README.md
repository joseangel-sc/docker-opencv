Inspired by [this](https://www.learnopencv.com/creating-a-virtual-pen-and-eraser-with-opencv/)
post, from the code [here](https://github.com/spmallick/learnopencv/tree/master/Creating-a-Virtual-Pen-and-Eraser)

I wanted to be able to use opencv inside a docker container 

Running: 

Have docker installed 
now `make build`

This was only tested on Kubuntu, but should work on other OS's 
`sudo xhost +`

and run it with a `make up`

If you want to debug, just do a `make shell` 

To calibrate the color of your pen use the `make calibrate` command, move the sliders until you only see 
the 'pen' you are going to be using. 

When you achieve this, type `s` and exit the container 

Just to make sure that the parameters are fine, type `make sure`, this is a cleaner and you should *almost* 
only see your 'pen'



