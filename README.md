### TODO

- look for coadaptations in weights, try dropout if so

- make forward pass runner script
  - python forward.py [run_name] [input_image]
  - if no input image, prompt for on screen draw
  - if no run name, use most recent
  - use imageio to save input to output along bean pole as gif

- try incrementally adding layers during training after <0.25MSE breakthroughs

- add skip stride exponential backoff
  - always attach the immediately behind layer
  - attach behind layers at exponentially increasing intervals
  - up to max skip maybe?  exponential backoff should prevent param explosion

- grab a batch of 50 images to visualize training progress on
  - every 1000 steps, write them to a summary collection "progress_images"
  - input, output, each module activation, and each module internal activation
  - use imageio to save same gif as above

- try recurrent CNN
