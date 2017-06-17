### TODO

- look for coadaptations in weights, try dropout if so

- add skip stride exponential backoff
  - always attach the immediately behind layer
  - attach behind layers at exponentially increasing intervals
  - up to max skip maybe?  exponential backoff should prevent param explosion

- use imageio to save input to output along bean pole as gif

- grab a batch of 50 images to visualize training progress on
  - every 1000 steps, write them to a summary collection "progress_images"
  - input, output, each module activation, and each module internal activation
  - use imageio to save same gif as above

- try recurrent CNN
