### TODO

- organize output images with some prefixes

- look for coadaptations in weights, try dropout if so

- add skip stride exponential backoff
  - always attach the immediately behind layer
  - attach behind layers at exponentially increasing intervals
  - up to max skip maybe?  exponential backoff should prevent param explosion

- try recurrent CNN
