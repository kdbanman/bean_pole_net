### TODO

- add skip stride
  - always attach the immediately behind layer
  - attach behind layers (up to max skip) if they are a multiple of skip stride
- add skip stride exponential backoff
  - always attach the immediately behind layer
  - attach behind layers at exponentially increasing intervals
  - up to max skip maybe?  exponential backoff should prevent param explosion
