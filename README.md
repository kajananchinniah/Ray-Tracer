# Ray-Tracer
Code is heavily inspired / taken at parts from the tutorial:
https://raytracing.github.io/books/RayTracingInOneWeekend.html but cuda
accelerated (although there are a few notable differences, such as how images are treated, lack of polymorphism in my code, etc)

Goal:
- Practice CUDA

Dependencies:
- CUDA + NVCC compiler
- OpenCV (only for image I/O)
- Gtest (tests)

Tested on:
- GPU: GeForce GTX 1050M

Note about the tests:
- Changed to testing to ensure behavior of my images don't change with additions
  rather than testing against website due to randomness. I instead made sure
  they looked visually similar.

Reference: https://raytracing.github.io/books/RayTracingInOneWeekend.html
