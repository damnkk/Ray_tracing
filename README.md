# Path_tracer
Path_tracer is a Path_tracing Off-line Renderer based on Disney principle's PBR material and some of basic noise reduction methods.
Which is implemented by OpenGL and C++.
# OverView

In this object , our workflow is simple scene & simple material --> complex scene & simple material --> complex scene & complex material(low quality) --> complex scene & complex material (high quality).

## part 1. basic ray tracing

In this part, we implemented a basic ray tracing program with  C++ on CPU, then save the final result as a png image.
<div align="center"><img src="ReadMe.asset/pathTracing.gif" width="256"><img src="ReadMe.asset/sample_4096.png" width="256"></div>

## part 2. BVH Accelerate Struct
In the last part,we had to go through all the triangles in the scene to get the path's hit result.For accelerate this processing,we implemented the BVH acceletrating structure in this part.Further more,we use SAH algorithm to optimize this structure to make sure the program can work in the scene which has tons of triangles.<div align="enter">
 <div align="center"><img src="ReadMe.asset/normalBVH.PNG" width="256"><img src="ReadMe.asset/SAHBVH.PNG" width="256"><img src="ReadMe.asset/三角形求交效果.PNG" width="256"></div>
&emsp;&emsp;&emsp;AABB-Box of BVH with out SAH(left) &emsp;&emsp;&emsp;    AABB-Box of BVH with SAH(right)&emsp; &emsp;&emsp;&emsp; Hit Result
 
## part 3. OpenGL ray tracing

Using OpenGL's fragment shader to run accelerate program. Transfer BVH and triangles in texture buffer, then send to shader. Finally tracing each pixel progressively, then display the dynamic tracing process in screen.

<div align="center"><img src="ReadMe.asset/part3.gif" width="256"></div>

## part 4. disney principle's BRDF

Learning micro facet theory, using Disney principle's BRDF to rendering various of physical material. <div align="enter">
For the low roughness surface, the hemispherical uniform sampling efficiency is very low, so there are many noise points in the image.

<div align="center"><img src="ReadMe.asset/part4.PNG" width="256"></div>

## part 5. Importance Sampling & Low Discrepancy Sequence

Methods to denoise, accelerate fitting progress.

Low Discrepancy Sequence (Sobol) :

<div align="center"><img src="ReadMe.asset/sobol.gif" width="320"> <img src="ReadMe.asset/fake_rand().gif" width="320"></div>

Importance Sampling, diffuse (left) and BRDF (right) :

<div align="center"><img src="ReadMe.asset/800c3511ae8043c08e3e43cb1e7ef8f6.png" width="512"></div>

Importance Sampling for HDR envmap :

<div align="center"><img src="ReadMe.asset/image-20211023172136160.png" width="512"></div>

Multi Importance Sampling with Heuristic power :

<div align="center"><img src="ReadMe.asset/29a21706e2664d57a2ca9a6089da632a.png" width="320"></div>


## part 6. Display
 <div align="center"><img src="ReadMe.asset/捕获 (2).PNG" width="256"><img src="ReadMe.asset/捕获2 (2).PNG" width="256"><img src="ReadMe.asset/捕获3 (2).PNG" width="256"></div><div align="enter">
  <div align="center"><img src="ReadMe.asset/捕获4 (2).PNG" width="256"><img src="ReadMe.asset/捕获5 (2).PNG" width="256"><img src="ReadMe.asset/捕获6 (2).PNG" width="256"></div><div align="enter">

# Requirement

environment:

* Windows 10 x64
* visual studio 2019
* vcpkg
* cmake


C++  lib:

* GLUT (freeglut) >= 3.0.0
* GLEW >= 2.1.0
* GLM  >= 0.9.9.5



Third part cpp lib:

* hdrloader
