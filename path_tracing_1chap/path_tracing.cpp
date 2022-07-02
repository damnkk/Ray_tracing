#include <iostream>
#include<string>
#include <vector>
#include <random>
#include <stdlib.h>
#include <glm/glm.hpp>  // 数学库支持
#include "svpng.inc"    // png输出 ref: https://github.com/miloyip/svpng
#include <omp.h>    // openmp多线程加速

using namespace glm;
using namespace std;

const int SAMPLE = 64;
const double BRIGHTNESS = (2.0f * 3.1415926f) * (1.0f / double(SAMPLE));//就是半球的面积除以采样数,由于半径为1,则半球面积就是2Π了。

const int WIDTH = 512;
const int HEIGHT = 512;
//相机参数
const double SCREEN_Z = 1.1;		//视平面Z坐标
const vec3 EYE = vec3(0, 0, 4.0);	//摄像机位置

//颜色
const vec3 RED(1, 0.5, 0.5);
const vec3 GREEN(0.5, 1, 0.5);
const vec3 BLUE(0.5, 0.5, 1);
const vec3 YELLOW(1.0, 1.0, 0.1);
const vec3 CYAN(0.1, 1.0, 1.0);
const vec3 MAGENTA(1.0, 0.1, 1.0);
const vec3 GRAY(0.5, 0.5, 0.5);
const vec3 WHITE(1, 1, 1);
const vec3 SPECIAL(0.078, 0.568, 0.658);


/*
* 明确一下求交操作要返回哪些信息:
* 1.是否相交
* 2.交点的位置,用于作为下一次弹射的起点
* 3.相交位置的表面属性:法向量,表面颜色,材质属性,发光度,粗糙度等。
*/
typedef struct Ray
{
	vec3 startpoint = vec3(0, 0, 0);//起点
	vec3 direction = vec3(0, 0, 0);//方向
}Ray;

typedef struct Material
{
	bool isEmissive = false;		//是否自发光
	vec3 normal = vec3(0, 0, 0);	//法向量
	vec3 color = vec3(0, 0, 0);		//颜色
	double specularRate = 0.0f;		//反射光占比
	double roughness = 1.0f;		//反射粗糙度,1.0肯定是最粗糙
	double refractRate = 0.0f;
	double refractAngle = 1.0f;
	double refractRoughtness = 0.0f;
}Material;

typedef struct HitResult
{
	bool isHit = false;				//是否相交
	double distance = 0.0f;			//与交点的距离
	vec3 HitPoint = vec3(0, 0, 0);	//交点位置
	Material material;				//命中点的材质
}HitResult;

class Shape//这里好好学习一下类的继承,以及虚函数的性质
{
public:
	Shape() {}
	virtual HitResult intersect(Ray ray) { return HitResult(); }
};
/*
	虚函数+指针+继承的编程习惯是很好的,这可以使得我们计算光线和任意图形求交的时候,都有一直的返回结果,也就是
	HitResult结构体,提高代码的可维护行,即使我们以后又加入了其他的图元,也不需要对主代码有任何改动,这里的父类只有一个虚函数
	而我们继承下来的三角形类当中有顶点颜色两种属性,带一个有参构造,和一个求交函数
*/

class Triangle :public Shape
{
public:
	Triangle() {}
	Triangle(vec3 p1, vec3 p2, vec3 p3,vec3 C)//代表着三角形的顶点和颜色
	{
		P1 = p1, P2 = p2, P3 = p3;								//初始化三个顶点
		material.normal = normalize(cross(p2 - p1, p3 - p1));	//利用两边向量叉乘再标准化得到法向量
		material.color = C;										//初始化三角形颜色
	}
	vec3 P1, P2, P3;
	Material material;
	/*
	这里的相交函数有三不交,和平面平行不交,交到起始点后面不交,交到三角形外面不交,而在正常状况下,起始点到三角形面的距离只
	需要三角形的法向量和顶点坐标即可计算得到。其余就是一堆点乘和叉乘
	*/
	HitResult intersect(Ray ray)
	{
		HitResult res;
		
		vec3 S = ray.startpoint;		//射线起点
		vec3 d = ray.direction;			//射线方向
		vec3 N = material.normal;		//法向量
		//if (dot(N, d) > 0.0f) N = -N;	//这个是为了在求t的时候不用管法向量的方向
		//if (dot(N, d) < 0.00001f&&dot(N,d)>0.0f) return res;//射线和三角形面法向量垂直,无交点,返回默认结果,交点距离为0,交点位置
		//if (dot(N, d) < 0.00001f && dot(N, d) > 0.0f) return res;
		if (fabs(dot(N, d)) < 0.00001f) return res;
		//为0,交点材质也为默认(无自发光,法向量为0,颜色为黑)
		float t = (dot(N, P1) - dot(S, N)) / dot(d, N);
		if (t < 0.0005f) return res;//交点在起始点的背面,也算是没有相交,和上面情况一样
		vec3 P = S + d*t;

		vec3 c1 = cross(P2 - P1, P - P1);
		vec3 c2 = cross(P3 - P2, P - P2);
		vec3 c3 = cross(P1 - P3, P - P3);

		vec3 n = material.normal;
		if (dot(c1, n) < 0 || dot(c2, n) < 0 || dot(c3, n) < 0) return res;

		res.isHit = true;
		res.distance = t;
		res.HitPoint = P;
		res.material = material;
		res.material.normal = N;

		return res;
	}
};

class Sphere :public Shape
{
public:
	Sphere() {}
	Sphere(vec3 o, double r, vec3 c) { O = o; R = r; material.color = c; }
	vec3 O;					//圆心
	double R;				//半径
	Material material;		//材质

	HitResult intersect(Ray ray)
	{
		HitResult res;

		vec3 S = ray.startpoint;	//射线起点
		vec3 d = ray.direction;		//射线方向

		float OS = length(O - S);
		float SH = dot(O - S, d);
		float OH = sqrt(pow(OS, 2) - pow(SH, 2));//射线到圆心的距离

		if (OH > R) return res;
		float PH = sqrt(pow(R, 2) - pow(OH, 2));//用来判断入射出射角

		float t1 = SH - PH;//我们知道从推导的公式来说,t1t2都是向量的模计算而得来的,而PH本身就是开方得到
		//不会有正负号的问题,而SH是点乘得到了,就会有正负问题,因此这里我们应该使用SH的绝对值形式
		float t2 = SH + PH;
		float t;
		if (t1 < 0 && t2 < 0) return res;
		else if (t1 > 0 && t2 > 0)
		{
			t = t1 < t2 ? t1 : t2;
		}
		else
		{
			t = t1 > t2 ? t1 : t2;
		}
		vec3 P = S + t * d;
		if (fabs(t1) < 0.0005f || fabs(t2) < 0.0005f) return res;
		//装填返回状态

		res.HitPoint = P;
		res.isHit = true;
		res.distance = t;
		res.material = material;
		res.material.normal = normalize(P-O);
		return res;
	}
};

//class Sphere :public Shape
//{
//public:
//	Sphere() {}
//	Sphere(vec3 o, float r, vec3 c) { O = o; R = r; material.color = c; }
//	vec3 O;//圆心
//	float R;
//	Material material;
//
//	HitResult intersect(Ray ray)
//	{
//		HitResult res;
//		vec3 S = ray.startpoint;
//		vec3 d = ray.direction;
//
//		float OS = length(O - S);
//		float SH = dot(O - S, d);
//		float OH = sqrt(pow(OS, 2) - pow(SH, 2));
//		if (OH > R) return res;
//		float PH = sqrt(pow(R, 2) - pow(OH, 2));
//
//		float t1 = SH - PH;
//		float t2 = SH + PH;
//		float t = (t1 < 0) ? (t2) : (t1);
//		vec3 P = S + t * d;
//		if (fabs(t1) < 0.0005f || fabs(t2) < 0.0005f) return res;
//
//		res.distance = t;
//		res.HitPoint = P;
//		res.isHit = true;
//		res.material = material;
//		res.material.normal = normalize(P-O);
//		return res;
//	}
//};



HitResult shoot(vector<Shape*> shapes, Ray ray)
{
	HitResult res, r;
	res.distance = 1145141919.810f; // inf
	// 遍历所有图形，求最近交点
	for (auto& shape : shapes)
	{
		r = shape->intersect(ray);
		if (r.isHit && r.distance < res.distance) res = r; // 记录距离最近的求交结果
	}
	return res;
}

std::uniform_real_distribution<> dis(0.0, 1.0);
random_device rd;
mt19937 gen(rd());
double randf()
{
	return dis(gen);
}



vec3 randomvec3()
{
	/*
	既然你想生成一个单位球向量,每个分量都是0~1肯定无法生成四面八方的向量,因此我们先将生成的向量乘2,使得分量范围在0~2,
	再整体减1,则分量的范围在-1~1,就是四面八方了,真好
	*/
	vec3 d;
	do
	{
		d = 2.0f * vec3(randf(), randf(), randf()) - vec3(1, 1, 1);
	} while (dot(d, d) > 1.0);
	return normalize(d);
}

vec3 randomDirection(vec3 n)
{
	return normalize(randomvec3() + n);
	
}
//vec3 randomdirection(vec3 n)
//{
//	 
//	vec3 d;
//	do
//	{
//		d = randomvec3();
//	} while (dot(d, n) < 0.0f);
//	return d;
//}

//vec3 pathTracing(vector<Shape*>& shapes, Ray ray)//直接光照的路径追踪
//{
//	float p = 0.5;
//	HitResult res = shoot(shapes, ray);
//	if (!res.isHit) return vec3(0); // 未命中
//	// 如果发光则返回颜色
//	if (res.material.isEmissive) return res.material.color/p;
//	// 否则直接返回
//	return vec3(0);
//}
vec3 pathTracing(vector<Shape*>& shapes, Ray ray, int depth)
{
	if (depth > 8) return vec3(0);//还能弹就弹
	HitResult res = shoot(shapes, ray);

	if (!res.isHit) return vec3(0);  //未命中

	//cout << depth << endl;
	//如果发光则返回颜色
	if (res.material.isEmissive) return res.material.color;

	//有P的概率保留
	double r = randf();//没有追踪到光源就可以再弹
	float P = 0.8;
	if (r > P) return vec3(0);

	//否则继续
	Ray randomRay;
	randomRay.startpoint = res.HitPoint;
	randomRay.direction = randomDirection(res.material.normal);

	vec3 color = vec3(0);
	float consine = fabs(dot(-ray.direction, res.material.normal));

	r = randf();
	//cout << r << endl;
	if (r < res.material.specularRate)												//反射
	{
		vec3 ref = normalize(reflect(ray.direction,res.material.normal));
		randomRay.direction = mix(ref, randomRay.direction, res.material.roughness);//反射向量和随机向量进行一个插值,反射率就是所谓干扰的向量所占的比例t
		color = pathTracing(shapes, randomRay, depth + 1) * consine;
		//cout << color.x << " " << color.y << " " << color.z << endl;
	}
	else if (res.material.specularRate <= r && r <= res.material.refractRate)		//折射
	{
		vec3 ref = normalize(refract(ray.direction, res.material.normal, float(res.material.refractAngle)));
		randomRay.direction = mix(ref, -randomRay.direction, res.material.refractRoughtness);
		color = pathTracing(shapes, randomRay, depth + 1) * consine;
	}
	else																			//漫反射
	{
		vec3 srcColor = res.material.color;
		vec3 ptColor = pathTracing(shapes, randomRay, depth + 1) * consine;
		color = srcColor * ptColor;		//和原颜色混合
	}
	
	return color /P;//这里要考虑保留概率,所谓这个就是伽马矫正
}

void imshow(double* SRC,string name)//我们计算材质得到的值都是浮点数,因此保存也是保存在浮点数组当中,因此这里传入的是浮点指针
{
	unsigned char* image = new unsigned char[WIDTH * HEIGHT * 3];
	unsigned char* p = image;
	double* S = SRC;
	FILE* fp;
	name += ".png";
	const char* filename = name.c_str();
	
	fopen_s(&fp, filename, "wb");
	for (int i = 0; i < HEIGHT; ++i)
	{
		for (int j = 0; j < WIDTH; ++j)
		{//我们在这里将浮点值计算以后转换为unsigned char
			*p++ = (unsigned char)clamp(pow(*S++, 1.0f / 1.9f) * 255, 0.0, 255.0); // R 通道,gamma校正在这里
			*p++ = (unsigned char)clamp(pow(*S++, 1.0f / 1.9f) * 255, 0.0, 255.0); // G 通道
			*p++ = (unsigned char)clamp(pow(*S++, 1.0f / 1.9f) * 255, 0.0, 255.0); // B 通道
		}
	}
	svpng(fp, WIDTH, HEIGHT, image, 0);
	delete[] image;
}



int main()
{
	
	
	vector<Shape*> Shapes;
	Shapes.push_back(new Triangle(vec3(-0.15, 0.4, -0.6), vec3(-0.15, -0.95, -0.6), vec3(0.15, 0.4, -0.6), YELLOW));
	Shapes.push_back(new Triangle(vec3(0.15, 0.4, -0.6), vec3(-0.15, -0.95, -0.6), vec3(0.15, -0.95, -0.6), YELLOW));
	
	//光源
	Triangle l1 = Triangle(vec3(0.4, 0.99, 0.4), vec3(-0.4, 0.99, 0.4), vec3(-0.4,0.99, -0.4),    WHITE);
	//cout << l1.material.normal.x<<" "<< l1.material.normal.y <<" "<< l1.material.normal.z << endl;
	Triangle l2 = Triangle(vec3(0.4, 0.99, 0.4), vec3(-0.4, 0.99, -0.4), vec3(0.4, 0.99, -0.4),    WHITE);
	l1.material.isEmissive = true;
	l2.material.isEmissive = true;
	Shapes.push_back(&l1);
	Shapes.push_back(&l2);
	// 背景盒子
	// bottom
	Shapes.push_back(new Triangle(vec3(1, -1, 1), vec3(-1, -1, -1), vec3(-1, -1, 1), WHITE));
	Shapes.push_back(new Triangle(vec3(1, -1, 1), vec3(1, -1, -1), vec3(-1, -1, -1), WHITE));
	// top
	Shapes.push_back(new Triangle(vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, 1, -1),  WHITE));
	Shapes.push_back(new Triangle(vec3(1, 1, 1), vec3(-1, 1, -1), vec3(1, 1, -1),  WHITE));
	// back
	Shapes.push_back(new Triangle(vec3(1, -1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1), CYAN));
	Shapes.push_back(new Triangle(vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), CYAN));
	// left
	Shapes.push_back(new Triangle(vec3(-1, -1, -1), vec3(-1, 1, 1), vec3(-1, -1, 1), BLUE));
	Shapes.push_back(new Triangle(vec3(-1, -1, -1), vec3(-1, 1, -1), vec3(-1, 1, 1), BLUE));
	// right
	Shapes.push_back(new Triangle(vec3(1, 1, 1), vec3(1, -1, -1), vec3(1, -1, 1), RED));
	Shapes.push_back(new Triangle(vec3(1, -1, -1), vec3(1, 1, 1), vec3(1, 1, -1), RED));
	//obj
	//Shapes.push_back(new Triangle(vec3(-0.7, 0.8, -0.8), vec3(-0.7, -0.7, -0.8), vec3(-0.1, -0.8, 0.8), WHITE));
	//Shapes.push_back(new Triangle(vec3(0.7, 0.8, -0.8), vec3(0.1, -0.8, 0.8), vec3(0.7, -0.7, -0.8),  WHITE));
	Sphere sph2 = Sphere(vec3(-0.6, -0.8, -0.5), 0.2, WHITE);
	sph2.material.specularRate = 0.9;
	sph2.material.roughness = 0.0;
	Shapes.push_back(&sph2);

	Sphere sph = Sphere(vec3(0.0, -0.3, 0.0), 0.4, WHITE);
	sph.material.specularRate = 0.1;
	sph.material.refractAngle = 0.1;
	sph.material.refractRate = 0.95;
	Shapes.push_back(&sph);

	Sphere sph1 = Sphere(vec3(0.6, -0.8, -0.5), 0.2, GREEN);
	sph1.material.specularRate = 0.3;
	sph1.material.roughness = 0.1;
	Shapes.push_back(&sph1);
	//Shapes.push_back(new Sphere(vec3(-0.1, -0.7, 0.2), 0.3, WHITE));
	

	double* image = new double[WIDTH * HEIGHT * 3];//取一块内存
	memset(image, 0.0, sizeof(double) * WIDTH * HEIGHT * 3);//内存初始化
	
	omp_set_num_threads(50);
#pragma omp parallel for//只有在实现光追的时候才能用,因为用的是最外层循环,每个像素的采样互不影响,如果仅有内部两层循环的情况下,使用这个多线程就会导致图像出错
	for (int k = 0; k < SAMPLE; ++k)
	{
		double* p = image;//复制一个image节点,用来迭代
		for (int i = 0; i < HEIGHT; ++i)
		{
			for (int j = 0; j < WIDTH; ++j)
			{
				double x = 2.0 * double(j) / double(WIDTH) - 1.0;
				double y = 2.0 * double(HEIGHT - i) / double(HEIGHT) - 1.0;//这里当i是0的时候,y是1,因此这里的换算公式是这样

				x += (randf() - 0.5f) / double(WIDTH);
				y += (randf() - 0.5f) / double(HEIGHT);

				
				vec3 coord = vec3(x, y, SCREEN_Z);          // 计算投影平面坐标
				vec3 direction = normalize(coord - EYE);    // 计算光线投射方向

				// 生成光线
				Ray ray;
				ray.startpoint = coord;
				ray.direction = direction;

				HitResult res;
				res = shoot(Shapes, ray);
				vec3 color = vec3(0,0,0);
				if (res.isHit)
				{
					if (res.material.isEmissive)//命中光源直接返回颜色
					{
						color = res.material.color;
					}
					//命中实体则选择一个随机方向重新发射光线并且进行路径追踪
					else
					{
						Ray randomRay;
						randomRay.startpoint = res.HitPoint;
						randomRay.direction = randomDirection(res.material.normal);

						//cout << randomRay.direction.x << " " << randomRay.direction.y << " " << randomRay.direction.z << endl;
						double r = randf();
						if (r < res.material.specularRate)
						{
							vec3 ref = normalize(reflect(ray.direction, res.material.normal));
							randomRay.direction = mix(ref, randomRay.direction, res.material.roughness);
							color = pathTracing(Shapes, randomRay, 0);
						}
						else if (res.material.specularRate <= r && r<=res.material.refractRate)
						{
							vec3 ref = normalize(refract(ray.direction, res.material.normal, float(res.material.refractAngle)));
							randomRay.direction = mix(ref, -randomRay.direction, res.material.refractRoughtness);
							color = pathTracing(Shapes, randomRay, 0);
						}
						else
						{
							//颜色积累
							vec3 srcColor = res.material.color;
							vec3 ptColor = pathTracing(Shapes, randomRay, 0);
							//cout << ptColor.x<<" "<<ptColor.y<<" "<<ptColor.z<<" " << endl;
							color = ptColor * srcColor;
						}
						color *= BRIGHTNESS;//这就是一个蒙特卡洛积分
					}
				}

				*p += color.x; ++p;//R
				*p += color.y; ++p;//G
				*p += color.z; ++p;//B
			}
		}
		/*string name = "test_path_tracing";
		name += "_spp";
		name += to_string(k+1);
		imshow(image, name);
		*/
	}
	string name = "test_path_tracingUU";
	imshow(image, name);
	
	delete[] image;
	
}























