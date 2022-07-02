#include <iostream>
#include<string>
#include <vector>
#include <random>
#include <stdlib.h>
#include <glm/glm.hpp>  // ��ѧ��֧��
#include "svpng.inc"    // png��� ref: https://github.com/miloyip/svpng
#include <omp.h>    // openmp���̼߳���

using namespace glm;
using namespace std;

const int SAMPLE = 64;
const double BRIGHTNESS = (2.0f * 3.1415926f) * (1.0f / double(SAMPLE));//���ǰ����������Բ�����,���ڰ뾶Ϊ1,������������2���ˡ�

const int WIDTH = 512;
const int HEIGHT = 512;
//�������
const double SCREEN_Z = 1.1;		//��ƽ��Z����
const vec3 EYE = vec3(0, 0, 4.0);	//�����λ��

//��ɫ
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
* ��ȷһ���󽻲���Ҫ������Щ��Ϣ:
* 1.�Ƿ��ཻ
* 2.�����λ��,������Ϊ��һ�ε�������
* 3.�ཻλ�õı�������:������,������ɫ,��������,�����,�ֲڶȵȡ�
*/
typedef struct Ray
{
	vec3 startpoint = vec3(0, 0, 0);//���
	vec3 direction = vec3(0, 0, 0);//����
}Ray;

typedef struct Material
{
	bool isEmissive = false;		//�Ƿ��Է���
	vec3 normal = vec3(0, 0, 0);	//������
	vec3 color = vec3(0, 0, 0);		//��ɫ
	double specularRate = 0.0f;		//�����ռ��
	double roughness = 1.0f;		//����ֲڶ�,1.0�϶�����ֲ�
	double refractRate = 0.0f;
	double refractAngle = 1.0f;
	double refractRoughtness = 0.0f;
}Material;

typedef struct HitResult
{
	bool isHit = false;				//�Ƿ��ཻ
	double distance = 0.0f;			//�뽻��ľ���
	vec3 HitPoint = vec3(0, 0, 0);	//����λ��
	Material material;				//���е�Ĳ���
}HitResult;

class Shape//����ú�ѧϰһ����ļ̳�,�Լ��麯��������
{
public:
	Shape() {}
	virtual HitResult intersect(Ray ray) { return HitResult(); }
};
/*
	�麯��+ָ��+�̳еı��ϰ���Ǻܺõ�,�����ʹ�����Ǽ�����ߺ�����ͼ���󽻵�ʱ��,����һֱ�ķ��ؽ��,Ҳ����
	HitResult�ṹ��,��ߴ���Ŀ�ά����,��ʹ�����Ժ��ּ�����������ͼԪ,Ҳ����Ҫ�����������κθĶ�,����ĸ���ֻ��һ���麯��
	�����Ǽ̳��������������൱���ж�����ɫ��������,��һ���вι���,��һ���󽻺���
*/

class Triangle :public Shape
{
public:
	Triangle() {}
	Triangle(vec3 p1, vec3 p2, vec3 p3,vec3 C)//�����������εĶ������ɫ
	{
		P1 = p1, P2 = p2, P3 = p3;								//��ʼ����������
		material.normal = normalize(cross(p2 - p1, p3 - p1));	//����������������ٱ�׼���õ�������
		material.color = C;										//��ʼ����������ɫ
	}
	vec3 P1, P2, P3;
	Material material;
	/*
	������ཻ������������,��ƽ��ƽ�в���,������ʼ����治��,�������������治��,��������״����,��ʼ�㵽��������ľ���ֻ
	��Ҫ�����εķ������Ͷ������꼴�ɼ���õ����������һ�ѵ�˺Ͳ��
	*/
	HitResult intersect(Ray ray)
	{
		HitResult res;
		
		vec3 S = ray.startpoint;		//�������
		vec3 d = ray.direction;			//���߷���
		vec3 N = material.normal;		//������
		//if (dot(N, d) > 0.0f) N = -N;	//�����Ϊ������t��ʱ���ùܷ������ķ���
		//if (dot(N, d) < 0.00001f&&dot(N,d)>0.0f) return res;//���ߺ��������淨������ֱ,�޽���,����Ĭ�Ͻ��,�������Ϊ0,����λ��
		//if (dot(N, d) < 0.00001f && dot(N, d) > 0.0f) return res;
		if (fabs(dot(N, d)) < 0.00001f) return res;
		//Ϊ0,�������ҲΪĬ��(���Է���,������Ϊ0,��ɫΪ��)
		float t = (dot(N, P1) - dot(S, N)) / dot(d, N);
		if (t < 0.0005f) return res;//��������ʼ��ı���,Ҳ����û���ཻ,���������һ��
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
	vec3 O;					//Բ��
	double R;				//�뾶
	Material material;		//����

	HitResult intersect(Ray ray)
	{
		HitResult res;

		vec3 S = ray.startpoint;	//�������
		vec3 d = ray.direction;		//���߷���

		float OS = length(O - S);
		float SH = dot(O - S, d);
		float OH = sqrt(pow(OS, 2) - pow(SH, 2));//���ߵ�Բ�ĵľ���

		if (OH > R) return res;
		float PH = sqrt(pow(R, 2) - pow(OH, 2));//�����ж���������

		float t1 = SH - PH;//����֪�����Ƶ��Ĺ�ʽ��˵,t1t2����������ģ�����������,��PH������ǿ����õ�
		//�����������ŵ�����,��SH�ǵ�˵õ���,�ͻ�����������,�����������Ӧ��ʹ��SH�ľ���ֵ��ʽ
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
		//װ���״̬

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
//	vec3 O;//Բ��
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
	// ��������ͼ�Σ����������
	for (auto& shape : shapes)
	{
		r = shape->intersect(ray);
		if (r.isHit && r.distance < res.distance) res = r; // ��¼����������󽻽��
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
	��Ȼ��������һ����λ������,ÿ����������0~1�϶��޷���������˷�������,��������Ƚ����ɵ�������2,ʹ�÷�����Χ��0~2,
	�������1,������ķ�Χ��-1~1,��������˷���,���
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

//vec3 pathTracing(vector<Shape*>& shapes, Ray ray)//ֱ�ӹ��յ�·��׷��
//{
//	float p = 0.5;
//	HitResult res = shoot(shapes, ray);
//	if (!res.isHit) return vec3(0); // δ����
//	// ��������򷵻���ɫ
//	if (res.material.isEmissive) return res.material.color/p;
//	// ����ֱ�ӷ���
//	return vec3(0);
//}
vec3 pathTracing(vector<Shape*>& shapes, Ray ray, int depth)
{
	if (depth > 8) return vec3(0);//���ܵ��͵�
	HitResult res = shoot(shapes, ray);

	if (!res.isHit) return vec3(0);  //δ����

	//cout << depth << endl;
	//��������򷵻���ɫ
	if (res.material.isEmissive) return res.material.color;

	//��P�ĸ��ʱ���
	double r = randf();//û��׷�ٵ���Դ�Ϳ����ٵ�
	float P = 0.8;
	if (r > P) return vec3(0);

	//�������
	Ray randomRay;
	randomRay.startpoint = res.HitPoint;
	randomRay.direction = randomDirection(res.material.normal);

	vec3 color = vec3(0);
	float consine = fabs(dot(-ray.direction, res.material.normal));

	r = randf();
	//cout << r << endl;
	if (r < res.material.specularRate)												//����
	{
		vec3 ref = normalize(reflect(ray.direction,res.material.normal));
		randomRay.direction = mix(ref, randomRay.direction, res.material.roughness);//���������������������һ����ֵ,�����ʾ�����ν���ŵ�������ռ�ı���t
		color = pathTracing(shapes, randomRay, depth + 1) * consine;
		//cout << color.x << " " << color.y << " " << color.z << endl;
	}
	else if (res.material.specularRate <= r && r <= res.material.refractRate)		//����
	{
		vec3 ref = normalize(refract(ray.direction, res.material.normal, float(res.material.refractAngle)));
		randomRay.direction = mix(ref, -randomRay.direction, res.material.refractRoughtness);
		color = pathTracing(shapes, randomRay, depth + 1) * consine;
	}
	else																			//������
	{
		vec3 srcColor = res.material.color;
		vec3 ptColor = pathTracing(shapes, randomRay, depth + 1) * consine;
		color = srcColor * ptColor;		//��ԭ��ɫ���
	}
	
	return color /P;//����Ҫ���Ǳ�������,��ν�������٤�����
}

void imshow(double* SRC,string name)//���Ǽ�����ʵõ���ֵ���Ǹ�����,��˱���Ҳ�Ǳ����ڸ������鵱��,������ﴫ����Ǹ���ָ��
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
		{//���������ｫ����ֵ�����Ժ�ת��Ϊunsigned char
			*p++ = (unsigned char)clamp(pow(*S++, 1.0f / 1.9f) * 255, 0.0, 255.0); // R ͨ��,gammaУ��������
			*p++ = (unsigned char)clamp(pow(*S++, 1.0f / 1.9f) * 255, 0.0, 255.0); // G ͨ��
			*p++ = (unsigned char)clamp(pow(*S++, 1.0f / 1.9f) * 255, 0.0, 255.0); // B ͨ��
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
	
	//��Դ
	Triangle l1 = Triangle(vec3(0.4, 0.99, 0.4), vec3(-0.4, 0.99, 0.4), vec3(-0.4,0.99, -0.4),    WHITE);
	//cout << l1.material.normal.x<<" "<< l1.material.normal.y <<" "<< l1.material.normal.z << endl;
	Triangle l2 = Triangle(vec3(0.4, 0.99, 0.4), vec3(-0.4, 0.99, -0.4), vec3(0.4, 0.99, -0.4),    WHITE);
	l1.material.isEmissive = true;
	l2.material.isEmissive = true;
	Shapes.push_back(&l1);
	Shapes.push_back(&l2);
	// ��������
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
	

	double* image = new double[WIDTH * HEIGHT * 3];//ȡһ���ڴ�
	memset(image, 0.0, sizeof(double) * WIDTH * HEIGHT * 3);//�ڴ��ʼ��
	
	omp_set_num_threads(50);
#pragma omp parallel for//ֻ����ʵ�ֹ�׷��ʱ�������,��Ϊ�õ��������ѭ��,ÿ�����صĲ�������Ӱ��,��������ڲ�����ѭ���������,ʹ��������߳̾ͻᵼ��ͼ�����
	for (int k = 0; k < SAMPLE; ++k)
	{
		double* p = image;//����һ��image�ڵ�,��������
		for (int i = 0; i < HEIGHT; ++i)
		{
			for (int j = 0; j < WIDTH; ++j)
			{
				double x = 2.0 * double(j) / double(WIDTH) - 1.0;
				double y = 2.0 * double(HEIGHT - i) / double(HEIGHT) - 1.0;//���ﵱi��0��ʱ��,y��1,�������Ļ��㹫ʽ������

				x += (randf() - 0.5f) / double(WIDTH);
				y += (randf() - 0.5f) / double(HEIGHT);

				
				vec3 coord = vec3(x, y, SCREEN_Z);          // ����ͶӰƽ������
				vec3 direction = normalize(coord - EYE);    // �������Ͷ�䷽��

				// ���ɹ���
				Ray ray;
				ray.startpoint = coord;
				ray.direction = direction;

				HitResult res;
				res = shoot(Shapes, ray);
				vec3 color = vec3(0,0,0);
				if (res.isHit)
				{
					if (res.material.isEmissive)//���й�Դֱ�ӷ�����ɫ
					{
						color = res.material.color;
					}
					//����ʵ����ѡ��һ������������·�����߲��ҽ���·��׷��
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
							//��ɫ����
							vec3 srcColor = res.material.color;
							vec3 ptColor = pathTracing(Shapes, randomRay, 0);
							//cout << ptColor.x<<" "<<ptColor.y<<" "<<ptColor.z<<" " << endl;
							color = ptColor * srcColor;
						}
						color *= BRIGHTNESS;//�����һ�����ؿ������
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























