#define GLUT_DISABLE_ATEXIT_HACK
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <sstream>
#include <iostream>
#include <algorithm>

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#define INF 114514.0
using namespace glm;

//using namespace std;

// ----------------------------------------------------------------------------- //
GLuint program;							//��ɫ���������
std::vector<vec3> vertices;				//��������
std::vector<GLuint> indices;			//��������
std::vector<vec3> lines;				//�߶ζ˵�����
vec3 rotateControl(0, 0, 0);			//��ת����
vec3 scaleControl(1, 1, 1);				//���Ų���

struct BVHNode
{
	BVHNode* left = NULL;//������
	BVHNode* right = NULL;//������
	int n, index;//Ҷ�ӽڵ����Ϣ
	vec3 AA, BB;//��ײ��
};



typedef struct Triangle
{
	vec3 p1, p2, p3;
	vec3 center;//���ĵ����ô����������ε�λ��,�����ǻ��������ε���ͬ�İ�Χ�е����ݾ������λ��,������ķ�ʽ��������ֱ�ȡƽ��
	Triangle(vec3 a, vec3 b, vec3 c)
	{
		p1 = a, p2 = b, p3 = c;
		center = (p1 + p2 + p3) / vec3(3, 3, 3);
	}
}Triangle;
std::vector<Triangle> triangles;


bool cmpx(const Triangle& t1, const Triangle& t2)
{
	return t1.center.x < t2.center.x;
}

bool cmpy(const Triangle& t1, const Triangle& t2)
{
	return t1.center.y < t2.center.y;
}

bool cmpz(const Triangle& t1, const Triangle& t2)
{
	return t1.center.z < t2.center.z;
}

struct HitResult
{
	Triangle* triangle = NULL;
	float distance = INF;
};
typedef struct Ray
{
	vec3 startPoint = vec3(0, 0, 0);//���
	vec3 direction = vec3(0, 0, 0);//����
}Ray;



std::string readShaderFile(std::string filepath)
{
	std::string res, line;
	std::ifstream fin(filepath);
	if (!fin.is_open())
	{
		std::cout << "�ļ�" << filepath << "��ʧ��" << std::endl;
		exit(-1);
	}
	while (getline(fin, line))
	{
		res += line + "\n";//һ��һ���س�
	}
	fin.close();
	return res;
}

GLuint getShaderProgram(std::string fshader, std::string vshader)
{
	std::string vSource = readShaderFile(vshader);
	std::string fSource = readShaderFile(fshader);//��ȡ��ɫ���ļ�
	const char* vpointer = vSource.c_str();//��ת��ΪC����ַ���
	const char* fpointer = fSource.c_str();

	//�ݴ�
	GLint success;
	GLchar infoLog[512];

	//���������붥����ɫ��
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, (const GLchar**)(&vpointer), NULL);//����һ����ɫ��
	glCompileShader(vertexShader);//������ɫ��
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);//���������ɫ���Ƿ����ɹ�
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);//�����ȡ������ı�����Ϣ�Ժ�,��洢�����char���鵱��
		std::cout << "������ɫ���������\n" << infoLog << std::endl;
		exit(-1);
	}
	//����������Ƭ����ɫ��
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, (const GLchar**)(&fpointer), NULL);//����һ��Ƭ����ɫ��
	glCompileShader(fragmentShader);//
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "Ƭ����ɫ���������\n" << infoLog << std::endl;
		exit(-1);
	}//���˱������
	
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	//ɾ����ɫ������women 

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	return shaderProgram;
}

void readObj(std::string filepath, std::vector<vec3>& vertices, std::vector<GLuint>& indices)
{
	std::ifstream fin(filepath);//���ļ�·���д��ļ���
	std::string line;
	if (!fin.is_open())//���ж�
	{
		std::cout << "ERROR::�ļ� " << filepath << " ��ʧ��" << std::endl;
		exit(-1);
	}

	//������ȡ
	int offset = vertices.size();

	//���ж�ȡ
	while (getline(fin, line))//��ȡһ����Ϣ��line
	{
		std::istringstream sin(line);//����һ�е�������Ϊstring stream��������ȡ
		std::string type;
		GLfloat x, y, z;
		int v0, v1, v2;

		//��ȡobj�ļ�
		sin >> type;//������������Ѿ���ȡ��һ��,һ�е���Ϣ����,��һ�������Ӧ����type,����type������ʼ��λ�û���������,
		//��Ϊ.obj�ļ�ͷ������һЩ��ע,��˶����������������͵���Ϣ,����ɶҲ����,ֱ�ӽ�����һ�еĶ�ȡ
		if (type == "v")
		{
			sin >> x >> y >> z;
			vertices.push_back(vec3(x, y, z));
		}
		if (type == "f")
		{
			sin >> v0 >> v1 >> v2;
			indices.push_back(v0 - 1 + offset);//obj�е������Ǵ�1��ʼ��,�����������������Ǵ�0��ʼ��,���Ҫ-1
			indices.push_back(v1 - 1 + offset);//����Ҳ������,���ǵ����Ǵ���Ķ������������һ��ʼ����ֵ,������ǵ���������-1֮��,��Ҫ���ݶ�������ĳ�ʼ��С����һ��ƫִ��
			indices.push_back(v2 - 1 + offset);
		}
		/*
		* ��������������f����,��ν��������ʲô��?������ʵ����ȷ�����������,ȷ������������������������λ������/��������/������,����ǰ����̳�
		* �ṩ��objֻ�ṩ��λ������,���,��������������鲢û�кܶ���Ҫ////���ָ�����ݿ�,ֻ�и����������,Ҳ�ͺܼ�����
		*/
	}
}

void addLine(vec3 p1, vec3 p2)
{
	lines.push_back(p1);
	lines.push_back(p2);
}
void addBox(BVHNode* root)
{
	float x1 = root->AA.x, y1 = root->AA.y, z1 = root->AA.z;
	float x2 = root->BB.x, y2 = root->BB.y, z2 = root->BB.z;
	lines.push_back(vec3(x1, y1, z1)), lines.push_back(vec3(x2, y1, z1));
	lines.push_back(vec3(x1, y1, z2)),lines.push_back(vec3(x1, y1, z1));
	lines.push_back(vec3(x1, y1, z1)), lines.push_back(vec3(x1, y2, z1));
	lines.push_back(vec3(x2, y1, z1)), lines.push_back(vec3(x2, y1, z2));
	lines.push_back(vec3(x2, y1, z1)), lines.push_back(vec3(x2, y2, z1));
	lines.push_back(vec3(x1, y2, z1)), lines.push_back(vec3(x2, y2, z1));
	lines.push_back(vec3(x1, y1, z2)), lines.push_back(vec3(x1, y2, z2));
	lines.push_back(vec3(x1, y2, z1)), lines.push_back(vec3(x1, y2, z2));
	lines.push_back(vec3(x1, y2, z2)), lines.push_back(vec3(x2, y2, z2));
	lines.push_back(vec3(x1, y1, z2)), lines.push_back(vec3(x2, y1, z2));
	lines.push_back(vec3(x2, y2, z1)), lines.push_back(vec3(x2, y2, z2));
	lines.push_back(vec3(x2, y1, z2)), lines.push_back(vec3(x2, y2, z2));
}
void addTriangle(Triangle* tri)
{
	//Ϊʲô������������,����Ҫpush_back()������,���һ�Ҫ��λ�ý���������ƫ����?��Ϊ����ȫ���ı�λ�õĻ�,ͬһ�������ξͻᱻ��ɫ��ɫ��Ⱦ����
	//���п�������ڵ�,�����������ĸ�������,��������һ��ƫ���Ժ�,�Ͳ�������ڵ�,���׿���
	if (tri)
	{
		lines.push_back(tri->p1 - vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p2 - vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p2 - vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p3 - vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p3 - vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p1 - vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p1 + vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p2 + vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p2 + vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p3 + vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p3 + vec3(0.0005, 0.0005, 0.0005));
		lines.push_back(tri->p1 + vec3(0.0005, 0.0005, 0.0005));
		
	}
}


float hitTriangle(Triangle* triangle, Ray ray)
{
	vec3 p1 = triangle->p1, p2 = triangle->p2, p3 = triangle->p3;
	vec3 S = ray.startPoint;			//�������
	vec3 d = ray.direction;				//���߷���
	vec3 N = normalize(cross(p2 - p1, p3 - p1));	//������
	if (dot(N, d) > 0.0f) N = -N;		//��ȡ��ȷ�ķ�����

	//���ʵ�ֺ�������ƽ��
	if (fabs(dot(d, N)) < 0.00001f) return INF;
	//����
	float t = (-dot(S, N) + dot(N, p1)) / dot(d, N);
	if (t < 0.0005f) return INF;//tΪ���϶����������ں�����,����0���ϵ���һС����,��Ϊ������һ��������ߵ����,
	//�����㱻����������������������,������Ҫ����û�ཻ,��Ϊ���������,�����Ϸ���Ĺ������������������ཻ��?

	//�������
	vec3 P = S + d * t;
	//�жϽ����Ƿ�������������
	vec3 c1 = cross(p2 - p1, P - p1);
	vec3 c2 = cross(p3 - p2, P - p2);
	vec3 c3 = cross(p1 - p3, P - p3);
	if (dot(c1, N) > 0 && dot(c2, N) > 0 && dot(c3, N) > 0) return t;
	if (dot(c1, N) < 0 && dot(c2, N) < 0 && dot(c3, N) < 0) return t;

	return INF;
	
}




BVHNode* buildBVH(std::vector<Triangle>& triangles, int l, int r, int n)//n����������Ŀ��ֵ,С��n�Ͳ������·���
{
	if (l > r) return 0;
	BVHNode* node = new BVHNode();
	node->AA = vec3(1145141919, 1145141919, 1145141919);//ע�⿴����ĳ�ʼ������,AA��С������,��ô��ʼ��Ϊ���
	node->BB = vec3(-1145141919, -1145141919, -1145141919);//BB�Ǵ������,��ô��ʼ��Ϊ��С

	//����AABB
	for (int i = l; i <= r; ++i)
	{
		//��С��AA
		float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
		float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
		float minz = min(triangles[i].p1.z,min(triangles[i].p2.z, triangles[i].p3.z));
		node->AA.x = min(node->AA.x, minx);
		node->AA.y = min(node->AA.y, miny);
		node->AA.z = min(node->AA.z, minz);
		float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
		float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
		float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
		node->BB.x = max(node->BB.x, maxx);
		node->BB.y = max(node->BB.y, maxy);
		node->BB.z = max(node->BB.z, maxz);
	}
	//С�ڵ���n��������,����Ҷ�ӽڵ�
	if ((r - l + 1) <= n)
	{
		node->n = r - l + 1;
		node->index = l;
		return node;
	}
	//����ݹ齨��
	float lenx = node->BB.x - node->AA.x;
	float leny = node->BB.y - node->AA.y;
	float lenz = node->BB.z - node->AA.z;
	//��x����
	if (lenx >= leny && lenx >= lenz)
		std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpx);
	if (leny >= lenx && leny >= lenz)
		std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpy);
	if (lenz >= leny && lenz >= lenx)
		std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpz);
	int mid = (l + r) / 2;
	node->left = buildBVH(triangles, l, mid, n);
	node->right = buildBVH(triangles, mid + 1, r, n);
	return node;
}

BVHNode* buildBVHwithSAH(std::vector<Triangle>& triangles, int l, int r, int n)
{
	
	if (l > r) return 0;

	BVHNode* node = new BVHNode();
	node->AA = vec3(1145141919, 1145141919, 1145141919);
	node->BB = vec3(-1145141919, -1145141919, -1145141919);
	for (int i = l; i <= r; ++i)
	{
		// ��С�� AA
		float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
		float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
		float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
		node->AA.x = min(node->AA.x, minx);
		node->AA.y = min(node->AA.y, miny);
		node->AA.z = min(node->AA.z, minz);
		// ���� BB
		float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
		float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
		float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
		node->BB.x = max(node->BB.x, maxx);
		node->BB.y = max(node->BB.y, maxy);
		node->BB.z = max(node->BB.z, maxz);
	}
	if ((r - l + 1) <= n) 
	{
		node->n = r - l + 1;
		node->index = l;
		return node;
	}
	//������ǻ��������εĹ���,����������б䶯
	float Cost = INF;
	int Axis = 0;
	int Split = (l + r) / 2;
	for (int axis = 0; axis < 3; ++axis)
	{
		//�ֱ�xyz������
		if (axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);//Ϊʲô�ڶ�������Ҫr+1��?��Ϊend()������ָ�����һ��Ԫ�صĺ���
		if (axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
		if (axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);
		//leftMax[i]:[l,i]������xyzֵ
		//leftMin[i]:[l,i]����С��xyzֵ
		/*
	�������һ�ֱȽϾ���ķ���,һ���ԾͰ�ÿ�ַָ�λ����ߵ������Сֵ�������
		*/
		std::vector<vec3> leftMax(r - l + 1, vec3(-INF, -INF, -INF));
		std::vector<vec3> leftMin(r - l + 1, vec3(INF, INF, INF));
		for (int i = l; i <= r; ++i)
		{
			Triangle& t = triangles[i];
			int bias = (i == l) ? 0 : 1;//��һ��Ԫ�����⴦��
			
			leftMax[i - l].x = max(leftMax[i - l - bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
			leftMax[i - l].y = max(leftMax[i - l - bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
			leftMax[i - l].z = max(leftMax[i - l - bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

			leftMin[i - l].x = min(leftMin[i - l - bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
			leftMin[i - l].y = min(leftMin[i - l - bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
			leftMin[i - l].z = min(leftMin[i - l - bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
			
		}

		//rightMax[i]:[i,r]������xyzֵ
		//rightMin[i]:[i,r]����С��xyzֵ
		std::vector<vec3> rightMax(r - l + 1, vec3(-INF, -INF, -INF));
		std::vector<vec3> rightMin(r - l + 1, vec3(INF, INF, INF));
		for (int i = r; i >= l; --i)
		{
			Triangle& t = triangles[i];
			int bias = (i == r) ? 0 : 1;
			rightMax[i - l].x = max(rightMax[i - l + bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
			rightMax[i - l].y = max(rightMax[i - l + bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
			rightMax[i - l].z = max(rightMax[i - l + bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));
												   
			rightMin[i - l].x = min(rightMin[i - l + bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
			rightMin[i - l].y = min(rightMin[i - l + bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
			rightMin[i - l].z = min(rightMin[i - l + bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
		}
		//����Ѱ�ҷָ�
		float cost = INF;
		int split = l;
		for (int i = l; i <= r - 1; ++i)
		{
			float lenx, leny, lenz;
			// ��� [l, i]
			vec3 leftAA = leftMin[i - l];
			vec3 leftBB = leftMax[i - l];
			lenx = leftBB.x - leftAA.x;
			leny = leftBB.y - leftAA.y;
			lenz = leftBB.z - leftAA.z;
			float leftS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));//��Χ�б����
			float leftCost = leftS * (i - l + 1);//��cost,���Ӵ�,���������λ���,�ǿ϶�������ӳɱ��͸���

			// �Ҳ� [i+1, r]
			vec3 rightAA = rightMin[i + 1 - l];
			vec3 rightBB = rightMax[i + 1 - l];
			lenx = rightBB.x - rightAA.x;
			leny = rightBB.y - rightAA.y;
			lenz = rightBB.z - rightAA.z;
			float rightS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
			float rightCost = rightS * (r - i);

			//��¼ÿ�ַָ����С��
			float totalCost = leftCost + rightCost;
			if (totalCost < cost)
			{
				cost = totalCost;
				split = i;
			}
		}
		if (cost < Cost)
		{
			Cost = cost;
			Split = split;
			Axis = axis;
		}
	}
	if (Axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
	if (Axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
	if (Axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);

	//�ݹ�
	node->left = buildBVHwithSAH(triangles, l, Split, n);
	node->right = buildBVHwithSAH(triangles, Split + 1, r, n);
	return node;

}


void dfsNlevel(BVHNode* root, int depth, int targetDepth)
{
	if (root == NULL) return;
	if (depth == targetDepth)
	{
		addBox(root);
		return;
	}
	dfsNlevel(root->left, depth + 1, targetDepth);
	dfsNlevel(root->right, depth + 1, targetDepth);
}

HitResult hitTriangleArray(Ray ray, std::vector<Triangle>& triangles, int l, int r)
{
	HitResult res;
	for (int i = l; i <= r; ++i)
	{
		float d = hitTriangle(&triangles[i], ray);
		if (d < INF && d < res.distance)
		{
			res.distance = d;
			res.triangle = &triangles[i];
		}
	}
	return res;
}

//AABB ������,û�н����򷵻�-1
float hitAABB(Ray r, vec3 AA, vec3 BB)
{
	vec3 invdir = vec3(1.0 / r.direction.x, 1.0 / r.direction.y, 1.0 / r.direction.z);
	/*
	***********************************************************
	ע�⿴����!! ,���Ⳣ���Ÿ�һ��,���ܲ��ܶ�
	���Ǹо��������BB��AA�Ƿ����ҲûɶӰ��,��Ϊ����Ҳ�п��ܴ���BB���ķ������밡,��������in,outҲֻ����ֻ��һ�������ļ���
	���滹Ҫ��������������ѡȡ�̵���Ϊ����㼯,ѡȡ������Ϊ����㼯,���Ҵ�������ѡȡ��Զ, �ӳ�����ѡȡ���
	*/

	vec3 in = (BB - r.startPoint) * invdir;//�����ʵ�ܺ����,����ÿһ�����ӵ�ÿһ��������˵,ʲô��ײ���ĸ�?������ʼ��ĳ�����+t*����������ĳ�����,Ȼ��t�ͺܺ����ˡ�
	vec3 out = (AA - r.startPoint) * invdir;

	vec3 tmax = max(in, out);
	vec3 tmin = min(in, out);
	float t1 = min(tmax.x, min(tmax.y, tmax.z));
	float t0 = max(tmin.x, max(tmin.y, tmin.z));

	return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1);//t1<t0,ֱ�Ӿ���û�������ڳ����������������,�������Ϊ��,��˵����������ں�����,�����Ƿ��������Ľ������Ϊ������
	//tֵ,�����������tֵ������0,��ô˵����������ڰ�Χ��֮��,��ô��Ȼ������ȡС,
}

//BVH�ϱ�����
HitResult hitBVH(Ray ray, std::vector<Triangle>& triangles, BVHNode* root)
{//�ⲿ�ִ������ǿ��Կ������������󽻵�˼�����̡���ǰ�ڵ�ѹ��������,����Ĭ�Ͻ��,�����ǰ�ڵ������������Ϊ0,��˵����Ҷ�ӽڵ�,��ôֱ�ӱ����ڵ��ڵ�
	//���������ж��Ƿ��ཻ,�����������ĿΪ0,˵����Ҫϸ��,��ô���Ǿ�Ҫ���ж�һ�¹����Ƿ��͵�ǰ�ڵ�������ӽڵ��AABB���ཻ,����ཻ,�ٶ���һ���ڵ����HitBVH�жϡ�
	//���շ��ؾ���Ͻ����Ǹ������
	if (root == NULL) return HitResult();
	if (root->n > 0)
	{
		return hitTriangleArray(ray, triangles, root->index, root->n + root->index - 1);
		/*
		�����Ҹо�������,��߽�Ϊɶ�����n��******������������,�󽻵ķ�Χ��ı�,Ӱ�����Ч��,���ǿ��ܲ�Ӱ����,�������ǻ���Ӧ��д�ԡ�
		*/
	}
	//����������AABB��
	float d1 = INF, d2 = INF;
	if (root->left) d1 = hitAABB(ray, root->left->AA, root->left->BB);//���ص���һ������,���Һ����0
	if (root->right) d2 = hitAABB(ray, root->right->AA, root->right->BB);

	//�ݹ���
	HitResult r1, r2;
	if (d1 > 0) r1 = hitBVH(ray, triangles, root->left);
	if (d2 > 0) r2 = hitBVH(ray, triangles, root->right);

	return r1.distance < r2.distance ? r1 : r2;
}


//��ʾ�ص�����
void display()
{
	mat4 unit(
		vec4(1, 0, 0, 0),
		vec4(0, 1, 0, 0),
		vec4(0, 0, 1, 0),
		vec4(0, 0, 0, 1));
	mat4 scaleMat = scale(unit, scaleControl);//xyz��������
	mat4 rotateMat = unit;
	rotateMat = rotate(rotateMat, radians(rotateControl.x), vec3(1, 0, 0));
	rotateMat = rotate(rotateMat, radians(rotateControl.y), vec3(0, 1, 0));
	rotateMat = rotate(rotateMat, radians(rotateControl.z), vec3(0, 0, 1));
	mat4 modelMat = rotateMat * scaleMat;

	GLuint mlocation = glGetUniformLocation(program, "model");
	glUniformMatrix4fv(mlocation, 1, GL_FALSE, value_ptr(modelMat));
	GLuint clocation = glGetUniformLocation(program, "color");
	//����
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//������������Ⱦģʽ,ȷ������Щ����Ա�����,������Ⱦ��ģʽΪ�߿�
	glUniform3fv(clocation, 1, value_ptr(vec3(1, 0, 0)));
	glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

	//����AABB����
	glUniform3fv(clocation, 1, value_ptr(vec3(1, 1, 1)));
	glDrawArrays(GL_LINES, vertices.size(), lines.size());
	glutSwapBuffers();//����������
}

//����˶�����
double lastX = 0.0, lastY = 0.0;
void mouse(int x, int y)
{
	//������ת
	rotateControl.y += -200 * (x - lastX) / 512;
	rotateControl.x += -200 * (y - lastY) / 512;
	/*
	*****************************�����xyΪʲôҪ������?
	* �������̫����,��Ϊ�����°�������ʱ��,ģ��ʵ������x��ת�İ�,ͬ��x�������Ұ�������ʱ��,ģ������y��ת��
	*/
	lastX = x, lastY = y;
	glutPostRedisplay();//�ػ�
}
void mouseDown(int button, int state, int x, int y)
{
	/*
	��ͺ�����Ȼ�ǲ�׽��갴�¶�����,������Ҫ�����ò�����Ϊ��ʵ�ֵ����תģ��,û�����������Ȼ�õ��������קģ��,
	����������������ڵ����ʱ��,ͬ����һ�ε����λ��,������������תģ�͵�ʱ�������ϴα���λ�ú͵�ǰλ�ò��̫��
	������ģ����̬�������䡣
	*/
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		lastX = x, lastY = y;
	}
}
void mouseWheel(int wheel, int direction, int x, int y)
{
	scaleControl.x += 1 * direction * 0.1;
	scaleControl.y += 1 * direction * 0.1;
	scaleControl.z += 1 * direction * 0.1;
	glutPostRedisplay();
}



int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(512, 512);
	glutCreateWindow("BVH");
	glewInit();
	readObj("./models/Stanford Bunny.obj", vertices, indices);
	for (auto& v : vertices)
	{
		v.x *= 5.0, v.y *= 5.0, v.z *= 5.0;
		v.y -= 0.5;//ֻ��һЩСС��ƽ�ƶ���
	}
	readObj("./models/quad.obj", vertices, indices);
	
	for (int i = 0; i < indices.size(); i += 3)
	{
		triangles.push_back(Triangle(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
	}
	//����BVH��
	BVHNode* root = buildBVHwithSAH(triangles, 0, triangles.size() - 1, 8);
	dfsNlevel(root, 0, 1);

	Ray ray;
	ray.startPoint = vec3(0, 0, 1);
	ray.direction = normalize(vec3(0.1, -0.1, -0.7));

	HitResult res = hitBVH(ray, triangles, root);
	addTriangle(res.triangle);
	addLine(ray.startPoint, ray.startPoint + ray.direction *vec3(5,5,5));

	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * (vertices.size() + lines.size()), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3) * vertices.size(), vertices.data());
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(vec3) * vertices.size(), sizeof(vec3) * lines.size(), lines.data());
	//����Ĵ������̶�Ҫ���ʼ�
	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	
	GLuint ebo;
	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices.size(), indices.data(), GL_STATIC_DRAW);

	std::string fshaderPath = "./shaders/fshader.fsh";
	std::string vshaderPath = "./shaders/vshader.vsh";
	program = getShaderProgram(fshaderPath, vshaderPath);
	glUseProgram(program);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);

	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glutDisplayFunc(display);
	glutMotionFunc(mouse);
	glutMouseFunc(mouseDown);
	glutMouseWheelFunc(mouseWheel);
	glutMainLoop();
	return 0;//
}
