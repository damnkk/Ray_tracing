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
GLuint program;							//着色器程序对象
std::vector<vec3> vertices;				//顶点坐标
std::vector<GLuint> indices;			//顶点索引
std::vector<vec3> lines;				//线段端点坐标
vec3 rotateControl(0, 0, 0);			//旋转参数
vec3 scaleControl(1, 1, 1);				//缩放参数

struct BVHNode
{
	BVHNode* left = NULL;//左子树
	BVHNode* right = NULL;//右子树
	int n, index;//叶子节点的信息
	vec3 AA, BB;//碰撞盒
};



typedef struct Triangle
{
	vec3 p1, p2, p3;
	vec3 center;//重心的作用代表了三角形的位置,而我们划分三角形到不同的包围盒的依据就是这个位置,而计算的方式就是三点分别取平均
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
	vec3 startPoint = vec3(0, 0, 0);//起点
	vec3 direction = vec3(0, 0, 0);//方向
}Ray;



std::string readShaderFile(std::string filepath)
{
	std::string res, line;
	std::ifstream fin(filepath);
	if (!fin.is_open())
	{
		std::cout << "文件" << filepath << "打开失败" << std::endl;
		exit(-1);
	}
	while (getline(fin, line))
	{
		res += line + "\n";//一行一个回车
	}
	fin.close();
	return res;
}

GLuint getShaderProgram(std::string fshader, std::string vshader)
{
	std::string vSource = readShaderFile(vshader);
	std::string fSource = readShaderFile(fshader);//读取着色器文件
	const char* vpointer = vSource.c_str();//并转换为C风格字符串
	const char* fpointer = fSource.c_str();

	//容错
	GLint success;
	GLchar infoLog[512];

	//创建并编译顶点着色器
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, (const GLchar**)(&vpointer), NULL);//创建一个着色器
	glCompileShader(vertexShader);//编译着色器
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);//用来检测着色器是否编译成功
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);//这里获取到具体的报错信息以后,会存储在这个char数组当中
		std::cout << "顶点着色器编译错误\n" << infoLog << std::endl;
		exit(-1);
	}
	//创建并编译片段着色器
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, (const GLchar**)(&fpointer), NULL);//创建一个片段着色器
	glCompileShader(fragmentShader);//
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "片段着色器编译错误\n" << infoLog << std::endl;
		exit(-1);
	}//到此编译结束
	
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	//删除着色器对象women 

	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);
	return shaderProgram;
}

void readObj(std::string filepath, std::vector<vec3>& vertices, std::vector<GLuint>& indices)
{
	std::ifstream fin(filepath);//从文件路径中打开文件流
	std::string line;
	if (!fin.is_open())//打开判断
	{
		std::cout << "ERROR::文件 " << filepath << " 打开失败" << std::endl;
		exit(-1);
	}

	//增量读取
	int offset = vertices.size();

	//按行读取
	while (getline(fin, line))//读取一行信息到line
	{
		std::istringstream sin(line);//将这一行的数据作为string stream解析并读取
		std::string type;
		GLfloat x, y, z;
		int v0, v1, v2;

		//读取obj文件
		sin >> type;//现在你就理解成已经读取了一行,一行的信息当中,第一个读入的应该是type,根据type决定初始化位置或者是索引,
		//因为.obj文件头还会有一些备注,因此读到不符合任意类型的信息,我们啥也不干,直接进行下一行的读取
		if (type == "v")
		{
			sin >> x >> y >> z;
			vertices.push_back(vec3(x, y, z));
		}
		if (type == "f")
		{
			sin >> v0 >> v1 >> v2;
			indices.push_back(v0 - 1 + offset);//obj中的索引是从1开始的,而我们容器的索引是从0开始的,因此要-1
			indices.push_back(v1 - 1 + offset);//我们也看到了,考虑到我们传入的顶点数组可能在一开始就有值,因此我们的索引除了-1之后,还要根据顶点数组的初始大小设置一个偏执量
			indices.push_back(v2 - 1 + offset);
		}
		/*
		* 此外继续深挖这个f类型,所谓的索引是什么呢?索引其实就是确定三角形面的,确定这个三角形面上三个顶点的位置坐标/纹理坐标/法向量,而当前这个教程
		* 提供的obj只提供了位置坐标,因此,它的索引数据这块并没有很多需要////来分割的数据块,只有个坐标的索引,也就很简单啦。
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
	//为什么三角形三条边,但是要push_back()六条边,而且还要把位置进行少量的偏移呢?因为你完全不改变位置的话,同一个三角形就会被红色白色渲染两次
	//就有可能造成遮挡,看不见到底哪个三角形,这样进行一点偏移以后,就不会造成遮挡,容易看清
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
	vec3 S = ray.startPoint;			//射线起点
	vec3 d = ray.direction;				//射线方向
	vec3 N = normalize(cross(p2 - p1, p3 - p1));	//法向量
	if (dot(N, d) > 0.0f) N = -N;		//获取正确的法向量

	//如果实现和三角形平行
	if (fabs(dot(d, N)) < 0.00001f) return INF;
	//距离
	float t = (-dot(S, N) + dot(N, p1)) / dot(d, N);
	if (t < 0.0005f) return INF;//t为负肯定是三角形在后面了,但是0以上的这一小部分,是为了消除一个反射光线的误差,
	//当交点被错误计算在了三角形面后面,则我们要假设没相交,因为按正常情况,在面上反射的光线怎们能算是与面相交呢?

	//交点计算
	vec3 P = S + d * t;
	//判断交点是否在三角形面中
	vec3 c1 = cross(p2 - p1, P - p1);
	vec3 c2 = cross(p3 - p2, P - p2);
	vec3 c3 = cross(p1 - p3, P - p3);
	if (dot(c1, N) > 0 && dot(c2, N) > 0 && dot(c3, N) > 0) return t;
	if (dot(c1, N) < 0 && dot(c2, N) < 0 && dot(c3, N) < 0) return t;

	return INF;
	
}




BVHNode* buildBVH(std::vector<Triangle>& triangles, int l, int r, int n)//n是三角形数目阈值,小于n就不再往下分了
{
	if (l > r) return 0;
	BVHNode* node = new BVHNode();
	node->AA = vec3(1145141919, 1145141919, 1145141919);//注意看这里的初始化技巧,AA是小的坐标,那么初始化为最大
	node->BB = vec3(-1145141919, -1145141919, -1145141919);//BB是大的坐标,那么初始化为最小

	//计算AABB
	for (int i = l; i <= r; ++i)
	{
		//最小点AA
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
	//小于等于n个三角形,建立叶子节点
	if ((r - l + 1) <= n)
	{
		node->n = r - l + 1;
		node->index = l;
		return node;
	}
	//否则递归建树
	float lenx = node->BB.x - node->AA.x;
	float leny = node->BB.y - node->AA.y;
	float lenz = node->BB.z - node->AA.z;
	//按x划分
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
		// 最小点 AA
		float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
		float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
		float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
		node->AA.x = min(node->AA.x, minx);
		node->AA.y = min(node->AA.y, miny);
		node->AA.z = min(node->AA.z, minz);
		// 最大点 BB
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
	//下面才是划分三角形的过程,因此是这里有变动
	float Cost = INF;
	int Axis = 0;
	int Split = (l + r) / 2;
	for (int axis = 0; axis < 3; ++axis)
	{
		//分别按xyz轴排序
		if (axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);//为什么第二个参数要r+1呢?因为end()迭代器指向最后一个元素的后面
		if (axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
		if (axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);
		//leftMax[i]:[l,i]中最大的xyz值
		//leftMin[i]:[l,i]中最小的xyz值
		/*
	这块用了一种比较绝妙的方法,一次性就把每种分割位置左边的最大最小值求出来了
		*/
		std::vector<vec3> leftMax(r - l + 1, vec3(-INF, -INF, -INF));
		std::vector<vec3> leftMin(r - l + 1, vec3(INF, INF, INF));
		for (int i = l; i <= r; ++i)
		{
			Triangle& t = triangles[i];
			int bias = (i == l) ? 0 : 1;//第一个元素特殊处理
			
			leftMax[i - l].x = max(leftMax[i - l - bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
			leftMax[i - l].y = max(leftMax[i - l - bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
			leftMax[i - l].z = max(leftMax[i - l - bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

			leftMin[i - l].x = min(leftMin[i - l - bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
			leftMin[i - l].y = min(leftMin[i - l - bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
			leftMin[i - l].z = min(leftMin[i - l - bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
			
		}

		//rightMax[i]:[i,r]中最大的xyz值
		//rightMin[i]:[i,r]中最小的xyz值
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
		//遍历寻找分割
		float cost = INF;
		int split = l;
		for (int i = l; i <= r - 1; ++i)
		{
			float lenx, leny, lenz;
			// 左侧 [l, i]
			vec3 leftAA = leftMin[i - l];
			vec3 leftBB = leftMax[i - l];
			lenx = leftBB.x - leftAA.x;
			leny = leftBB.y - leftAA.y;
			lenz = leftBB.z - leftAA.z;
			float leftS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));//包围盒表面积
			float leftCost = leftS * (i - l + 1);//总cost,盒子大,里面三角形还多,那肯定这个盒子成本就高了

			// 右侧 [i+1, r]
			vec3 rightAA = rightMin[i + 1 - l];
			vec3 rightBB = rightMax[i + 1 - l];
			lenx = rightBB.x - rightAA.x;
			leny = rightBB.y - rightAA.y;
			lenz = rightBB.z - rightAA.z;
			float rightS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
			float rightCost = rightS * (r - i);

			//记录每种分割的最小答案
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

	//递归
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

//AABB 盒子求交,没有交点则返回-1
float hitAABB(Ray r, vec3 AA, vec3 BB)
{
	vec3 invdir = vec3(1.0 / r.direction.x, 1.0 / r.direction.y, 1.0 / r.direction.z);
	/*
	***********************************************************
	注意看这里!! ,我这尝试着改一下,看能不能对
	但是感觉这下面的BB和AA是否调换也没啥影响,因为光线也有可能从离BB近的方向射入啊,因此这里的in,out也只不过只是一个初步的计算
	下面还要从两个向量当中选取短的作为入射点集,选取长的作为出射点集,并且从入射中选取最远, 从出射中选取最近
	*/

	vec3 in = (BB - r.startPoint) * invdir;//这个其实很好理解,对于每一个盒子的每一个分量来说,什么叫撞上哪个?就是起始点某轴分量+t*方向向量的某轴分量,然后t就很好求了。
	vec3 out = (AA - r.startPoint) * invdir;

	vec3 tmax = max(in, out);
	vec3 tmin = min(in, out);
	float t1 = min(tmax.x, min(tmax.y, tmax.z));
	float t0 = max(tmin.x, max(tmin.y, tmin.z));

	return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1);//t1<t0,直接就算没交到。在出射大于入射的情况下,如果入射为负,则说明光线起点在盒子内,则我们返回真正的交点距离为出射点的
	//t值,如果入射出射的t值都大于0,那么说明光线起点在包围盒之外,那么自然是两者取小,
}

//BVH上遍历求交
HitResult hitBVH(Ray ray, std::vector<Triangle>& triangles, BVHNode* root)
{//这部分代码我们可以看到整个光线求交的思考过程。当前节点压根不存在,返回默认结果,如果当前节点的三角形数不为0,则说明是叶子节点,那么直接遍历节点内的
	//三角形来判断是否相交,如果三角形数目为0,说明还要细分,那么我们就要先判断一下光线是否会和当前节点的左右子节点的AABB盒相交,如果相交,再对下一级节点进行HitBVH判断。
	//最终返回距离较近的那个结果。
	if (root == NULL) return HitResult();
	if (root->n > 0)
	{
		return hitTriangleArray(ray, triangles, root->index, root->n + root->index - 1);
		/*
		这里我感觉有问题,左边界为啥是这个n呢******这里是有问题,求交的范围会改变,影响求教效率,但是可能不影响结果,但是我们还是应该写对。
		*/
	}
	//和左右子树AABB求交
	float d1 = INF, d2 = INF;
	if (root->left) d1 = hitAABB(ray, root->left->AA, root->left->BB);//返回的是一个距离,并且恒大于0
	if (root->right) d2 = hitAABB(ray, root->right->AA, root->right->BB);

	//递归结果
	HitResult r1, r2;
	if (d1 > 0) r1 = hitBVH(ray, triangles, root->left);
	if (d2 > 0) r2 = hitBVH(ray, triangles, root->right);

	return r1.distance < r2.distance ? r1 : r2;
}


//显示回调函数
void display()
{
	mat4 unit(
		vec4(1, 0, 0, 0),
		vec4(0, 1, 0, 0),
		vec4(0, 0, 1, 0),
		vec4(0, 0, 0, 1));
	mat4 scaleMat = scale(unit, scaleControl);//xyz进行缩放
	mat4 rotateMat = unit;
	rotateMat = rotate(rotateMat, radians(rotateControl.x), vec3(1, 0, 0));
	rotateMat = rotate(rotateMat, radians(rotateControl.y), vec3(0, 1, 0));
	rotateMat = rotate(rotateMat, radians(rotateControl.z), vec3(0, 0, 1));
	mat4 modelMat = rotateMat * scaleMat;

	GLuint mlocation = glGetUniformLocation(program, "model");
	glUniformMatrix4fv(mlocation, 1, GL_FALSE, value_ptr(modelMat));
	GLuint clocation = glGetUniformLocation(program, "color");
	//绘制
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//这里设置了渲染模式,确定了哪些面可以被看到,还有渲染的模式为线框
	glUniform3fv(clocation, 1, value_ptr(vec3(1, 0, 0)));
	glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);

	//绘制AABB盒子
	glUniform3fv(clocation, 1, value_ptr(vec3(1, 1, 1)));
	glDrawArrays(GL_LINES, vertices.size(), lines.size());
	glutSwapBuffers();//交换缓冲区
}

//鼠标运动函数
double lastX = 0.0, lastY = 0.0;
void mouse(int x, int y)
{
	//调整旋转
	rotateControl.y += -200 * (x - lastX) / 512;
	rotateControl.x += -200 * (y - lastY) / 512;
	/*
	*****************************这里的xy为什么要换着用?
	* 这就是你太菜了,因为你上下扒拉鼠标的时候,模型实际是绕x轴转的啊,同样x方向左右扒拉鼠标的时候,模型是绕y轴转的
	*/
	lastX = x, lastY = y;
	glutPostRedisplay();//重绘
}
void mouseDown(int button, int state, int x, int y)
{
	/*
	这和函数虽然是捕捉鼠标按下动作的,但是主要的作用并不是为了实现点击拖转模型,没有这个函数依然得点击才能拖拽模型,
	这个函数的作用是在点击的时候,同步上一次的鼠标位置,这样不会在拖转模型的时候由于上次保存位置和当前位置差别太大
	而导致模型姿态发生跳变。
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
		v.y -= 0.5;//只是一些小小的平移而已
	}
	readObj("./models/quad.obj", vertices, indices);
	
	for (int i = 0; i < indices.size(); i += 3)
	{
		triangles.push_back(Triangle(vertices[indices[i]], vertices[indices[i + 1]], vertices[indices[i + 2]]));
	}
	//建立BVH树
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
	//这里的创建流程都要做笔记
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
