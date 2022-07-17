#include<iostream>
#include<iomanip>
#include<string>
#include<fstream>
#include<vector>
#include<sstream>
#include<algorithm>
#include<ctime>

#include<GL/glew.h>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <SOIL2/SOIL2.h>

#include"lib\hdrloader.h"
#define INF 114514.0
using namespace glm;


struct Material
{
	vec3 emissive = vec3(0, 0, 0);//作为光源的时候的发光颜色,默认为黑
	vec3 baseColor = vec3(1, 1, 1);//物体本身颜色,默认为白
	float subsurface = 0.0;//次表面,实现玉石效果
	float metallic = 0.0;//金属度
	float specular = 0.0;//镜面反射
	float specularTint = 0.0;//决定了镜面反射的颜色是否更接近物体本身的颜色还是更接近白色
	float roughness = 0.0;//粗糙度
	float anisotropic = 0.0;
	float sheen = 0.0;
	float sheenTint = 0.0;
	float clearcoat = 0.0;
	float clearcoatGloss = 0.0;
	float IOR = 1.0;
	float transmission = 0.0;
};

struct Triangle
{
	vec3 p1, p2, p3;
	vec3 n1, n2, n3;
	Material material;
};

struct BVHNode
{
	int left, right;
	vec3 AA, BB;
	int n, index;
};
struct Triangle_encoded//将三角形的属性编码成一堆三维向量
{
	vec3 p1, p2, p3;//本身就是三维向量的成员属性直接照搬即可
	vec3 n1, n2, n3;
	vec3 emissive;
	vec3 baseColor;
	vec3 param1;		//(subSurface,metallic,specular)
	vec3 param2;		//(specularTint,roughness,anisotropic)
	vec3 param3;		//(sheen,sheenTint,clearcoat)
	vec3 param4;		//(clearcoatCloss,IOR,transmission)
};
struct BVHNode_encoded
{
	vec3 childs;
	vec3 leafInfo;
	vec3 AA, BB;
};

class RenderPass
{
public:
	GLuint FBO = 0;
	GLuint vao, vbo;
	std::vector<GLuint>colorAttachments;//这个容器存储着要传入下一pass的纹理id
	GLuint program;
	int height = 512;
	int width = 512;
	void bindData(bool finalPass = false)
	{
		if (!finalPass) glGenFramebuffers(1, &FBO);//如果不是最后一个过程,就建立一个帧缓冲
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);//绑定到当前的帧缓冲,对应下面处理完了就解绑

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		std::vector<vec3> square = { vec3(-1, -1, 0), vec3(1, -1, 0), vec3(-1, 1, 0), vec3(1, 1, 0), vec3(-1, 1, 0), vec3(1, -1, 0) };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * square.size(), NULL, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3) * square.size(), &square[0]);

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
		//不是finalPass则生成帧缓冲的颜色附件
		if (!finalPass)
		{
			std::vector<GLuint> attachments;
			for (int i = 0; i < colorAttachments.size(); ++i)
			{
				glBindTexture(GL_TEXTURE_2D, colorAttachments[i]);//给要传入下一pass的纹理ID绑定到一个2D纹理上
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorAttachments[i], 0); //将颜色纹理绑定到 i 号颜色附件,所谓颜色附件就是GL_COLOR_ATTACHMENT0
				attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
			}
			glDrawBuffers(attachments.size(), &attachments[0]);//虽然带Draw,但并不是一个绘制指令,只是确定了绘制颜色组件的顺序
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);//解绑
	}
	void draw(std::vector<GLuint>texPassArray = { })
	{
		glUseProgram(program);//管线配置的时候设定了当前pass的shader
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);//渲染当前pass就绑定当前pass的帧缓冲,每个pass自带一个帧缓冲
		glBindVertexArray(vao);//将每一帧的顶点数组对象绑定好,也就是支棱好画布
		//传上一帧的帧缓冲颜色附件
		for (int i = 0; i < texPassArray.size(); ++i)
		{
			glActiveTexture(GL_TEXTURE0 + i);//激活纹理
			glBindTexture(GL_TEXTURE_2D, texPassArray[i]);//这里是将上一个pass的颜色组件传进来绑定到纹理,注意这个顺序,先激活,现在绑定
			std::string nName = "texPass" + std::to_string(i);
			glUniform1i(glGetUniformLocation(program, nName.c_str()), i);
		}
		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//每次清空颜色缓冲和深度缓冲
		glDrawArrays(GL_TRIANGLES, 0, 6);//画布的两个三角形,6个顶点

		glBindVertexArray(0);//这三行对应于函数开头前三行,前面各种激活,各种绑定,现在各种解绑
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glUseProgram(0);


	}
};
//----------------------------------------------------------------//
GLuint trianglesTextureBuffer;
GLuint nodesTextureBuffer;
GLuint lastFrame;
GLuint hdrMap;
GLuint hdrCache;
int hdrResolution;

RenderPass pass1;
RenderPass pass2;
RenderPass pass3;

//相机参数
float upAngle = 0.0;//调整上下
float rotatAngle = 0.0;//调整左右
float r = 4.0;





bool cmpx(const Triangle& t1, const Triangle& t2) {
	vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
	vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
	return center1.x < center2.x;
}
bool cmpy(const Triangle& t1, const Triangle& t2) {
	vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
	vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
	return center1.y < center2.y;
}
bool cmpz(const Triangle& t1, const Triangle& t2) {
	vec3 center1 = (t1.p1 + t1.p2 + t1.p3) / vec3(3, 3, 3);
	vec3 center2 = (t2.p1 + t2.p2 + t2.p3) / vec3(3, 3, 3);
	return center1.z < center2.z;
}

std::string readShaderFile(std::string filepath)
{
	std::string res, line;
	std::ifstream fin(filepath);
	if (!fin.is_open())
	{
		std::cout << "ERROR:: 文件" << filepath << "  打开失败" << std::endl;
		exit(-1);
	}
	while (std::getline(fin, line))
	{
		res += line + "\n";
	}
	fin.close();
	return res;
}

GLuint getTextureRGB32F(int width, int height)
{
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	return tex;
}

// 获取着色器对象
GLuint getShaderProgram(std::string fshader, std::string vshader) {
	// 读取shader源文件
	std::string vSource = readShaderFile(vshader);
	std::string fSource = readShaderFile(fshader);
	const char* vpointer = vSource.c_str();
	const char* fpointer = fSource.c_str();

	//错误保存
	GLint success;
	GLchar infoLog[512];

	//创建编译顶点着色器
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, (const GLchar**)(&vpointer), NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);   // 错误检测
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "顶点着色器编译错误\n" << infoLog << std::endl;
		exit(-1);
	}

	//片段着色器
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, (const GLchar**)(&fpointer), NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);   // 错误检测
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "片段着色器编译错误\n" << infoLog << std::endl;
		exit(-1);
	}

	// 链接两个着色器到program对象
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// 删除着色器对象
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}
mat4 getTransformMatrix(vec3 rotateCtrl, vec3 translateCtrl, vec3 scaleCtrl)
{
	glm::mat4 uint(
		glm::vec4(1, 0, 0, 0),
		glm::vec4(0, 1, 0, 0),
		glm::vec4(0, 0, 1, 0),
		glm::vec4(0, 0, 0, 1)
	);
	mat4 scale = glm::scale(uint, scaleCtrl);
	mat4 translate = glm::translate(uint, translateCtrl);
	mat4 rotate = uint;
	rotate = glm::rotate(rotate, glm::radians(rotateCtrl.x), glm::vec3(1, 0, 0));//x分量转换为角度,对x轴进行旋转
	rotate = glm::rotate(rotate, glm::radians(rotateCtrl.y), glm::vec3(0, 1, 0));
	rotate = glm::rotate(rotate, glm::radians(rotateCtrl.z), glm::vec3(0, 0, 1));
	mat4 model = translate * rotate * scale;//顺序为位移->旋转->缩放
	return model;
}

void readObj(std::string filepath, std::vector<Triangle>& triangles, Material material, mat4 trans, bool smoothNormal)
{
	//顶点位置,索引
	std::vector<vec3> vertices;
	std::vector<GLuint> indices;
	//打开文件流
	std::ifstream fin(filepath);
	std::string line;

	if (!fin.is_open())
	{
		std::cout << "ERROR::文件 " << filepath << "  打开失败" << std::endl;
		exit(-1);
	}

	float maxx = -11451419.19;
	float maxy = -11451419.19;
	float maxz = -11451419.19;
	float minx = 11451419.19;
	float miny = 11451419.19;
	float minz = 11451419.19;

	//按行读取
	while (std::getline(fin, line))
	{
		std::istringstream sin(line);
		std::string type;
		GLfloat x, y, z;
		int v0, v1, v2;
		int vn0, vn1, vn2;
		int vt0, vt1, vt2;
		char slash;

		//统计斜杠数目
		int slashcnt = 0;
		for (int i = 0; i < line.size(); ++i)
		{
			if (line[i] == '/')
			{
				++slashcnt;
			}
		}
		//读取obj文件
		sin >> type;
		if (type == "v")
		{
			sin >> x >> y >> z;
			vertices.push_back(vec3(x, y, z));
			maxx = max(maxx, x), maxy = max(maxy, y), maxz = max(maxz, z);
			minx = min(minx, x), miny = min(miny, y), minz = min(minz, z);
		}
		if (type == "f")
		{
			if (slashcnt == 6)
			{
				sin >> v0 >> slash >> vt0 >> slash >> vn0;//分别是顶点索引,法向量索引和纹理坐标索引
				sin >> v1 >> slash >> vt1 >> slash >> vn1;
				sin >> v2 >> slash >> vt2 >> slash >> vn2;
			}
			else if (slashcnt == 3)//如果索引/分为2部分的话,只有顶点索引和纹理坐标索引了
			{
				sin >> v0 >> slash >> vt0;
				sin >> v1 >> slash >> vt1;
				sin >> v2 >> slash >> vt2;
			}
			else
			{
				sin >> v0 >> v1 >> v2;
			}
			indices.push_back(v0 - 1);//因为顶点数组的索引是从0开始的,而obj文件里的索引是从1开始,因此整体左移一位
			indices.push_back(v1 - 1);//此外,这里好像只提取顶点位置的索引,什么法向量,什么纹理坐标啥的都舍弃了
			indices.push_back(v2 - 1);
		}
	}

	//模型大小归一化
	float lenx = maxx - minx;
	float leny = maxy - miny;
	float lenz = maxz - minz;
	float maxaxis = max(lenx, max(leny, lenz));
	for (auto& v : vertices)
	{
		v.x /= maxaxis;
		v.y /= maxaxis;
		v.z /= maxaxis;
	}
	for (auto& v : vertices)
	{
		vec4 vv = vec4(v.x, v.y, v.z, 1);//先升维成四维齐次向量
		vv = trans * vv;
		v = vec3(vv.x, vv.y, vv.z);
	}

	std::vector<vec3> normals(vertices.size(), vec3(0, 0, 0));
	for (int i = 0; i < indices.size(); i += 3)
	{
		vec3 p1 = vertices[indices[i]];
		vec3 p2 = vertices[indices[i + 1]];
		vec3 p3 = vertices[indices[i + 2]];
		vec3 n = normalize(cross(p2 - p1, p3 - p1));
		normals[indices[i]] += n;
		normals[indices[i + 1]] += n;
		normals[indices[i + 2]] += n;
	}
	//构建Triangles对象数组

	int offset = triangles.size();  // 增量更新
	triangles.resize(offset + indices.size() / 3);

	for (int i = 0; i < indices.size(); i += 3)
	{
		Triangle& t = triangles[offset + i / 3];
		//传递顶点属性
		t.p1 = vertices[indices[i]];
		t.p2 = vertices[indices[i + 1]];
		t.p3 = vertices[indices[i + 2]];

		if (!smoothNormal)
		{
			vec3 n = normalize(cross(t.p2 - t.p1, t.p3 - t.p1));//这里是根据同一个三角形的三个点进行计算的
			t.n1 = n; t.n2 = n; t.n3 = n;
		}
		else
		{
			t.n1 = normalize(normals[indices[i]]);//这里是依据之前,不是同一个三角形的三个点计算得到的法向量进行赋值
			t.n2 = normalize(normals[indices[i + 1]]);//也正是因为存在这个选项,我们再片段着色器中计算交点法向量的时候需要专门使用插值计算
			t.n3 = normalize(normals[indices[i + 2]]);//因为三角形三个点的法向量不一定是一个方向的。
		}
		t.material = material;
	}
}

int buildBVH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n)
{
	//其实很好理解,之前BVH建树是因为可以使用指针,因此传进来一个节点即可,现在不行了,那就维护一个BVHNode数组
	if (l > r)
	{
		return 0;
	}
	// 注：
	// 此处不可通过指针，引用等方式操作，必须用 nodes[id] 来操作
	// 因为 std::vector<> 扩容时会拷贝到更大的内存，那么地址就改变了
	// 而指针，引用均指向原来的内存，所以会发生错误
	nodes.push_back(BVHNode()); //如果左右边界没有相撞,那么就建立一个节点,这里的建立方式就是默认构造,然后push_back();
	int id = nodes.size() - 1;//得到当前节点的索引
	nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
	nodes[id].AA = vec3(1145141919, 1145141919, 1145141919);
	nodes[id].BB = vec3(-1145141919, -1145141919, -1145141919);

	for (int i = l; i <= r; i++) {
		// 最小点 AA
		float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
		float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
		float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
		nodes[id].AA.x = min(nodes[id].AA.x, minx);
		nodes[id].AA.y = min(nodes[id].AA.y, miny);
		nodes[id].AA.z = min(nodes[id].AA.z, minz);
		// 最大点 BB
		float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
		float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
		float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
		nodes[id].BB.x = max(nodes[id].BB.x, maxx);
		nodes[id].BB.y = max(nodes[id].BB.y, maxy);
		nodes[id].BB.z = max(nodes[id].BB.z, maxz);
	}
	if ((r - l + 1) <= n) {
		nodes[id].n = r - l + 1;
		nodes[id].index = l;
		return id;
	}
	float lenx = nodes[id].BB.x - nodes[id].AA.x;
	float leny = nodes[id].BB.y - nodes[id].AA.y;
	float lenz = nodes[id].BB.z - nodes[id].AA.z;

	if (lenx >= leny && lenx >= lenz)
	{
		std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpx);
	}
	if (leny >= lenx && leny >= lenz)
	{
		std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpy);
	}
	if (lenz >= lenx && lenz >= leny)
	{
		std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpz);
	}
	int mid = (l + r) / 2;
	int left = buildBVH(triangles, nodes, l, mid, n);
	int right = buildBVH(triangles, nodes, mid + 1, r, n);
	nodes[id].left = left;
	std::cout << nodes[id].left << std::endl;
	nodes[id].right = right;
	return id;
}

int buildBVHwithSAH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n)
{
	if (l > r)
	{
		std::cout << "weisha" << std::endl;
		return 0;
	}

	nodes.push_back(BVHNode());
	int id = nodes.size() - 1;
	nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
	nodes[id].AA = vec3(1145141919, 1145141919, 1145141919);
	nodes[id].BB = vec3(-1145141919, -1145141919, -1145141919);
	// 计算 AABB
	for (int i = l; i <= r; i++) {
		// 最小点 AA
		float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
		float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
		float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
		nodes[id].AA.x = min(nodes[id].AA.x, minx);
		nodes[id].AA.y = min(nodes[id].AA.y, miny);
		nodes[id].AA.z = min(nodes[id].AA.z, minz);
		// 最大点 BB
		float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
		float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
		float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
		nodes[id].BB.x = max(nodes[id].BB.x, maxx);
		nodes[id].BB.y = max(nodes[id].BB.y, maxy);
		nodes[id].BB.z = max(nodes[id].BB.z, maxz);
	}
	// 不多于 n 个三角形 返回叶子节点
	if ((r - l + 1) <= n) {
		nodes[id].n = r - l + 1;
		nodes[id].index = l;
		return id;
	}

	// 否则递归建树
	float Cost = INF;
	int Axis = 0;
	int Split = (l + r) / 2;
	for (int axis = 0; axis < 3; axis++) {
		// 分别按 x，y，z 轴排序
		if (axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
		if (axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
		if (axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);

		// leftMax[i]: [l, i] 中最大的 xyz 值
		// leftMin[i]: [l, i] 中最小的 xyz 值
		std::vector<vec3> leftMax(r - l + 1, vec3(-INF, -INF, -INF));
		std::vector<vec3> leftMin(r - l + 1, vec3(INF, INF, INF));
		// 计算前缀 注意 i-l 以对齐到下标 0
		for (int i = l; i <= r; i++) {
			Triangle& t = triangles[i];
			int bias = (i == l) ? 0 : 1;  // 第一个元素特殊处理

			leftMax[i - l].x = max(leftMax[i - l - bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
			leftMax[i - l].y = max(leftMax[i - l - bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
			leftMax[i - l].z = max(leftMax[i - l - bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

			leftMin[i - l].x = min(leftMin[i - l - bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
			leftMin[i - l].y = min(leftMin[i - l - bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
			leftMin[i - l].z = min(leftMin[i - l - bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
		}

		std::vector<vec3> rightMax(r - l + 1, vec3(-INF, -INF, -INF));
		std::vector<vec3> rightMin(r - l + 1, vec3(INF, INF, INF));
		// 计算后缀 注意 i-l 以对齐到下标 0
		for (int i = r; i >= l; i--) {
			Triangle& t = triangles[i];
			int bias = (i == r) ? 0 : 1;  // 第一个元素特殊处理

			rightMax[i - l].x = max(rightMax[i - l + bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
			rightMax[i - l].y = max(rightMax[i - l + bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
			rightMax[i - l].z = max(rightMax[i - l + bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

			rightMin[i - l].x = min(rightMin[i - l + bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
			rightMin[i - l].y = min(rightMin[i - l + bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
			rightMin[i - l].z = min(rightMin[i - l + bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
		}

		//遍历寻找分割
		float cost = INF;
		float split = l;
		for (int i = l; i <= r - 1; ++i)
		{//i是什么?i是分隔界限啊,在这个地方,i是左节点的有边界,i+1是右节点的左边界,也正因为如此,我们的i可以从l开始,但是要在r-1结束
			float lenx, leny, lenz;
			vec3 leftAA = leftMin[i - l];
			vec3 leftBB = leftMax[i - l];
			lenx = leftBB.x - leftAA.x;
			leny = leftBB.y - leftAA.y;
			lenz = leftBB.z - leftAA.z;
			float leftS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
			float leftcost = leftS * (i - l + 1);

			vec3 rightAA = rightMin[i - l + 1];
			vec3 rightBB = rightMax[i - l + 1];
			lenx = rightBB.x - rightAA.x;
			leny = rightBB.y - rightAA.y;
			lenz = rightBB.z - rightAA.z;
			float rightS = 2.0 * ((lenx * leny) + (lenx * lenz) + (leny * lenz));
			float rightcost = rightS * (r - i);

			float totalCost = rightcost + leftcost;
			if (totalCost < cost)
			{
				cost = totalCost;
				split = i;
			}
		}
		if (cost < Cost)//比较的核心就是这个耗费
		{
			Cost = cost;
			Axis = axis;
			Split = split;
		}
	}
	if (Axis == 0) std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpx);
	if (Axis == 1) std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpy);
	if (Axis == 2) std::sort(triangles.begin() + l, triangles.begin() + r + 1, cmpz);


	//std::cout << buildBVHwithSAH(triangles, nodes, Split + 1, r, n) << std::endl;
	//int left = buildBVHwithSAH(triangles, nodes, l, Split, n);
	//int right = buildBVHwithSAH(triangles, nodes, Split + 1, r, n);
	nodes[id].left = buildBVHwithSAH(triangles, nodes, l, Split, n); //注意了,这里
	//std::cout << "nodes[id].left = " << nodes[id].left << std::endl;

	nodes[id].right = buildBVHwithSAH(triangles, nodes, Split + 1, r, n);;

	return id;
}

float* calculateHdrCache(float* HDR, int width, int height)
{
	float lumSum = 0.0;

	//初始化h行w列的概率密度pdf并统计总亮度
	std::vector<std::vector<float>> pdf(height);
	for (auto& line : pdf) line.resize(width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float R = HDR[3 * (i * width + j)];
			float G = HDR[3 * (i * width + j) + 1];
			float B = HDR[3 * (i * width + j) + 2];
			float lum = 0.2 * R + 0.7 * G + 0.1 * B;//这个是RGB转换亮度的固定公式
			pdf[i][j] = lum;//统计每个像素的亮度
			lumSum += lum;//统计亮度之和
		}
	}

	//概率密度归一化
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			pdf[i][j] /= lumSum;
		}
	}
	//累加每一列得到x的边缘概率密度
	std::vector<float>pdf_x_margin;
	pdf_x_margin.resize(width);
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
			pdf_x_margin[j] += pdf[i][j];

	//计算x的边缘分布函数
	std::vector<float> cdf_x_margin = pdf_x_margin;
	for (int i = 1; i < width; i++)
		cdf_x_margin[i] += cdf_x_margin[i - 1];//这就是所谓前缀和

	//计算y在X=x下的条件概率密度函数
	std::vector<std::vector<float>> pdf_y_condiciton = pdf;
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
			pdf_y_condiciton[i][j] /= pdf_x_margin[j];

	//计算y在X=x下的条件概率分布函数
	std::vector<std::vector<float>> cdf_y_condiciton = pdf_y_condiciton;
	for (int j = 0; j < width; j++)
		for (int i = 1; i < height; i++)
			cdf_y_condiciton[i][j] += cdf_y_condiciton[i - 1][j];

	//cdf_y_condiciton 转置未按列存储
	//cdf_y_condiciton[i]表示y在X=i下的条件概率分布函数
	std::vector<std::vector<float>> temp = cdf_y_condiciton;
	cdf_y_condiciton = std::vector<std::vector<float>>(width);
	for (auto& line : cdf_y_condiciton) line.resize(height);
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
			cdf_y_condiciton[j][i] = temp[i][j];

	// 穷举 xi_1, xi_2 预计算样本 xy
	// sample_x[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 x
	// sample_y[i][j] 表示 xi_1=i/height, xi_2=j/width 时 (x,y) 中的 y
	// sample_p[i][j] 表示取 (i, j) 点时的概率密度

	std::vector<std::vector<float>> sample_x(height);
	for (auto& line : sample_x) line.resize(width);
	std::vector<std::vector<float>> sample_y(height);
	for (auto& line : sample_y) line.resize(width);
	std::vector<std::vector<float>> sample_p(height);
	for (auto& line : sample_p) line.resize(width);
	for (int j = 0; j < width; ++j)
	{
		for (int i = 0; i < height; ++i)
		{
			float xi_1 = float(i) / height;//这里可以看到相当于将整个HDR图像映射到[0,1]^2的范围,并且对这个范围均匀预计算
			float xi_2 = float(j) / width;//了一遍,将结果存储到一个纹理当中,到时候直接在纹理里头采样插值很快得结果

			//用xi_1在cdf_x_margin中lowerbound 得到样本x,别忘了cdf_x_margin是那个1维的累计数组
			int x = std::lower_bound(cdf_x_margin.begin(), cdf_x_margin.end(), xi_1) - cdf_x_margin.begin();//lower_bound返回一个最小大于的迭代器,
			//和begin()迭代器一减,就是距离,也就是对应的x坐标了,真的很妙
			//用xi_2在X=x的情况下得到样本y
			int y = std::lower_bound(cdf_y_condiciton[x].begin(), cdf_y_condiciton[x].end(), xi_2) - cdf_y_condiciton[x].begin();
			//存储纹理坐标xy和xy位置对应的概率密度
			sample_x[i][j] = float(x) / width;//这里应该是为了归一化,到时候直接拿来构建立体角旋转角来确立采样向量。
			sample_y[i][j] = float(y) / height;
			sample_p[i][j] = pdf[i][j];
		}
	}
	//整合结果到纹理
	//R,G通道存储样本(x,y),而B通道存储pdf(i,j)
	//我们是将计算结果存储到一块纹理当中,作为一个纹理,我们用其RGB三个通道分别存储三个计算值
	//简单来说,我们预先均匀地计算了很多随机数采样对应的xy值,存在纹理中,到时候直接查表就快了。
	float* cache = new float[width * height * 3];
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			cache[3 * (i * width + j)] = sample_x[i][j];	
			cache[3 * (i * width + j) + 1] = sample_y[i][j];
			cache[3 * (i * width + j) + 2] = sample_p[i][j];
		}
	}
	return cache;
}



//绘制
clock_t t1, t2;
double dt, fps;
int  frameCounter = 0;
void display()
{
	//帧计时
	t2 = clock();
	dt = (double)(t2 - t1) / CLOCKS_PER_SEC;//计时粒度为1000毫秒
	fps = 1.0 / dt;
	std::cout << "\r";
	std::cout << std::fixed << std::setprecision(2) << "FPS: " << fps << "    迭代次数" << frameCounter;//这里禁用了科学计数法,精度保留到小数点2位之后
	t1 = t2;//别忘了更新计时器
	//相机参数
	vec3 eye = vec3(-sin(radians(rotatAngle)) * cos(radians(upAngle)), sin(radians(upAngle)), cos(radians(rotatAngle)) * cos(radians(upAngle)));
	eye.x *= r; eye.y *= r; eye.z *= r;//这里的eye我们一定明确是摄像机在世界坐标系当中的坐标,由此可以看出,我们的摄像机是围绕世界坐标系当中的(0,0,0)为圆心,4为半径的球面上移动
	mat4 cameraRotate = lookAt(eye, vec3(0, 0, 0), vec3(0, 1, 0));//第二个参数本应该是eye+front,但是我们一直看向原点,因此front = -eye,最后抵消为(0,0,0),并且头顶向量为(0,1,0)
	//更新一下理解:lookAt得到的就是视图变换矩阵,也就是世界坐标->摄像机坐标的变换矩阵
	cameraRotate = inverse(cameraRotate);//而这里对于lookAt的结果又进行了一次转置,因此cameraRotate就变成了摄像机坐标->世界坐标的变换矩阵,我们又知道,当时在读取模型的时候,我们进行了模型变换,也就是说,
	//我们读进来的模型是处于世界坐标系。我们知道光线是从摄像机为起点,指向画布的每一个像素的,但是shader中直接以画布的xy轴和画布的深度作为光线变换前的方向,这个很好理解,这里可以看出画布是在摄像机坐标系当中, 
	//那么摄像机的位置就是原点(0,0,0)了,因此可以这么表示,而我们又知道shader中设定的光线的起点是以摄像机在世界坐标系当中设定的,因此我们直接用画布像素坐标得来的这个光线方向还要从摄像机坐标转换到世界坐标系当中,
	//才能和我们世界坐标系当中的光线起点相吻合,因此,我们直接lookAt取反,即可得到世界坐标系当中的光线,因此它和世界坐标系中的模型求交,没有任何违和感

	//传uniform给pass1
	glUseProgram(pass1.program);
	glUniform3fv(glGetUniformLocation(pass1.program, "eye"), 1, value_ptr(eye));
	glUniformMatrix4fv(glGetUniformLocation(pass1.program, "cameraRotate"), 1, GL_FALSE, value_ptr(cameraRotate));
	glUniform1i(glGetUniformLocation(pass1.program, "frameCounter"), frameCounter++);// 传计数器用作随机种子
	glUniform1i(glGetUniformLocation(pass1.program, "hdrResolution"), hdrResolution);    //hdr cache



	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);
	glUniform1i(glGetUniformLocation(pass1.program, "triangles"), 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
	glUniform1i(glGetUniformLocation(pass1.program, "nodes"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, lastFrame);
	glUniform1i(glGetUniformLocation(pass1.program, "lastFrame"), 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, hdrMap);
	glUniform1i(glGetUniformLocation(pass1.program, "hdrMap"), 3);

	glActiveTexture(GL_TEXTURE4);
	glBindTexture(GL_TEXTURE_2D, hdrCache);
	glUniform1i(glGetUniformLocation(pass1.program, "hdrCache"), 4);

	//绘制
	pass1.draw();
	pass2.draw(pass1.colorAttachments);
	pass3.draw(pass2.colorAttachments);

	glutSwapBuffers();
}
void framefunc()
{
	glutPostRedisplay();//可选地通知程序重新绘制画面
}
double lastX = 0.0, lastY = 0.0;
void mouse(int x, int y)
{
	frameCounter = 0;
	//调整旋转
	rotatAngle += 150 * (x - lastX) / 512;
	upAngle += 150 * (y - lastY) / 512;
	upAngle = min(upAngle, 89.0f);
	upAngle = max(upAngle, -89.0f);
	lastX = x; lastY = y;
	glutPostRedisplay();//鼠标控制动了,就重绘,从实际的表现来看,这个重绘不是刷新画面,而是完全从头开始累计光追
}

void mouseDown(int button, int state, int x, int y)
{
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)
	{
		lastX = x, lastY = y;
	}
}

void mouseWheel(int wheel, int direction, int x, int y)
{
	frameCounter = 0;
	r += -direction * 0.5;
	glutPostRedisplay();//依然是完全从头绘制
}

int main(int argc, char** argv)
{

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(350, 50);
	glutInitWindowSize(512, 512);
	glutCreateWindow("path_tracing_GPU");
	glewInit();

	std::vector<Triangle> triangles;

	Material m;
	m.baseColor = vec3(1.0,0.84,0.0);
	m.roughness = 0.6;
	m.metallic = 1.0;
	m.specular = 0.8;
	m.clearcoat = 0.2;
	m.clearcoatGloss = 0.2;
	//m.anisotropic = 0.1;
	//m.subsurface = 1.0;
	//readObj("models/Stanford Bunny.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(-0.2, -1.6, 0), vec3(1.5, 1.5, 1.5)), true);
	//readObj("models/room.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0.0, -2.5, 0), vec3(10, 10, 10)), true);

	readObj("models/nanosuit.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(-0.8, 0.2, 0), vec3(3.0, 3.0, 3.0)), true);
	m.baseColor = vec3(1.0,1.0,1.0);
	
	m.roughness = 0.05;
	m.clearcoat = 0.2;
	m.metallic = 0.0;
	m.clearcoatGloss = 1.0;
	readObj("models/quad.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0, 0.2, 0), vec3(18.83, 0.01, 18.83)), false);

	//m.baseColor = vec3(1, 1, 1);
	////m.emissive = vec3(30, 20, 10);
	////readObj("models/quad.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0.0, 1.38, -0.0), vec3(0.7, 0.01, 0.7)), false);
	//readObj("models/sphere.obj", triangles, m, getTransformMatrix(vec3(0, 0, 0), vec3(0.0, 0.9, -0.0), vec3(1, 1, 1)), false);

	int nTriangles = triangles.size();
	std::cout << "模型读取完成: 共 " << nTriangles << " 个三角形" << std::endl;

	// 建立 bvh
	BVHNode testNode;
	testNode.left = 255;
	testNode.right = 128;
	testNode.n = 30;
	testNode.AA = vec3(1, 1, 0);
	testNode.BB = vec3(0, 1, 0);
	std::vector<BVHNode> nodes{ testNode };
	//buildBVH(triangles, nodes, 0, triangles.size() - 1, 8);
	buildBVHwithSAH(triangles, nodes, 0, triangles.size() - 1, 8);
	int nNodes = nodes.size();
	std::cout << "BVH 建立完成: 共 " << nNodes << " 个节点" << std::endl;

	//编码三角形材质
	std::vector<Triangle_encoded> triangles_encoded(nTriangles);
	for (int i = 0; i < nTriangles; i++) {
		Triangle& t = triangles[i];
		Material& m = t.material;
		// 顶点位置
		triangles_encoded[i].p1 = t.p1;
		triangles_encoded[i].p2 = t.p2;
		triangles_encoded[i].p3 = t.p3;
		// 顶点法线
		triangles_encoded[i].n1 = t.n1;
		triangles_encoded[i].n2 = t.n2;
		triangles_encoded[i].n3 = t.n3;
		// 材质
		triangles_encoded[i].emissive = m.emissive;
		triangles_encoded[i].baseColor = m.baseColor;
		triangles_encoded[i].param1 = vec3(m.subsurface, m.metallic, m.specular);
		triangles_encoded[i].param2 = vec3(m.specularTint, m.roughness, m.anisotropic);
		triangles_encoded[i].param3 = vec3(m.sheen, m.sheenTint, m.clearcoat);
		triangles_encoded[i].param4 = vec3(m.clearcoatGloss, m.IOR, m.transmission);
	}
	//编码BVHNode
	std::vector<BVHNode_encoded> nodes_encoded(nNodes);
	for (int i = 0; i < nNodes; i++) {
		nodes_encoded[i].childs = vec3(nodes[i].left, nodes[i].right, 0);
		nodes_encoded[i].leafInfo = vec3(nodes[i].n, nodes[i].index, 0);
		nodes_encoded[i].AA = nodes[i].AA;
		nodes_encoded[i].BB = nodes[i].BB;
	}

	//简单来说就是先创建了一块显存,将编码之后的三角形写进去,接下来创建一个纹理,将纹理类型设置为GL_TEXTURE_BUFFER,并且将纹理的缓冲设置为我们之前存
	//放编码三角形的的那块显存,这样这个纹理对应的内容就是我们的编码三角形了


	GLuint tbo0;
	glGenBuffers(1, &tbo0);//生成一个缓冲,绑定到tbo0上
	glBindBuffer(GL_TEXTURE_BUFFER, tbo0);//这一步才真正定义了缓冲的类型
	glBufferData(GL_TEXTURE_BUFFER, sizeof(Triangle_encoded) * triangles_encoded.size(), triangles_encoded.data(), GL_STATIC_DRAW);//缓冲的具体数据写进来
	glGenTextures(1, &trianglesTextureBuffer);//类似的方式又建立了一块纹理
	glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);//纹理类型绑定为纹理缓冲
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo0);//纹理的缓冲定为tbo0,在这里就将我们的纹理具体的内容和前面创建的缓冲绑定在一起了,纹理和数据的关系就是这么建立的

	GLuint tbo1;
	glGenBuffers(1, &tbo1);
	glBindBuffer(GL_TEXTURE_BUFFER, tbo1);
	glBufferData(GL_TEXTURE_BUFFER, nodes_encoded.size() * sizeof(BVHNode_encoded), &nodes_encoded[0], GL_STATIC_DRAW);
	glGenTextures(1, &nodesTextureBuffer);
	glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo1);

	// hdr 全景图
	HDRLoaderResult hdrRes;
	bool r = HDRLoader::load("./HDR/peppermint_powerplant_4k.hdr", hdrRes);
	//bool r = HDRLoader::load("./HDR/chinese_garden_2k.hdr", hdrRes);
	hdrMap = getTextureRGB32F(hdrRes.width, hdrRes.height);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, hdrRes.cols);
	
	//hdr重要性采样 cache
	std::cout << "计算HDR贴图重要性采样Cache,当前分辨率: " << hdrRes.width << " " << hdrRes.height << std::endl;
	hdrCache = getTextureRGB32F(hdrRes.width, hdrRes.height);
	float* cache = calculateHdrCache(hdrRes.cols, hdrRes.width, hdrRes.height);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, cache);
	hdrResolution = hdrRes.width;
	std::cout << cache[23] << std::endl;
	// ----------------------------------------------------------------------------- //

	// 管线配置

	pass1.program = getShaderProgram("./shaders/fshader.fsh", "./shaders/vshader.vsh");
	//pass1.width = pass1.height = 256;

	//pass1.colorAttachments.push_back(getTextureRGB32F(pass1.width, pass1.height));
	pass1.colorAttachments.push_back(getTextureRGB32F(pass1.width, pass1.height));
	//pass1.colorAttachments.push_back(getTextureRGB32F(pass1.width, pass1.height));
	pass1.bindData();

	glUseProgram(pass1.program);
	glUniform1i(glGetUniformLocation(pass1.program, "nTriangles"), triangles.size());
	glUniform1i(glGetUniformLocation(pass1.program, "nNodes"), nodes.size());
	glUniform1i(glGetUniformLocation(pass1.program, "width"), pass1.width);
	glUniform1i(glGetUniformLocation(pass1.program, "height"), pass1.height);
	glUseProgram(0);

	pass2.program = getShaderProgram("./shaders/pass2.fsh", "./shaders/vshader.vsh");
	lastFrame = getTextureRGB32F(pass2.width, pass2.height);
	pass2.colorAttachments.push_back(lastFrame);
	pass2.bindData();

	pass3.program = getShaderProgram("./shaders/pass3.fsh", "./shaders/vshader.vsh");
	pass3.bindData(true);


	// ----------------------------------------------------------------------------- //

	std::cout << "开始...::" << std::endl << std::endl;
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0, 0.0, 0.0, 1.0);



	glutDisplayFunc(display);
	glutIdleFunc(framefunc);
	glutMotionFunc(mouse);
	glutMouseFunc(mouseDown);
	glutMouseWheelFunc(mouseWheel);
	glutMainLoop();//永不结束,根据需要,自行调取应该执行的回调函数,而最主要的回调函数就是哪个DisplayFunc。

	return 0;

}