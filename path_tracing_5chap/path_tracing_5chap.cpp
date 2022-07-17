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
	vec3 emissive = vec3(0, 0, 0);//��Ϊ��Դ��ʱ��ķ�����ɫ,Ĭ��Ϊ��
	vec3 baseColor = vec3(1, 1, 1);//���屾����ɫ,Ĭ��Ϊ��
	float subsurface = 0.0;//�α���,ʵ����ʯЧ��
	float metallic = 0.0;//������
	float specular = 0.0;//���淴��
	float specularTint = 0.0;//�����˾��淴�����ɫ�Ƿ���ӽ����屾�����ɫ���Ǹ��ӽ���ɫ
	float roughness = 0.0;//�ֲڶ�
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
struct Triangle_encoded//�������ε����Ա����һ����ά����
{
	vec3 p1, p2, p3;//���������ά�����ĳ�Ա����ֱ���հἴ��
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
	std::vector<GLuint>colorAttachments;//��������洢��Ҫ������һpass������id
	GLuint program;
	int height = 512;
	int width = 512;
	void bindData(bool finalPass = false)
	{
		if (!finalPass) glGenFramebuffers(1, &FBO);//����������һ������,�ͽ���һ��֡����
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);//�󶨵���ǰ��֡����,��Ӧ���洦�����˾ͽ��

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		std::vector<vec3> square = { vec3(-1, -1, 0), vec3(1, -1, 0), vec3(-1, 1, 0), vec3(1, 1, 0), vec3(-1, 1, 0), vec3(1, -1, 0) };
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * square.size(), NULL, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3) * square.size(), &square[0]);

		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
		//����finalPass������֡�������ɫ����
		if (!finalPass)
		{
			std::vector<GLuint> attachments;
			for (int i = 0; i < colorAttachments.size(); ++i)
			{
				glBindTexture(GL_TEXTURE_2D, colorAttachments[i]);//��Ҫ������һpass������ID�󶨵�һ��2D������
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, colorAttachments[i], 0); //����ɫ����󶨵� i ����ɫ����,��ν��ɫ��������GL_COLOR_ATTACHMENT0
				attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
			}
			glDrawBuffers(attachments.size(), &attachments[0]);//��Ȼ��Draw,��������һ������ָ��,ֻ��ȷ���˻�����ɫ�����˳��
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);//���
	}
	void draw(std::vector<GLuint>texPassArray = { })
	{
		glUseProgram(program);//�������õ�ʱ���趨�˵�ǰpass��shader
		glBindFramebuffer(GL_FRAMEBUFFER, FBO);//��Ⱦ��ǰpass�Ͱ󶨵�ǰpass��֡����,ÿ��pass�Դ�һ��֡����
		glBindVertexArray(vao);//��ÿһ֡�Ķ����������󶨺�,Ҳ����֧��û���
		//����һ֡��֡������ɫ����
		for (int i = 0; i < texPassArray.size(); ++i)
		{
			glActiveTexture(GL_TEXTURE0 + i);//��������
			glBindTexture(GL_TEXTURE_2D, texPassArray[i]);//�����ǽ���һ��pass����ɫ����������󶨵�����,ע�����˳��,�ȼ���,���ڰ�
			std::string nName = "texPass" + std::to_string(i);
			glUniform1i(glGetUniformLocation(program, nName.c_str()), i);
		}
		glViewport(0, 0, width, height);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);//ÿ�������ɫ�������Ȼ���
		glDrawArrays(GL_TRIANGLES, 0, 6);//����������������,6������

		glBindVertexArray(0);//�����ж�Ӧ�ں�����ͷǰ����,ǰ����ּ���,���ְ�,���ڸ��ֽ��
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

//�������
float upAngle = 0.0;//��������
float rotatAngle = 0.0;//��������
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
		std::cout << "ERROR:: �ļ�" << filepath << "  ��ʧ��" << std::endl;
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

// ��ȡ��ɫ������
GLuint getShaderProgram(std::string fshader, std::string vshader) {
	// ��ȡshaderԴ�ļ�
	std::string vSource = readShaderFile(vshader);
	std::string fSource = readShaderFile(fshader);
	const char* vpointer = vSource.c_str();
	const char* fpointer = fSource.c_str();

	//���󱣴�
	GLint success;
	GLchar infoLog[512];

	//�������붥����ɫ��
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, (const GLchar**)(&vpointer), NULL);
	glCompileShader(vertexShader);
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);   // ������
	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
		std::cout << "������ɫ���������\n" << infoLog << std::endl;
		exit(-1);
	}

	//Ƭ����ɫ��
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, (const GLchar**)(&fpointer), NULL);
	glCompileShader(fragmentShader);
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);   // ������
	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
		std::cout << "Ƭ����ɫ���������\n" << infoLog << std::endl;
		exit(-1);
	}

	// ����������ɫ����program����
	GLuint shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	// ɾ����ɫ������
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
	rotate = glm::rotate(rotate, glm::radians(rotateCtrl.x), glm::vec3(1, 0, 0));//x����ת��Ϊ�Ƕ�,��x�������ת
	rotate = glm::rotate(rotate, glm::radians(rotateCtrl.y), glm::vec3(0, 1, 0));
	rotate = glm::rotate(rotate, glm::radians(rotateCtrl.z), glm::vec3(0, 0, 1));
	mat4 model = translate * rotate * scale;//˳��Ϊλ��->��ת->����
	return model;
}

void readObj(std::string filepath, std::vector<Triangle>& triangles, Material material, mat4 trans, bool smoothNormal)
{
	//����λ��,����
	std::vector<vec3> vertices;
	std::vector<GLuint> indices;
	//���ļ���
	std::ifstream fin(filepath);
	std::string line;

	if (!fin.is_open())
	{
		std::cout << "ERROR::�ļ� " << filepath << "  ��ʧ��" << std::endl;
		exit(-1);
	}

	float maxx = -11451419.19;
	float maxy = -11451419.19;
	float maxz = -11451419.19;
	float minx = 11451419.19;
	float miny = 11451419.19;
	float minz = 11451419.19;

	//���ж�ȡ
	while (std::getline(fin, line))
	{
		std::istringstream sin(line);
		std::string type;
		GLfloat x, y, z;
		int v0, v1, v2;
		int vn0, vn1, vn2;
		int vt0, vt1, vt2;
		char slash;

		//ͳ��б����Ŀ
		int slashcnt = 0;
		for (int i = 0; i < line.size(); ++i)
		{
			if (line[i] == '/')
			{
				++slashcnt;
			}
		}
		//��ȡobj�ļ�
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
				sin >> v0 >> slash >> vt0 >> slash >> vn0;//�ֱ��Ƕ�������,������������������������
				sin >> v1 >> slash >> vt1 >> slash >> vn1;
				sin >> v2 >> slash >> vt2 >> slash >> vn2;
			}
			else if (slashcnt == 3)//�������/��Ϊ2���ֵĻ�,ֻ�ж�����������������������
			{
				sin >> v0 >> slash >> vt0;
				sin >> v1 >> slash >> vt1;
				sin >> v2 >> slash >> vt2;
			}
			else
			{
				sin >> v0 >> v1 >> v2;
			}
			indices.push_back(v0 - 1);//��Ϊ��������������Ǵ�0��ʼ��,��obj�ļ���������Ǵ�1��ʼ,�����������һλ
			indices.push_back(v1 - 1);//����,�������ֻ��ȡ����λ�õ�����,ʲô������,ʲô��������ɶ�Ķ�������
			indices.push_back(v2 - 1);
		}
	}

	//ģ�ʹ�С��һ��
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
		vec4 vv = vec4(v.x, v.y, v.z, 1);//����ά����ά�������
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
	//����Triangles��������

	int offset = triangles.size();  // ��������
	triangles.resize(offset + indices.size() / 3);

	for (int i = 0; i < indices.size(); i += 3)
	{
		Triangle& t = triangles[offset + i / 3];
		//���ݶ�������
		t.p1 = vertices[indices[i]];
		t.p2 = vertices[indices[i + 1]];
		t.p3 = vertices[indices[i + 2]];

		if (!smoothNormal)
		{
			vec3 n = normalize(cross(t.p2 - t.p1, t.p3 - t.p1));//�����Ǹ���ͬһ�������ε���������м����
			t.n1 = n; t.n2 = n; t.n3 = n;
		}
		else
		{
			t.n1 = normalize(normals[indices[i]]);//����������֮ǰ,����ͬһ�������ε����������õ��ķ��������и�ֵ
			t.n2 = normalize(normals[indices[i + 1]]);//Ҳ������Ϊ�������ѡ��,������Ƭ����ɫ���м��㽻�㷨������ʱ����Ҫר��ʹ�ò�ֵ����
			t.n3 = normalize(normals[indices[i + 2]]);//��Ϊ������������ķ�������һ����һ������ġ�
		}
		t.material = material;
	}
}

int buildBVH(std::vector<Triangle>& triangles, std::vector<BVHNode>& nodes, int l, int r, int n)
{
	//��ʵ�ܺ����,֮ǰBVH��������Ϊ����ʹ��ָ��,��˴�����һ���ڵ㼴��,���ڲ�����,�Ǿ�ά��һ��BVHNode����
	if (l > r)
	{
		return 0;
	}
	// ע��
	// �˴�����ͨ��ָ�룬���õȷ�ʽ������������ nodes[id] ������
	// ��Ϊ std::vector<> ����ʱ�´����������ڴ棬��ô��ַ�͸ı���
	// ��ָ�룬���þ�ָ��ԭ�����ڴ棬���Իᷢ������
	nodes.push_back(BVHNode()); //������ұ߽�û����ײ,��ô�ͽ���һ���ڵ�,����Ľ�����ʽ����Ĭ�Ϲ���,Ȼ��push_back();
	int id = nodes.size() - 1;//�õ���ǰ�ڵ������
	nodes[id].left = nodes[id].right = nodes[id].n = nodes[id].index = 0;
	nodes[id].AA = vec3(1145141919, 1145141919, 1145141919);
	nodes[id].BB = vec3(-1145141919, -1145141919, -1145141919);

	for (int i = l; i <= r; i++) {
		// ��С�� AA
		float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
		float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
		float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
		nodes[id].AA.x = min(nodes[id].AA.x, minx);
		nodes[id].AA.y = min(nodes[id].AA.y, miny);
		nodes[id].AA.z = min(nodes[id].AA.z, minz);
		// ���� BB
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
	// ���� AABB
	for (int i = l; i <= r; i++) {
		// ��С�� AA
		float minx = min(triangles[i].p1.x, min(triangles[i].p2.x, triangles[i].p3.x));
		float miny = min(triangles[i].p1.y, min(triangles[i].p2.y, triangles[i].p3.y));
		float minz = min(triangles[i].p1.z, min(triangles[i].p2.z, triangles[i].p3.z));
		nodes[id].AA.x = min(nodes[id].AA.x, minx);
		nodes[id].AA.y = min(nodes[id].AA.y, miny);
		nodes[id].AA.z = min(nodes[id].AA.z, minz);
		// ���� BB
		float maxx = max(triangles[i].p1.x, max(triangles[i].p2.x, triangles[i].p3.x));
		float maxy = max(triangles[i].p1.y, max(triangles[i].p2.y, triangles[i].p3.y));
		float maxz = max(triangles[i].p1.z, max(triangles[i].p2.z, triangles[i].p3.z));
		nodes[id].BB.x = max(nodes[id].BB.x, maxx);
		nodes[id].BB.y = max(nodes[id].BB.y, maxy);
		nodes[id].BB.z = max(nodes[id].BB.z, maxz);
	}
	// ������ n �������� ����Ҷ�ӽڵ�
	if ((r - l + 1) <= n) {
		nodes[id].n = r - l + 1;
		nodes[id].index = l;
		return id;
	}

	// ����ݹ齨��
	float Cost = INF;
	int Axis = 0;
	int Split = (l + r) / 2;
	for (int axis = 0; axis < 3; axis++) {
		// �ֱ� x��y��z ������
		if (axis == 0) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpx);
		if (axis == 1) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpy);
		if (axis == 2) std::sort(&triangles[0] + l, &triangles[0] + r + 1, cmpz);

		// leftMax[i]: [l, i] ������ xyz ֵ
		// leftMin[i]: [l, i] ����С�� xyz ֵ
		std::vector<vec3> leftMax(r - l + 1, vec3(-INF, -INF, -INF));
		std::vector<vec3> leftMin(r - l + 1, vec3(INF, INF, INF));
		// ����ǰ׺ ע�� i-l �Զ��뵽�±� 0
		for (int i = l; i <= r; i++) {
			Triangle& t = triangles[i];
			int bias = (i == l) ? 0 : 1;  // ��һ��Ԫ�����⴦��

			leftMax[i - l].x = max(leftMax[i - l - bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
			leftMax[i - l].y = max(leftMax[i - l - bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
			leftMax[i - l].z = max(leftMax[i - l - bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

			leftMin[i - l].x = min(leftMin[i - l - bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
			leftMin[i - l].y = min(leftMin[i - l - bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
			leftMin[i - l].z = min(leftMin[i - l - bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
		}

		std::vector<vec3> rightMax(r - l + 1, vec3(-INF, -INF, -INF));
		std::vector<vec3> rightMin(r - l + 1, vec3(INF, INF, INF));
		// �����׺ ע�� i-l �Զ��뵽�±� 0
		for (int i = r; i >= l; i--) {
			Triangle& t = triangles[i];
			int bias = (i == r) ? 0 : 1;  // ��һ��Ԫ�����⴦��

			rightMax[i - l].x = max(rightMax[i - l + bias].x, max(t.p1.x, max(t.p2.x, t.p3.x)));
			rightMax[i - l].y = max(rightMax[i - l + bias].y, max(t.p1.y, max(t.p2.y, t.p3.y)));
			rightMax[i - l].z = max(rightMax[i - l + bias].z, max(t.p1.z, max(t.p2.z, t.p3.z)));

			rightMin[i - l].x = min(rightMin[i - l + bias].x, min(t.p1.x, min(t.p2.x, t.p3.x)));
			rightMin[i - l].y = min(rightMin[i - l + bias].y, min(t.p1.y, min(t.p2.y, t.p3.y)));
			rightMin[i - l].z = min(rightMin[i - l + bias].z, min(t.p1.z, min(t.p2.z, t.p3.z)));
		}

		//����Ѱ�ҷָ�
		float cost = INF;
		float split = l;
		for (int i = l; i <= r - 1; ++i)
		{//i��ʲô?i�Ƿָ����ް�,������ط�,i����ڵ���б߽�,i+1���ҽڵ����߽�,Ҳ����Ϊ���,���ǵ�i���Դ�l��ʼ,����Ҫ��r-1����
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
		if (cost < Cost)//�Ƚϵĺ��ľ�������ķ�
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
	nodes[id].left = buildBVHwithSAH(triangles, nodes, l, Split, n); //ע����,����
	//std::cout << "nodes[id].left = " << nodes[id].left << std::endl;

	nodes[id].right = buildBVHwithSAH(triangles, nodes, Split + 1, r, n);;

	return id;
}

float* calculateHdrCache(float* HDR, int width, int height)
{
	float lumSum = 0.0;

	//��ʼ��h��w�еĸ����ܶ�pdf��ͳ��������
	std::vector<std::vector<float>> pdf(height);
	for (auto& line : pdf) line.resize(width);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float R = HDR[3 * (i * width + j)];
			float G = HDR[3 * (i * width + j) + 1];
			float B = HDR[3 * (i * width + j) + 2];
			float lum = 0.2 * R + 0.7 * G + 0.1 * B;//�����RGBת�����ȵĹ̶���ʽ
			pdf[i][j] = lum;//ͳ��ÿ�����ص�����
			lumSum += lum;//ͳ������֮��
		}
	}

	//�����ܶȹ�һ��
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			pdf[i][j] /= lumSum;
		}
	}
	//�ۼ�ÿһ�еõ�x�ı�Ե�����ܶ�
	std::vector<float>pdf_x_margin;
	pdf_x_margin.resize(width);
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
			pdf_x_margin[j] += pdf[i][j];

	//����x�ı�Ե�ֲ�����
	std::vector<float> cdf_x_margin = pdf_x_margin;
	for (int i = 1; i < width; i++)
		cdf_x_margin[i] += cdf_x_margin[i - 1];//�������νǰ׺��

	//����y��X=x�µ����������ܶȺ���
	std::vector<std::vector<float>> pdf_y_condiciton = pdf;
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
			pdf_y_condiciton[i][j] /= pdf_x_margin[j];

	//����y��X=x�µ��������ʷֲ�����
	std::vector<std::vector<float>> cdf_y_condiciton = pdf_y_condiciton;
	for (int j = 0; j < width; j++)
		for (int i = 1; i < height; i++)
			cdf_y_condiciton[i][j] += cdf_y_condiciton[i - 1][j];

	//cdf_y_condiciton ת��δ���д洢
	//cdf_y_condiciton[i]��ʾy��X=i�µ��������ʷֲ�����
	std::vector<std::vector<float>> temp = cdf_y_condiciton;
	cdf_y_condiciton = std::vector<std::vector<float>>(width);
	for (auto& line : cdf_y_condiciton) line.resize(height);
	for (int j = 0; j < width; j++)
		for (int i = 0; i < height; i++)
			cdf_y_condiciton[j][i] = temp[i][j];

	// ��� xi_1, xi_2 Ԥ�������� xy
	// sample_x[i][j] ��ʾ xi_1=i/height, xi_2=j/width ʱ (x,y) �е� x
	// sample_y[i][j] ��ʾ xi_1=i/height, xi_2=j/width ʱ (x,y) �е� y
	// sample_p[i][j] ��ʾȡ (i, j) ��ʱ�ĸ����ܶ�

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
			float xi_1 = float(i) / height;//������Կ����൱�ڽ�����HDRͼ��ӳ�䵽[0,1]^2�ķ�Χ,���Ҷ������Χ����Ԥ����
			float xi_2 = float(j) / width;//��һ��,������洢��һ��������,��ʱ��ֱ����������ͷ������ֵ�ܿ�ý��

			//��xi_1��cdf_x_margin��lowerbound �õ�����x,������cdf_x_margin���Ǹ�1ά���ۼ�����
			int x = std::lower_bound(cdf_x_margin.begin(), cdf_x_margin.end(), xi_1) - cdf_x_margin.begin();//lower_bound����һ����С���ڵĵ�����,
			//��begin()������һ��,���Ǿ���,Ҳ���Ƕ�Ӧ��x������,��ĺ���
			//��xi_2��X=x������µõ�����y
			int y = std::lower_bound(cdf_y_condiciton[x].begin(), cdf_y_condiciton[x].end(), xi_2) - cdf_y_condiciton[x].begin();
			//�洢��������xy��xyλ�ö�Ӧ�ĸ����ܶ�
			sample_x[i][j] = float(x) / width;//����Ӧ����Ϊ�˹�һ��,��ʱ��ֱ�����������������ת����ȷ������������
			sample_y[i][j] = float(y) / height;
			sample_p[i][j] = pdf[i][j];
		}
	}
	//���Ͻ��������
	//R,Gͨ���洢����(x,y),��Bͨ���洢pdf(i,j)
	//�����ǽ��������洢��һ��������,��Ϊһ������,��������RGB����ͨ���ֱ�洢��������ֵ
	//����˵,����Ԥ�Ⱦ��ȵؼ����˺ܶ������������Ӧ��xyֵ,����������,��ʱ��ֱ�Ӳ��Ϳ��ˡ�
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



//����
clock_t t1, t2;
double dt, fps;
int  frameCounter = 0;
void display()
{
	//֡��ʱ
	t2 = clock();
	dt = (double)(t2 - t1) / CLOCKS_PER_SEC;//��ʱ����Ϊ1000����
	fps = 1.0 / dt;
	std::cout << "\r";
	std::cout << std::fixed << std::setprecision(2) << "FPS: " << fps << "    ��������" << frameCounter;//��������˿�ѧ������,���ȱ�����С����2λ֮��
	t1 = t2;//�����˸��¼�ʱ��
	//�������
	vec3 eye = vec3(-sin(radians(rotatAngle)) * cos(radians(upAngle)), sin(radians(upAngle)), cos(radians(rotatAngle)) * cos(radians(upAngle)));
	eye.x *= r; eye.y *= r; eye.z *= r;//�����eye����һ����ȷ�����������������ϵ���е�����,�ɴ˿��Կ���,���ǵ��������Χ����������ϵ���е�(0,0,0)ΪԲ��,4Ϊ�뾶���������ƶ�
	mat4 cameraRotate = lookAt(eye, vec3(0, 0, 0), vec3(0, 1, 0));//�ڶ���������Ӧ����eye+front,��������һֱ����ԭ��,���front = -eye,������Ϊ(0,0,0),����ͷ������Ϊ(0,1,0)
	//����һ�����:lookAt�õ��ľ�����ͼ�任����,Ҳ������������->���������ı任����
	cameraRotate = inverse(cameraRotate);//���������lookAt�Ľ���ֽ�����һ��ת��,���cameraRotate�ͱ�������������->��������ı任����,������֪��,��ʱ�ڶ�ȡģ�͵�ʱ��,���ǽ�����ģ�ͱ任,Ҳ����˵,
	//���Ƕ�������ģ���Ǵ�����������ϵ������֪�������Ǵ������Ϊ���,ָ�򻭲���ÿһ�����ص�,����shader��ֱ���Ի�����xy��ͻ����������Ϊ���߱任ǰ�ķ���,����ܺ����,������Կ��������������������ϵ����, 
	//��ô�������λ�þ���ԭ��(0,0,0)��,��˿�����ô��ʾ,��������֪��shader���趨�Ĺ��ߵ�����������������������ϵ�����趨��,�������ֱ���û����������������������߷���Ҫ�����������ת������������ϵ����,
	//���ܺ�������������ϵ���еĹ���������Ǻ�,���,����ֱ��lookAtȡ��,���ɵõ���������ϵ���еĹ���,���������������ϵ�е�ģ����,û���κ�Υ�͸�

	//��uniform��pass1
	glUseProgram(pass1.program);
	glUniform3fv(glGetUniformLocation(pass1.program, "eye"), 1, value_ptr(eye));
	glUniformMatrix4fv(glGetUniformLocation(pass1.program, "cameraRotate"), 1, GL_FALSE, value_ptr(cameraRotate));
	glUniform1i(glGetUniformLocation(pass1.program, "frameCounter"), frameCounter++);// �������������������
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

	//����
	pass1.draw();
	pass2.draw(pass1.colorAttachments);
	pass3.draw(pass2.colorAttachments);

	glutSwapBuffers();
}
void framefunc()
{
	glutPostRedisplay();//��ѡ��֪ͨ�������»��ƻ���
}
double lastX = 0.0, lastY = 0.0;
void mouse(int x, int y)
{
	frameCounter = 0;
	//������ת
	rotatAngle += 150 * (x - lastX) / 512;
	upAngle += 150 * (y - lastY) / 512;
	upAngle = min(upAngle, 89.0f);
	upAngle = max(upAngle, -89.0f);
	lastX = x; lastY = y;
	glutPostRedisplay();//�����ƶ���,���ػ�,��ʵ�ʵı�������,����ػ治��ˢ�»���,������ȫ��ͷ��ʼ�ۼƹ�׷
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
	glutPostRedisplay();//��Ȼ����ȫ��ͷ����
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
	std::cout << "ģ�Ͷ�ȡ���: �� " << nTriangles << " ��������" << std::endl;

	// ���� bvh
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
	std::cout << "BVH �������: �� " << nNodes << " ���ڵ�" << std::endl;

	//���������β���
	std::vector<Triangle_encoded> triangles_encoded(nTriangles);
	for (int i = 0; i < nTriangles; i++) {
		Triangle& t = triangles[i];
		Material& m = t.material;
		// ����λ��
		triangles_encoded[i].p1 = t.p1;
		triangles_encoded[i].p2 = t.p2;
		triangles_encoded[i].p3 = t.p3;
		// ���㷨��
		triangles_encoded[i].n1 = t.n1;
		triangles_encoded[i].n2 = t.n2;
		triangles_encoded[i].n3 = t.n3;
		// ����
		triangles_encoded[i].emissive = m.emissive;
		triangles_encoded[i].baseColor = m.baseColor;
		triangles_encoded[i].param1 = vec3(m.subsurface, m.metallic, m.specular);
		triangles_encoded[i].param2 = vec3(m.specularTint, m.roughness, m.anisotropic);
		triangles_encoded[i].param3 = vec3(m.sheen, m.sheenTint, m.clearcoat);
		triangles_encoded[i].param4 = vec3(m.clearcoatGloss, m.IOR, m.transmission);
	}
	//����BVHNode
	std::vector<BVHNode_encoded> nodes_encoded(nNodes);
	for (int i = 0; i < nNodes; i++) {
		nodes_encoded[i].childs = vec3(nodes[i].left, nodes[i].right, 0);
		nodes_encoded[i].leafInfo = vec3(nodes[i].n, nodes[i].index, 0);
		nodes_encoded[i].AA = nodes[i].AA;
		nodes_encoded[i].BB = nodes[i].BB;
	}

	//����˵�����ȴ�����һ���Դ�,������֮���������д��ȥ,����������һ������,��������������ΪGL_TEXTURE_BUFFER,���ҽ�����Ļ�������Ϊ����֮ǰ��
	//�ű��������εĵ��ǿ��Դ�,������������Ӧ�����ݾ������ǵı�����������


	GLuint tbo0;
	glGenBuffers(1, &tbo0);//����һ������,�󶨵�tbo0��
	glBindBuffer(GL_TEXTURE_BUFFER, tbo0);//��һ�������������˻��������
	glBufferData(GL_TEXTURE_BUFFER, sizeof(Triangle_encoded) * triangles_encoded.size(), triangles_encoded.data(), GL_STATIC_DRAW);//����ľ�������д����
	glGenTextures(1, &trianglesTextureBuffer);//���Ƶķ�ʽ�ֽ�����һ������
	glBindTexture(GL_TEXTURE_BUFFER, trianglesTextureBuffer);//�������Ͱ�Ϊ������
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo0);//����Ļ��嶨Ϊtbo0,������ͽ����ǵ������������ݺ�ǰ�洴���Ļ������һ����,��������ݵĹ�ϵ������ô������

	GLuint tbo1;
	glGenBuffers(1, &tbo1);
	glBindBuffer(GL_TEXTURE_BUFFER, tbo1);
	glBufferData(GL_TEXTURE_BUFFER, nodes_encoded.size() * sizeof(BVHNode_encoded), &nodes_encoded[0], GL_STATIC_DRAW);
	glGenTextures(1, &nodesTextureBuffer);
	glBindTexture(GL_TEXTURE_BUFFER, nodesTextureBuffer);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_RGB32F, tbo1);

	// hdr ȫ��ͼ
	HDRLoaderResult hdrRes;
	bool r = HDRLoader::load("./HDR/peppermint_powerplant_4k.hdr", hdrRes);
	//bool r = HDRLoader::load("./HDR/chinese_garden_2k.hdr", hdrRes);
	hdrMap = getTextureRGB32F(hdrRes.width, hdrRes.height);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, hdrRes.cols);
	
	//hdr��Ҫ�Բ��� cache
	std::cout << "����HDR��ͼ��Ҫ�Բ���Cache,��ǰ�ֱ���: " << hdrRes.width << " " << hdrRes.height << std::endl;
	hdrCache = getTextureRGB32F(hdrRes.width, hdrRes.height);
	float* cache = calculateHdrCache(hdrRes.cols, hdrRes.width, hdrRes.height);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, hdrRes.width, hdrRes.height, 0, GL_RGB, GL_FLOAT, cache);
	hdrResolution = hdrRes.width;
	std::cout << cache[23] << std::endl;
	// ----------------------------------------------------------------------------- //

	// ��������

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

	std::cout << "��ʼ...::" << std::endl << std::endl;
	glEnable(GL_DEPTH_TEST);
	glClearColor(0.0, 0.0, 0.0, 1.0);



	glutDisplayFunc(display);
	glutIdleFunc(framefunc);
	glutMotionFunc(mouse);
	glutMouseFunc(mouseDown);
	glutMouseWheelFunc(mouseWheel);
	glutMainLoop();//��������,������Ҫ,���е�ȡӦ��ִ�еĻص�����,������Ҫ�Ļص����������ĸ�DisplayFunc��

	return 0;

}