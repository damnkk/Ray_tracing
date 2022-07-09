#version 330 core

in vec3 pix;
//--------------------------------------------------------//
uniform int nTriangles;






uniform vec3 eye;
uniform samplerBuffer triangles;
uniform samplerBuffer nodes;
//-------------------------------------------------------//
#define INF 114514.0
#define SIZE_TRIANGLE 12
#define SIZE_BVHNODE 4
//-------------------------------------------------------//
//Triangle数据格式
struct Triangle
{
	vec3 p1,p2,p3;
	vec3 n1,n2,n3;
};
struct BVHNode
{
	int left;
	int right;
	int n;
	int index;
	vec3 AA,BB;
};

//材质
struct Material
{
	vec3 emissive;
	vec3 baseColor;
	float subsurface;
	float metallic;
	float specular;
	float specularTint;
	float roughness;
	float anisotropic;
	float sheen;
	float sheenTint;
	float clearcoat;
	float clearcoatGloss;
	float IOR;
	float transmission;
};
//光线
struct Ray 
{
	vec3 startPoint;
	vec3 direction;
};
//光线求交结果
struct Hitresult
{
	bool isHit;			//是否命中
	bool isInside;		//是否从内部命中
	float distance;		//与交点的距离
	vec3 hitPoint;		//光线命中点
	vec3 normal;		//命中点法向量
	vec3 viewDir;		//射线方向
	Material material;	//命中点的表面材质
};

//获取第i下标的三角形
Triangle getTriangle(int i)
{
	int offset = i*SIZE_TRIANGLE;
	Triangle t;

	//顶点坐标
	t.p1 = texelFetch(triangles,offset+0).xyz;
	t.p2 = texelFetch(triangles,offset+1).xyz;
	t.p3 = texelFetch(triangles,offset+2).xyz;
	//normal
	t.n1 = texelFetch(triangles,offset+3).xyz;
	t.n2 = texelFetch(triangles,offset+4).xyz;
	t.n3 = texelFetch(triangles,offset+5).xyz;

	return t;
}
//获取第i下标三角形的材质
Material getMaterial(int i)
{
	Material m;
	int offset = i*SIZE_TRIANGLE;
	vec3 param1 = texelFetch(triangles,offset+8).xyz;
	vec3 param2 = texelFetch(triangles,offset+9).xyz;
	vec3 param3 = texelFetch(triangles,offset+10).xyz;
	vec3 param4 = texelFetch(triangles,offset+11).xyz;
	m.emissive = texelFetch(triangles,offset+6).xyz;
	m.baseColor = texelFetch(triangles,offset+7).xyz;
	m.subsurface = param1.x;
	m.metallic = param1.y;
	m.specular = param1.z;
	m.specularTint = param2.x;
	m.roughness = param2.y;
	m.anisotropic = param2.z;
	m.sheen = param3.x;
    m.sheenTint = param3.y;
    m.clearcoat = param3.z;
    m.clearcoatGloss = param4.x;
    m.IOR = param4.y;
    m.transmission = param4.z;

	return m;
}

BVHNode getBVHNode(int i )
{
	BVHNode node;
	
	//左右子树
	int offset = i*SIZE_BVHNODE;
	ivec3 childs = ivec3(texelFetch(nodes,offset+0).xyz);
	ivec3 leafInfo = ivec3(texelFetch(nodes,offset+1).xyz);
	node.left = int(childs.x);
	node.right = int(childs.y);
	node.n = int(leafInfo.x);
	//包围盒
	node.AA = texelFetch(nodes,offset+2).xyz;
	node.BB = texelFetch(nodes,offset+3).xyz;
	return node;
}


Hitresult hitTriangle(Triangle triangle, Ray ray) {
    Hitresult res;
    res.distance = INF;
    res.isHit = false;
    res.isInside = false;

    vec3 p1 = triangle.p1;
    vec3 p2 = triangle.p2;
    vec3 p3 = triangle.p3;

    vec3 S = ray.startPoint;    // 射线起点
    vec3 d = ray.direction;     // 射线方向
    vec3 N = normalize(cross(p2-p1, p3-p1));    // 法向量

    // 从三角形背后（模型内部）击中
    if (dot(N, d) > 0.0f) {
        N = -N;   
        res.isInside = true;
    }

    // 如果视线和三角形平行
    if (abs(dot(N, d)) < 0.00001f) return res;

    // 距离
    float t = (dot(N, p1) - dot(S, N)) / dot(d, N);
    if (t < 0.0005f) return res;    // 如果三角形在光线背面

    // 交点计算
    vec3 P = S + d * t;

    // 判断交点是否在三角形中
    vec3 c1 = cross(p2 - p1, P - p1);
    vec3 c2 = cross(p3 - p2, P - p2);
    vec3 c3 = cross(p1 - p3, P - p3);
    bool r1 = (dot(c1, N) > 0 && dot(c2, N) > 0 && dot(c3, N) > 0);
    bool r2 = (dot(c1, N) < 0 && dot(c2, N) < 0 && dot(c3, N) < 0);

    // 命中，封装返回结果
    if (r1 || r2) {
        res.isHit = true;
        res.hitPoint = P;
        res.distance = t;
        res.normal = N;
        res.viewDir = d;
        // 根据交点位置插值顶点法线
        float alpha = (-(P.x-p2.x)*(p3.y-p2.y) + (P.y-p2.y)*(p3.x-p2.x)) / (-(p1.x-p2.x-0.00005)*(p3.y-p2.y+0.00005) + (p1.y-p2.y+0.00005)*(p3.x-p2.x+0.00005));
        float beta  = (-(P.x-p3.x)*(p1.y-p3.y) + (P.y-p3.y)*(p1.x-p3.x)) / (-(p2.x-p3.x-0.00005)*(p1.y-p3.y+0.00005) + (p2.y-p3.y+0.00005)*(p1.x-p3.x+0.00005));
        float gama  = 1.0 - alpha - beta;
        vec3 Nsmooth = alpha * triangle.n1 + beta * triangle.n2 + gama * triangle.n3;
        Nsmooth = normalize(Nsmooth);
        res.normal = (res.isInside) ? (-Nsmooth) : (Nsmooth);
    }

    return res;
}
//AABB 盒子求交
float hitAABB(Ray r,vec3 AA,vec3 BB)
{
	vec3 invdir = 1.0/r.direction;
	vec3 f = (AA-r.startPoint)*invdir;
	vec3 n = (BB-r.startPoint)*invdir;
	vec3 tmax = max(f,n);
	vec3 tmin = min(f,n);
	float t1 = min(tmax.x,min(tmax.y,tmax.z));
	float t0 = max(tmin.x,min(tmin.y,tmin.z));
	return (t1>=t0)?((t0>0.0)?(t0):(t1)):(-1);
}
//-----------------------------------------------------------------//
//暴力遍历数组下标范围[l,r]求最近交点
Hitresult hitArray(Ray ray, int l, int r) {
    Hitresult res;
    res.isHit = false;
    res.distance = INF;
    for(int i=l; i<=r; i++) {
        Triangle triangle = getTriangle(i);
        Hitresult r = hitTriangle(triangle,ray); 
        if(r.isHit && r.distance<res.distance) {
            res = r;
            res.material = getMaterial(i);
        }
    }
    return res;
}

Hitresult hitBVH(Ray ray)
{
	Hitresult res;
	res.isHit = false;
	res.distance = INF;
	//栈
	int stack[256];
	int sp = 0;
	stack[sp++] = 1;
	while(sp>0)
	{

}

void main()
{
	Ray ray;
	ray.startPoint= eye;
	vec3 dir = vec3(pix.xy,2)-ray.startPoint;
	ray.direction  = normalize(dir);
}

