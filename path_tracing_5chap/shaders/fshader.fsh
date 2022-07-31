#version 330 core

in vec3 pix;
out vec4 fragColor;

// ----------------------------------------------------------------------------- //

uniform int frameCounter;
uniform int nTriangles;
uniform int nNodes;
uniform int width;
uniform int height;
uniform int hdrResolution;

uniform samplerBuffer triangles;
uniform samplerBuffer nodes;

uniform sampler2D lastFrame;
uniform sampler2D hdrMap;
uniform sampler2D hdrCache;

uniform vec3 eye;
uniform mat4 cameraRotate;

// ----------------------------------------------------------------------------- //

#define PI              3.1415926
#define INF             114514.0
#define SIZE_TRIANGLE   12
#define SIZE_BVHNODE    4

// ----------------------------------------------------------------------------- //

// Triangle 数据格式
struct Triangle {
    vec3 p1, p2, p3;    // 顶点坐标
    vec3 n1, n2, n3;    // 顶点法线
};

// BVH 树节点
struct BVHNode {
    int left;           // 左子树
    int right;          // 右子树
    int n;              // 包含三角形数目
    int index;          // 三角形索引
    vec3 AA, BB;        // 碰撞盒
};

// 物体表面材质定义
struct Material {
    vec3 emissive;          // 作为光源时的发光颜色
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

// 光线
struct Ray {
    vec3 startPoint;
    vec3 direction;
};

// 光线求交结果
struct HitResult {
    bool isHit ;             // 是否命中
    bool isInside;          // 是否从内部命中
    float distance;         // 与交点的距离
    vec3 hitPoint;          // 光线命中点
    vec3 normal;            // 命中点法线
    vec3 viewDir;           // 击中该点的光线的方向
    Material material;      // 命中点的表面材质
};

/*
vec3 toNormalHemisphere(vec3 v, vec3 N) {
    vec3 tangent = vec3(0);
    if(N.yz==vec2(0)) tangent = vec3(0, 0, -N.x);
    else if(N.xz==vec2(0)) tangent = vec3(0, 0, N.y);
    else if(N.xy==vec2(0)) tangent = vec3(-N.z, 0, 0);
    else if(abs(N.x)>abs(N.y)) tangent = normalize(vec3(0, N.z, -N.y));
    else tangent = normalize(vec3(-N.z, 0, N.x)); 
    vec3 bitangent = cross(N, tangent);
    return normalize(v.x * tangent + v.y * bitangent + v.z * N);
}
*/


// ----------------------------------------------------------------------------- //

// 获取第 i 下标的三角形
Triangle getTriangle(int i) {
    int offset = i * SIZE_TRIANGLE;
    Triangle t;

    // 顶点坐标
    t.p1 = texelFetch(triangles, offset + 0).xyz;
    t.p2 = texelFetch(triangles, offset + 1).xyz;
    t.p3 = texelFetch(triangles, offset + 2).xyz;
    // 法线
    t.n1 = texelFetch(triangles, offset + 3).xyz;
    t.n2 = texelFetch(triangles, offset + 4).xyz;
    t.n3 = texelFetch(triangles, offset + 5).xyz;

    return t;
}

// 获取第 i 下标的三角形的材质
Material getMaterial(int i) {
    Material m;

    int offset = i * SIZE_TRIANGLE;
    vec3 param1 = texelFetch(triangles, offset + 8).xyz;
    vec3 param2 = texelFetch(triangles, offset + 9).xyz;
    vec3 param3 = texelFetch(triangles, offset + 10).xyz;
    vec3 param4 = texelFetch(triangles, offset + 11).xyz;
    
    m.emissive = texelFetch(triangles, offset + 6).xyz;
    m.baseColor = texelFetch(triangles, offset + 7).xyz;
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

// 获取第 i 下标的 BVHNode 对象
BVHNode getBVHNode(int i) {
    BVHNode node;

    // 左右子树
    int offset = i * SIZE_BVHNODE;
    ivec3 childs = ivec3(texelFetch(nodes, offset + 0).xyz);
    ivec3 leafInfo = ivec3(texelFetch(nodes, offset + 1).xyz);
    node.left = int(childs.x);
    node.right = int(childs.y);
    node.n = int(leafInfo.x);
    node.index = int(leafInfo.y);

    // 包围盒
    node.AA = texelFetch(nodes, offset + 2).xyz;
    node.BB = texelFetch(nodes, offset + 3).xyz;

    return node;
}

// ----------------------------------------------------------------------------- //

// 光线和三角形求交 
HitResult hitTriangle(Triangle triangle, Ray ray) {
    HitResult res;
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
        float alpha = (-(P.x-p2.x)*(p3.y-p2.y) + (P.y-p2.y)*(p3.x-p2.x)) / (-(p1.x-p2.x)*(p3.y-p2.y) + (p1.y-p2.y)*(p3.x-p2.x)+1e-7);
        float beta  = (-(P.x-p3.x)*(p1.y-p3.y) + (P.y-p3.y)*(p1.x-p3.x)) / (-(p2.x-p3.x)*(p1.y-p3.y) + (p2.y-p3.y)*(p1.x-p3.x)+1e-7);
        float gama  = 1.0 - alpha - beta;
        vec3 Nsmooth = alpha * triangle.n1 + beta * triangle.n2 + gama * triangle.n3;
        Nsmooth = normalize(Nsmooth);
        res.normal = (res.isInside) ? (-Nsmooth) : (Nsmooth);
    }

    return res;
}

// 和 aabb 盒子求交，没有交点则返回 -1
float hitAABB(Ray r, vec3 AA, vec3 BB) {
    vec3 invdir = 1.0 / r.direction;

    vec3 f = (BB - r.startPoint) * invdir;
    vec3 n = (AA - r.startPoint) * invdir;

    vec3 tmax = max(f, n);
    vec3 tmin = min(f, n);

    float t1 = min(tmax.x, min(tmax.y, tmax.z));
    float t0 = max(tmin.x, max(tmin.y, tmin.z));

    return (t1 >= t0) ? ((t0 > 0.0) ? (t0) : (t1)) : (-1);
}

// ----------------------------------------------------------------------------- //

// 暴力遍历数组下标范围 [l, r] 求最近交点
HitResult hitArray(Ray ray, int l, int r) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;
    for(int i=l; i<=r; i++) {
        Triangle triangle = getTriangle(i);
        HitResult r = hitTriangle(triangle, ray);
        if(r.isHit && r.distance<res.distance) {
            res = r;
            res.material = getMaterial(i);
        }
    }
    return res;
}

// 遍历 BVH 求交
HitResult hitBVH(Ray ray) {
    HitResult res;
    res.isHit = false;
    res.distance = INF;

    // 栈
    int stack[256];
    int sp = 0;

    stack[sp++] = 1;
    while(sp>0) {
        int top = stack[--sp];
        BVHNode node = getBVHNode(top);
        
        // 是叶子节点，遍历三角形，求最近交点
        if(node.n>0) {
            int L = node.index;
            int R = node.index + node.n - 1;
            HitResult r = hitArray(ray, L, R);
            if(r.isHit && r.distance<res.distance) res = r;
            continue;
        }
        
        // 和左右盒子 AABB 求交
        float d1 = INF; // 左盒子距离
        float d2 = INF; // 右盒子距离
        if(node.left>0) {
            BVHNode leftNode = getBVHNode(node.left);
            d1 = hitAABB(ray, leftNode.AA, leftNode.BB);
        }
        if(node.right>0) {
            BVHNode rightNode = getBVHNode(node.right);
            d2 = hitAABB(ray, rightNode.AA, rightNode.BB);
        }

        // 在最近的盒子中搜索
        if(d1>0 && d2>0) {
            if(d1<d2) { // d1<d2, 左边先
                stack[sp++] = node.right;
                stack[sp++] = node.left;
            } else {    // d2<d1, 右边先
                stack[sp++] = node.left;
                stack[sp++] = node.right;
            }
        } else if(d1>0) {   // 仅命中左边
            stack[sp++] = node.left;
        } else if(d2>0) {   // 仅命中右边
            stack[sp++] = node.right;
        }
    }

    return res;
}

// ----------------------------------------------------------------------------- //

/*
 * 生成随机向量，依赖于 frameCounter 帧计数器
 * 代码来源：https://blog.demofox.org/2020/05/25/casual-shadertoy-path-tracing-1-basic-camera-diffuse-emissive/
*/

uint seed = uint(
    uint((pix.x * 0.5 + 0.5) * width)  * uint(1973) + 
    uint((pix.y * 0.5 + 0.5) * height) * uint(9277) + 
    uint(frameCounter) * uint(26699)) | uint(1);

uint wang_hash(inout uint seed) {
    seed = uint(seed ^ uint(61)) ^ uint(seed >> uint(16));
    seed *= uint(9);
    seed = seed ^ (seed >> 4);
    seed *= uint(0x27d4eb2d);
    seed = seed ^ (seed >> 15);
    return seed;
}
 
float rand() {
    return float(wang_hash(seed)) / 4294967296.0;
}

uint seed_sync = uint(
    uint((pix.x * 0.0 + 0.5) * width)  * uint(1973) + 
    uint((pix.y * 0.0 + 0.5) * height) * uint(9277) + 
    uint(114514) * uint(26699)) | uint(1);

float rand_sync() {
    return float(wang_hash(seed_sync)) / 4294967296.0;
}

const uint V[]= const uint [8*32](
  2147483648, 1073741824, 536870912, 268435456, 134217728, 67108864, 33554432, 16777216, 8388608, 4194304, 2097152, 1048576, 524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 2147483648, 3221225472, 2684354560, 4026531840, 2281701376, 3422552064, 2852126720, 4278190080, 2155872256, 3233808384, 2694840320, 4042260480, 2290614272, 3435921408, 2863267840, 4294901760, 2147516416, 3221274624, 2684395520, 4026593280, 2281736192, 3422604288, 2852170240, 4278255360, 2155905152, 3233857728, 2694881440, 4042322160, 2290649224, 3435973836, 2863311530, 4294967295, 2147483648, 3221225472, 1610612736, 2415919104, 3892314112, 1543503872, 2382364672, 3305111552, 1753219072, 2629828608, 3999268864, 1435500544, 2154299392, 3231449088, 1626210304, 2421489664, 3900735488, 1556135936, 2388680704, 3314585600, 1751705600, 2627492864, 4008611328, 1431684352, 2147543168, 3221249216, 1610649184, 2415969680, 3892340840, 1543543964, 2382425838, 3305133397, 2147483648, 3221225472, 536870912, 1342177280, 4160749568, 1946157056, 2717908992, 2466250752, 3632267264, 624951296, 1507852288, 3872391168, 2013790208, 3020685312, 2181169152, 3271884800, 546275328, 1363623936, 4226424832, 1977167872, 2693105664, 2437829632, 3689389568, 635137280, 1484783744, 3846176960, 2044723232, 3067084880, 2148008184, 3222012020, 537002146, 1342505107, 2147483648, 1073741824, 536870912, 2952790016, 4160749568, 3690987520, 2046820352, 2634022912, 1518338048, 801112064, 2707423232, 4038066176, 3666345984, 1875116032, 2170683392, 1085997056, 579305472, 3016343552, 4217741312, 3719483392, 2013407232, 2617981952, 1510979072, 755882752, 2726789248, 4090085440, 3680870432, 1840435376, 2147625208, 1074478300, 537900666, 2953698205, 2147483648, 1073741824, 1610612736, 805306368, 2818572288, 335544320, 2113929216, 3472883712, 2290089984, 3829399552, 3059744768, 1127219200, 3089629184, 4199809024, 3567124480, 1891565568, 394297344, 3988799488, 920674304, 4193267712, 2950604800, 3977188352, 3250028032, 129093376, 2231568512, 2963678272, 4281226848, 432124720, 803643432, 1633613396, 2672665246, 3170194367, 2147483648, 3221225472, 2684354560, 3489660928, 1476395008, 2483027968, 1040187392, 3808428032, 3196059648, 599785472, 505413632, 4077912064, 1182269440, 1736704000, 2017853440, 2221342720, 3329785856, 2810494976, 3628507136, 1416089600, 2658719744, 864310272, 3863387648, 3076993792, 553150080, 272922560, 4167467040, 1148698640, 1719673080, 2009075780, 2149644390, 3222291575, 2147483648, 1073741824, 2684354560, 1342177280, 2281701376, 1946157056, 436207616, 2566914048, 2625634304, 3208642560, 2720006144, 2098200576, 111673344, 2354315264, 3464626176, 4027383808, 2886631424, 3770826752, 1691164672, 3357462528, 1993345024, 3752330240, 873073152, 2870150400, 1700563072, 87021376, 1097028000, 1222351248, 1560027592, 2977959924, 23268898, 437609937);

int grayCode(int i)
{
    return i^(i>>1);
}
// 生成第 d 维度的第 i 个 sobol 数
float sobol(int d,int i)
{
    uint result = 0u;
    int offset = 32*d;
    for(int j = 0;i!=0;i>>=1,j++)
    {
        if((i&1)!=0)
        {
            result^=V[offset+j];
        }
    }
    return float(result)*(1.0f/float(0xFFFFFFFFU));
}

vec2 sobolVec2(int i,int b)
{
    float u = sobol(b*2,grayCode(i));
    float v = sobol(b*2+1,grayCode(i));
    return vec2(u,v);
}

vec2 CranleyPattersonRotation(vec2 p)
{
    uint pseed = uint(
    uint((pix.x*0.5+0.5)*width) *uint(1973)+
    uint((pix.y*0.5+0.5)*height)*uint(9277)+
    uint(114514/1919)*uint(26699))|uint(1);
    float u  =float(wang_hash(pseed))/4294967296.0;
    float v = float(wang_hash(pseed))/4294967296.0;
    p.x +=u ;
    if(p.x>1) p.x-=1;
    if(p.x<0) p.x+=1;
    p.y +=v;
    if(p.y>1) p.y-=1;
    if(p.y<0) p.y+=1;

    return p;
}

//-------------------------------------------------//

float sqr(float x)
{
    return x*x;
}

float SchlickFresnel(float u)
{//菲涅尔方程的一个近似计算函数,其实就是一个求5次方
    float m = clamp(1-u,0,1);
    float m2 = m*m;
    return m2*m2*m;//pow(m,5)
}

float GTR1(float NdotH,float a)
{//两种法线分布项
    if(a>=1) return 1/PI;
    float a2 = a*a;
    float t = 1+(a2-1)*NdotH*NdotH;
    return (a2-1)/(PI*log(a2)*t);
}

float GTR2(float NdotH,float a)
{//两种法线分布项
    float a2 = a*a;
    float t = 1+(a2-1)*NdotH*NdotH;
    return a2/(PI*t*t);
}
float GTR2_aniso(float NdotH,float HdotX,float HdotY,float ax,float ay)
{
    return 1/(PI*ax*ay*sqr(sqr(HdotX/ax)+sqr(HdotY/ay)+NdotH*NdotH));
}

float smithG_GGX(float NdotV,float alphaG)
{
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1/(NdotV+sqrt(a+b-a*b));
}

float smithG_GGX_aniso(float NdotV,float VdotX,float VdotY,float ax,float ay)
{
    return 1/(NdotV+sqrt(sqr(VdotX*ax)+sqr(VdotY*ay)+sqr(NdotV)));
}

vec3 BRDF_Evaluate(vec3 V,vec3 N,vec3 L,in Material material)
{
    float NdotL = dot(N,L);
    float NdotV = dot(N,V);
    if(NdotL<0||NdotV<0) return vec3(0);

    vec3 H = normalize(L+V);
    float NdotH = dot(N,H);
    float LdotH = dot(L,H);
    //各种颜色
    vec3 Cdlin = material.baseColor;
    float Cdlum = 0.3*Cdlin.r+0.6*Cdlin.g+0.1*Cdlin.b;
    vec3 Ctint = (Cdlum>0)?(Cdlin/Cdlum):(vec3(1));//用来偏转反射的色调,虽然偏转了,但是在后面根据
    //specularTint进行插值的时候,这部分我们依然看作是物体的原底色
    vec3 Cspec = material.specular * mix(vec3(1),Ctint,material.specularTint);
    vec3 Cspec0 = mix(0.08*Cspec,Cdlin,material.metallic);//菲涅尔项,这个确定了从0度观察物体时
    //得到反射的一个程度,非金属就是0.08*Cspec,而纯金属就是纯镜面反射,因此金属度来调整这个比例。
    vec3 Csheen = mix(vec3(1),Ctint,material.sheenTint);


    //漫反射
    float Fd90 = 0.5+2.0*material.roughness*LdotH*LdotH;
    float FL = SchlickFresnel(NdotL);
    float FV = SchlickFresnel(NdotV);
    float Fd = mix(1.0,Fd90,FL)*mix(1.0,Fd90,FV);//原公式的这里居然是插值我都看不出来,太菜了
    //次表面散射
    float Fss90 = LdotH*LdotH*material.roughness;
    float Fss = mix(1.0,Fss90,FL)*mix(1.0,Fss90,FV);
    float ss = 1.25*(Fss*(1.0/(NdotL+NdotV)-0.5)+0.5);

    //镜面反射--各向同性
    float alpha = max(0.001, sqr(material.roughness));//问题,你这里头传入之前就已经把粗糙度平方了,再在公式里平方会不会出错
    float Ds = GTR2(NdotH,alpha);
    float FH = SchlickFresnel(LdotH);
    vec3 Fs = mix(Cspec0,vec3(1),FH);
    float Gs = smithG_GGX(NdotL,material.roughness);
    Gs *= smithG_GGX(NdotV, material.roughness);//入射出射相乘才行
    //没有添加论文里分母上的那个校正因子
   
    //清漆
    float Dr = GTR1(NdotH,mix(0.1,0.001,material.clearcoatGloss));
    float Fr = mix(0.04,1.0,FH);
    float Gr = smithG_GGX(NdotL,0.25)*smithG_GGX(NdotV,0.25);

    //sheen
    vec3 Fsheen = FH*material.sheen*Csheen;


    vec3 diffuse = (1.0/PI)*mix(Fd,ss,material.subsurface)*Cdlin+Fsheen;
    vec3 specular = Gs*Fs*Ds;
    vec3 clearcoat = vec3(0.25*Gr*Fr*Dr*material.clearcoat);
    return diffuse*(1.0-material.metallic)+specular+clearcoat;//金属度满了以后就没有漫反射了
}

// ----------------------------------------------------------------------------- //

void getTagent(vec3 N,inout vec3 tangent,inout vec3 bitangent)
{
    vec3 helper = vec3(1,0,0);
    if(abs(N.x)>0.999) helper = vec3 (0,0,1);
    bitangent = normalize(cross(N,helper));
    tangent = normalize(cross(N,bitangent));
}

// 将向量 v 投影到 N 的法向半球
vec3 toNormalHemisphere(vec3 v, vec3 N) {
    vec3 helper = vec3(1, 0, 0);
    if(abs(N.x)>0.999) helper = vec3(0, 0, 1);
    vec3 tangent = normalize(cross(N, helper));
    vec3 bitangent = normalize(cross(N, tangent));
    return v.x * tangent + v.y * bitangent + v.z * N;
}

// ----------------------------------------------------------------------------- //

// 半球均匀采样
//这里的半球均匀采样的函数也是PBRT书里面给的,具体的数学原理我们先模糊掉,我们只知道公式当中
//需要两个随机数来进行计算,而这两个随机数就造成了我们采样的区别,我们只要在这个地方将之前生
//成的伪随机数替换为sobol底差异序列,就能实现更均匀的采样了。
vec3 SampleHemisphere(float xi_1,float xi_2) {
    float z = xi_1;
    float r = max(0, sqrt(1.0 - z*z));
    float phi = 2.0 * PI * xi_2;
    return vec3(r * cos(phi), r * sin(phi), z);
}
//半球普通采样
vec3 SampleHemisphere() {
    float z = rand();
    float r = max(0, sqrt(1.0 - z*z));
    float phi = 2.0 * PI * rand();
    return vec3(r * cos(phi), r * sin(phi), z);
}

//余弦加权的法向半球采样
vec3 SampleCosineHemisphere(float xi_1,float xi_2,vec3 N)
{
    float r = sqrt(xi_1);
    float theta = xi_2*2.0*PI;
    float x = r*cos(theta);
    float y = r*sin(theta);
    float z = sqrt(1.0-x*x-y*y);

    //从z半球投影到法相半球
    vec3 L = toNormalHemisphere(vec3(x,y,z),N);
    return L;
}
//清漆重要性采样
vec3 SampleGTR1(float xi_1,float xi_2,vec3 V,vec3 N,float alpha)
{
    float phi_h = 2.0 * PI * xi_1;
    float sin_phi_h = sin(phi_h);
    float cos_phi_h = cos(phi_h);

    float cos_theta_h = sqrt((1.0-pow(alpha*alpha, 1.0-xi_2))/(1.0-alpha*alpha));
    float sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));

    // 采样 "微平面" 的法向量 作为镜面反射的半角向量 h 
    vec3 H = vec3(sin_theta_h*cos_phi_h, sin_theta_h*sin_phi_h, cos_theta_h);
    H = toNormalHemisphere(H, N);   // 投影到真正的法向半球

    // 根据 "微法线" 计算反射光方向
    vec3 L = reflect(-V, H);

    return L;
}

vec3 SampleGTR2(float xi_1, float xi_2, vec3 V, vec3 N, float alpha)
{
    float phi_h = 2.0 * PI * xi_1;
    float sin_phi_h = sin(phi_h);
    float cos_phi_h = cos(phi_h);

    float cos_theta_h = sqrt((1.0-xi_2)/(1.0+(alpha*alpha-1.0)*xi_2));
    float sin_theta_h = sqrt(max(0.0, 1.0 - cos_theta_h * cos_theta_h));

    // 采样 "微平面" 的法向量 作为镜面反射的半角向量 h 
    vec3 H = vec3(sin_theta_h*cos_phi_h, sin_theta_h*sin_phi_h, cos_theta_h);
    H = toNormalHemisphere(H, N);   // 投影到真正的法向半球

    // 根据 "微法线" 计算反射光方向
    vec3 L = reflect(-V, H);

    return L;
}

vec3 SampleBRDF(float xi_1,float xi_2,float xi_3,vec3 V,vec3 N,in Material material)
{
    float alpha_GTR1 = mix(0.1,0.001,material.clearcoatGloss);
    float alpha_GTR2 = max(0.001,sqr(material.roughness));

    //辐射度统计
    float r_diffuse = (1.0-material.metallic);
    float r_specular = 1.0;
    float r_clearcoat = 0.25*material.clearcoat;
    float r_sum = r_diffuse+r_specular+r_clearcoat;
    //根据辐射度计算概率
    float p_diffuse = r_diffuse/r_sum;
    float p_specular = r_specular/r_sum;
    float p_clearcoat = r_clearcoat /r_sum;

    //按照概率采样
    float rd = xi_3;
    //漫反射
    if(rd<=p_diffuse)
    {
        return SampleCosineHemisphere(xi_1,xi_2,N);
    }
    //镜面反射
    else if(p_diffuse<rd&&rd<=p_diffuse+p_specular)
    {
        return SampleGTR2(xi_1,xi_2,V,N,alpha_GTR2);
    }
    //清漆
    else if(p_diffuse+p_specular<rd)
    {
        return SampleGTR1(xi_1,xi_2,V,N,alpha_GTR1);
    }
    return vec3(0,1,0);
}


//采样预计算的hdr cache
vec3 SampleHdr(float xi_1,float xi_2)
{
    vec2 xy = texture2D(hdrCache, vec2(xi_1, xi_2)).rg; // x, y
    xy.y = 1.0 - xy.y; // flip y

    // 获取角度
    float phi = 2.0 * PI * (xy.x - 0.5);    // [-pi ~ pi]
    float theta = PI * (xy.y - 0.5);        // [-pi/2 ~ pi/2]   

    // 球坐标计算方向
    vec3 L = vec3(cos(theta)*cos(phi), sin(theta), cos(theta)*sin(phi));

    return L;
}
// ----------------------------------------------------------------------------- //

// 将三维向量 v 转为 HDR map 的纹理坐标 uv
vec2 toSphericalCoord(vec3 v) {
    vec2 uv = vec2(atan(v.z, v.x), asin(v.y));
    uv /= vec2(2.0 * PI, PI);
    uv += 0.5;
    uv.y = 1.0 - uv.y;
    return uv;
}

// 获取 HDR 环境颜色
vec3 hdrColor(vec3 v) {
    vec2 uv = toSphericalCoord(normalize(v));
    vec3 color = texture2D(hdrMap, uv).rgb;
    //color = min(color, vec3(50000));
    
    return color;
}

//输入光线方向L获取HDR在该位置的概率密度
//hdr分辨率维4096*2048-->hdrResolution = 4096
float hdrPdf(vec3 L,int hdrResolution)
{
    vec2 uv = toSphericalCoord(normalize(L));//方向向量 uv 纹理坐标

    float pdf = texture2D(hdrCache,uv).b;
    float theta = PI*(0.5-uv.y);//图片域到积分域的转换系数和立体角有关
    float sin_theta = max(sin(theta),1e-10);

    //球坐标和图片积分与的转换系数
    float p_convert = float(hdrResolution*hdrResolution/2)/(2.0*PI*PI*sin_theta);

    return pdf*p_convert;
}

float BRDF_Pdf(vec3 V,vec3 N,vec3 L,in Material material)
{
    float NdotL = dot(N,L);
    float NdotV = dot(N,V);
    if(NdotL<0||NdotV<0) return 0;

    vec3 H = normalize(L+V);
    float NdotH = dot(N,H);
    float LdotH = dot(L,H);

    //镜面反射 --各项同性
    float alpha_Ds = max(0.001,sqr(material.roughness));
    float Ds = GTR2(NdotH,alpha_Ds);
    float Dr = GTR1(NdotH,mix(0.1,0.001,material.clearcoatGloss));//       清漆

    //三种BRDF的概率密度
    float pdf_diffuse = NdotL/PI;
    float pdf_specular= Ds * NdotH / (4.0 * dot(L, H));
    float pdf_clearcoat = Dr * NdotH / (4.0 * dot(L, H));

    //辐射度统计
    float r_diffuse = (1.0-material.metallic);
    float r_specular = 1.0;
    float r_clearcoat = 0.25* material.clearcoat;
    float r_sum = r_diffuse+r_specular+r_clearcoat;

    //根据辐射度计算选择某种采样方式的概率
    float p_diffuse  = r_diffuse/r_sum;
    float p_specular = r_specular/r_sum;
    float p_clearcoat = r_clearcoat/r_sum;

    //根据概率混合
    float pdf = p_diffuse* pdf_diffuse+p_specular*pdf_specular+ p_clearcoat*pdf_clearcoat;

    pdf = max(1e-10,pdf);

    return pdf;
}

float misMixWeight(float a,float b)
{
    float t = a*a;
    return t/(b*b+t);
}

// ----------------------------------------------------------------------------- //

// 路径追踪
//vec3 pathTracing(HitResult hit, int maxBounce) {
//      这个是漫反射重要性采样版本的光线追踪
//    vec3 Lo = vec3(0);      // 最终的颜色
//    vec3 history = vec3(1); // 递归积累的颜色
//    for(int bounce = 0;bounce<maxBounce;++bounce)
//    {
//        vec3 V = -hit.viewDir;
//        vec3 N = hit.normal;
//        vec2 uv = sobolVec2(frameCounter+1,bounce);
//        uv = CranleyPattersonRotation(uv);
//        //vec3 L = toNormalHemisphere(SampleHemisphere(uv.x,uv.y),hit.normal);
//        vec3 L = SampleCosineHemisphere(uv.x,uv.y,hit.normal);
//        float cosine_i = max(0,dot(N,L));
//        float cosine_o = max(0,dot(N,V));
//
//        //float pdf = 1.0/(2.0*PI);
//        
//        vec3 tangent,bitangent;
//        getTagent(N,tangent,bitangent);
//        vec3 f_r = BRDF_Evaluate(V,N,L,tangent,bitangent,hit.material);
//        vec3 H = normalize(V+L);
//        float NdotH = dot(hit.normal,H);
//        float pdf = NdotH/PI;
//
//        Ray randomRay;
//        randomRay.startPoint = hit.hitPoint;
//        randomRay.direction = L;
//        HitResult newhit = hitBVH(randomRay);
//        
//        if(!newhit.isHit)
//        {
//            vec3 skyColor = sampleHdr(randomRay.direction);
//            Lo+= history*skyColor*f_r*cosine_i/pdf;
//            break;
//        }
//
//        vec3 Le = newhit.material.emissive;
//        Lo +=history*Le*cosine_i/pdf;
//
//        hit = newhit;
//        history*= f_r*cosine_i/pdf;
//    }
//    return Lo;
//}
//vec3 pathTracing(HitResult hit, int maxBounce) {
//    //这个是BRDF重要性采样版本的光线追踪程序,其实主要是对镜面反射材质做了优化
//    vec3 Lo = vec3(0);      // 最终的颜色
//    vec3 history = vec3(1); // 递归积累的颜色
//    for(int bounce = 0;bounce<maxBounce;++bounce)
//    {
//        vec3 V = -hit.viewDir;
//        vec3 N = hit.normal;
//        //vec3 L = toNormalHemisphere(SampleHemisphere(),hit.normal);
//        float alpha = max(0.001,sqr(hit.material.roughness));
//        vec2 uv = sobolVec2(frameCounter+1,bounce);
//        uv = CranleyPattersonRotation(uv);
//        vec3 L = SampleGTR2(uv.x,uv.y,V,N,alpha);
//        //vec3 L = SampleCosineHemisphere(uv.x,uv.y,hit.normal);
//        float cosine_i = max(0,dot(N,L));
//        float cosine_o = max(0,dot(N,V));
//
//        //float pdf = 1.0/(2.0*PI);
//        
//        vec3 tangent,bitangent;
//        getTagent(N,tangent,bitangent);
//        vec3 f_r = BRDF_Evaluate(V,N,L,tangent,bitangent,hit.material);
//        vec3 H = normalize(V+L);
//        float NdotH = dot(hit.normal,H);
//        float pdf = BRDF_Pdf(V,N,L,hit.material);
//        if(pdf==0.0) break;//同样可以产生作用,所以又获得一个经验,那就是,这里当像素计算中遇到除法的时候,要保证分母不为0,
//        //否则会出现着色器错误,而这个错误不会以报错的形式丢出,而是当前像素直接变黑,并且在程序结束前都不会再显示任何东西。
//
//        Ray randomRay;
//        randomRay.startPoint = hit.hitPoint;
//        randomRay.direction = L;
//        HitResult newhit = hitBVH(randomRay);
//
//        if(!newhit.isHit)
//        {
//            vec3 skyColor = sampleHdr(randomRay.direction);
//            Lo+= history*skyColor*f_r*cosine_i/pdf;
//            break;
//        }
//
//        vec3 Le = newhit.material.emissive;
//        Lo +=history*Le*cosine_i/pdf;
//
//        hit = newhit;
//        history*= f_r*cosine_i/pdf;
//    }
//    return Lo;
//}
vec3 pathTracing(HitResult hit, int maxBounce) {
    //这个是BRDF重要性采样版本的光线追踪程序,其实主要是对镜面反射材质做了优化
    vec3 Lo = vec3(0);      // 最终的颜色
    vec3 history = vec3(1); // 递归积累的颜色
    for(int bounce = 0;bounce<maxBounce;++bounce)
    {
        vec3 V = -hit.viewDir;
        vec3 N = hit.normal;
        //vec3 L = toNormalHemisphere(SampleHemisphere(),hit.normal);
        float alpha = max(0.001,sqr(hit.material.roughness));
        vec2 uv = sobolVec2(frameCounter+1,bounce);
        uv = CranleyPattersonRotation(uv);
        vec3 L = SampleGTR2(uv.x,uv.y,V,N,alpha);
        //vec3 L = SampleCosineHemisphere(uv.x,uv.y,hit.normal);
        float cosine_i = max(0,dot(N,L));
        float cosine_o = max(0,dot(N,V));

        //float pdf = 1.0/(2.0*PI);
        
        vec3 tangent,bitangent;
        getTagent(N,tangent,bitangent);
        vec3 f_r = BRDF_Evaluate(V,N,L,hit.material);
        vec3 H = normalize(V+L);
        float NdotH = dot(hit.normal,H);
        float pdf = BRDF_Pdf(V,N,L,hit.material);
        if(pdf==0.0) break;

        Ray randomRay;
        randomRay.startPoint = hit.hitPoint;
        randomRay.direction = L;
        HitResult newhit = hitBVH(randomRay);

        if(!newhit.isHit)
        {
            vec3 skyColor = hdrColor(randomRay.direction);
            Lo+= history*skyColor*f_r*cosine_i/pdf;
            break;
        }

        vec3 Le = newhit.material.emissive;
        Lo +=history*Le*cosine_i/pdf;

        hit = newhit;
        history*= f_r*cosine_i/pdf;
    }
    return Lo;
}
//重要性采样版本
vec3 pathTracingImportanceSampling(HitResult hit ,int maxBounce)
{
    vec3 Lo = vec3(0);//最终颜色
    vec3 history = vec3(1);//递归累计项
    for(int bounce = 0;bounce<maxBounce;++bounce)
    {
        vec3 V = -hit.viewDir;
        vec3 N = hit.normal;

        //HDR 环境贴图重要性采样
        Ray hdrTestRay;
        hdrTestRay.startPoint = hit.hitPoint;
        hdrTestRay.direction = SampleHdr(rand(),rand());
        if(dot(N,hdrTestRay.direction)>0.0)
        {
            //如果采样方向背向点p则放弃测试,因为NdotL<0
            HitResult hdrHit = hitBVH(hdrTestRay);

            //天空光仅在没有遮挡的情况下积累亮度
            if(!hdrHit.isHit)
            {
                //获取采样方向L上的:1.光照贡献,2.环境贴图在该位置的pdf,3.BRDF函数值,4.BRDF在该方向的pdf
                vec3 L = hdrTestRay.direction;
                vec3 color = hdrColor(L);
                float pdf_light = hdrPdf(L,hdrResolution);
                vec3 f_r = BRDF_Evaluate(V,N,L,hit.material);
                float pdf_brdf = BRDF_Pdf(V,N,L,hit.material);//这个是给后面的多重重要性采样用的。

                //多重重要性采样
                float mis_weight = misMixWeight(pdf_light,pdf_brdf);
                Lo += mis_weight*history * color * f_r * dot(N, L) / pdf_light; // 累计亮度
            }
        }
        vec2 sobol = sobolVec2(frameCounter+1,bounce);
        vec2 uv = CranleyPattersonRotation(sobol);
        float xi_1 = uv.x;
        float xi_2 = uv.y;
        float xi_3 = rand();

        vec3 L = SampleBRDF(xi_1,xi_2,xi_3,V,N,hit.material);
        float NdotL = dot(N,L);
        if(NdotL<=0.0) break;
        //发射光线
        Ray randomRay;
        randomRay.startPoint = hit.hitPoint;
        randomRay.direction = L;
        HitResult newhit = hitBVH(randomRay);

        //获取L方向上的BRDF值和概率密度
        vec3 f_r = BRDF_Evaluate(V,N,L,hit.material);
        float pdf_brdf = BRDF_Pdf(V,N,L,hit.material);
        if(pdf_brdf<=0.0) break;

        //未命中
        if(!newhit.isHit)
        {
            vec3 color = hdrColor(L);
            float pdf_light = hdrPdf(L,hdrResolution);

            //多重重要性采样
            float mis_weight = misMixWeight(pdf_brdf,pdf_light);
            Lo +=mis_weight*history*color*NdotL*f_r/pdf_brdf;
            break;
        }
        vec3 Le = newhit.material.emissive;
        Lo +=history*Le*NdotL*f_r/pdf_brdf;

        //递归

        hit = newhit;
        history*=NdotL*f_r/pdf_brdf;
    }
    return Lo;
}
//-----------------------------------------------------------------------------//

void main() {



    
    // 投射光线
    Ray ray;
    
    ray.startPoint = eye;
    vec2 AA = vec2((rand()-0.5)/float(width), (rand()-0.5)/float(height));
    vec4 dir = cameraRotate * vec4(pix.xy+AA, -1.5, 0.0);
    ray.direction = normalize(dir.xyz);

    // primary hit
    HitResult firstHit = hitBVH(ray);
    vec3 color;
    
    if(!firstHit.isHit) {
        //color = vec3(0.2,0.3,0.2);
        color = hdrColor(ray.direction);
    } else {
        vec3 Le = firstHit.material.emissive;
        //vec3 Li = pathTracing(firstHit, 4);
        vec3 Li = pathTracingImportanceSampling(firstHit,4);
        color = Le + Li;
    }  
    
    // 和上一帧混合
    vec3 lastColor = texture2D(lastFrame, pix.xy*0.5+0.5).rgb;//一直是黑的
    color = mix(lastColor, color, 1.0/(frameCounter+1));//frameCounter一直是0
    //color = vec3(frameCounter*0.001,frameCounter*0.001,frameCounter*0.001+2);



///*采样可视化部分代码,想看了就屏蔽上面,单独留下这一部分
//    vec3 color = vec3(0);
//    for(int i=0; i<50; i++) {//循环次数根据采样种类进行调节,重要性采样可以循环个几次到几十次,这样可视化的速率正好,如果是伪随机数的化,就设为5000次
//        /*  //重要性采样可视化
//        vec2 uv;
//        uv=sobolVec2(frameCounter+1,1);
//        uv.x = (uv.x-0.5)*2;
//        uv.y = (uv.y-0.5)*2;
//        if(distance(pix.xy, uv)<0.006) color.rgb = vec3(1, 0, 0);
//        vec3 lastColor = texture2D(lastFrame, pix.xy*0.5+0.5).rgb;//一直是黑的
//        color = max(lastColor,color);*/
//        
//        /*   //伪随机数采样可视化
//        if(i==4999)
//        {
//            uv = vec2(rand(),rand());
//            uv.x = (uv.x-0.5)*2;
//            uv.y = (uv.y-0.5)*2;
//            if(distance(pix.xy, uv)<0.01) color.rgb = vec3(1, 0, 0);
//        vec3 lastColor = texture2D(lastFrame, pix.xy*0.5+0.5).rgb;//一直是黑的
//        color = max(lastColor,color);
//        }*/ 
//    }
//    */

   
    // 输出
    gl_FragData[0] = vec4(color, 1.0);
}
