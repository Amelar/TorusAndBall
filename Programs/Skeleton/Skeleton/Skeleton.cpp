//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2016-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kivéve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : David Adam
// Neptun : AO4S5C
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded 
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 0;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

const char *vertexSource2 = R"(
#version 130
precision highp float;
uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
uniform vec4  wLiPos;       // pos of light source 
uniform vec4  wLiPos2;
uniform vec3  wEye;         // pos of eye


in  vec3 vtxPos;            // pos in modeling space
in  vec3 vtxNorm;           // normal in modeling space
in float u;
in float v;

out vec3 wNormal;           // normal in world space
out vec3 wView;             // view in world space
out vec3 wLight;            // light dir in world space
out vec2 texcoord;
out vec3 wLight2;

void main() {
	gl_Position = vec4(vtxPos, 1) * MVP; // to NDC

		vec4 wPos = vec4(vtxPos, 1) * M;
	wLight = wLiPos.xyz * wPos.w - wPos.xyz*1;
	wLight2 =  wLiPos2.xyz * wPos.w - wPos.xyz*1;
	wView = wEye * wPos.w - wPos.xyz;
	wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
	texcoord = vec2(u,v);
})";

const char *fragmentSource2 = R"(
#version 130
precision highp float;
uniform sampler2D samplerUnit;
uniform vec3 kd, ks, ka;// diffuse, specular, ambient ref
uniform vec3 La, Le;    // ambient and point source rad
uniform vec3 La2, Le2;
uniform float shine;    // shininess for specular ref

in vec2 texcoord;
in  vec3 wNormal;       // interpolated world sp normal
in  vec3 wView;         // interpolated world sp view
in  vec3 wLight;        // interpolated world sp illum dir
in vec3 wLight2;
out vec4 fragmentColor; // output goes to frame buffer

void main() {
   vec3 N = normalize(wNormal);
   vec3 V = normalize(wView); 
   vec3 L2 = normalize(wLight2); 
   vec3 L = normalize(wLight);
   vec3 H = normalize(L + V);
   vec3 H2 = normalize(L2 + V);
   float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
   float cost2 = max(dot(N,L2), 0), cosd2 = max(dot(N,H2), 0);
   vec3 color = ka * La + 
               (kd * cost + ks * pow(cosd,shine)) * Le;
   color += ka * La2 + 
               (kd * cost2 + ks * pow(cosd2,shine)) * Le2;

	   fragmentColor = vec4(color, 1)*texture(samplerUnit, texcoord);
}

)";

float minn(float v1, float v2) {
	if (v1 > v2)
		return v2;
	else
		return v1;
}

struct mat4 {
	float m[4][4];
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}



	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }

	void SetUniform(unsigned shaderProg, char * name) {
		int loc = glGetUniformLocation(shaderProg, name);
		glUniformMatrix4fv(loc, 1, GL_TRUE, &m[0][0]);
	}
};

struct vec3 {
	float v[3];

	vec3(float x = 0, float y = 0, float z = 0) {
		v[0] = x; v[1] = y; v[2] = z;
	}

	vec3 operator+(const vec3& vec) const {
		vec3 result;
		result.v[0] = v[0] + vec.v[0];
		result.v[1] = v[1] + vec.v[1];
		result.v[2] = v[2] + vec.v[2];
		return result;
	}

	float dot(const vec3 &vec) const {
		float d = v[0] * vec.v[0] + v[1] * vec.v[1] + v[2] * vec.v[2];
		return d;
	}


	vec3 operator-(const vec3& vec) const {
		vec3 result;
		result.v[0] = v[0] - vec.v[0];
		result.v[1] = v[1] - vec.v[1];
		result.v[2] = v[2] - vec.v[2];
		return result;
	}

	vec3 operator*(const vec3& vec) const {
		vec3 result;
		result.v[0] = v[0] * vec.v[0];
		result.v[1] = v[1] * vec.v[1];
		result.v[2] = v[2] * vec.v[2];
		return result;
	}

	vec3 operator*(float f) const {
		vec3 result;
		result.v[0] = v[0] * f;
		result.v[1] = v[1] * f;
		result.v[2] = v[2] * f;
		return result;
	}

	vec3 operator/(float f) const {
		vec3 result;
		result.v[0] = v[0] / f;
		result.v[1] = v[1] / f;
		result.v[2] = v[2] / f;
		return result;
	}

	float length() const {
		float lg = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		return lg;
	}

	vec3 normalize() const {
		float f = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
		return *this / f;
	}
};

struct Texture {
	unsigned int textureId;
	Texture() {}
	Texture(vec3 vec1, vec3 vec2) {
		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);    // binding
		int width, height;
		float image[36 * 36 * 3];
		for (int i = 0; i < 36; i++)
			for (int j = 0; j < 108; j = j + 12) {
				image[i * 36 * 3 + j] = vec1.v[0];
				image[i * 36 * 3 + j + 1] = vec1.v[1];
				image[i * 36 * 3 + j + 2] = vec1.v[2];
				image[i * 36 * 3 + j + 3] = vec2.v[0];
				image[i * 36 * 3 + j + 4] = vec2.v[1];
				image[i * 36 * 3 + j + 5] = vec2.v[2];
				image[i * 36 * 3 + j + 6] = vec2.v[0];
				image[i * 36 * 3 + j + 7] = vec2.v[1];
				image[i * 36 * 3 + j + 8] = vec2.v[2];
				image[i * 36 * 3 + j + 9] = vec1.v[0];
				image[i * 36 * 3 + j + 10] = vec1.v[1];
				image[i * 36 * 3 + j + 11] = vec1.v[2];

			}
		//float image[12] = { vec1.v[0], vec1.v[1], vec1.v[2], vec2.v[0], vec2.v[1], vec2.v[2], vec2.v[0], vec2.v[1], vec2.v[2], vec1.v[0], vec1.v[1], vec1.v[2] };
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 36, 36,
			0, GL_RGB, GL_FLOAT, image); //Texture -> OpenGL
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	}
};



mat4 Translate(float tx, float ty, float tz) {
	return mat4(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		tx, ty, tz, 1);
}

mat4 Rotate(float angle, float wx, float wy, float wz) {
	return mat4(1 - (wx*wx - 1)*(cosf(angle) - 1), -wz*sinf(angle) - wx*wy*(cosf(angle) - 1), wx*sinf(angle) - wx*wz*(cosf(angle) - 1), 0,
		wz*sinf(angle) - wx*wy*(cosf(angle) - 1), 1 - (wy*wy - 1)*(cosf(angle) - 1), -wx*sinf(angle) - wy*wz*(cosf(angle) - 1), 0,
		-wy*sinf(angle) - wx*wz*(cosf(angle) - 1), wx*sinf(angle) - wy*wz*(cosf(angle) - 1), 1 - (wz*wz - 1)*(cosf(angle) - 1), 0,
		0, 0, 0, 1);
}
mat4 Scale(float sx, float sy, float sz) {
	return mat4(sx, 0, 0, 0,
		0, sy, 0, 0,
		0, 0, sz, 0,
		0, 0, 0, 1);
}



struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}
};

vec3 cross(vec3 vec1, vec3 vec) {
	vec3 result;
	result.v[0] = vec1.v[1] * vec.v[2] - vec1.v[2] * vec.v[1];
	result.v[1] = vec1.v[2] * vec.v[0] - vec1.v[0] * vec.v[2];
	result.v[2] = vec1.v[0] * vec.v[1] - vec1.v[1] * vec.v[0];
	return result;
}

struct Light {
	vec3 wLiPos;
	vec3 dir;
	vec4 wLight;
	float t = 0;
	vec3 La;
	vec3 Le;
	Light() {}

	Light(vec3 wLiPos, vec4 wLight, vec3 La, vec3 Le) :wLiPos(wLiPos), wLight(wLight), La(La), Le(Le) {
		dir = vec3(0, 1, 1).normalize();
	}



	void Animate(float dt) {
		vec3 lastpos = wLiPos;
		bool change = false;
		float time = (dt - t) * 20;

		float TorusWall = powf(8 - sqrtf((powf(wLiPos.v[0] + 8, 2) + (powf(wLiPos.v[2] + 4, 2)))), 2) + powf(wLiPos.v[1], 2) - 4 * 4;
		if (TorusWall >= 0.0f)
			change = true;
		if (change) {
			vec3 normal;
			normal.v[0] = wLiPos.v[0] *
				(2 - (2 * 8) / (sqrt(powf(wLiPos.v[0], 2) + powf(wLiPos.v[2], 2))));
			normal.v[1] = 2 * wLiPos.v[1];
			normal.v[2] = wLiPos.v[2] *
				(2 - (2 * 8) / (sqrt(powf(wLiPos.v[0], 2) + powf(wLiPos.v[2], 2))));
			normal = (normal*-1).normalize();
			dir = (dir - (normal * normal.dot(dir)) *2.0f);
			dir = dir.normalize();
			wLiPos = wLiPos + dir*(time * 2);
		}
		wLiPos = wLiPos + dir*(time);
		t = dt;


	}



};

struct Material {
	vec3 kd, ks, ka;// diffuse, specular, ambient ref
	vec3 La, Le;    // ambient and point source rad
	float shine;    // shininess for specular ref
	Material() {
		kd = vec3(0.5, 0.0, 0.5);
		ks = vec3(0.5, 0.0, 0.5);
		ka = vec3(0.5, 0.0, 0.5);
		shine = 5.0f;
	}

	Material(vec3 kd, vec3 ks, vec3 ka, float shine)
		:kd(kd), ks(ks), ka(ka), shine(shine)
	{}

	void create(unsigned int shaderProg) {
		int loc = glGetUniformLocation(shaderProg, "kd");
		glUniform3f(loc, kd.v[0], kd.v[1], kd.v[2]);
		int loc2 = glGetUniformLocation(shaderProg, "ks");
		glUniform3f(loc2, ks.v[0], ks.v[1], ks.v[2]);
		int loc3 = glGetUniformLocation(shaderProg, "ka");
		glUniform3f(loc3, ka.v[0], ka.v[1], ka.v[2]);
		int loc6 = glGetUniformLocation(shaderProg, "shine");
		glUniform1f(loc6, shine);
	}


};

struct RenderState {
	RenderState() {}

	mat4 M, V, P, Minv;
	Light light;
	Light light2;
	Material *material;
	Texture * texture;
	vec3 lookat;
	vec3 wEye;

};

struct Geometry {
	unsigned int vao, nVtx;

	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
	}
	void Draw(RenderState state, unsigned int shaderProg) {
		int samplerUnit = 0; // GL_TEXTURE1, …
		int location = glGetUniformLocation(shaderProg, "samplerUnit");
		glUniform1i(location, samplerUnit);
		glActiveTexture(GL_TEXTURE0 + samplerUnit);
		glBindTexture(GL_TEXTURE_2D, state.texture->textureId);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, nVtx);

	}
};

struct VertexData {
	vec3 position, normal;
	float u, v;

	VertexData() {}
};

struct ParamSurface : Geometry {
	virtual VertexData GenVertexData(float u, float v) = 0;

	void Create(int N, int M) {
		nVtx = N * M * 6;
		unsigned int vbo;
		glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);

		VertexData *vtxData = new VertexData[nVtx], *pVtx = vtxData;
		for (int i = 0; i < N; i++) for (int j = 0; j < M; j++) {
			*pVtx++ = GenVertexData((float)i / N, (float)j / M);
			*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
			*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
			*pVtx++ = GenVertexData((float)(i + 1) / N, (float)j / M);
			*pVtx++ = GenVertexData((float)(i + 1) / N, (float)(j + 1) / M);
			*pVtx++ = GenVertexData((float)i / N, (float)(j + 1) / M);
			VertexData vtx = GenVertexData((float)i / N, (float)j / M);
		}

		int stride = sizeof(VertexData), sVec3 = sizeof(vec3);
		glBufferData(GL_ARRAY_BUFFER, nVtx * stride, vtxData, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);  // AttribArray 0 = POSITION
		glEnableVertexAttribArray(1);  // AttribArray 1 = NORMAL
		glEnableVertexAttribArray(2);  // AttribArray 2 = UV
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)sVec3);
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(2 * sVec3));
	}

};

class Sphere : public ParamSurface {
	vec3 center;
	float radius;
public:
	Sphere(vec3 c, float r) : center(c), radius(r) {
		Create(32, 16); // tessellation level
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.normal = vec3(cos(u * 2 * M_PI) * sin(v*M_PI),
			sin(u * 2 * M_PI) * sin(v*M_PI),
			cos(v*M_PI));
		vd.position = vd.normal * radius + center;
		vd.u = u; vd.v = v;
		return vd;
	}
};

class Torus : public ParamSurface {
	vec3 center;
	float R;
	float r;
public:
	Torus(vec3 c, float R, float r) : center(c), R(R), r(r)
	{
		Create(32, 32);
	}

	VertexData GenVertexData(float u, float v) {
		VertexData vd;
		vd.position.v[0] = ((R + r*cos(u * 2 * M_PI))*cos(v * 2 * M_PI));
		vd.position.v[1] = r * sin(u * 2 * M_PI);
		vd.position.v[2] = (R + r * cos(u * 2 * M_PI))*sin(v * 2 * M_PI);
		vd.position = vd.position + center;
		vd.normal.v[0] = vd.position.v[0] *
			(2 - (2 * R) / (sqrt(powf(vd.position.v[0], 2) + powf(vd.position.v[2], 2))));
		vd.normal.v[1] = 2 * vd.position.v[1];
		vd.normal.v[2] = vd.position.v[2] *
			(2 - (2 * R) / (sqrt(powf(vd.position.v[0], 2) + powf(vd.position.v[2], 2))));
		vd.normal = vd.normal*-1;
		vd.u = u;
		vd.v = v;
		return vd;
	}
};





class Camera {
	vec3 wLookat, wVup;
	float fov, asp, fp, bp;
public:
	vec3  wEye;

	Camera()
	{
		wEye = vec3(16, 8, 8);
		wLookat = vec3(0, 0, 0);
		wVup = vec3(0, 1, 0);
		fov = 1;
		asp = 1;
		fp = 1;
		bp = 100;
	}

	Camera(vec3 wEye, float fov, float asp, float fp, float bp)
		:wEye(wEye), fov(fov), asp(asp), fp(fp), bp(bp)
	{
		wLookat = vec3(0, 0, 0);
		wVup = vec3(0, 1, 0);
	}

	mat4 V() { // view matrix
		vec3 w = (wEye - wLookat).normalize();
		vec3 u = cross(wVup, w).normalize();
		vec3 v = cross(w, u);
		return Translate(-wEye.v[0], -wEye.v[1], -wEye.v[2]) *
			mat4(u.v[0], v.v[0], w.v[0], 0.0f,
				u.v[1], v.v[1], w.v[1], 0.0f,
				u.v[2], v.v[2], w.v[2], 0.0f,
				0.0f, 0.0f, 0.0f, 1.0f);
	}
	mat4 P() { // projection matrix
		float sy = 1 / tan(fov / 2);
		return mat4(sy / asp, 0.0f, 0.0f, 0.0f,
			0.0f, sy, 0.0f, 0.0f,
			0.0f, 0.0f, -(fp + bp) / (bp - fp), -1.0f,
			0.0f, 0.0f, -2 * fp*bp / (bp - fp), 0.0f);
	}
};



struct Shader {
	unsigned int shaderProg;


	void Create(const char * vsSrc, const char * vsAttrNames[],
		const char * fsSrc, const char * fsOuputName) {
		unsigned int vs = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vs, 1, &vsSrc, NULL); glCompileShader(vs);
		checkShader(vs, "Vertex shader error");
		unsigned int fs = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fs, 1, &fsSrc, NULL); glCompileShader(fs);
		checkShader(fs, "Fragment shader error");
		shaderProg = glCreateProgram();
		glAttachShader(shaderProg, vs);
		glAttachShader(shaderProg, fs);

		for (int i = 0; i < sizeof(vsAttrNames) / sizeof(char*); i++)
			glBindAttribLocation(shaderProg, i, vsAttrNames[i]);
		glBindFragDataLocation(shaderProg, 0, fsOuputName);
		glLinkProgram(shaderProg);
	}
	virtual
		void Bind(RenderState& state) {
		glUseProgram(shaderProg);
	}
};


struct PerShader : Shader {
public:
	PerShader() {
		const char* valami[2] = { "vtxPos", "vtxNorm" };
		Create(vertexSource2, valami, fragmentSource2, "fragmentColor");
	}
	void Bind(RenderState& state) {
		int loc = glGetUniformLocation(shaderProg, "wLiPos");
		glUniform4f(loc, state.light.wLiPos.v[0], state.light.wLiPos.v[1], state.light.wLiPos.v[2], 1);
		int loc3 = glGetUniformLocation(shaderProg, "wLiPos2");
		glUniform4f(loc3, state.light2.wLiPos.v[0], state.light2.wLiPos.v[1], state.light2.wLiPos.v[2], 1);
		int loc4 = glGetUniformLocation(shaderProg, "La");
		glUniform3f(loc4, state.light.La.v[0], state.light.La.v[1], state.light.La.v[2]);
		int loc5 = glGetUniformLocation(shaderProg, "Le");
		glUniform3f(loc5, state.light.Le.v[0], state.light.Le.v[1], state.light.Le.v[2]);
		int loc6 = glGetUniformLocation(shaderProg, "La2");
		glUniform3f(loc6, state.light2.La.v[0], state.light2.La.v[1], state.light2.La.v[2]);
		int loc7 = glGetUniformLocation(shaderProg, "Le2");
		glUniform3f(loc7, state.light2.Le.v[0], state.light2.Le.v[1], state.light2.Le.v[2]);
		int loc2 = glGetUniformLocation(shaderProg, "wEye");
		glUniform3f(loc2, state.wEye.v[0], state.wEye.v[1], state.wEye.v[2]);
		state.M.SetUniform(shaderProg, "M");
		state.Minv.SetUniform(shaderProg, "Minv");
		(state.M*state.V*state.P).SetUniform(shaderProg, "MVP");
		state.material->create(shaderProg);
		glUseProgram(shaderProg);

	}
};


class Object {
protected:
	Shader *shader;
	Geometry *geometry;
	Material *material;
	Texture *texture;
	vec3 scale, pos, rotAxis;
	float rotAngle;

public:

	Object(Shader *shader, Texture *text, Geometry *geo, Material * material, vec3 pos, vec3 rotAxis, vec3 scale, float rotAngle)
		:shader(shader), texture(text), geometry(geo), material(material), pos(pos), rotAxis(rotAxis), scale(scale), rotAngle(rotAngle)
	{
	}
	virtual void Draw(RenderState state) {
		state.M = Scale(scale.v[0], scale.v[1], scale.v[2]) *
			Rotate(rotAngle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]) *
			Translate(pos.v[0], pos.v[1], pos.v[2]);
		state.Minv = Translate(-pos.v[0], -pos.v[1], -pos.v[2]) *
			Rotate(-rotAngle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]) *
			Scale(1 / scale.v[0], 1 / scale.v[1], 1 / scale.v[2]);
		state.material = material; state.texture = texture;
		shader->Bind(state);
		geometry->Draw(state, shader->shaderProg);
	}
	virtual void Animate(float dt) {
	}
};

class AnimeObject : public Object {
public:
	AnimeObject(Shader *shader, Texture *text, Geometry *geo, Material * material, vec3 pos, vec3 rotAxis, vec3 scale, float rotAngle)
		:Object(shader, text, geo, material, pos, rotAxis, scale, rotAngle) {}
	void Draw(RenderState state) {
		state.M = Scale(scale.v[0], scale.v[1], scale.v[2]) *
			Rotate(rotAngle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]) *
			Translate(pos.v[0], pos.v[1], pos.v[2]);
		state.Minv = Translate(-pos.v[0], -pos.v[1], -pos.v[2]) *
			Rotate(-rotAngle, rotAxis.v[0], rotAxis.v[1], rotAxis.v[2]) *
			Scale(1 / scale.v[0], 1 / scale.v[1], 1 / scale.v[2]);
		state.material = material; state.texture = texture;
		shader->Bind(state);
		geometry->Draw(state, shader->shaderProg);
		char ch = '3';
		int c = (int)ch;
		printf("%d", c);
	}
	void Animate(float dt) {
		vec3 normal;
		dt = fmod(dt, 1);
		float sdt = fmod(fabs(sinf(dt*M_PI)), 1);
		float cdt = fmod(fabs(cosf(dt)), 1);
		vec3 lastpos = pos;
		pos.v[0] = ((8 + 3 * cos(sdt * 2 * M_PI))*cos(dt * 2 * M_PI)) + 8;
		pos.v[1] = 3 * sin(sdt * 2 * M_PI) + 2;
		pos.v[2] = (8 + 3 * cos(sdt * 2 * M_PI))*sin(dt * 2 * M_PI) - 4;
		normal.v[0] = pos.v[0] *
			(2 - (2 * 8) / (sqrt(powf(pos.v[0], 2) + powf(pos.v[2], 2))));
		normal.v[1] = 2 * pos.v[1];
		normal.v[2] = pos.v[2] *
			(2 - (2 * 8) / (sqrt(powf(pos.v[0], 2) + powf(pos.v[2], 2))));
		normal = normal*-1;
		vec3 w = cross(normal*-1, (lastpos - pos));
		rotAxis = w.normalize();
		rotAngle = M_PI;
	}
};



class Scene {
	Camera camera;
	std::vector<Object *> objects;
	Light light;
	Light light2;
	RenderState state;
public:

	Scene()
	{
		camera = Camera(vec3(0, 0, -10), 1.45, 1, 1, 100);
		light = Light(vec3(0, 0, 0), vec4(1, 1, 1), vec3(0.0f, 0.0f, 0.0f), vec3(0.5f, 0.5f, 0.5f));
		light2 = Light(vec3(0, 0, -10), vec4(1, 0, 1), vec3(0.0f, 0.0f, 0.0f), vec3(0.25f, 0.89f, 0.96f));
		Shader *shader = new PerShader();
		objects.push_back(new Object(shader, new Texture(vec3(0.0, 1.0, 1.0), vec3(1.0, 1.0, 1.0)), new Torus(vec3(0, 0, 0), 4, 2), new Material(), vec3(8, 0, -4), vec3(1, 1, 0), vec3(2, 2, 2), 0));
		objects.push_back(new AnimeObject(shader, new Texture(vec3(1.0, 1.0, 1.0), vec3(1.0, 0.0, 0.0)), new Sphere(vec3(0, 0, 0), 1), new Material(), vec3(0, 0, -2), vec3(1, 0, 0), vec3(1, 1, 1), 0));
	}

	void Render() {

		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.light = light;
		state.light2 = light2;
		for (Object * obj : objects) obj->Draw(state);
	}

	void Animate(float dt) {
		for (Object * obj : objects) obj->Animate(dt);
		light.Animate(dt);
		light2.Animate(dt);
	}
};


// handle of the shader program
unsigned int shaderProgram;
Scene *scene;



// Initialization, create an OpenGL context
void onInitialization() {
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glViewport(0, 0, windowWidth, windowHeight);
	scene = new Scene();
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene->Render();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

float tend = 0.0f;

// Idle event indicating that some time elapsed: do animation here
void onIdle() {

	const float dt = 0.1f;
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 4000.0f;

	for (float t = tstart; t < tend; t += dt) {
		if (tend - t < dt)
			scene->Animate(tend);
		else
			scene->Animate(t);
	}
	glutPostRedisplay();


}

// Idaig modosithatod...
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_2_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}
