#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define MAX_UNITS (1 << 29)
#define UNINIT -1

enum Op : uint8_t
{
	OP_ADD,
	OP_MUL,
	OP_POW,
	OP_RELU,
	OP_NULL,
	OP_COUNT
};

double *v;
double *g;
uint32_t *c;
uint8_t *op;
int *topo;
int topo_count;
uint64_t *topo_visited;

static int next;

static inline int CU()
{
	if (next > (MAX_UNITS - 2))
	{
		printf("Out of units bb\n");
	}
	return next++;
}

static void BAdd(int x)
{
	g[c[x << 1]] += g[x]; 
	g[c[(x << 1) | 1]] += g[x]; 
}

static void BMul(int x)
{
	g[c[x << 1]] += v[c[(x << 1) | 1]] * g[x]; 
	g[c[(x << 1) | 1]] += v[c[x << 1]] * g[x]; 
}

static void BPow(int x)
{
	double exp = v[c[(x << 1) | 1]];
	g[c[x << 1]] += (exp * pow(v[c[x << 1]], exp - 1.0)) * g[x];
}

static void BRelu(int x)
{
	if (v[x] > 0.0) g[c[x << 1]] += g[x];
}

void (*Back[])(int) = {
	&BAdd,
	&BMul,
	&BPow,
	&BRelu
};

const char* OpNames[] = {
	"+",
	"*",
	"^",
	"ReLU",
	"NULL"
};

static int New(double x)
{
	v[next] = x;
	g[next] = 0.0;
	c[next << 1] = UNINIT;
	c[(next << 1) | 1] = UNINIT;
	op[next] = OP_NULL;
	return CU();
}

static int Add(int x, int y)
{
	v[next] = v[x] + v[y];
	c[next << 1] = x;
	c[(next << 1) | 1] = y;
	op[next] = OP_ADD;

	return CU();
}

static int Mul(int x, int y)
{
	v[next] = v[x] * v[y];
	c[next << 1] = x;
	c[(next << 1) | 1] = y;
	op[next] = OP_MUL;

	return CU();
}

static int Pow(int x, int y)
{
	v[next] = pow(v[x], v[y]);
	c[next << 1] = x;
	c[(next << 1) | 1] = y;
	op[next] = OP_POW;

	return CU();
}

static int Relu(int x)
{
	if (v[x] > 0.0)
		v[next] = v[x];
	else
		v[next] = 0.0;
	
	c[next << 1] = x;
	c[(next << 1) | 1] = UNINIT;
	op[next] = OP_RELU;

	return CU();
}

static int Neg(int x)
{
	return Mul(x, New(-1.0));
}

static int Sub(int x, int y)
{
	return Add(x, Neg(y));
}

static int Div(int x, int y)
{
	return Mul(x, Pow(y, New(-1.0)));
}

void Topo(int x)
{
	if (topo_visited[x >> 6] & (1L << (x & 63)))
	{
		return;
	}

	topo_visited[x >> 6] |= (1L << (x & 63));
	if (op[x] < OP_NULL)
	{
		Topo(c[x << 1]);
		if (op[x] < OP_POW)
			Topo(c[(x << 1) | 1]);

		topo[topo_count++] = x;
	}
}

void Backward()
{
	g[topo[topo_count - 1]] = 1.0;
	int i = topo_count;
	while (i--)
	{
		Back[op[topo[i]]](topo[i]);
	}
}

void Print(int x)
{
	printf("%d(v: %.3lf, g:%.3lf)(%d %s %d)\n", x, v[x], g[x], c[x << 1], OpNames[op[x]], c[(x << 1) | 1]);
}

int Count()
{
	return next;
}

void ClearTopoData()
{
	memset(topo, 0, MAX_UNITS * sizeof(int));
	memset(topo_visited, 0, MAX_UNITS / 8);
	topo_count = 0;
}

void ClearGrad(int count)
{
	memset(g, 0, count * sizeof(double));
}

void FreeFrom(int c)
{
	next = c;
}
