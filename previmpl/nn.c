#include <stdlib.h>
#include "val.c"

double RNG(double min, double max)
{
	return rand() / (RAND_MAX / (max - min)) + min;
}

struct Neuron
{
	int c;
	int *w;
	int b;
};

struct Layer
{
	int c;
	struct Neuron *neurons;
	int *out;
};

struct MLP
{
	struct Layer *layers;
	int c;
};

void InitNeuron(struct Neuron *this, int count)
{
	this->c = count;
	this->w = (int*)malloc(count * sizeof(int));
	for (int i = 0; i < count; ++i)
		this->w[i] = New(RNG(-1.0, 1.0));
	this->b = New(RNG(-1.0, 1.0));
}

int Neuron(struct Neuron *this, int *x)
{
	int dot = New(0.0);
	for (int i = 0; i < this->c; ++i)
	{
		dot = Add(dot, Mul(this->w[i], x[i]));
	}

	return Add(dot, this->b);
}

void InitLayer(struct Layer *this, int in, int out)
{
	this->neurons = malloc(out * sizeof(struct Neuron));
	this->out = malloc(out * sizeof(int));
	this->c = out;
	for (int i = 0; i < out; ++i)
	{
		InitNeuron(this->neurons + i, in);
	}
}

int *Layer(struct Layer *this, int *x, int nonlin)
{
	if (nonlin)
	{
		for (int i = 0; i < this->c; ++i)
		{
			this->out[i] = Relu(Neuron(this->neurons + i, x));
		}
	}
	else
	{
		for (int i = 0; i < this->c; ++i)
		{
			this->out[i] = Neuron(this->neurons + i, x);
		}
	}
	return this->out;
}

void InitMLP(struct MLP *this, int *sizes, int count)
{
	this->layers = malloc(count * sizeof(struct Layer));
	this->c = count;
	for (int i = 0; i < count; ++i)
	{
		InitLayer(this->layers + i, sizes[i], sizes[i + 1]);
	}
}

int *MLP(struct MLP *this, int *x)
{
	int *data = x;
	for (int i = 0; i < this->c; ++i)
	{
		data = Layer(this->layers + i, data, i != this->c - 1);
	}
	return data;
}

void MLP_SaveParams(struct MLP *this, const char *filename)
{
	FILE *f;
	f = fopen(filename, "wb");
	for (int i = 0; i < this->c; ++i)
	{
		for (int j = 0; j < this->layers[i].c; ++j)
		{
			for (int k = 0; k < this->layers[i].neurons[j].c; ++k)
			{
				fwrite(&v[this->layers[i].neurons[j].w[k]], sizeof(double), 1, f);
			}
			fwrite(&v[this->layers[i].neurons[j].b], sizeof(double), 1, f);
		}
	}
	fclose(f);
}

void MLP_LoadParams(struct MLP *this, const char *filename)
{
	FILE *f;
	f = fopen(filename, "rb");
	for (int i = 0; i < this->c; ++i)
	{
		for (int j = 0; j < this->layers[i].c; ++j)
		{
			for (int k = 0; k < this->layers[i].neurons[j].c; ++k)
			{
				fread(&v[this->layers[i].neurons[j].w[k]], sizeof(double), 1, f);
			}
			fread(&v[this->layers[i].neurons[j].b], sizeof(double), 1, f);
		}
	}
	fclose(f);
}

void MLP_Step(struct MLP *this, double step)
{
	for (int i = 0; i < this->c; ++i)
	{
		for (int j = 0; j < this->layers[i].c; ++j)
		{
			for (int k = 0; k < this->layers[i].neurons[j].c; ++k)
			{
				v[this->layers[i].neurons[j].w[k]] -= step * g[this->layers[i].neurons[j].w[k]];
			}
			v[this->layers[i].neurons[j].b] -= step * g[this->layers[i].neurons[j].b];
		}
	}
}

int Loss(int *res, int *pred, int count)
{
	int loss = New(0.0);
	for (int i = 0; i < count; ++i)
	{
		loss = Add(loss, Pow(Sub(res[i], pred[i]), New(2.0)));
	}
	return loss;
}

void Learn(struct MLP *this, double *input, int input_count, double *predic, double step, double error)
{
	int *res = malloc(input_count * sizeof(int));
	int input_size = this->layers[0].neurons[0].c;
	int *in = (int*)input;
	int *pred = (int*)predic;

	for (int i = 0; i < input_count * input_size; ++i)
	{
		in[i] = New(input[i]);
	}

	for (int i = 0; i < input_count; ++i)
	{
		pred[i] = New(predic[i]);
	}

	int initial_units = Count();
	ClearTopoData();
	ClearGrad(initial_units);

	for (int i = 0; i < input_count; ++i)
	{
		res[i] = *MLP(this, in + (i * input_size));
	}
	
	int loss = Loss(res, pred, input_count);
	Topo(loss);

	while (fabs(v[loss]) > error)
	{
		Backward();
		MLP_Step(this, step);
		ClearGrad(Count());
		FreeFrom(initial_units);

		for (int i = 0; i < input_count; ++i)
		{
			res[i] = *MLP(this, in + (i * input_size));
		}
		
		loss = Loss(res, pred, input_count);
	}
	
	printf("Training completed with error margin: %lf\n", fabs(v[loss]));
	for (int i = 0; i < input_count; ++i)
	{
		printf("Prediction:  %lf\n", v[pred[i]]);
		printf("Last result: %lf\n", v[res[i]]);
		printf("---------------------------\n");
	}

	free(res);
	FreeFrom(initial_units);
}

void Test(struct MLP *this, double *input, int input_count, double *result)
{

	int *res = (int*)result;
	int input_size = this->layers[0].neurons[0].c;
	int *in = (int*)input;

	for (int i = 0; i < input_count * input_size; ++i)
	{
		in[i] = New(input[i]);
	}

	int initial_units = Count();
	ClearTopoData();
	ClearGrad(initial_units);

	for (int i = 0; i < input_count; ++i)
	{
		res[i] = *MLP(this, in + (i * input_size));
	}

	for (int i = input_count - 1; i >= 0; --i)
	{
		result[i] = v[res[i]];
	}

	FreeFrom(initial_units);
}