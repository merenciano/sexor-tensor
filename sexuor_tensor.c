#include <stdint.h>
#include <stdio.h>
#include <immintrin.h>

typedef float data_t;
typedef float grad_t;
typedef uint32_t Neuron;

typedef struct
{
	data_t **weights;
	grad_t **wgrads;
	Neuron **wprevs;
	uint32_t *wcount;

	data_t *bias;
	grad_t *bgrad;
	Neuron *bprev;
	void *data_pool;
} ST_NeuronPool;

typedef struct
{
	Neuron first_neuron;
	size_t wcount; // For each neuron.
	size_t size;
} ST_Layer;

typedef struct
{
	ST_NeuronPool neurons;
	ST_Layer *layers;
	size_t layer_count;
} ST_Network;

static inline float Normalize(float x, float max)
{
	return x / (max / 2.0f) - 1.0f;
}

static data_t ST_ProcessNeuron(data_t *restrict w, data_t *restrict in, data_t bias, size_t wcount)
{
	__m256 vout = _mm256_setzero_ps();
	for (int i = 0; i < wcount; i+= 8)
	{
		__m256 x = _mm256_load_ps(w + 8 * i);
		__m256 y = _mm256_load_ps(in + 8 * i);
		__m256 b = _mm256_set1_ps(bias);
		__m256 z = _mm256_fmadd_ps(x, y, b);
		vout = _mm256_add_pd(vout, z);
	}

	__m128 high = _mm256_extractf128_ps(vout, 1);
	__m128 r4 = _mm_add_ps(high, _mm256_castps256_ps128(vout));
	__m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
	__m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
	return _mm_cvtss_f32(r1); // TODO: Alguna activation func con input vec?
}

static void ST_CreateNetwork(ST_Network *this, size_t input_count, size_t *layer_sizes, size_t layer_count)
{
	this->layers = malloc(layer_count * sizeof(ST_Layer));

	size_t neuron_count = 0;
	size_t total_wcount = 0;	
	for (int i = 0; i < layer_count; ++i)
	{
		neuron_count += layer_sizes[i];
		this->layers[i].size = layer_sizes[i];
	}


	for (int i = 1; i < layer_count - 1; ++i)
	{
		total_wcount += layer_sizes[i] * layer_sizes[i - 1];
		this->layers[i].wcount = layer_sizes[i - 1];
	}
	total_wcount += input_count * layer_sizes[0];
	this->layers[0].wcount = input_count;

	this->neurons.weights = (data_t**)malloc(neuron_count * sizeof(data_t*));
	this->neurons.wgrads = (grad_t**)malloc(neuron_count * sizeof(grad_t*));
	this->neurons.data_pool = aligned_alloc(256, total_wcount * sizeof(data_t) + total_wcount * sizeof(grad_t)); // AVX2
	this->neurons.bias = malloc(neuron_count * sizeof(data_t));
	this->neurons.bgrad = malloc(neuron_count * sizeof(grad_t));

	for (int i = 0; i < neuron_count; ++i)
	{
		*((data_t*)this->neurons.data_pool + i) = Normalize(rand(), RAND_MAX);
	}

	data_t *walloc = this->neurons.data_pool;
	grad_t *galloc = this->neurons.data_pool + total_wcount * sizeof(data_t);

	for (int i = 0; i < layer_count; ++i)
	{
		this->layers[i].first_neuron = i ? this->layers[i - 1].first_neuron + this->layers[i - 1].wcount : 0;
		
		for (int j = 0; j < this->layers[i].size; ++j)
		{
			this->neurons.weights[this->layers[i].first_neuron + j] = walloc;
			this->neurons.wgrads[this->layers[i].first_neuron + j] = galloc;

			walloc += this->layers[i].wcount;
			galloc += this->layers[i].wcount;
		}
	}

	for (int i = 0; i < layer_count; ++i)
	{
		printf("Layer %d: First neuron: %d, Wcount for each neuron: %ld\n", i, this->layers[i].first_neuron, this->layers[i].wcount);
	}
}

static void ST_ForwardPass(ST_Network *this, data_t *input)
{
	for (int l = 0; l< this->layer_count; ++l)
	{
		for (int n = 0; n < this->layers[l].size; ++n)
		{
			Neuron neu = this->layers[l].first_neuron + n;
			float out = ST_ProcessNeuron(this->neurons.weights[neu], input, this->neurons.bias[neu], this->layers[l].wcount);
		}
	}
}
