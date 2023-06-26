#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <system_error>
#include <math.h>
#include <vector>
#include <algorithm>

static float GetRand(float min, float max)
{
	return std::rand() / ((float)RAND_MAX / (max - min)) + min;
}

class Tensor
{
public:
	Tensor() {}
	Tensor(size_t size) : data(size, 0.0f), grad(size, 0.0f) {}
	Tensor(const std::vector<float> &input) : data(input), grad(input.size(), 0.0f) {}
	Tensor(std::vector<float> &&input) : data(std::move(input)), grad(input.size(), 0.0f) {}

	void Rand(float min, float max) {
		for (int i = 0; i < data.size(); ++i) data[i] = GetRand(min, max);
	}
	void ZeroGrad() { for (int i = 0; i < grad.size(); ++i) grad[i] = 0.0f; }
	std::vector<float> data;
	std::vector<float> grad;
};

void Neuron(Tensor &n, const Tensor &in, Tensor *out)
{
	float sum = 0.0f;
	for (int i = 0; i < in.data.size(); ++i)
	{
		sum += n.data[i] * in.data[i];
	}
	out->data.emplace_back(sum + n.data.back());
	out->grad.emplace_back(0.0f);
}

Tensor Layer(std::vector<Tensor> &l, const Tensor &in)
{
	Tensor ret;
	ret.data.reserve(l.size() + 1);
	ret.grad.reserve(l.size() + 1);
	for (int i = 0; i < l.size(); ++i)
	{
		Neuron(l[i], in, &ret);
	}
	return ret;
}

void B_Layer(std::vector<Tensor> &l, const Tensor &in, Tensor &prev)
{
	for (int i = 0; i < l.size(); ++i)
	{
		for (int j = 0; j < prev.grad.size(); ++j)
		{
			l[i].grad[j] += prev.data[j] * in.grad[i];
			prev.grad[j] += l[i].data[j] * in.grad[i];
		}
		l[i].grad[prev.grad.size()] = in.grad[i]; // bias.
	}
}

Tensor ReLU(const Tensor &in)
{
	assert(!in.data.empty());
	Tensor ret(in.data.size());
	for (int i = 0; i < ret.data.size(); ++i)
	{
		ret.data[i] = std::max(0.0f, in.data[i]);
	}

	assert(ret.data.size() == in.data.size());
	for (int i = 0; i < ret.data.size(); ++i)
	{
		assert(ret.data[i] >= 0.0f);
	}

	return ret;
}

void B_ReLU(const Tensor &in, Tensor &prev)
{
	for (int i = 0; i < prev.grad.size(); ++i)
	{
		if (in.data[i] > 0.0f)
		{
			prev.grad[i] += in.grad[i];
		}
	}
}

Tensor Linear(const Tensor &in)
{
	return in;
}

void B_Linear(const Tensor &in, Tensor &prev)
{
	assert(in.grad.size() == prev.grad.size());
	for (int i = 0; i < in.grad.size(); ++i)
	{
		assert(prev.grad[i] == 0.0f);
		prev.grad[i] += in.grad[i];
	}
}

enum EActivationFunc
{
	AF_RELU = 0,
	AF_LINEAR,
	AF_COUNT
};

Tensor (*const ActivFunc[AF_COUNT])(const Tensor&) = {
	ReLU,
	Linear
};

void (*const B_ActivFunc[AF_COUNT])(const Tensor&, Tensor&) = {
	B_ReLU,
	B_Linear
};

void PrintSoftmax(Tensor &in)
{
	std::vector<float> softmax(in.data.size());
	float max_val = *std::max_element(in.data.begin(), in.data.end());
	float sum = 0.0f;
	int big = -1;
	float bigf = 0.0f;

	for (int i = 0; i < in.data.size(); i++)
	{
		softmax[i] = expf(in.data[i] - max_val);
		sum += softmax[i];
	}

	printf("Softmax: ");
	for (int i = 0; i < softmax.size(); i++)
	{
		softmax[i] = softmax[i] / sum;
		printf("%f ", softmax[i]);
		if (bigf <= softmax[i])
		{
			big = i;
			bigf = softmax[i];
		}
	}
	printf("\n");
	printf("Label: %d\n", big);
}

class NN
{
public:
	NN() : _activ(AF_RELU), _last_activ(AF_LINEAR) {}
	
	NN& SetDimensions(size_t input_size, std::vector<size_t> dimension)
	{
		for (const auto &d : dimension)
		{
			auto &l = _neurons.emplace_back();
			for (int i = 0; i < d; ++i)
			{
				l.emplace_back(input_size + 1); // +1 por el bias.
				l.back().Rand(-1.0f, 1.0f);
			}
			input_size = d;
		}
		return *this;
	}

	NN& SetActivation(EActivationFunc activ) { _activ = activ; return *this; }
	NN& SetLastActivation(EActivationFunc lactiv) { _last_activ = lactiv; return *this; }

	const Tensor& Predict(const std::vector<float> &in)
	{
		auto &tensor_stack = _op_inputs.emplace_back();
		tensor_stack.emplace_back(in);
		for (int l = 0; l < _neurons.size() - 1; ++l)
		{
			tensor_stack.emplace_back(Layer(_neurons[l], tensor_stack.back()));
			tensor_stack.emplace_back(ActivFunc[_activ](tensor_stack.back()));
		}
		tensor_stack.emplace_back(Layer(_neurons.back(), tensor_stack.back()));
		tensor_stack.emplace_back(ActivFunc[_last_activ](tensor_stack.back()));
		return tensor_stack.back();
	}

	Tensor Loss(const std::vector<float> &pred)
	{
		Tensor ret(1);
		for (int i = 0; i < pred.size(); ++i)
		{
			float diff = _op_inputs[i].back().data[0] - pred[i];
			_op_inputs[i].back().grad[0] = diff * 2.0f;
			ret.data[0] += diff * diff;
		}
		ret.grad[0] = 1.0f;
		return ret;
	}

	Tensor SoftmaxLoss(const std::vector<float> &pred)
	{
		Tensor ret(1);
		ret.data[0] = 0.0f;

		for (int batch = 0; batch < pred.size(); ++batch)
		{
			Tensor &in = _op_inputs[batch].back();
			std::vector<float> softmax(in.data.size());
			float max_val = *std::max_element(in.data.begin(), in.data.end());
			float sum = 0.0f;

			for (int i = 0; i < in.data.size(); i++)
			{
				softmax[i] = expf(in.data[i] - max_val);
				sum += softmax[i];
			}

			for (int i = 0; i < softmax.size(); i++)
			{
				softmax[i] = softmax[i] / sum;

				if (i == (int)pred[batch])
				{
					ret.data[0] += -log(softmax[i] + 1e-8f);
					in.grad[i] = softmax[i] - 1;
				}
				else
				{
					in.grad[i] = softmax[i];
				}
			}
		}

		return ret;
	}

	void Backwards()
	{
		for (auto &tensor_stack : _op_inputs)
		{
			auto tensor_it = tensor_stack.rbegin();
			B_ActivFunc[_last_activ](*tensor_it, *(tensor_it + 1));
			++tensor_it;
			B_Layer(_neurons.back(), *tensor_it, *(tensor_it + 1));
			++tensor_it;
			for (int l = _neurons.size() - 2; l >= 0; --l)
			{
				B_ActivFunc[_activ](*tensor_it, *(tensor_it + 1));
				++tensor_it;
				B_Layer(_neurons[l], *tensor_it, *(tensor_it + 1));
				++tensor_it;
			}
			assert(++tensor_it == tensor_stack.rend());
		}
	}

	void Optimize(float step)
	{
		for (auto &l : _neurons)
		{
			for (auto &n : l)
			{
				for (int i = 0; i < n.data.size(); ++i)
				{
					n.data[i] -= n.grad[i] * step;
				}
				n.ZeroGrad();
			}
		}

		_op_inputs.clear();
	}

	void Train(std::vector<std::vector<float>> input, std::vector<float> pred)
	{
		for (;;)
		{
			for (int i = 0; i < input.size(); ++i)
			{
				Predict(input[i]);
			}
			float loss = SoftmaxLoss(pred).data[0];
			printf("Loss: %f\n", loss);
			if (loss < 1.0f)
			{
				break;
			}
			Backwards();
			Optimize(0.001f);
		}
	}

	void SaveParams(const char *filename)
	{
		FILE *f;
		f = fopen(filename, "wb");

		for (auto &l : _neurons)
		{
			for (auto &n : l)
			{
				for (auto &d : n.data)
				{
					fwrite(&d, sizeof(float), 1, f);
				}
			}
		}
		fclose(f);
	}

	void LoadParams(const char *filename)
	{
		FILE *f;
		f = fopen(filename, "rb");

		for (auto &l : _neurons)
		{
			for (auto &n : l)
			{
				for (auto &d : n.data)
				{
					fread(&d, sizeof(float), 1, f);
				}
			}
		}
		fclose(f);
	}

private:
	std::vector<std::vector<Tensor>> _neurons;
	std::vector<std::vector<Tensor>> _op_inputs;
	EActivationFunc _activ;
	EActivationFunc _last_activ;
};

void LoadTrainingData(std::vector<std::vector<float>> &pix, std::vector<float> &label)
{
	static const int BATCHES = 100;
	FILE *f = fopen("mnist/train-digits", "rb");
	pix.resize(BATCHES);
	label.resize(BATCHES);

	fseek(f, 16, SEEK_SET);
	for (int i = 0; i < BATCHES; ++i)
	{
		pix[i].resize(13 * 13);
		char img[26][26];
		float *img1 = &pix[i][0];
		fseek(f, 28, SEEK_CUR);
		for (int j = 0; j < 26; ++j)
		{
			fseek(f, 2, SEEK_CUR);
			fread(img[j], 26, 1, f);
			fseek(f, 2, SEEK_CUR);
		}
		fseek(f, 28, SEEK_CUR);

		for (int y = 0; y < 26; y += 2)
		{
			for (int x = 0; x < 26; x += 2)
			{
				float val = (float)img[y][x] + (float)img[y][x+1];
				val += (float)img[y+1][x] + (float)img[y+1][x+1];
				val /= 510.0;
				val -= 1.0;
				img1[(y/2) * 13 + (x/2)] = val;
			}
		}
	}
	fclose(f);

	f = fopen("mnist/train-labels", "rb");
	char labs[BATCHES];
	fseek(f, 8, SEEK_SET);
	fread(labs, BATCHES, 1, f);
	for (int i = 0; i < BATCHES; ++i)
	{
		label[i] = (float)labs[i];
	}
	fclose(f);
}

NN nn;

int main()
{
	std::vector<std::vector<float>> pix;
	std::vector<float> label;
	LoadTrainingData(pix, label);

	nn.SetDimensions(13 * 13, {26, 26, 10});
	nn.Train(pix, label);

	nn.SaveParams("model.bin");



	/*std::vector<std::vector<float>> input = {
		{1.0f, 1.0f, 1.0f},
		{1.0f, 0.0f, 0.0f},
		{0.0f, 1.0f, 0.0f},
		{0.0f, 0.0f, 0.0f},
		{1.0f, 1.0f, 0.0f},
		{1.0f, 0.0f, 1.0f},
		{0.0f, 1.0f, 1.0f},
		{0.0f, 0.0f, 1.0f}
	};

	std::vector<float> pred = {
		7.0f, 4.0f, 2.0f, 0.0f, 6.0f, 5.0f, 3.0f, 1.0f
	};

	nn.SetDimensions(3, {3, 3, 8});

	nn.Train(input, pred);

	auto res = nn.Predict({0.0f, 1.0f, 0.0f});
	printf("\n Se espera 2: ");
	PrintSoftmax(res);

	res = nn.Predict({1.0f, 1.0f, 1.0f});
	printf("\n Se espera 7: ");
	PrintSoftmax(res);

	res = nn.Predict({1.0f, 0.0f, 0.0f});
	printf("\n Se espera 4: ");
	PrintSoftmax(res);*/

	return 0;
}