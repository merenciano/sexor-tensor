#include <cstdlib>
#include <system_error>
#include <vector>
#include <deque>
#include <set>
#include <stack>
#include <functional>
#include <math.h>

using fp_t = float;

static fp_t GetRand(fp_t min, fp_t max)
{
	return std::rand() / ((fp_t)RAND_MAX / (max - min)) + min;
}

class Neuron;

class Tensor
{
public:

	Tensor(size_t size)
	{
		data.resize(size + 1, 0.0f);
		grad.resize(size + 1, 0.0f);

		//RandomData(-1.0f, 1.0f);
		//ZeroGrad();
	}

	void RandomData(fp_t min, fp_t max)
	{
		for (auto &d : data)
		{
			d = GetRand(min, max);
		}
	}

	void ZeroGrad()
	{
		for (auto &g : grad)
		{
			g = 0.0f;
		}
	}
	std::vector<fp_t> data;
	std::vector<fp_t> grad;
};

using op_t = Tensor* (*)(Tensor*, const std::vector<Tensor*>&);

static Tensor ProcessLayer(Tensor *in, const std::vector<Tensor*> &layer)
{
	Tensor out(layer.size());
	for (int n = 0; n < layer.size(); ++n)
	{
		for (int i = 0; i < in->data.size(); ++i)
		{
			out.data[n] += layer[n]->data[i] * in->data[i]; // w dot.
		}
		out.data[n] += layer[n]->data.back(); // + bias.
	}
	return out;
}

class Net
{
public:
	Net& Layer(size_t neuron_count, size_t input_count = 0)
	{
		std::vector<Tensor> layer(neuron_count, input_count ? input_count : output->data.size() - 1);
		for (auto &n : layer)
		{
			n.RandomData(-1.0f, 1.0f);
		}
		ops.push()
		return *this;
	}

	Tensor *output;
	std::vector<std::vector<Tensor>> layers;
	std::stack<Tensor> data;
	std::stack<op_t> ops;
};

class Value
{
public:
	static std::deque<Value> graph;
	enum Operation
	{
		OP_NEURON,
		OP_SIGMOID,
		OP_SOFTMAX,
		OP_SOFTMAX_LOSS,
		OP_CATEG_CROSS_ENTROPY,
		OP_LOSS,
		OP_INPUT,
	};

	Value(fp_t val) : _value(val)
	{
		_neuron = nullptr;
		_grad = 0.0f;
		_op = OP_INPUT;
	}

	Value(fp_t val, Neuron *neu, std::vector<Value*> in, Operation op) : _value(val), _grad(0.0f), _neuron(neu), _in(in), _op(op) {}
	Value(const Value&) = delete;

	Value(Value &&other)
	{
		_neuron = other._neuron;
		other._neuron = nullptr;
		_in.swap(other._in);
		_value = other._value;
		_grad = other._grad;
		_op = other._op;
	}

	void Backward();

	Value &Sigmoid()
	{
		return Value::graph.emplace_back(1.0f / (1.0f + exp(-_value)), nullptr, std::vector<Value*>(1, this), OP_SIGMOID);
	}

	fp_t operator*(fp_t x) const
	{
		return _value * x;
	}

	fp_t _value;
	fp_t _grad;
	Neuron *_neuron; //TODO: Layer?
	std::vector<Value*> _in;
	Operation _op;
};

std::deque<Value> Value::graph;

class Neuron
{
public:
	Neuron() = delete;
	Neuron(const Neuron&) = delete;
	Neuron(size_t weights_count)
	{
		_weights.reserve(weights_count);
		_grads.resize(weights_count, 0.0f);
		_bias = std::rand() / ((fp_t)RAND_MAX / 2.0f) - 1.0f;
		_bgrad = 0.0f;

		for (int i = 0; i < weights_count; ++i)
		{
			_weights.emplace_back(std::rand() / ((fp_t)RAND_MAX / 2.0f) - 1.0f);
		}
	}

	Neuron(Neuron&& other)
	{
		_weights.swap(other._weights);
		_grads.swap(other._grads);
		_bias = other._bias;
		_bgrad = other._bgrad;
	}

	Value* operator()(const std::vector<Value*>& input)
	{
		fp_t v = 0.0f;
		for (int i = 0; i < input.size(); ++i)
		{
			v += (*(input[i]) * _weights[i]) + _bias;
		}
		return &Value::graph.emplace_back(v, this, input, Value::OP_NEURON);
	}

	void Optimize(fp_t step)
	{
		for (int i = 0; i <  _weights.size(); ++i)
		{
			_weights[i] -= step * _grads[i];
		}
		_bias -= step * _bgrad;
	}

	void ZeroGrad()
	{
		for (auto& g : _grads)
		{
			g = 0.0f;
		}
		_bgrad = 0.0f;
	}

	std::vector<fp_t> _weights;
	std::vector<fp_t> _grads;
	fp_t _bias;
	fp_t _bgrad;
};

class Layer
{
public:
	Layer() = delete;
	Layer(const Layer&) = delete;
	Layer(size_t neuron_count, size_t neuron_size) {
		for (int i = 0; i < neuron_count; ++i)
		{
			_neurons.emplace_back(neuron_size);
		}
	}

	Layer(Layer&& other)
	{
		_neurons.swap(other._neurons);
	}

	std::vector<Value*> Process(const std::vector<Value*>& in)
	{
		std::vector<Value*> ret;

		for (auto& n : _neurons)
		{
			ret.emplace_back(n(in));
		}

		return ret;
	}

	std::vector<Neuron> _neurons;
};

std::vector<Value*> Sigmoid(const std::vector<Value*>& in)
{
	std::vector<Value*> ret;
	for (auto& v : in)
	{
		ret.emplace_back(&(v->Sigmoid()));
	}
	return ret;
}

std::vector<Value*> SoftMax(const std::vector<Value*> &in)
{
	std::vector<Value*> ret;
	fp_t sum = 0.0f;
	for (int i = 0; i < in.size(); ++i)
	{
		auto n = new Neuron(0); // TODO memleak
		n->_bias = (fp_t)i;
		ret.emplace_back(&Value::graph.emplace_back(exp(in[i]->_value), n, in, Value::OP_SOFTMAX));
		sum += ret.back()->_value;
	}

	for (auto val : ret)
	{
		val->_value /= sum;
	}
	return ret;
}

class Network
{
public:
	Network() = delete;
	Network(const Network&) = delete;
	Network(std::vector<size_t> dimension)
	{
		for (int i = 1; i < dimension.size(); ++i)
		{
			_layers.emplace_back(dimension[i], dimension[i - 1]);
		}
	}

	Network(Network &&other)
	{
		_layers.swap(other._layers);
	}

	std::vector<Value*> Process(const std::vector<Value*>& in)
	{
		std::vector<Value*> out = in;
		for (auto& l : _layers)
		{
			out = l.Process(out);
		}
		for (int l = 0; l < _layers.size(); ++l)
		{
			out = _layers[l].Process(out);
			if (l < (_layers.size() - 1))
			{
				out = Sigmoid(out);
			}
			else
			{
				out = SoftMax(out);
			}
		}

		return out;
	}

	void Optimize(fp_t step)
	{
		for (auto &l : _layers)
		{
			for (auto &n : l._neurons)
			{
				n.Optimize(step);
				n.ZeroGrad();
			}
		}
		Value::graph.clear();
	}

	std::vector<Layer> _layers;
};

Value &SoftmaxLoss(const std::vector<std::vector<Value*>>& score, const std::vector<std::vector<fp_t>> pred)
{
	const fp_t eps = 10e-8;
	std::vector<Value*> rets;
	int batch_size = score.size();
	for (int batch = 0; batch < batch_size; ++batch)
	{
		auto ret = rets.emplace_back(&Value::graph.emplace_back(0.0f, nullptr, score[batch], Value::OP_SOFTMAX_LOSS));
		ret->_grad = 1.0f;
		fp_t sum = 0.0f;
		for (int i = 0; i < score[batch].size(); ++i)
		{
			sum += exp(score[batch][i]->_value);
		}
		
		for (int i = 0; i < pred[batch].size(); ++i)
		{
			auto softmax = exp(score[batch][i]->_value) / sum;
			if (pred[batch][i])
			{
				ret->_value += log(softmax) + eps;
				ret->_in[i]->_grad += (softmax - 1.0f) * (log(softmax) + eps);
			}
			else
			{
				ret->_value += eps;
				ret->_in[i]->_grad += softmax * eps;
			}
		}
	}

	fp_t sum = 0.0f;
	for (int i = 0; i < rets.size(); ++i)
	{
		sum += rets[i]->_value;
	}
	auto& ret = Value::graph.emplace_back(sum, nullptr, rets, Value::OP_SOFTMAX_LOSS);
	ret._grad = 1.0f;
	printf("\nLoss: %lf\n", ret._value);
	return ret;
}

Value &Loss(const std::vector<Value*>& score, const std::vector<fp_t> pred)
{
	auto &ret = Value::graph.emplace_back(0.0f, nullptr, score, Value::OP_LOSS);
	for (int i = 0; i < pred.size(); ++i)
	{
		if (pred[i] != 0.0f)
		{
			ret._value = -log(score[i]->_value);
			return ret;
		}
	}
	return ret;
}

Value &CrossEntropy(const std::vector<std::vector<Value*>>& score, const std::vector<std::vector<fp_t>> pred)
{
	const fp_t eps = 10e-8;
	fp_t sum = 0.0f;
	std::vector<Value*> in;
	for (int b = 0; b < pred.size(); ++b)
	{
		for (int i = 0; i < pred[b].size(); ++i)
		{
			sum += pred[b][i] * log(score[b][i]->_value) + eps;
			score[b][i]->_grad = -pred[b][i] * (1.0f / (score[b][i]->_value + eps));
			in.emplace_back(score[b][i]);
		}
	}
	auto &ret = Value::graph.emplace_back(-sum, nullptr, in, Value::OP_CATEG_CROSS_ENTROPY);
	ret._grad = 1.0f;
	return ret;
}

void Value::Backward()
{
	switch(_op)
	{
		case OP_NEURON:
		{
			for (int i = 0; i < _in.size(); ++i)
			{
				_neuron->_grads[i] += _in.at(i)->_value * _grad;
				_in[i]->_grad += _neuron->_weights[i] * _grad;
				_neuron->_bgrad += _grad;
			}
			break;
		}

		case OP_SIGMOID:
		{
			for (auto i : _in)
			{
				i->_grad += _value * (1.0f - _value) * _grad;
			}
			break;
		}
		case OP_SOFTMAX:
		{
			for (int i = 0; i < _in.size(); ++i)
			{
				_in[i]->_grad += ((fp_t)i == _neuron->_bias) ? _value * (1.0f - _value) * _grad : 0.0f;
			}
			break;
		}

		case OP_SOFTMAX_LOSS:
		case OP_CATEG_CROSS_ENTROPY:
		{
			// Ya he puesto los valores en SoftMax.
			break;
		}

		default: break;
	}
}

void Topo(Value *value, std::set<Value*> &visited, std::vector<Value*> &out)
{
	if (visited.count(value))
	{
		return;
	}

	visited.emplace(value);
	for (auto i : value->_in)
	{
		Topo(i, visited, out);
	}

	out.emplace_back(value);
}

int main()
{
	//Network net({28*28, 56, 10});
	Network net({3, 3});

	std::vector<std::vector<Value*>> inputs(5);
	inputs[0].emplace_back(new Value(0.9f));
	inputs[0].emplace_back(new Value(0.0f));
	inputs[0].emplace_back(new Value(0.0f));

	inputs[1].emplace_back(new Value(0.0f));
	inputs[1].emplace_back(new Value(0.9f));
	inputs[1].emplace_back(new Value(0.0f));

	inputs[2].emplace_back(new Value(0.1f));
	inputs[2].emplace_back(new Value(0.9f));
	inputs[2].emplace_back(new Value(0.1f));

	inputs[3].emplace_back(new Value(0.0f));
	inputs[3].emplace_back(new Value(0.0f));
	inputs[3].emplace_back(new Value(0.9f));

	inputs[4].emplace_back(new Value(0.0f));
	inputs[4].emplace_back(new Value(0.0f));
	inputs[4].emplace_back(new Value(0.8f));

	std::vector<std::vector<fp_t>> pred(5);
	pred[0].emplace_back(1.0f);
	pred[0].emplace_back(0.0f);
	pred[0].emplace_back(0.0f);

	pred[1].emplace_back(0.0f);
	pred[1].emplace_back(1.0f);
	pred[1].emplace_back(0.0f);

	pred[2].emplace_back(0.0f);
	pred[2].emplace_back(1.0f);
	pred[2].emplace_back(0.0f);

	pred[3].emplace_back(0.0f);
	pred[3].emplace_back(0.0f);
	pred[3].emplace_back(1.0f);

	pred[4].emplace_back(0.0f);
	pred[4].emplace_back(0.0f);
	pred[4].emplace_back(1.0f);

	for (int i = 0; i < 5000; ++i)
	{
		std::vector<std::vector<Value*>> outs;
		for (int batch = 0; batch < inputs.size(); ++batch)
		{
			outs.emplace_back(net.Process(inputs[batch]));
		}

		//auto &p = SoftmaxLoss(outs, pred);
		auto &l = CrossEntropy(outs, pred);
		printf("  %f  ", l._value);
		std::set<Value*> vis;
		std::vector<Value*> topo;
		Topo(&l, vis, topo);
		for (auto it = topo.rbegin(); it != topo.rend(); ++it)
		{
			(*it)->Backward();
		}
		/*for (auto it = Value::graph.rbegin(); it != Value::graph.rend(); ++it)
		{
			it->Backward();
		}*/
		net.Optimize(0.001f);
	}

	auto out = net.Process({new Value(0.9f) , new Value(0.3f), new Value(0.5f)});
	printf("Esperado: %d, %d, %d. Resultado: %f, %f, %f.\n", 1, 0, 0, out[0]->_value, out[1]->_value, out[2]->_value);
	Value::graph.clear();

	out = net.Process({new Value(0.1f) , new Value(0.2f), new Value(0.3f)});
	printf("Esperado: %d, %d, %d. Resultado: %f, %f, %f.\n", 0, 0, 1, out[0]->_value, out[1]->_value, out[2]->_value);
	Value::graph.clear();

	out = net.Process({new Value(0.7f) , new Value(0.8f), new Value(0.3f)});
	printf("Esperado: %d, %d, %d. Resultado: %f, %f, %f.\n", 0, 1, 0, out[0]->_value, out[1]->_value, out[2]->_value);
	Value::graph.clear();

	out = net.Process({new Value(0.5f) , new Value(0.2f), new Value(0.3f)});
	printf("Esperado: %d, %d, %d. Resultado: %f, %f, %f.\n", 1, 0, 0, out[0]->_value, out[1]->_value, out[2]->_value);
	Value::graph.clear();

	return 0; 
}
