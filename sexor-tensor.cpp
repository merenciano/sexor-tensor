#include <cstdlib>
#include <system_error>
#include <vector>
#include <deque>
#include <set>
#include <math.h>

using fp_t = float;

class Neuron;

class Value
{
public:
	static std::deque<Value> graph;
	enum Operation
	{
		OP_NEURON,
		OP_SIGMOID,
		OP_SOFTMAX_LOSS, 
		OP_INPUT,
	};

	Value(fp_t val) : _value(val)
	{
		_neuron = nullptr;
		_grad = 0.0f;
		_op = OP_INPUT;
	}

	Value(fp_t val, Neuron *neu, std::vector<Value*> in, Operation op) : _value(val), _neuron(neu), _in(in), _op(op) {}
	Value(const Value&) = delete;

	void Backward();

	Value &Sigmoid()
	{
		return Value::graph.emplace_back(1.0f / (1.0f + exp(-_value)), nullptr, std::vector<Value*>(1, this), OP_SIGMOID);
	}

	fp_t operator*(fp_t x) const
	{
		return _value * x;
	}
	Neuron *_neuron; //TODO: Layer?
	std::vector<Value*> _in;
	fp_t _value;
	fp_t _grad;
	Operation _op;
};

class Neuron
{
public:
	Neuron() = delete;
	Neuron(const Neuron&) = delete;
	Neuron(size_t weights_count)
	{
		_weights.reserve(weights_count);
		_grads.resize(weights_count, 0.0f);
		_bias = std::rand() / (RAND_MAX / 2.0f) - 1.0f;
		_bgrad = 0.0f;

		for (int i = 0; i < weights_count; ++i)
		{
			_weights.emplace_back(std::rand() / (RAND_MAX / 2.0f) - 1.0f);
		}
	}

	Value* operator()(const std::vector<Value*>& input)
	{
		fp_t v = 0.0f;
		for (int i = 0; i < input.size(); ++i)
		{
			v += (*(input[i]) * _weights[i]) + _bias;
		}
		return &Value::graph.emplace_back(v, this, &input, Value::OP_NEURON);
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

	std::vector<Value*> Process(const std::vector<Value*>& in)
	{
		std::vector<Value*> ret;

		for (auto& n : _neurons)
		{
			ret.emplace_back(&(n(in)->Sigmoid()));
		}

		return ret;
	}

	std::vector<Neuron> _neurons;
};

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

	std::vector<Value*> Process(const std::vector<Value*>& in)
	{
		std::vector<Value*> out = in;
		for (auto& l : _layers)
		{
			out = l.Process(out);
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

Value &SoftmaxLoss(const std::vector<Value*>& score, const std::vector<fp_t> pred)
{
	auto &ret = Value::graph.emplace_back(0.0f, nullptr, score, Value::OP_SOFTMAX_LOSS);
	fp_t sum = 0.0f;
	for (int i = 0; i < score.size(); ++i)
	{
		sum += exp(score[i]->_value);
	}
	
	for (int i = 0; i < pred.size(); ++i)
	{
		auto softmax = exp(score[i]->_value) / sum;
		if (pred[i])
		{
			ret._value = -log(softmax);
			ret._in[i]->_grad = softmax - 1.0f;
		}
		else
		{
			ret._in[i]->_grad = softmax;
		}
	}

	printf("Score: ");
	for (auto s : score)
	{
		printf("%f ,", s->_value);
	}
	printf("\nLoss: %lf\n", ret._value);

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
			_grad += _value * (1.0f - _value);
			break;
		}

		case OP_SOFTMAX_LOSS:
		{
			// Ya he puesto los valores en SoftMax.
			break;
		}
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
	Network net({28*28, 56, 10});

	std::vector<Value*> inputs;
	for (int i = 0; i < 28*28; ++i)
	{
		inputs.emplace_back(new Value(1.0f));
	}

	std::vector<fp_t> pred(10, 0.0f);
	pred[4] = 1.0f;

	for (int i = 0; i < 50; ++i)
	{
		auto out = net.Process(inputs);
		auto &f = SoftmaxLoss(out, pred);
		std::set<Value*> vis;
		std::vector<Value*> topo;
		Topo(&f, vis, topo);
		for (auto& val : topo)
		{
			val->Backward();
		}
		net.Optimize(0.3f);
	}

	return 0; 
}
