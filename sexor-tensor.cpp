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
		OP_LOSS,
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
	std::vector<Value*> rets;
	int batch_size = score.size();
	for (int batch = 0; batch < batch_size; ++batch)
	{
		auto ret = rets.emplace_back(&Value::graph.emplace_back(0.0f, nullptr, score[batch], Value::OP_SOFTMAX_LOSS));
		ret->_grad = (1.0f / batch_size) * -1.0f;
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
				ret->_value = -log(softmax);
				ret->_in[i]->_grad += (softmax - 1.0f) * ret->_grad;
			}
			else
			{
				//ret->_in[i]->_grad = softmax * ret->_grad;
			}
		}
	}

	fp_t sum = 0.0f;
	for (int i = 0; i < rets.size(); ++i)
	{
		sum += rets[i]->_value;
	}
	auto& ret = Value::graph.emplace_back(sum / batch_size, nullptr, rets, Value::OP_SOFTMAX_LOSS);
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
			_in[0]->_grad += _value * (1.0f - _value) * _grad;
			break;
		}

		case OP_SOFTMAX_LOSS:
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
	Network net({3, 3, 3});

	std::vector<std::vector<Value*>> inputs(5);
	inputs[0].emplace_back(new Value(0.7f));
	inputs[0].emplace_back(new Value(0.3f));
	inputs[0].emplace_back(new Value(0.2f));

	inputs[1].emplace_back(new Value(0.1f));
	inputs[1].emplace_back(new Value(0.6f));
	inputs[1].emplace_back(new Value(0.2f));

	inputs[2].emplace_back(new Value(0.6f));
	inputs[2].emplace_back(new Value(0.9f));
	inputs[2].emplace_back(new Value(0.5f));

	inputs[3].emplace_back(new Value(0.0f));
	inputs[3].emplace_back(new Value(0.1f));
	inputs[3].emplace_back(new Value(0.3f));

	inputs[4].emplace_back(new Value(0.3f));
	inputs[4].emplace_back(new Value(0.3f));
	inputs[4].emplace_back(new Value(0.4f));

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

	for (int i = 0; i < 50; ++i)
	{
		std::vector<std::vector<Value*>> outs;
		for (int batch = 0; batch < inputs.size(); ++batch)
		{
			outs.emplace_back(net.Process(inputs[batch]));
		}

		SoftmaxLoss(outs, pred);
		for (auto it = Value::graph.rbegin(); it != Value::graph.rend(); ++it)
		{
			it->Backward();
		}
		net.Optimize(0.3f);
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
