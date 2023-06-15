#include <vector>
#include <deque>

using fp_t = float;

class Neuron;

class Value
{
public:
	static std::deque<Value> graph;
	enum Operation
	{
		OP_NEURON,
		OP_LOSS, 
	};

	Value(fp_t val, Neuron *neu, std::vector<Value> *in, Operation op) : _value(val), _neuron(neu), _in(in), _op(op) {}

	void Backwards();

	fp_t operator*(fp_t x) const
	{
		return _value * x;
	}
	Neuron *_neuron;
	std::vector<Value> *_in;
	fp_t _value;
	fp_t _grad;
	Operation _op;
};

class Neuron
{
public:
	Value& operator()(const std::vector<Value>& input)
	{
		fp_t v = 0.0f;
		for (int i = 0; i < input.size(); ++i)
		{
			v += (input[i] * _weights[i]) + _bias;
		}
		return Value::graph.emplace_back(v, this, &input, Value::OP_NEURON);
	}

	void AddGrads(const std::vector<Value>& input, fp_t grad)
	{
	}

	std::vector<fp_t> _weights;
	std::vector<fp_t> _grads;
	fp_t _bias;
	fp_t _bgrad;
};

void Value::Backwards()
{
		switch(_op)
		{
			case OP_NEURON:
			{
				for (int i = 0; i < _in->size(); ++i)
				{
					_neuron->_grads[i] += _in->at(i)._value * _grad;
					(*_in)[i]._grad += _neuron->_weights[i] * _grad;
				}
			}
		}
}
