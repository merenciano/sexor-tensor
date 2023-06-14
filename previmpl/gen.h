#include <stdint.h>
#include <vector>
#include <limits>

namespace gen
{
	/*template<typename Tval = double,
		 typename Tgrad = double,
		 typename Thand = int32_t>*/
	template<typename Tval,
		 typename Tgrad,
		 typename Thand>
	class Unit
	{
		enum Operation : uint8_t
		{
			OP_ADD,
			OP_MUL,
			OP_POW,
			OP_RELU,
			OP_NULL
		};
		static const Thand UNINIT = std::numeric_limits<Thand>::max;
		static std::vector<Tval> values;
		static std::vector<Tgrad> grad;
		static std::vector<std::pair<Thand, Thand>> childs;
		static std::vector<Operation> ops;
		static int next;
	public:
		Unit(Tval value)
		{
			values.emplace_back(value);
			grad.emplace_back(0.0);
			childs.emplace_back({UNINIT, UNINIT});
			ops.emplace_back(OP_NULL);
			_handle = values.size() - 1;
		}

		~Unit() {};

		Unit<Tval, Tgrad, Thand> operator+(const Unit<Tval, Tgrad, Thand>& other)
		{
			return Unit(values[_handle] + values[other._handle]);
		}

		Unit<Tval, Tgrad, Thand> operator-(const Unit<Tval, Tgrad, Thand>& other)
		{
			return values[_handle] - values[other._handle];
		}
	private:
		Thand _handle;
	};
}