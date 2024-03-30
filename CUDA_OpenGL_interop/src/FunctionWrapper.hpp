#include <memory>

template<typename T>
class Function;

template<typename Ret, typename... Param>
class Function<Ret(Param...)>
{
public:
    Function(Ret(*f)(Param...))
        : callable{ std::make_unique<callable_impl<Ret(*)(Param...)>>(f) } {}

    template<typename FunctionObject>
    Function(FunctionObject fo)
        : callable{ std::make_unique<callable_impl<FunctionObject>>(std::move(fo)) } {}

    Ret operator()(Param... param)
    {
        return callable->call(param...);
    }
private:
    struct callable_interface
    {
        virtual Ret call(Param...) = 0;
        virtual ~callable_interface() = default;
    };

    template <typename Callable>
    struct callable_impl : callable_interface
    {
        callable_impl(Callable c) : callable{ std::move(c) } {}
        Ret call(Param... param) { return callable(param...); }
        Callable callable;
    };

    std::unique_ptr<callable_interface> callable;
};