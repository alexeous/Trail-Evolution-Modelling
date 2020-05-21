#pragma once
#include <utility>

template< typename TLambda >
ref class LambdaWrapper {
private:
    TLambda* lambda_;

public:
    LambdaWrapper(TLambda&& lambda) : lambda_(new TLambda(lambda)) { }
    ~LambdaWrapper() {
        this->!LambdaWrapper();
    }
    !LambdaWrapper() {
        delete lambda_;
    }

    template< typename TReturn, typename... TArgs >
    TReturn Call(TArgs... args) {
        return (*lambda_)(args...);
    }
};

//Support for lambdas that use the mutable keyword
template< typename TDelegate, typename TLambda, typename TReturn, typename... TArgs >
TDelegate^ CreateDelegateHelper(
    TLambda&& lambda,
    TReturn(__thiscall TLambda::*)(TArgs...)) {
    LambdaWrapper<TLambda>^ wrapper =
        gcnew LambdaWrapper<TLambda>(std::forward<TLambda>(lambda));
    return gcnew TDelegate(wrapper, &LambdaWrapper<TLambda>::Call<TReturn, TArgs...>);
}

template< typename TDelegate, typename TLambda, typename TReturn, typename... TArgs >
TDelegate^ CreateDelegateHelper(
    TLambda&& lambda,
    TReturn(__clrcall TLambda::*)(TArgs...)) {
    LambdaWrapper<TLambda>^ wrapper =
        gcnew LambdaWrapper<TLambda>(std::forward<TLambda>(lambda));
    return gcnew TDelegate(wrapper, &LambdaWrapper<TLambda>::Call<TReturn, TArgs...>);
}

//Support for lambdas that are not mutable
template< typename TDelegate, typename TLambda, typename TReturn, typename... TArgs >
TDelegate^ CreateDelegateHelper(
    TLambda&& lambda,
    TReturn(__thiscall TLambda::*)(TArgs...) const) {
    LambdaWrapper<TLambda>^ wrapper =
        gcnew LambdaWrapper<TLambda>(std::forward<TLambda>(lambda));
    return gcnew TDelegate(wrapper, &LambdaWrapper<TLambda>::Call<TReturn, TArgs...>);
}

template< typename TDelegate, typename TLambda, typename TReturn, typename... TArgs >
TDelegate^ CreateDelegateHelper(
    TLambda&& lambda,
    TReturn(__clrcall TLambda::*)(TArgs...) const) {
    LambdaWrapper<TLambda>^ wrapper =
        gcnew LambdaWrapper<TLambda>(std::forward<TLambda>(lambda));
    return gcnew TDelegate(wrapper, &LambdaWrapper<TLambda>::Call<TReturn, TArgs...>);
}

template< typename TDelegate, typename TLambda >
TDelegate^ CreateDelegate(TLambda&& lambda) {
    return CreateDelegateHelper<TDelegate>(
        std::forward<TLambda>(lambda),
        &TLambda::operator());
}