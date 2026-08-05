//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/math/detail/Alg1010Impl.hh
//---------------------------------------------------------------------------//
#pragma once

#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/math/Algorithms.hh"

namespace celeritas::detail
{

inline CELER_FUNCTION real_type copysignr(real_type x, real_type y)
{
#if (CELERITAS_REAL_TYPE == CELERITAS_REAL_TYPE_FLOAT)
    return std::copysignf(x, y);
#else
    return std::copysign(x, y);
#endif
}

struct Complex
{
    real_type real;
    real_type imag;

    inline CELER_FUNCTION Complex(real_type r, real_type i)
    {
        this->real = r;
        this->imag = i;
    }

    inline CELER_FUNCTION Complex() : Complex(0.0, 0.0) {}

    inline CELER_FUNCTION Complex sqrt() const
    {
        real_type a = this->real, b = this->imag;
        real_type absv = this->abs();
        return Complex{
            std::sqrt(static_cast<real_type>(0.5) * (absv + a)),
            copysignr(std::sqrt(static_cast<real_type>(0.5) * (absv - a)), b)};
    }

    inline CELER_FUNCTION real_type abs() const
    {
        return hypot<real_type>(this->real, this->imag);
    }

    inline CELER_FUNCTION Complex conj() const
    {
        return Complex{this->real, -this->imag};
    }

    inline CELER_FUNCTION Complex operator+(Complex const& other) const
    {
        return Complex{this->real + other.real, this->imag + other.imag};
    }

    inline CELER_FUNCTION Complex operator-(Complex const& other) const
    {
        return Complex{this->real - other.real, this->imag - other.imag};
    }

    inline CELER_FUNCTION Complex operator+(real_type other) const
    {
        return Complex{this->real + other, this->imag};
    }

    inline CELER_FUNCTION Complex operator-(real_type other) const
    {
        return (*this) + (-other);
    }

    inline CELER_FUNCTION Complex operator*(real_type factor_real) const
    {
        return Complex{this->real * factor_real, this->imag * factor_real};
    }

    inline CELER_FUNCTION Complex operator/(real_type divisor_real) const
    {
        return Complex{this->real / divisor_real, this->imag / divisor_real};
    }

    inline CELER_FUNCTION Complex operator/(Complex divisor_complex) const
    {
        Complex d = divisor_complex;
        real_type d2 = d.real * d.real + d.imag * d.imag;
        Complex c = d.conj();
        return ((*this) * c) / d2;
    }

    inline CELER_FUNCTION Complex operator*(Complex const& other) const
    {
        real_type r1 = this->real, i1 = this->imag, r2 = other.real,
                  i2 = other.real;
        return Complex{r1 * r2 - i1 * i2, i1 * r2 + r1 * i2};
    }

    inline CELER_FUNCTION Complex& operator=(real_type right)
    {
        real = right;
        imag = 0.0;
        return *this;
    }

    inline CELER_FUNCTION Complex& operator*=(real_type right)
    {
        real *= right;
        imag *= right;
        return *this;
    }
};

}  // namespace celeritas::detail
