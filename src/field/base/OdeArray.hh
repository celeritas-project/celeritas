//----------------------------------*-C++-*----------------------------------//
// Copyright 2020 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file OdePointers.hh
//---------------------------------------------------------------------------//
#pragma once

#include "base/Array.hh"
#include "base/Range.hh"
#include "base/Types.hh"

#include <cmath>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * A state array of the equation of motion: pos = v_[0-2], mom = v_[3-5]
 *
 * XXX TODO: extend this to include time and spin or to template with a
 * variable ode_dim and array type
 */

class OdeArray
{
    static constexpr size_type ode_dim = 6;
    using ode_type                     = Array<real_type, ode_dim>;

  private:
    ode_type v_;

  public:
    // Constructors
    CELER_FUNCTION
    OdeArray()
    {
        for (auto i : range(ode_dim))
            v_[i] = 0;
    }

    // Copy constructor
    CELER_FUNCTION
    OdeArray(OdeArray const& v)
    {
        for (auto i : range(ode_dim))
            v_[i] = v[i];
    }

    // Assignment operator
    CELER_FUNCTION
    OdeArray operator=(OdeArray const& v)
    {
        for (auto i : range(ode_dim))
            v_[i] = v[i];
        return *this;
    }

    // Access by an index
    CELER_FUNCTION
    real_type& operator[](const size_type i)
    {
        CELER_ASSERT(i < ode_dim);
        return v_[i];
    }

    CELER_FUNCTION
    real_type const& operator[](const size_type i) const
    {
        CELER_ASSERT(i < ode_dim);
        return v_[i];
    }

    CELER_FUNCTION
    ode_type get() const { return v_; }

    // Inplace binary operators
#define INPLACE_BINARY_OP(OPERATOR)                  \
    CELER_FUNCTION                                   \
    OdeArray& operator OPERATOR(const OdeArray& rhs) \
    {                                                \
        for (auto i : range(ode_dim))                \
            v_[i] OPERATOR rhs[i];                   \
        return *this;                                \
    }
    INPLACE_BINARY_OP(+=)
    INPLACE_BINARY_OP(-=)
    INPLACE_BINARY_OP(*=)
    INPLACE_BINARY_OP(/=)
#undef INPLACE_BINARY_OP

    // Inplace binary operators
#define INPLACE_SCALAR_OP(OPERATOR)                 \
    CELER_FUNCTION                                  \
    OdeArray& operator OPERATOR(const real_type& c) \
    {                                               \
        for (auto i : range(ode_dim))               \
            v_[i] OPERATOR c;                       \
        return *this;                               \
    }
    INPLACE_SCALAR_OP(*=)
    INPLACE_SCALAR_OP(/=)
#undef INPLACE_SCALAR_OP

    // Derived accessors
    CELER_FUNCTION
    const Real3 position() const { return {v_[0], v_[1], v_[2]}; }

    CELER_FUNCTION
    const Real3 momentum() const { return {v_[3], v_[4], v_[5]}; }

    // Setter
    CELER_FUNCTION
    void position(const Real3 pos)
    {
        for (auto i : range(3))
            v_[i] = pos[i];
    }

    CELER_FUNCTION
    void momentum(const Real3 mon)
    {
        for (auto i : range(3))
            v_[i + 3] = mon[i];
    }

    CELER_FUNCTION
    real_type position_square() const
    {
        return v_[0] * v_[0] + v_[1] * v_[1] + v_[2] * v_[2];
    }

    CELER_FUNCTION
    real_type distance_square(const OdeArray& y) const
    {
        return (v_[0] - y[0]) * (v_[0] - y[0]) + (v_[1] - y[1]) * (v_[1] - y[1])
               + (v_[2] - y[2]) * (v_[2] - y[2]);
    }

    CELER_FUNCTION
    real_type distance_closest(const OdeArray& y1, const OdeArray& y2) const
    {
        real_type d2y1 = (*this).distance_square(y1);
        real_type d2y2 = (*this).distance_square(y2);
        return sqrt(d2y1 * d2y2 / (d2y1 + d2y2));
    }

    CELER_FUNCTION
    real_type momentum_square() const
    {
        return v_[3] * v_[3] + v_[4] * v_[4] + v_[5] * v_[5];
    }

    CELER_FUNCTION
    real_type momentum_mag() const
    {
        return sqrt(v_[3] * v_[3] + v_[4] * v_[4] + v_[5] * v_[5]);
    }

    CELER_FUNCTION
    real_type momentum_inv() const
    {
        real_type mom = momentum_mag();
        CELER_ASSERT(mom > 0);
        return 1.0 / mom;
    }
};

#define ODE_BINARY_OP(OPERATOR, ASSIGNMENT)                              \
    CELER_FUNCTION                                                       \
    OdeArray operator OPERATOR(const OdeArray& lhs, const OdeArray& rhs) \
    {                                                                    \
        OdeArray result(lhs);                                            \
        result ASSIGNMENT rhs;                                           \
        return result;                                                   \
    }
ODE_BINARY_OP(+, +=)
ODE_BINARY_OP(-, -=)
ODE_BINARY_OP(*, *=)
ODE_BINARY_OP(/, /=)
#undef ODE_BINARY_OP

#define ODE_SCALAR_OP(OPERATOR, ASSIGNMENT)                              \
    CELER_FUNCTION                                                       \
    OdeArray operator OPERATOR(OdeArray const& lhs, const real_type rhs) \
    {                                                                    \
        OdeArray result(lhs);                                            \
        result ASSIGNMENT rhs;                                           \
        return result;                                                   \
    }                                                                    \
    CELER_FUNCTION                                                       \
    OdeArray operator OPERATOR(const real_type lhs, OdeArray const& rhs) \
    {                                                                    \
        OdeArray result(rhs);                                            \
        result ASSIGNMENT lhs;                                           \
        return result;                                                   \
    }
ODE_SCALAR_OP(*, *=)
ODE_SCALAR_OP(/, /=)
#undef ODE_SCALAR_OP

OdeArray operator-(OdeArray const& rhs)
{
    return -1.0 * rhs;
}

//---------------------------------------------------------------------------//
} // namespace celeritas
