//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Scaler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <G4TwoVector.hh>

#include "corecel/cont/Array.hh"
#include "geocel/detail/LengthUnits.hh"

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//
/*!
 * Convert a unit from Geant4 scale to another.
 *
 * Currently the scale is hardcoded as mm (i.e., CLHEP units) but could easily
 * be a class attribute.
 */
class Scaler
{
  public:
    //!@{
    //! \name Type aliases
    using Real3 = Array<double, 3>;
    //!@}

  public:
    //! Convert a positional scalar
    double operator()(double val) const { return val * scale_; }

    //! Convert a two-vector
    std::pair<double, double> operator()(G4TwoVector const& vec) const
    {
        return this->to<std::pair<double, double>>(vec.x(), vec.y());
    }

    //! Convert a three-vector
    Array<double, 3> operator()(G4ThreeVector const& vec) const
    {
        return this->to<Array<double, 3>>(vec.x(), vec.y(), vec.z());
    }

    //! Create an array
    template<class S, class... Ts>
    S to(Ts&&... args) const
    {
        return S{(*this)(std::forward<Ts>(args))...};
    }

  private:
    inline static constexpr double scale_
        = ::celeritas::lengthunits::millimeter;
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
