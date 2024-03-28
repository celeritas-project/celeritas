//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Scaler.hh
//---------------------------------------------------------------------------//
#pragma once

#include <utility>
#include <G4ThreeVector.hh>
#include <G4TwoVector.hh>

#include "corecel/Assert.hh"
#include "corecel/cont/Array.hh"
#include "geocel/detail/LengthUnits.hh"

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//
/*!
 * Convert a unit from Geant4 scale to another.
 */
class Scaler
{
  public:
    //! Default scale to CLHEP units (mm)
    Scaler() : scale_{celeritas::lengthunits::millimeter} {}

    //! Scale with an explicit factor, probably for testing
    explicit Scaler(double sc) : scale_{sc} { CELER_EXPECT(scale_ > 0); }

    //! Convert a positional scalar
    double operator()(double val) const { return val * scale_; }

    //! Convert a two-vector
    Array<double, 2> operator()(G4TwoVector const& vec) const
    {
        return this->to<Array<double, 2>>(vec.x(), vec.y());
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
    double scale_;
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
