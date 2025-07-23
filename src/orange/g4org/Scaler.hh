//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 *
 * The input is the length scale of the original input in the new units.
 */
class Scaler
{
  public:
    //!@{
    //! \name Type aliases
    using Real2 = Array<real_type, 2>;
    using Real3 = Array<real_type, 3>;
    //!@}

  public:
    //! Scale with an explicit factor from Geant4 or for testing
    explicit Scaler(double sc) : scale_{sc} { CELER_EXPECT(scale_ > 0); }

    //! Default scale to CLHEP units (mm)
    Scaler() : scale_{celeritas::lengthunits::millimeter} {}

    //! Multiply a value by the scale, converting to Celeritas precision
    real_type operator()(double val) const
    {
        return static_cast<real_type>(val * scale_);
    }

    //! Convert and scale a 2D point
    Real2 operator()(G4TwoVector const& vec) const
    {
        return this->to<Real2>(vec.x(), vec.y());
    }

    //! Convert and scale a 3D point
    Real3 operator()(G4ThreeVector const& vec) const
    {
        return this->to<Real3>(vec.x(), vec.y(), vec.z());
    }

    //! Create an array or other object by scaling each argument
    template<class S, class... Ts>
    S to(Ts&&... args) const
    {
        return S{(*this)(std::forward<Ts>(args))...};
    }

    //! Scaling value in Geant4 precision
    double value() const { return scale_; }

  private:
    double scale_;
};

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
