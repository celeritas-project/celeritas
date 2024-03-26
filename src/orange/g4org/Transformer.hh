//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Transformer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4AffineTransform.hh>
#include <G4RotationMatrix.hh>
#include <G4ThreeVector.hh>

#include "orange/transform/VariantTransform.hh"

#include "Scaler.hh"

namespace celeritas
{
namespace g4org
{
//---------------------------------------------------------------------------//
/*!
 * Return a VecGeom transformation from a Geant4 transformation.
 */
class Transformer
{
  public:
    //!@{
    //! \name Type aliases
    using Real3 = Array<double, 3>;
    //!@}

  public:
    // Construct with a scale
    inline explicit Transformer(Scaler const& scale);

    // Convert a translation
    Translation operator()(G4ThreeVector const& t) const;

    // Convert a translation + rotation
    Transformation
    operator()(G4ThreeVector const& t, G4RotationMatrix const& rot) const;

    //! Convert a translation + optional rotation
    inline VariantTransform
    operator()(G4ThreeVector const& t, G4RotationMatrix const* rot) const;

    //! Convert an affine transform
    inline Transformation operator()(G4AffineTransform const& at) const;

  private:
    //// DATA ////

    Scaler const& scale_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a scaling function.
 */
Transformer::Transformer(Scaler const& scale) : scale_{scale} {}

//---------------------------------------------------------------------------//
/*!
 * Create a transform from a translation plus optional rotation.
 */
auto Transformer::operator()(G4ThreeVector const& t,
                             G4RotationMatrix const* rot) const
    -> VariantTransform
{
    if (rot)
    {
        return (*this)(t, *rot);
    }
    else
    {
        return (*this)(t);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Create a transform from an affine transform.
 */
auto Transformer::operator()(G4AffineTransform const& affine) const
    -> Transformation
{
    return (*this)(affine.NetTranslation(), affine.NetRotation());
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
