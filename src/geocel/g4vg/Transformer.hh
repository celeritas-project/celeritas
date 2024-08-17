//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/g4vg/Transformer.hh
//---------------------------------------------------------------------------//
#pragma once

#include <G4AffineTransform.hh>
#include <G4RotationMatrix.hh>
#include <G4ThreeVector.hh>
#include <VecGeom/base/Transformation3D.h>

#include "Scaler.hh"

namespace celeritas
{
namespace g4vg
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
    using result_type = vecgeom::Transformation3D;
    //!@}

  public:
    // Construct with a scale
    inline explicit Transformer(Scaler const& convert_scale_);

    //! Convert a translation
    inline result_type operator()(G4ThreeVector const& t) const;

    //! Convert a translation + rotation
    inline result_type
    operator()(G4ThreeVector const& t, G4RotationMatrix const& rot) const;

    //! Convert a translation + optional rotation
    inline result_type
    operator()(G4ThreeVector const& t, G4RotationMatrix const* rot) const;

    //! Convert an affine transform
    inline result_type operator()(G4AffineTransform const& at) const;

  private:
    //// DATA ////

    Scaler const& convert_scale_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a scaling function.
 */
Transformer::Transformer(Scaler const& convert_scale)
    : convert_scale_{convert_scale}
{
}

//---------------------------------------------------------------------------//
/*!
 * Create a transform from a translation.
 */
auto Transformer::operator()(G4ThreeVector const& t) const -> result_type
{
    return {convert_scale_(t[0]), convert_scale_(t[1]), convert_scale_(t[2])};
}

//---------------------------------------------------------------------------//
/*!
 * Create a transform from a translation plus rotation.
 */
auto Transformer::operator()(G4ThreeVector const& t,
                             G4RotationMatrix const& rot) const -> result_type
{
    return {convert_scale_(t[0]),
            convert_scale_(t[1]),
            convert_scale_(t[2]),
            rot.xx(),
            rot.yx(),
            rot.zx(),
            rot.xy(),
            rot.yy(),
            rot.zy(),
            rot.xz(),
            rot.yz(),
            rot.zz()};
}

//---------------------------------------------------------------------------//
/*!
 * Create a transform from a translation plus optional rotation.
 */
auto Transformer::operator()(G4ThreeVector const& t,
                             G4RotationMatrix const* rot) const -> result_type
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
auto Transformer::operator()(G4AffineTransform const& affine) const -> result_type
{
    return (*this)(affine.NetTranslation(), affine.NetRotation());
}

//---------------------------------------------------------------------------//
}  // namespace g4vg
}  // namespace celeritas
