//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
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
 * Return an ORANGE transformation from a Geant4 transformation.
 *
 * In Geant4, "object" or "direct" transform refers to daughter-to-parent, how
 * to place the daughter object in the parent. The "frame" transform (raw \c
 * GetTransform or the \c fPtrTransform object) is how to transform from parent
 * to daughter and is the inverse of that transform.
 *
 * Even though the affine transform's matrix has a \c operator() which does a
 * matrix-vector multiply (aka \c gemv), this is *not* the same as the affine
 * transform's rotation, which applies the *inverse* of the stored matrix.
 *
 * All ORANGE/Celeritas transforms are "daughter to parent". The transforms
 * returned from this function \em must be daughter-to-parent!
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
    inline Translation operator()(G4ThreeVector const& t) const;

    // Convert a pure rotation
    inline Transformation operator()(G4RotationMatrix const& rot) const;

    // Convert a translation + rotation
    inline Transformation
    operator()(G4ThreeVector const& t, G4RotationMatrix const& rot) const;

    // Convert a translation + optional rotation
    inline VariantTransform
    operator()(G4ThreeVector const& t, G4RotationMatrix const* rot) const;

    // Convert an affine transform
    inline Transformation operator()(G4AffineTransform const& at) const;

  private:
    //// DATA ////

    Scaler const& scale_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Convert a rotation matrix
inline SquareMatrixReal3 convert_from_geant(G4RotationMatrix const& rot);

//---------------------------------------------------------------------------//
// Get the transpose/inverse of a rotation matrix
inline SquareMatrixReal3 transposed_from_geant(G4RotationMatrix const& rot);

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Construct with a scaling function.
 */
Transformer::Transformer(Scaler const& scale) : scale_{scale} {}

//---------------------------------------------------------------------------//
/*!
 * Create a transform from a translation.
 */
auto Transformer::operator()(G4ThreeVector const& t) const -> Translation
{
    return Translation{scale_.to<Real3>(t[0], t[1], t[2])};
}

//---------------------------------------------------------------------------//
/*!
 * Create a transform from a translation plus rotation.
 */
auto Transformer::operator()(G4ThreeVector const& trans,
                             G4RotationMatrix const& rot) const
    -> Transformation
{
    return Transformation{convert_from_geant(rot), scale_.to<Real3>(trans)};
}

//---------------------------------------------------------------------------//
/*!
 * Create a transform from an affine transform.
 *
 * The affine transform's stored rotation matrix is \em inverted!
 */
auto Transformer::operator()(G4AffineTransform const& affine) const
    -> Transformation
{
    return Transformation{transposed_from_geant(affine.NetRotation()),
                          scale_.to<Real3>(affine.NetTranslation())};
}

//---------------------------------------------------------------------------//
/*!
 * Convert a rotation matrix.
 */
SquareMatrixReal3 convert_from_geant(G4RotationMatrix const& rot)
{
    return {Real3{{rot.xx(), rot.xy(), rot.xz()}},
            Real3{{rot.yx(), rot.yy(), rot.yz()}},
            Real3{{rot.zx(), rot.zy(), rot.zz()}}};
}

//---------------------------------------------------------------------------//
/*!
 * Get a transposed rotation matrix.
 */
SquareMatrixReal3 transposed_from_geant(G4RotationMatrix const& rot)
{
    return {Real3{{rot.xx(), rot.yx(), rot.zx()}},
            Real3{{rot.xy(), rot.yy(), rot.zy()}},
            Real3{{rot.xz(), rot.yz(), rot.zz()}}};
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
