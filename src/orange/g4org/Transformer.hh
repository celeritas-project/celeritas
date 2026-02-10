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
#include <G4Transform3D.hh>

#include "geocel/Types.hh"
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
    using Real3 = Array<real_type, 3>;
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

    // Convert a more general transform (includes reflection)
    inline Transformation operator()(G4Transform3D const& tran) const;

    // Convert an affine transform
    inline Transformation operator()(G4AffineTransform const& at) const;

    // Construct dynamically
    inline VariantTransform
    variant(G4ThreeVector const& t, G4RotationMatrix const* rot) const;

  private:
    //// DATA ////

    Scaler const& scale_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
// Convert a ThreeVector
inline Real3 convert_from_geant(G4ThreeVector const& vec);

//---------------------------------------------------------------------------//
// Convert three doubles to a Real3
inline Real3 convert_from_geant(double x, double y, double z);

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
 * Convert a more general transform (including possibly reflection).
 */
Transformation Transformer::operator()(G4Transform3D const& tran) const
{
    SquareMatrixReal3 rot{convert_from_geant(tran.xx(), tran.xy(), tran.xz()),
                          convert_from_geant(tran.yx(), tran.yy(), tran.yz()),
                          convert_from_geant(tran.zx(), tran.zy(), tran.zz())};

    return Transformation{rot,
                          scale_.to<Real3>(tran.dx(), tran.dy(), tran.dz())};
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
 * Create a transform from a translation and optional rotation.
 */
auto Transformer::variant(G4ThreeVector const& trans,
                          G4RotationMatrix const* rot) const -> VariantTransform
{
    if (rot)
    {
        // Do another check for the identity matrix (parameterized volumes
        // often have one)
        if (!rot->isIdentity())
        {
            return (*this)(trans, *rot);
        }
    }
    if (trans[0] != 0 || trans[1] != 0 || trans[2] != 0)
    {
        return (*this)(trans);
    }
    return NoTransformation{};
}

//---------------------------------------------------------------------------//
/*!
 * Convert a ThreeVector.
 */
Real3 convert_from_geant(G4ThreeVector const& vec)
{
    return convert_from_geant(vec[0], vec[1], vec[2]);
}

//---------------------------------------------------------------------------//
/*!
 * Convert three doubles to a Real3.
 */
Real3 convert_from_geant(double x, double y, double z)
{
    return Real3{{static_cast<real_type>(x),
                  static_cast<real_type>(y),
                  static_cast<real_type>(z)}};
}

//---------------------------------------------------------------------------//
/*!
 * Convert a rotation matrix.
 */
SquareMatrixReal3 convert_from_geant(G4RotationMatrix const& rot)
{
    return {convert_from_geant(rot.xx(), rot.xy(), rot.xz()),
            convert_from_geant(rot.yx(), rot.yy(), rot.yz()),
            convert_from_geant(rot.zx(), rot.zy(), rot.zz())};
}

//---------------------------------------------------------------------------//
/*!
 * Get a transposed rotation matrix.
 */
SquareMatrixReal3 transposed_from_geant(G4RotationMatrix const& rot)
{
    return {convert_from_geant(rot.xx(), rot.yx(), rot.zx()),
            convert_from_geant(rot.xy(), rot.yy(), rot.zy()),
            convert_from_geant(rot.xz(), rot.yz(), rot.zz())};
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
