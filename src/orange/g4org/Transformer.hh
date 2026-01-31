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
#include "geocel/g4/Convert.hh"
#include "orange/transform/Transformation.hh"
#include "orange/transform/Translation.hh"

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
    inline Transformation operator()(G4RotationMatrix const& g4rm) const;

    // Convert a translation + rotation
    inline Transformation
    operator()(G4ThreeVector const& t, G4RotationMatrix const& g4rm) const;

    // Convert a more general transform (includes reflection)
    inline Transformation operator()(G4Transform3D const& g4tr) const;

    // Convert an affine transform
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
auto Transformer::operator()(G4ThreeVector const& g4t,
                             G4RotationMatrix const& g4rm) const
    -> Transformation
{
    SquareMatrixReal3 mat{Real3(g4rm.xx(), g4rm.xy(), g4rm.xz()),
                          Real3(g4rm.yx(), g4rm.yy(), g4rm.yz()),
                          Real3(g4rm.zx(), g4rm.zy(), g4rm.zz())};

    return Transformation{mat, scale_.to<Real3>(g4t)};
}

//---------------------------------------------------------------------------//
/*!
 * Convert a more general transform (including possibly reflection).
 */
Transformation Transformer::operator()(G4Transform3D const& g4tr) const
{
    SquareMatrixReal3 mat{Real3(g4tr.xx(), g4tr.xy(), g4tr.xz()),
                          Real3(g4tr.yx(), g4tr.yy(), g4tr.yz()),
                          Real3(g4tr.zx(), g4tr.zy(), g4tr.zz())};

    return Transformation{mat,
                          scale_.to<Real3>(g4tr.dx(), g4tr.dy(), g4tr.dz())};
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
    // *Transpose* the rotation matrix
    auto const& g4rm = affine.NetRotation();
    SquareMatrixReal3 mat{Real3(g4rm.xx(), g4rm.yx(), g4rm.zx()),
                          Real3(g4rm.xy(), g4rm.yy(), g4rm.zy()),
                          Real3(g4rm.xz(), g4rm.yz(), g4rm.zz())};

    return Transformation{mat, scale_.to<Real3>(affine.NetTranslation())};
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
