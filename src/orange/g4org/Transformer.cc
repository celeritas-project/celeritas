//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/Transformer.cc
//---------------------------------------------------------------------------//
#include "Transformer.hh"

namespace celeritas
{
namespace g4org
{
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
auto Transformer::operator()(G4ThreeVector const& t,
                             G4RotationMatrix const& rot) const
    -> Transformation
{
    return {Transformation::Mat3{Real3{{rot.xx(), rot.yx(), rot.zx()}},
                                 Real3{{rot.xy(), rot.yy(), rot.zy()}},
                                 Real3{{rot.xz(), rot.yz(), rot.zz()}}},
            scale_.to<Real3>(t[0], t[1], t[2])};
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
