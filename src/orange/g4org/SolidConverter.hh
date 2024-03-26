//----------------------------------*-C++-*----------------------------------//
// Copyright 2023-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/g4org/SolidConverter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <unordered_map>

#include "corecel/Types.hh"
#include "corecel/cont/Array.hh"

class G4VSolid;
class G4BooleanSolid;

namespace celeritas
{
namespace orangeinp
{
class ObjectInterface;
}

namespace g4org
{
//---------------------------------------------------------------------------//
class Scaler;
class Transformer;

//---------------------------------------------------------------------------//
/*!
 * Convert a Geant4 solid to a VecGeom "unplaced volume".
 */
class SolidConverter
{
  public:
    //!@{
    //! \name Type aliases
    using arg_type = G4VSolid const&;
    using result_type = std::shared_ptr<orangeinp::ObjectInterface const>;
    //!@}

  public:
    inline SolidConverter(Scaler const& convert_scale,
                          Transformer const& convert_transform);

    // Return a VecGeom-owned 'unplaced volume'
    result_type operator()(arg_type);

    // Return a sphere with equivalent capacity
    result_type to_sphere(arg_type) const;

  private:
    //// DATA ////

    Scaler const& scale_;
    Transformer const& transform_;
    std::unordered_map<G4VSolid const*, result_type> cache_;

    //// HELPER FUNCTIONS ////

    // Convert a solid that's not in the cache
    result_type convert_impl(arg_type);

    // Conversion functions
    result_type box(arg_type);
    result_type cons(arg_type);
    result_type cuttubs(arg_type);
    result_type ellipsoid(arg_type);
    result_type ellipticalcone(arg_type);
    result_type ellipticaltube(arg_type);
    result_type extrudedsolid(arg_type);
    result_type genericpolycone(arg_type);
    result_type generictrap(arg_type);
    result_type hype(arg_type);
    result_type intersectionsolid(arg_type);
    result_type orb(arg_type);
    result_type para(arg_type);
    result_type paraboloid(arg_type);
    result_type polycone(arg_type);
    result_type polyhedra(arg_type);
    result_type reflectedsolid(arg_type);
    result_type sphere(arg_type);
    result_type subtractionsolid(arg_type);
    result_type tessellatedsolid(arg_type);
    result_type tet(arg_type);
    result_type torus(arg_type);
    result_type trap(arg_type);
    result_type trd(arg_type);
    result_type tubs(arg_type);
    result_type unionsolid(arg_type);

    // Construct bool daughters
    result_type convert_bool_impl(G4BooleanSolid const&);
    // Calculate solid capacity in native celeritas units
    double calc_capacity(G4VSolid const&) const;
};

//---------------------------------------------------------------------------//
/*!
 * Construct with transform helper.
 */
SolidConverter::SolidConverter(Scaler const& convert_scale,
                               Transformer const& convert_transform)
    : scale_(convert_scale), transform_(convert_transform)
{
}

//---------------------------------------------------------------------------//
}  // namespace g4org
}  // namespace celeritas
