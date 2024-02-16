//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ConvexSurfaceBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "orange/OrangeTypes.hh"
#include "orange/surf/VariantSurface.hh"

#include "CsgTypes.hh"

namespace celeritas
{
namespace orangeinp
{
namespace detail
{
class CsgUnitBuilder;
struct ConvexSurfaceState;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Build a set of intersecting surfaces within a CSG node.
 *
 * This is the building block for constructing shapes, solids, and so forth.
 * The result of this class is:
 * - CSG nodes describing the inserted surfaces which have been transformed
 *   into the global reference frame
 * - Metadata combining the surfaces names with the object name
 * - Bounding boxes (interior and exterior)
 *
 * \todo Should we require that the user implicitly guarantee that the result
 * is convex, e.g. prohibit quadrics outside "saddle" points? What about a
 * torus, which (unless degenerate) is never convex?
 */
class ConvexSurfaceBuilder
{
  public:
    //! Add a surface with negative quadric value being "inside"
    template<class S>
    void operator()(S const& surf)
    {
        return (*this)(Sense::inside, surf);
    }

    // Add a surface with specified sense (usually inside except for planes)
    template<class S>
    void operator()(Sense sense, S const& surf);

  public:
    // "Private", to be used by testing and detail
    using State = detail::ConvexSurfaceState;
    using UnitBuilder = detail::CsgUnitBuilder;
    using VecNode = std::vector<NodeId>;

    // Construct with persistent unit builder and less persistent state
    ConvexSurfaceBuilder(UnitBuilder* ub, State* state);

  private:
    //// TYPES ////

    // Helper for constructing surfaces
    UnitBuilder* ub_;

    // State being modified by this builder
    State* state_;

    //// HELPER FUNCTION ////

    void insert_transformed(std::string&& ext,
                            Sense sense,
                            VariantSurface const& surf);
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Apply a convex surface builder to an unknown type
void visit(ConvexSurfaceBuilder& csb, Sense sense, VariantSurface const& surf);

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
