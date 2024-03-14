//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/ConvexSurfaceBuilder.hh
//---------------------------------------------------------------------------//
#pragma once

#include <vector>

#include "geocel/BoundingBox.hh"
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
 * Internally, this class uses:
 * - \c SurfaceClipper to update bounding boxes for closed quadric surfaces
 *   (axis aligned cylinders, spheres, planes)
 * - \c apply_transform (which calls \c detail::SurfaceTransformer and its
 *   siblings) to generate new surfaces based on the local transformed
 *   coordinate system
 * - \c RecursiveSimplifier to take transformed or user-input surfaces and
 *   reduce them to more compact quadric representations
 *
 * \todo Should we require that the user implicitly guarantee that the result
 * is convex, e.g. prohibit quadrics outside "saddle" points? What about a
 * torus, which (unless degenerate) is never convex? Or should we just require
 * that "exiting a surface exits the region"? (Think about application to OR-ed
 * combinations for safety calculation.)
 */
class ConvexSurfaceBuilder
{
  public:
    // Add a surface with negative quadric value being "inside"
    template<class S>
    inline void operator()(S const& surf);

    // Add a surface with specified sense (usually inside except for planes)
    template<class S>
    void operator()(Sense sense, S const& surf);

    // Promise that the convex surface is inside/outside this bbox
    inline void operator()(Sense sense, BBox const& bbox);

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
    void shrink_exterior(BBox const& bbox);
    void grow_interior(BBox const& bbox);
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Apply a convex surface builder to an unknown type
void visit(ConvexSurfaceBuilder& csb, Sense sense, VariantSurface const& surf);

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Add a surface with negative quadric value being "inside".
 */
template<class S>
void ConvexSurfaceBuilder::operator()(S const& surf)
{
    return (*this)(Sense::inside, surf);
}

//---------------------------------------------------------------------------//
/*!
 * Promise that all bounding surfaces are inside/outside this bbox.
 *
 * "inside" will shrink the exterior bbox, and "outside" will grow the interior
 * bbox. All bounding surfaces within the region must be *inside* the exterior
 * region and *outside* the interior region.
 */
void ConvexSurfaceBuilder::operator()(Sense sense, BBox const& bbox)
{
    if (sense == Sense::inside)
    {
        this->shrink_exterior(bbox);
    }
    else
    {
        this->grow_interior(bbox);
    }
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
