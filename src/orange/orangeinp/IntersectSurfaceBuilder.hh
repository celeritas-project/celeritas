//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/IntersectSurfaceBuilder.hh
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
struct IntersectSurfaceState;
}  // namespace detail

//---------------------------------------------------------------------------//
/*!
 * Build a region of intersecting surfaces as a CSG node.
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
 */
class IntersectSurfaceBuilder
{
  public:
    //!@{
    //! \name Type aliases
    using Tol = Tolerance<>;
    //!@}

  public:
    // Get the construction tolerance
    Tol const& tol() const;

    // Add a surface with negative quadric value being "inside"
    template<class S>
    inline void operator()(S const& surf);

    // Add a surface with specified sense (usually inside except for planes)
    template<class S>
    void operator()(Sense sense, S const& surf);

    // Add a surface with specified sense and explicit face name
    template<class S>
    void operator()(Sense sense, S const& surf, std::string&& face_name);

    // Promise that the resulting region is inside/outside this bbox
    inline void operator()(Sense sense, BBox const& bbox);

  public:
    // "Private", to be used by testing and detail
    using State = detail::IntersectSurfaceState;
    using UnitBuilder = detail::CsgUnitBuilder;
    using VecNode = std::vector<NodeId>;

    // Construct with persistent unit builder and less persistent state
    IntersectSurfaceBuilder(UnitBuilder* ub, State* state);

  private:
    //// TYPES ////

    // Helper for constructing surfaces
    UnitBuilder* ub_;

    // State being modified by this builder
    State* state_;

    //// HELPER FUNCTION ////

    void insert_transformed(Sense sense,
                            VariantSurface const& surf,
                            std::string&& ext);
    void shrink_exterior(BBox const& bbox);
    void grow_interior(BBox const& bbox);
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//

// Apply a surface builder to an unknown type
void visit(IntersectSurfaceBuilder& csb,
           Sense sense,
           VariantSurface const& surf);

//---------------------------------------------------------------------------//
// INLINE FUNCTION DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Add a surface with negative quadric value being "inside".
 */
template<class S>
void IntersectSurfaceBuilder::operator()(S const& surf)
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
void IntersectSurfaceBuilder::operator()(Sense sense, BBox const& bbox)
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
