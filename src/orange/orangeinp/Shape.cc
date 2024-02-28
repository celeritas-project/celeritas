//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Shape.cc
//---------------------------------------------------------------------------//
#include "Shape.hh"

#include "ConvexSurfaceBuilder.hh"
#include "CsgTreeUtils.hh"

#include "detail/ConvexSurfaceState.hh"
#include "detail/CsgUnitBuilder.hh"
#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this shape.
 */
NodeId ShapeBase::build(VolumeBuilder& vb) const
{
    // Set input attributes for surface state
    detail::ConvexSurfaceState css;
    css.transform = &vb.local_transform();
    css.object_name = this->label();
    css.make_face_name = {};  // No prefix for standalone shapes

    // Construct surfaces
    auto sb = ConvexSurfaceBuilder(&vb.unit_builder(), &css);
    this->interior().build(sb);

    // Intersect the given surfaces to create a new CSG node
    return vb.insert_region(Label{std::move(css.object_name)},
                            Joined{op_and, std::move(css.nodes)},
                            calc_merged_bzone(css));
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class Shape<Box>;
template class Shape<Cone>;
template class Shape<Cylinder>;
template class Shape<Ellipsoid>;
template class Shape<Prism>;
template class Shape<Sphere>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
