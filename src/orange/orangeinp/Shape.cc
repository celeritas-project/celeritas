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
NodeId Shape::build(VolumeBuilder& vb) const
{
    // Set input attributes for surface state
    detail::ConvexSurfaceState css;
    css.transform = &vb.local_transform();
    css.object_name = this->label();
    css.make_face_name = {};  // No prefix for standalone shapes

    // Construct surfaces
    auto sb = ConvexSurfaceBuilder(&vb.unit_builder(), &css);
    this->build_interior(sb);

    // Intersect the given surfaces to create a new CSG node
    auto node_id = vb.insert_region(Label{std::move(css.object_name)},
                                    Joined{op_and, std::move(css.nodes)},
                                    calc_merged_bzone(css));

    return node_id;
}

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

template class ShapeImpl<Box>;
template class ShapeImpl<Cone>;
template class ShapeImpl<Cylinder>;
template class ShapeImpl<Ellipsoid>;
template class ShapeImpl<Prism>;
template class ShapeImpl<Sphere>;

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
