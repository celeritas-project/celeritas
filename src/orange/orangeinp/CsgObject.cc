//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/CsgObject.cc
//---------------------------------------------------------------------------//
#include "CsgObject.hh"

#include <utility>

#include "detail/BoundingZone.hh"
#include "detail/CsgUnitBuilder.hh"
#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Construct with the object to negate and an empty name.
 */
NegatedObject::NegatedObject(SPConstObject obj)
    : NegatedObject{{}, std::move(obj)}
{
}

//---------------------------------------------------------------------------//
/*!
 * Construct with a name and object.
 */
NegatedObject::NegatedObject(std::string&& label, SPConstObject obj)
    : label_{std::move(label)}, obj_{std::move(obj)}
{
    CELER_EXPECT(obj_);
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this object.
 */
NodeId NegatedObject::build(VolumeBuilder& vb) const
{
    // Build object to be negated
    auto daughter_id = obj_->build(vb);
    // Get bounding zone for daughter and negate it
    detail::BoundingZone bz = vb.unit_builder().bounds(daughter_id);
    bz.negate();
    // Add the new region (or anti-region)
    return vb.insert_region(Label{label_}, Negated{daughter_id}, std::move(bz));
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
