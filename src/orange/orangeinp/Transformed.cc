//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Transformed.cc
//---------------------------------------------------------------------------//
#include "Transformed.hh"

#include "corecel/io/JsonPimpl.hh"

#include "detail/CsgUnitBuilder.hh"
#include "detail/VolumeBuilder.hh"

#if CELERITAS_USE_JSON
#    include "ObjectIO.json.hh"
#endif

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Construct with daughter object and transform.
 *
 * The input transform should *not* be "no transform" because that would be
 * silly.
 *
 * TODO: require that the input is simplified as well?
 */
Transformed::Transformed(SPConstObject obj, VariantTransform&& transform)
    : obj_{std::move(obj)}, transform_{std::move(transform)}
{
    CELER_EXPECT(obj_);
    CELER_EXPECT(!transform_.valueless_by_exception());
    CELER_EXPECT(!std::holds_alternative<NoTransformation>(transform_));
}

//---------------------------------------------------------------------------//
/*!
 * Construct a volume from this transformed shape.
 */
NodeId Transformed::build(VolumeBuilder& vb) const
{
    // Apply the transform through the life of this object
    auto scoped_transform = vb.make_scoped_transform(transform_);
    return obj_->build(vb);
}

//---------------------------------------------------------------------------//
/*!
 * Output to JSON.
 */
void Transformed::output(JsonPimpl* j) const
{
    to_json_pimpl(j, *this);
}

//---------------------------------------------------------------------------//
}  // namespace orangeinp
}  // namespace celeritas
