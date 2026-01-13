//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file orange/orangeinp/Transformed.cc
//---------------------------------------------------------------------------//
#include "Transformed.hh"

#include "corecel/io/JsonPimpl.hh"

#include "ObjectIO.json.hh"

#include "detail/VolumeBuilder.hh"

namespace celeritas
{
namespace orangeinp
{
//---------------------------------------------------------------------------//
/*!
 * Construct a transformed object if nontrivial, or return the original.
 */
auto Transformed::or_object(SPConstObject obj,
                            VariantTransform const& transform) -> SPConstObject
{
    if (std::holds_alternative<NoTransformation>(transform))
    {
        return obj;
    }
    return std::make_shared<Transformed>(std::move(obj), transform);
}

//---------------------------------------------------------------------------//
/*!
 * Construct with daughter object and transform.
 *
 * The input transform should *not* be "no transform". If you don't know
 * whether it is or not, use the \c Transformed::or_object factory function.
 */
Transformed::Transformed(SPConstObject obj, VariantTransform const& transform)
    : obj_{std::move(obj)}, transform_{transform}
{
    CELER_EXPECT(!transform_.valueless_by_exception());
    CELER_EXPECT(!std::holds_alternative<NoTransformation>(transform_));
    CELER_EXPECT(obj_);
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
